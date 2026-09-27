// Wires the world, the brain, the 3D view and the chat UI together.

import { OllamaBackend, WEBLLM_MODEL, WebLLMBackend, type Backend } from "./brain";
import { ScratchBackend, fetchBrain } from "./scratch";
import { DreamPeek } from "./dream";
import { Errands, STORY_LENGTH, type Errand } from "./errands";
import { Game, type GameEvent } from "./game";
import { Memory, localStore, type Store } from "./memory";
import { View } from "./render";
import { makeWorld, objById, restoreWorld, tick } from "./world";

// ---------- settings & saves ----------

// Only one tab plays at a time: two would each overwrite the other's saves. The newest tab wins; older
// ones stop saving and step aside (see the bottom of this file).
let activeTab = true;
const whileActive = (s: Store): Store => ({ load: () => s.load(), save: (d) => { if (activeTab) s.save(d); } });

const BRAINS = ["own", "webllm", "ollama"] as const;
type Brain = (typeof BRAINS)[number];

interface Settings {
  name: string;
  backend: Brain;
  /** Which of its own brains: "" = the one it hatched with, "genN" = a dream's (served by serve.mjs). */
  ownBrain: string;
  ollamaUrl: string;
  ollamaModel: string;
  autonomy: boolean;
  thoughts: boolean;
  useMemory: boolean;
}

const DEFAULTS: Settings = {
  name: "Blobb",
  backend: "own",
  ownBrain: "",
  ollamaUrl: "http://localhost:11434",
  ollamaModel: "qwen2.5:0.5b",
  autonomy: true,
  thoughts: true,
  useMemory: true,
};

function read(key: string): unknown {
  try { const s = localStorage.getItem(key); return s ? JSON.parse(s) : null; } catch { return null; }
}
function write(key: string, value: unknown): void {
  if (!activeTab) return;
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage blocked: play without saving */ }
}

/** Saved settings, keeping only keys that still exist and still have the right type. */
function loadSettings(): Settings {
  const saved = read("blobb.settings");
  const out: Settings = { ...DEFAULTS };
  if (typeof saved !== "object" || saved === null) return out;
  // Saves from before it had its own brain stored the old default (the 350 MB LLM) as if it were a choice:
  // start those on its own brain. The LLM is still one click away in ⚙.
  const fromBeforeOwnBrain = !("ownBrain" in saved);
  for (const [k, v] of Object.entries(saved as Record<string, unknown>)) {
    if (!(k in DEFAULTS) || typeof v !== typeof DEFAULTS[k as keyof Settings]) continue;
    if (k === "backend" && !(BRAINS as readonly unknown[]).includes(v)) continue;
    if (k === "ownBrain" && !/^(gen\d+)?$/.test(String(v))) continue;
    Object.assign(out, { [k]: v });
  }
  if (!out.name.trim()) out.name = DEFAULTS.name;
  if (fromBeforeOwnBrain) out.backend = DEFAULTS.backend;
  return out;
}
const settings = loadSettings();
const saveSettings = () => { write("blobb.settings", settings); };

const saveWorld = () => { write("blobb.world", { ...world, savedAt: Date.now() }); };

// ---------- DOM ----------

// eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};
const canvas = $<HTMLCanvasElement>("scene");
const log = $("log");
const bubble = $("bubble");
const thinking = $("thinking");

function addLog(cls: string, text: string): void {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.textContent = text;
  log.append(el);
  while (log.children.length > 80) log.firstElementChild?.remove();
  log.scrollTop = log.scrollHeight;
}

let bubbleUntil = 0;
function speak(text: string): void {
  if (!text.trim()) return;
  bubble.textContent = text;
  bubble.hidden = false;
  bubbleUntil = performance.now() + 2500 + text.length * 60;
}

// ---------- game ----------

// Time kept passing while you were away (gently - see catchUp).
let world = restoreWorld(read("blobb.world"), Date.now());
const memory = new Memory(whileActive(localStore()));
const view = new View(canvas, world);

function onEvent(e: GameEvent): void {
  switch (e.t) {
    case "owner": addLog("owner", e.text); break;
    case "thinking": thinking.hidden = !e.on; break;
    case "decision":
      if (e.d.thought) addLog("thought", e.d.thought);
      addLog("plan", `plan: ${e.expanded || "(nothing)"}`);
      if (e.d.say) { addLog("blob", `${settings.name}: ${e.d.say}`); speak(e.d.say); }
      break;
    case "step": addLog(`step ${e.ok ? "ok" : "fail"}`, e.msg); break;
    case "learned": addLog("learned", `💡 ${e.text}`); break;
    case "system": addLog("system", e.text); break;
    case "attempt": break;
    case "errand":
      addLog("errand", `✅ ${e.done.from === "yourself" ? "Done" : `Done for ${e.done.from}`}: ${e.done.thanks}`);
      addLog("system", `🕊️ A pigeon lands with a letter from ${e.next.from}.`);
      showLetter(e.next, window.innerWidth >= 600);
      break;
  }
}

/** `open`: unfold it. On a phone letters arrive folded (with a pulse) so they don't push the chat off-screen. */
function showLetter(e: Errand, open: boolean): void {
  $("letterFrom").textContent = e.from;
  $("letterText").textContent = e.letter;
  $("letterGrumble").textContent = e.grumble;
  const card = $<HTMLDetailsElement>("letter");
  card.open = open;
  card.classList.toggle("new", !open);
  renderSuggestions();
}

const game = new Game(world, memory, null, (a) => view.play(a), onEvent);
game.opts = { name: settings.name, autonomy: settings.autonomy, useMemory: settings.useMemory };
const errands = new Errands(whileActive(localStore("blobb.errands.v1")));
game.errands = errands;

// ---------- brain ----------

const brainStatus = $("brainStatus");
const loader = $("loader");

function makeBackend(): Backend {
  if (settings.backend === "own") {
    const gen = settings.ownBrain;
    return new ScratchBackend(gen ? `Own brain · ${gen}` : "Own brain", fetchBrain(gen ? `brains/${gen}/` : "brain/"), 0.7,
      () => new Worker(new URL("./scratch-worker.js", import.meta.url), { type: "module" }));
  }
  return settings.backend === "ollama"
    ? new OllamaBackend(settings.ollamaUrl.replace(/\/$/, ""), settings.ollamaModel, {
      get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
      set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* not remembered: tunes again next time */ } },
    })
    : new WebLLMBackend(WEBLLM_MODEL);
}

async function loadBrain(): Promise<void> {
  const backend = makeBackend();
  game.brain = null;
  loader.hidden = false;
  $("loaderTitle").textContent = `Summoning ${settings.name}…`;
  $("loaderGo").hidden = true;
  const bar = $<HTMLProgressElement>("loaderBar");
  bar.hidden = false;
  try {
    await backend.init((p, text) => { bar.value = p; $("loaderText").textContent = text; });
    game.brain = backend;
    brainStatus.textContent = `🧠 ${backend.label}`;
    loader.hidden = true;
    addLog("system", `✨ The spellbook glows and ${settings.name} plops out of another dimension. Tell it what to do!`);
  } catch (e) {
    // A dreamed brain lives next to the trainer; opened elsewhere (or deleted), wake the one it hatched with.
    if (settings.backend === "own" && settings.ownBrain) {
      addLog("system", `${settings.name}'s ${settings.ownBrain} brain isn't here, so it wakes with the one it hatched with.`);
      settings.ownBrain = "";
      saveSettings();
      await loadBrain();
      return;
    }
    $("loaderTitle").textContent = "The summoning fizzled";
    $("loaderText").textContent = e instanceof Error ? e.message : String(e);
    $("loaderGo").hidden = false;
    $("loaderGo").textContent = "Try again";
    bar.hidden = true;
  }
}

async function maybeAutoLoad(): Promise<void> {
  // Ollama is local and cheap to connect to; WebLLM is a big download, so ask first unless it's cached.
  if (settings.backend === "webllm") {
    let cached: boolean;
    try {
      const webllm = await import("@mlc-ai/web-llm");
      cached = await webllm.hasModelInCache(WEBLLM_MODEL);
    } catch { cached = false; }
    if (!cached) {
      loader.hidden = false;
      $("loaderTitle").textContent = `Summon ${settings.name}`;
      $("loaderText").textContent =
        "The spell runs right here in your browser, on your GPU. The first summoning downloads about 350 MB " +
        "(cached after that) - use Wi-Fi.";
      $("loaderBar").hidden = true;
      $("loaderGo").hidden = false;
      return;
    }
  }
  await loadBrain();
}

$("loaderGo").addEventListener("click", () => { void loadBrain(); });
$("loaderSettings").addEventListener("click", () => { openSettings(); });

// ---------- chat & feedback ----------

$("letter").addEventListener("toggle", () => { $("letter").classList.remove("new"); });

$<HTMLFormElement>("chat").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const input = $<HTMLInputElement>("msg");
  if (!game.brain) { addLog("system", `Summon ${settings.name} first (⚙).`); return; }
  game.say(input.value);
  input.value = "";
});

const SUGGESTIONS = [
  "you look hungry, eat something", "pick up the stick", "deliver it to the tower", "get the berry up high",
  "go into the walled garden", "cross the gap", "get fruit from the tree", "rest a bit",
];
/** Chips under the log: the current errand's hint first, then the usual. */
function renderSuggestions(): void {
  const hint = errands.active().hint;
  const chips = [...new Set([...(hint ? [hint] : []), ...SUGGESTIONS])];
  $("suggest").replaceChildren(...chips.map((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = i === 0 && hint ? `📜 ${s}` : s;
    b.addEventListener("click", () => { if (game.brain) game.say(s); else addLog("system", `Summon ${settings.name} first (⚙).`); });
    return b;
  }));
}

$("good").addEventListener("click", () => { game.rate(true); });
$("bad").addEventListener("click", () => { game.rate(false); });
$("skillBtn").addEventListener("click", () => {
  const f = $("skillForm");
  f.hidden = !f.hidden;
  if (!f.hidden) $<HTMLInputElement>("skillName").focus();
});
$<HTMLFormElement>("skillForm").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const input = $<HTMLInputElement>("skillName");
  game.teachSkill(input.value);
  input.value = "";
  $("skillForm").hidden = true;
});

// ---------- settings ----------

function openSettings(): void {
  // The loader sheet sits on top of this one; Done brings it back if there's still no brain.
  loader.hidden = true;
  $<HTMLInputElement>("setName").value = settings.name;
  for (const r of document.querySelectorAll<HTMLInputElement>('input[name="backend"]')) r.checked = r.value === settings.backend;
  $<HTMLInputElement>("setOllamaUrl").value = settings.ollamaUrl;
  $<HTMLInputElement>("setOllamaModel").value = settings.ollamaModel;
  $<HTMLInputElement>("setAutonomy").checked = settings.autonomy;
  $<HTMLInputElement>("setThoughts").checked = settings.thoughts;
  $<HTMLInputElement>("setMemory").checked = settings.useMemory;
  $("settings").hidden = false;
  void fetch(`${settings.ollamaUrl}/api/tags`).then((r) => r.json() as Promise<{ models?: { name: string }[] }>)
    .then((t) => {
      $("ollamaModels").replaceChildren(...(t.models ?? []).map((m) => Object.assign(document.createElement("option"), { value: m.name })));
    }).catch(() => undefined);
}

$("settingsBtn").addEventListener("click", openSettings);
$("closeSettings").addEventListener("click", () => {
  const before = `${settings.backend}|${settings.ownBrain}|${settings.ollamaUrl}|${settings.ollamaModel}`;
  settings.name = $<HTMLInputElement>("setName").value.trim() || "Blobb";
  const picked = document.querySelector<HTMLInputElement>('input[name="backend"]:checked')?.value;
  settings.backend = BRAINS.find((b) => b === picked) ?? "own";
  settings.ollamaUrl = $<HTMLInputElement>("setOllamaUrl").value.trim() || DEFAULTS.ollamaUrl;
  settings.ollamaModel = $<HTMLInputElement>("setOllamaModel").value.trim() || DEFAULTS.ollamaModel;
  settings.autonomy = $<HTMLInputElement>("setAutonomy").checked;
  settings.thoughts = $<HTMLInputElement>("setThoughts").checked;
  settings.useMemory = $<HTMLInputElement>("setMemory").checked;
  saveSettings();
  applySettings();
  $("settings").hidden = true;
  if (before !== `${settings.backend}|${settings.ownBrain}|${settings.ollamaUrl}|${settings.ollamaModel}` || !game.brain) {
    loader.hidden = true;
    void maybeAutoLoad();
  }
});

function applySettings(): void {
  game.opts = { name: settings.name, autonomy: settings.autonomy, useMemory: settings.useMemory };
  document.body.classList.toggle("hide-thoughts", !settings.thoughts);
  $<HTMLInputElement>("msg").placeholder = `Tell ${settings.name} what to do…`;
}

$("exportBtn").addEventListener("click", () => {
  const data = memory.exportForDream(settings.name) as { samples: { id: string }[] };
  if (!data.samples.length) { addLog("system", "No new memories to dream about since the last export."); return; }
  const blob = new Blob([JSON.stringify(data, null, 1)], { type: "application/json" });
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob), download: `blobb-memories-${new Date().toISOString().slice(0, 19).replace(/:/g, "")}.json`,
  });
  a.click();
  // Revoking right away can cancel the download in some browsers.
  setTimeout(() => { URL.revokeObjectURL(a.href); }, 10_000);
  memory.markDreamed(data.samples.map((x) => x.id));
  addLog("system", `Exported ${String(data.samples.length)} memories for dreaming.`);
});

$("resetWorld").addEventListener("click", () => {
  world = makeWorld();
  game.world = world;
  view.setWorld(world);
  saveWorld();
  addLog("system", "The island is fresh again.");
  $("settings").hidden = true;
});

let forgetArmed = false;
$("forget").addEventListener("click", () => {
  const btn = $("forget");
  if (!forgetArmed) { forgetArmed = true; btn.textContent = "Tap again to forget everything"; return; }
  memory.forget();
  forgetArmed = false;
  btn.textContent = "Forget everything";
  addLog("system", `${settings.name} forgot everything it learned.`);
});

$("skillsBtn").addEventListener("click", () => {
  const body = $("knowledgeBody");
  body.replaceChildren();
  const section = (title: string, items: HTMLElement[]) => {
    const h = document.createElement("h3");
    h.textContent = title;
    const ul = document.createElement("ul");
    if (!items.length) { const li = document.createElement("li"); li.textContent = "nothing yet"; items = [li]; }
    ul.append(...items);
    body.append(h, ul);
  };
  section("Skills", memory.skills.map((s) => {
    const li = document.createElement("li");
    li.textContent = `${s.name}: ${s.steps.map((x) => (x.arg === "none" ? x.do : `${x.do} ${x.arg}`)).join(", ")} (used ${String(s.uses)}×)`;
    const del = Object.assign(document.createElement("button"), { textContent: "forget" });
    del.addEventListener("click", () => { memory.removeSkill(s.name); li.remove(); });
    li.append(del);
    return li;
  }));
  section("Lessons from getting unstuck", memory.insights.map((i) => {
    const li = document.createElement("li");
    li.textContent = `${i.reason.replace("_", " ")} → ${i.fix} (${String(i.count)}×)`;
    return li;
  }));
  const liked = memory.attempts.filter((a) => a.rated).slice(-12).reverse();
  section("Recently rated", liked.map((a) => {
    const li = document.createElement("li");
    li.textContent = `${a.reward > 0 ? "👍" : "👎"} "${a.command ?? "(on its own)"}" → ${a.executed.map((x) => `${x.do} ${x.arg}`).join(", ")}`;
    return li;
  }));
  $("settings").hidden = true;
  $("knowledge").hidden = false;
});
$("closeKnowledge").addEventListener("click", () => { $("knowledge").hidden = true; });

// ---------- loop ----------

const food = $("food"), energy = $("energy"), shape = $("shape"), held = $("held");
function hud(): void {
  const b = world.blob;
  food.style.width = `${b.food.toFixed(0)}%`;
  food.classList.toggle("low", b.food < 25);
  energy.style.width = `${b.energy.toFixed(0)}%`;
  energy.classList.toggle("low", b.energy < 20);
  shape.textContent = b.shape;
  const h = objById(world, b.holding);
  held.textContent = h ? `holding ${h.kind}` : "";
}

function place(el: HTMLElement, dy: number): void {
  const p = view.bubbleAnchor();
  el.style.left = `${String(p.x)}px`;
  el.style.top = `${String(p.y - dy)}px`;
}

function resize(): void {
  view.resize(window.innerWidth, window.innerHeight);
}
window.addEventListener("resize", resize);
resize();
applySettings();
view.snapCamera();

let last = performance.now(), sinceSave = 0, sinceHud = 0;
function loop(now: number): void {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  tick(world, dt);
  game.update(dt);
  view.frame(dt);
  if (!bubble.hidden) { place(bubble, 6); if (now > bubbleUntil) bubble.hidden = true; }
  if (!thinking.hidden) place(thinking, bubble.hidden ? 6 : 44);
  sinceHud += dt;
  if (sinceHud > 0.25) { hud(); sinceHud = 0; }
  sinceSave += dt;
  if (sinceSave > 5) { saveWorld(); sinceSave = 0; }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
document.addEventListener("visibilitychange", () => { if (document.hidden) saveWorld(); });

addLog("system", `You are a very old wizard, and you are done with people. Your spellbook pulled ${settings.name} out of another ` +
  "dimension to run the villagers' errands for you. Tell it what to do - it will try, fail, and learn. " +
  "👍/👎 teach it what you like; ⭐ names a plan that worked. When it sleeps (🌙), what it learned sinks into its brain.");
showLetter(errands.active(), window.innerWidth >= 600);
if (errands.storyDone) addLog("system", `All ${String(STORY_LENGTH)} of the first letters are answered. The villagers, sadly, keep writing.`);

// ---------- one tab at a time ----------

try {
  const tabId = Math.random().toString(36).slice(2);
  const channel = new BroadcastChannel("blobb");
  channel.onmessage = (ev: MessageEvent<{ hello?: string }>) => {
    if (!activeTab || !ev.data.hello || ev.data.hello === tabId) return;
    activeTab = false;
    game.opts.autonomy = false;
    game.brain = null;
    $("elsewhere").hidden = false;
  };
  channel.postMessage({ hello: tabId });
} catch { /* no BroadcastChannel: old browser, one tab it is */ }
$("playHere").addEventListener("click", () => { location.reload(); });

// ---------- dreams ----------

new DreamPeek(() => settings.name, (model) => {
  // Its own brain's dreams are "own:genN"; the LLM's are Ollama tags.
  if (model.startsWith("own:")) {
    const gen = model.slice(4);
    if (!/^gen\d+$/.test(gen)) return;
    settings.backend = "own";
    settings.ownBrain = gen;
  } else {
    settings.backend = "ollama";
    settings.ollamaModel = model;
  }
  saveSettings();
  addLog("system", `${settings.name} wakes up with a new brain: ${model}.`);
  void loadBrain();
});
void maybeAutoLoad();
