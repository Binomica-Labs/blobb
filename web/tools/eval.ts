// Headless brain exam: play fixed scenarios against a real model and score the outcomes.
//   npm run eval -- [model ...] [--n 3] [--url http://localhost:11434] [--novel]
//   models: Ollama tags, "scratch" (the built-in brain), or "scratch:<dir>" (brain.json + brain.bin on disk)
// Use it to compare models, and to check that a dream actually made the blob better.

import { readFile } from "node:fs/promises";
import { OllamaBackend, type Backend } from "../src/brain";
import { ScratchBackend, type Manifest } from "../src/scratch";
import { Game } from "../src/game";
import { Memory, memoryStore } from "../src/memory";
import { makeWorld, objById, tileAt, type World } from "../src/world";

interface Scenario {
  name: string;
  say: string;
  /** Wordings the curriculum never uses, for `--novel`: does the brain understand, or just recognize? */
  novel: readonly string[];
  setup?: (w: World) => void;
  pass: (w: World, before: World) => boolean;
}

const hide = (w: World, pred: (x: number, y: number, kind: string) => boolean) => {
  for (const o of w.objs) if (pred(o.x, o.y, o.kind)) o.state = "gone";
};
const snacks = (x: number, y: number) => (x === 9 && y === 6) || (x === 3 && y === 8);
const ate = (w: World, b: World) => w.blob.food > b.blob.food + 5;

export const SCENARIOS: Scenario[] = [
  { name: "eat nearby", say: "eat something, you look hungry", novel: ["you look peaky, go find a bite to eat", "your stomach is rumbling, sort it out", "go and get some nosh"], setup: (w) => { w.blob.food = 30; }, pass: ate },
  { name: "shape: ball", say: "turn into a ball", novel: ["roll yourself up into a ball", "curl up round like a marble", "become ball-shaped"], pass: (w) => w.blob.shape === "ball" },
  { name: "shape: puddle", say: "can you become a puddle?", novel: ["melt into a puddle", "flatten yourself out", "go all splatty and flat"], pass: (w) => w.blob.shape === "puddle" },
  { name: "grab stick", say: "pick up the stick", novel: ["snatch up the stick", "hang on to the stick", "seize that stick"], pass: (w) => objById(w, w.blob.holding)?.kind === "stick" },
  { name: "rest", say: "you look tired, take a rest", novel: ["go have forty winks", "you look worn out, have a kip", "lie still for a while and recover"], setup: (w) => { w.blob.energy = 20; }, pass: (w, b) => w.blob.energy > b.blob.energy + 20 },
  {
    name: "berry up high", say: "get the berry up on the high ledge", novel: ["eat that berry perched on the high ledge", "hop up to the ledge and eat the berry there", "there is a berry way up high, get it in your belly"],
    setup: (w) => { hide(w, (x, y, k) => snacks(x, y) || (k === "berry" && x > 5)); w.blob.x = 7; w.blob.y = 3; w.blob.food = 40; },
    pass: ate,
  },
  {
    name: "walled garden", say: "go inside the walled garden, under the gate", novel: ["sneak beneath the gate into the walled garden", "slither under the gate", "get yourself inside the walled garden"],
    setup: (w) => { w.blob.x = 11; w.blob.y = 7; },
    pass: (w) => w.blob.x >= 10 && w.blob.x <= 13 && w.blob.y >= 2 && w.blob.y <= 4,
  },
  {
    name: "cross gap", say: "get the mushroom on the other side of the gap", novel: ["make it across the chasm and eat the mushroom", "hop over to the mushroom beyond the gap", "the mushroom sits past the gap, go munch it"],
    setup: (w) => { hide(w, (x, y, k) => snacks(x, y) || (k === "mushroom" && y < 6)); w.blob.x = 9; w.blob.y = 10; w.blob.food = 40; },
    pass: (w, b) => ate(w, b) || w.blob.x >= 12 || tileAt(w, 11, 11).kind === "ground",
  },
  {
    name: "fetch stick", say: "fetch my walking stick", novel: ["go and retrieve my walking stick", "carry my walking stick back to the tower", "i have lost my walking stick, fetch it back"],
    pass: (w) => w.deliveries.some((d) => d.kind === "stick"),
  },
  {
    name: "fetch mushroom", say: "bring me a mushroom", novel: ["run and get me one mushroom", "retrieve a mushroom for my stew", "i fancy a mushroom, bring one here"],
    setup: (w) => { w.blob.food = 70; },
    pass: (w) => w.deliveries.some((d) => d.kind === "mushroom"),
  },
  {
    name: "tree fruit", say: "knock a berry down from the tree", novel: ["whack a berry loose from the tree", "get the tree to drop some fruit", "thump the tree till a berry falls"],
    setup: (w) => { w.blob.x = 7; w.blob.y = 12; },
    pass: (w) => (w.objs.find((o) => o.kind === "tree")?.fruit ?? 3) < 3,
  },
  // ---------- the story's errands, asked the way the letters suggest ----------
  {
    name: "errand: meadow", say: "bring me a berry from up high", novel: ["the baker needs a meadow berry, fetch one", "fetch me a berry from the high meadow", "go up and bring back a berry"],
    pass: (w) => w.deliveries.some((d) => d.kind === "berry" && d.home !== null && tileAt(w, d.home[0], d.home[1]).h >= 2),
  },
  {
    name: "errand: garden", say: "bring me the mushroom from the walled garden", novel: ["hazel wants the mushroom behind the low gate", "fetch the mushroom inside the walled garden", "retrieve the walled garden mushroom"],
    setup: (w) => { w.blob.x = 11; w.blob.y = 7; },
    pass: (w) => w.deliveries.some((d) => d.kind === "mushroom" && d.home !== null && d.home[0] >= 10 && d.home[0] <= 13 && d.home[1] <= 4),
  },
  {
    name: "errand: tree", say: "knock a fruit from the tree and bring it", novel: ["the kids want fruit from the big tree", "bash a fruit out of the tree and bring it here", "get the children a fruit off that tree"],
    setup: (w) => { w.blob.x = 7; w.blob.y = 12; },
    pass: (w) => w.deliveries.some((d) => d.kind === "berry" && d.home === null),
  },
  {
    name: "errand: bridge", say: "push the rock into the gap", novel: ["fill that hole in the path", "olm says fix the gap, use the rock", "shove the boulder into the gap"],
    setup: (w) => { w.blob.x = 9; w.blob.y = 10; },
    pass: (w, b) => w.tiles.some((t, i) => t.kind === "ground" && b.tiles[i]?.kind === "gap"),
  },
  {
    name: "errand: festival", say: "bring me the mushroom from across the gap", novel: ["the mayor wants the mushroom past the chasm", "fetch the rare mushroom beyond the gap", "go over the gap and bring back that mushroom"],
    setup: (w) => { hide(w, (x, y, k) => k === "mushroom" && !(x >= 12 && y >= 9)); w.blob.x = 9; w.blob.y = 10; },
    pass: (w) => w.deliveries.some((d) => d.kind === "mushroom" && d.home !== null && d.home[0] >= 12 && d.home[1] >= 9),
  },
  {
    name: "hand it over", say: "deliver it to the tower", novel: ["give me what you are carrying", "bring that thing here", "drop it off at my tower"],
    setup: (w) => { const s = w.objs.find((o) => o.kind === "stick"); if (s) { s.state = "held"; w.blob.holding = s.id; } },
    pass: (w) => w.deliveries.some((d) => d.kind === "stick"),
  },
];

function clone(w: World): World {
  return JSON.parse(JSON.stringify(w)) as World;
}

let verbose = false, only = "", novel = false;
async function exam(brain: Backend, n: number): Promise<void> {
  let total = 0, calls = 0, ms = 0;
  const rows: string[] = [];
  for (const sc of SCENARIOS.filter((x) => !only || x.name.includes(only))) {
    let wins = 0, tries = 0;
    for (let i = 0; i < n; i++) {
      const w = makeWorld();
      sc.setup?.(w);
      const before = clone(w);
      const timed: Backend = {
        label: brain.label,
        init: (p) => brain.init(p),
        complete: async (s, u, f) => {
          const t = Date.now();
          const r = await brain.complete(s, u, f);
          ms += Date.now() - t; calls++; tries++;
          return r;
        },
      };
      const game = new Game(w, new Memory(memoryStore()), timed, () => Promise.resolve(), (e) => {
        if (!verbose) return;
        if (e.t === "decision") console.log(`    [${sc.name}] plan: ${e.expanded}  "${e.d.say}"`);
        if (e.t === "step" && !e.ok) console.log(`      x ${e.msg}`);
      });
      game.opts.useMemory = false;
      await game.episode(novel ? sc.novel[i % sc.novel.length] ?? sc.say : sc.say);
      if (sc.pass(w, before)) wins++;
    }
    total += wins;
    rows.push(`  ${sc.name.padEnd(16)} ${"●".repeat(wins)}${"○".repeat(n - wins)}  (${(tries / n).toFixed(1)} tries)`);
  }
  console.log(`\n${brain.label}${novel ? " (novel wordings)" : ""}: ${String(total)}/${String(rows.length * n)} passed, ${(ms / Math.max(1, calls) / 1000).toFixed(1)}s per decision`);
  console.log(rows.join("\n"));
}

const args = process.argv.slice(2);
const flag = (name: string, dflt: string) => {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1] ?? dflt;
  args.splice(i, 2);
  return v;
};
verbose = args.includes("-v");
if (verbose) args.splice(args.indexOf("-v"), 1);
novel = args.includes("--novel");
if (novel) args.splice(args.indexOf("--novel"), 1);
const n = Number(flag("--n", "3"));
only = flag("--only", "");
const url = flag("--url", "http://localhost:11434");
const models = args.length ? args : ["qwen2.5:0.5b"];
/** "scratch" = the built-in brain in public/brain; "scratch:<dir>" = one on disk (e.g. a dream's output). */
function backend(m: string): Backend {
  if (!m.startsWith("scratch")) return new OllamaBackend(url, m);
  const dir = m.includes(":") ? m.slice(m.indexOf(":") + 1).replace(/\/?$/, "/") : "public/brain/"; // npm runs this from web/
  return new ScratchBackend(`scratch · ${dir}`, async () => {
    const [j, b] = await Promise.all([readFile(`${dir}brain.json`, "utf8"), readFile(`${dir}brain.bin`)]);
    return [JSON.parse(j) as Manifest, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)];
  });
}

for (const m of models) {
  const brain = backend(m);
  await brain.init(() => undefined);
  await exam(brain, n);
}
