// Generates the blob's "instinct" curriculum: randomized situations paired with plans that are
// verified to work by running them in the simulator. Also generates retry situations
// ("your last plan failed because X") paired with the right fix, which teaches the retry loop.
//   npm run -s curriculum -- [count] > ../trainer/data/curriculum.jsonl   (-s: keep npm's banner out of the file)

import { runStep, type Step } from "../src/actions";
import { canonical, describePlan, systemPrompt, type Decision } from "../src/brain";
import { look, observe } from "../src/perception";
import { SHAPES, THINGS, makeWorld, objById, standable, tileAt, type Shape, type World } from "../src/world";

const pick = <T>(xs: readonly T[]): T => {
  const x = xs[Math.floor(Math.random() * xs.length)];
  if (x === undefined) throw new Error("empty");
  return x;
};
const chance = (p: number) => Math.random() < p;
const clone = (w: World): World => JSON.parse(JSON.stringify(w)) as World;
const S = (d: Step["do"], arg = "none"): Step => ({ do: d, arg });

/** Run a plan on a copy of the world. */
function simulate(w: World, plan: readonly Step[]): { ok: boolean; world: World; failed: { step: Step; msg: string } | null; notes: string[] } {
  const c = clone(w);
  const notes: string[] = [];
  for (const step of plan) {
    const out = runStep(c, step);
    notes.push(out.msg);
    if (!out.ok) return { ok: false, world: c, failed: { step, msg: out.msg }, notes };
  }
  return { ok: true, world: c, failed: null, notes };
}

interface Family {
  name: string;
  weight: number;
  says: readonly string[];
  setup?: (w: World) => void;
  /** Candidate plans, best first. The first that works (and meets the goal) is the answer. */
  plans: (w: World) => Step[][];
  /** A naive first try, used to make retry examples when it fails. */
  naive?: (w: World) => Step[];
  goal: (after: World, before: World) => boolean;
  thought: (plan: Step[]) => string;
  reply: readonly string[] | ((plan: Step[]) => readonly string[]);
}

/** Only train on plans that name things the blob can actually see (or is holding). */
function visible(w: World, plan: readonly Step[]): boolean {
  const seen = new Set<string>(look(w).map((s) => s.kind));
  const held = objById(w, w.blob.holding)?.kind;
  if (held) seen.add(held);
  let willHold: string | null = null;
  return plan.every((s) => {
    const ok = !(THINGS as readonly string[]).includes(s.arg) || seen.has(s.arg) || s.arg === willHold
      || (s.do === "eat" && plan.some((x) => x.do === "use" && x.arg === s.arg))
      || (s.do === "eat" && s.arg === "berry" && plan.some((x) => x.arg === "tree"));
    // Set after the check, so "grab X" itself still needs X in sight.
    if (s.do === "grab") willHold = s.arg;
    return ok;
  });
}

const ate = (a: World, b: World) => a.blob.food > b.blob.food + 5;
const morphTo = (w: World, s: Shape): Step[] => (w.blob.shape === s ? [] : [S("morph", s)]);
const toBlob = (w: World): Step[] => (w.blob.shape === "spring" || w.blob.shape === "blob" ? [] : [S("morph", "blob")]);
const hideNear = (w: World, f: (x: number, y: number, k: string) => boolean) => {
  for (const o of w.objs) if (o.state === "world" && f(o.x, o.y, o.kind)) o.state = "gone";
};
const snack = (x: number, y: number) => (x === 9 && y === 6) || (x === 3 && y === 8);
/** The wizard got a `kind` (from somewhere `from` likes, if given). */
const delivered = (kind: string, from: (home: readonly [number, number] | null) => boolean = () => true) =>
  (a: World, b: World) => a.deliveries.length > b.deliveries.length && a.deliveries.some((d) => d.kind === kind && from(d.home));
const empty = (w: World) => { w.blob.holding = null; for (const o of w.objs) if (o.state === "held") o.state = "gone"; };

const FAMILIES: Family[] = [
  {
    name: "eat", weight: 3,
    says: ["eat something", "you look hungry, eat", "go eat", "find some food", "have a snack", "eat a berry", "dinner time!", "grab a bite", "eat the mushroom"],
    plans: (w) => [[S("eat", "berry")], [S("eat", "mushroom")], [...toBlob(w), S("eat", "berry")], [...toBlob(w), S("eat", "mushroom")]],
    goal: ate,
    thought: (p) => `Food! There's a ${p[p.length - 1]?.arg ?? "snack"} I can reach.`,
    reply: ["Nom nom!", "Yummy!", "Food!", "Munch munch~"],
  },
  ...SHAPES.map((shape): Family => ({
    name: `morph-${shape}`, weight: 1,
    says: [`turn into a ${shape}`, `become a ${shape}!`, `can you be a ${shape}?`, `${shape} shape please`, `change into a ${shape}`, `morph into a ${shape}`],
    setup: (w) => { if (w.blob.shape === shape) w.blob.shape = pick(SHAPES.filter((s) => s !== shape)); },
    plans: () => [[S("morph", shape)]],
    goal: (a) => a.blob.shape === shape,
    thought: () => `The wizard wants me to be a ${shape}. Squish!`,
    reply: ["Squish!", "Ta-da!", `I'm a ${shape}!`, "Look at me!"],
  })),
  {
    name: "grab-stick", weight: 1.5,
    says: ["pick up the stick", "grab the stick", "get that stick", "take the stick", "go get the stick"],
    setup: (w) => { w.blob.holding = null; for (const o of w.objs) if (o.kind === "stick") o.state = "world"; },
    plans: (w) => [[S("grab", "stick")], [...toBlob(w), S("grab", "stick")]],
    naive: () => [S("use", "stick")],
    goal: (a) => objById(a, a.blob.holding)?.kind === "stick",
    thought: () => "A stick! I'll grab it.",
    reply: ["Got it!", "Stick!", "Mine now!", "Ooh, a stick!"],
  },
  {
    name: "drop", weight: 0.5,
    says: ["drop it", "put that down", "let go of the stick", "drop the stick"],
    setup: (w) => { const s = w.objs.find((o) => o.kind === "stick"); if (s) { s.state = "held"; w.blob.holding = s.id; } },
    plans: () => [[S("drop")]],
    goal: (a) => a.blob.holding === null,
    thought: () => "Okay, I'll put it down.",
    reply: ["Okay!", "Plop.", "There you go."],
  },
  {
    name: "rest", weight: 1,
    says: ["take a rest", "you look tired, rest", "have a nap", "rest a bit", "sleep for a while", "get some energy back"],
    setup: (w) => { w.blob.energy = 10 + Math.random() * 40; w.blob.food = Math.max(w.blob.food, 20); },
    plans: () => [[S("rest")]],
    goal: (a, b) => a.blob.energy > b.blob.energy + 20,
    thought: () => "I'm tired. Time to rest.",
    reply: ["Zzz...", "Yawn~", "Nap time.", "So sleepy..."],
  },
  {
    name: "high-berry", weight: 3,
    says: ["get the berry up high", "get the berry on the ledge", "eat the berry on the high ledge", "can you reach that high berry?", "climb up and get the berry", "go get the berries up there"],
    setup: (w) => {
      hideNear(w, (x, y, k) => snack(x, y) || (k === "berry" && x > 5));
      w.blob.x = 5 + Math.floor(Math.random() * 4); w.blob.y = 1 + Math.floor(Math.random() * 7);
      if (w.blob.food > 70) w.blob.food = 40;
    },
    plans: (w) => chance(0.8)
      ? [[...morphTo(w, "spring"), S("eat", "berry")], [S("grab", "stick"), S("use", "berry"), S("eat", "berry")]]
      : [[S("grab", "stick"), S("use", "berry"), S("eat", "berry")], [...morphTo(w, "spring"), S("eat", "berry")]],
    naive: () => [S("eat", "berry")],
    goal: ate,
    thought: (p) => p.some((s) => s.arg === "spring") ? "The berry is up too high. A spring can jump up there!" : "It's too high to climb. The stick can reach it!",
    reply: (p) => p.some((s) => s.arg === "spring") ? ["Boing!", "Boing boing!", "Up I go!"] : ["Hook it!", "Stick power!", "Gotcha!"],
  },
  {
    name: "garden", weight: 2.5,
    says: ["go into the walled garden", "go under the gate", "get into the garden", "get the mushroom in the garden", "squeeze under the gate", "eat the mushroom behind the gate"],
    setup: (w) => {
      hideNear(w, (x, y) => snack(x, y) || x >= 12 && y >= 9);
      w.blob.x = 8 + Math.floor(Math.random() * 6); w.blob.y = 6 + Math.floor(Math.random() * 2);
      if (w.blob.food > 70) w.blob.food = 45;
    },
    plans: (w) => [[...morphTo(w, "puddle"), S("eat", "mushroom")], [...morphTo(w, "puddle"), S("eat", "berry")]],
    naive: () => [S("eat", "mushroom")],
    goal: (a) => a.blob.y <= 4 && a.blob.x >= 10 && a.blob.x <= 13,
    thought: () => "The gate is very low. If I'm a flat puddle I can slide under it!",
    reply: ["Sploosh!", "Squeeze...", "Slide slide~", "Under I go!"],
  },
  {
    name: "gap", weight: 2.5,
    says: ["cross the gap", "get the mushroom across the gap", "get to the other side of the gap", "eat the mushroom over there", "jump the gap"],
    setup: (w) => {
      hideNear(w, (x, y, k) => snack(x, y) || (k === "mushroom" && y < 6) || (k === "berry" && x > 5 && y < 6));
      w.blob.x = 6 + Math.floor(Math.random() * 5); w.blob.y = 8 + Math.floor(Math.random() * 6);
      if (w.blob.food > 70) w.blob.food = 45;
    },
    plans: (w) => chance(0.6)
      ? [[...morphTo(w, "spring"), S("eat", "mushroom")], [S("push", "rock"), ...toBlob(w), S("eat", "mushroom")]]
      : [[...toBlob(w), S("push", "rock"), S("eat", "mushroom")], [...morphTo(w, "spring"), S("eat", "mushroom")]],
    naive: () => [S("eat", "mushroom")],
    goal: (a, b) => ate(a, b),
    thought: (p) => p.some((s) => s.do === "push") ? "There's a gap. If I push the rock into it, it becomes a bridge!" : "There's a gap in the way. A spring can jump over it!",
    reply: (p) => p.some((s) => s.do === "push") ? ["Bridge time!", "Heave ho!", "Rock and roll!"] : ["Boing!", "Wheee!", "Over I go!"],
  },
  {
    name: "tree", weight: 2,
    says: ["get fruit from the tree", "knock a berry down from the tree", "shake the tree", "get a berry out of the tree", "the tree has fruit, get some"],
    setup: (w) => {
      w.blob.x = 3 + Math.floor(Math.random() * 7); w.blob.y = 9 + Math.floor(Math.random() * 5);
      w.blob.holding = null;
    },
    plans: (w) => chance(0.6)
      ? [[...morphTo(w, "ball"), S("push", "tree")], [S("grab", "stick"), S("use", "tree")]]
      : [[S("grab", "stick"), S("use", "tree")], [...morphTo(w, "ball"), S("push", "tree")]],
    naive: () => [S("push", "tree")],
    goal: (a) => (a.objs.find((o) => o.kind === "tree")?.fruit ?? 3) < 3,
    thought: (p) => p.some((s) => s.do === "use") ? "The fruit is up in the tree. I can whack it with the stick!" : "The fruit is up in the tree. If I'm a hard ball I can ram it!",
    reply: (p) => p.some((s) => s.do === "use") ? ["Thwack!", "Fruit, fall down!", "Whack!"] : ["Bonk!", "Ram!", "Rolling in!"],
  },
  // ---------- errands for the wizard ----------
  {
    name: "fetch-stick", weight: 2,
    says: ["fetch my walking stick", "bring me my stick", "go get my stick and bring it here", "blob. stick. now.",
      "where's my walking stick? bring it to the tower", "I need my stick, fetch it"],
    setup: (w) => { empty(w); for (const o of w.objs) if (o.kind === "stick") o.state = "world"; },
    plans: (w) => [[S("grab", "stick"), S("deliver")], [...toBlob(w), S("grab", "stick"), S("deliver")]],
    naive: () => [S("deliver")],
    goal: delivered("stick"),
    thought: () => "The wizard wants his stick. Grab it, then bring it to the tower.",
    reply: ["Here's your stick!", "Stick delivery!", "Fetched!", "For you, wizard!"],
  },
  {
    name: "fetch-food", weight: 2.5,
    says: ["bring me a mushroom", "fetch a mushroom for the tower", "a villager wants a mushroom, go get one",
      "bring me a berry", "fetch a berry and deliver it", "get a berry and bring it to the tower", "mrs pennywhistle wants a mushroom"],
    setup: (w) => { empty(w); if (w.blob.food < 40) w.blob.food = 60; },
    plans: (w) => (["mushroom", "berry"] as const).flatMap((k) => [[S("grab", k), S("deliver")], [...toBlob(w), S("grab", k), S("deliver")]]),
    naive: () => [S("deliver")],
    goal: (a, b) => a.deliveries.length > b.deliveries.length,
    thought: (p) => `The wizard wants a ${p.find((s) => s.do === "grab")?.arg ?? "thing"}. I'll carry it to the tower instead of eating it.`,
    reply: ["Special delivery!", "Here you go!", "Plop! For the tower.", "Fetched!"],
  },
  {
    name: "fetch-meadow-berry", weight: 1.5,
    says: ["bring me a berry from up high", "the baker wants a berry from the high meadow", "fetch a berry from the ledge and deliver it"],
    setup: (w) => { empty(w); hideNear(w, (x, y, k) => k === "berry" && (x > 5 || snack(x, y))); at(w, 5 + Math.floor(Math.random() * 3), 2 + Math.floor(Math.random() * 5)); },
    plans: (w) => [[...morphTo(w, "spring"), S("grab", "berry"), S("deliver")]],
    naive: () => [S("grab", "berry"), S("deliver")],
    goal: delivered("berry", (h) => h !== null && h[0] <= 4 && h[1] <= 4),
    thought: () => "The berry is up high. As a spring I can jump up, grab it, and bring it back.",
    reply: ["Boing, delivery!", "Up and back!", "Meadow berry!"],
  },
  {
    name: "fetch-garden-mushroom", weight: 1.5,
    says: ["get the mushroom from the walled garden and bring it", "the herbalist wants the garden mushroom", "fetch the mushroom behind the gate"],
    setup: (w) => { empty(w); hideNear(w, (x, y, k) => k === "mushroom" && !(x >= 10 && x <= 13 && y <= 4)); at(w, 8 + Math.floor(Math.random() * 5), 6 + Math.floor(Math.random() * 2)); },
    plans: (w) => [[...morphTo(w, "puddle"), S("grab", "mushroom"), S("deliver")]],
    naive: () => [S("grab", "mushroom"), S("deliver")],
    goal: delivered("mushroom", (h) => h !== null && h[0] >= 10 && h[1] <= 4),
    thought: () => "The gate is low. As a puddle I can slide under, grab the mushroom and slide back out.",
    reply: ["Slide and fetch!", "Sploosh, got it!", "Under and back!"],
  },
];

const at = (w: World, x: number, y: number) => { w.blob.x = x; w.blob.y = y; };

const family = (name: string): Family => {
  const f = FAMILIES.find((x) => x.name === name);
  if (!f) throw new Error(`no family ${name}`);
  return f;
};

// On its own (no command): look after needs.
const SELF: Family[] = [
  { ...family("eat"), name: "self-eat", says: [""], setup: (w) => { w.blob.food = 5 + Math.random() * 30; }, thought: (p) => `My tummy is rumbling. I'll get that ${p[p.length - 1]?.arg ?? "food"}.` },
  { ...family("rest"), name: "self-rest", says: [""], setup: (w) => { w.blob.energy = 3 + Math.random() * 15; w.blob.food = 50 + Math.random() * 40; } },
];

function randomWorld(): World {
  const w = makeWorld();
  w.blob.food = 20 + Math.random() * 80;
  w.blob.energy = 30 + Math.random() * 70;
  w.blob.shape = chance(0.6) ? "blob" : pick(SHAPES);
  for (let i = 0; i < 40; i++) {
    const x = 1 + Math.floor(Math.random() * 10), y = 5 + Math.floor(Math.random() * 10);
    if (standable(w, x, y) && tileAt(w, x, y).kind === "ground" && tileAt(w, x, y).h === 0) { w.blob.x = x; w.blob.y = y; break; }
  }
  for (const o of w.objs) if ((o.kind === "berry" || o.kind === "mushroom") && chance(0.15)) o.state = "gone";
  const tree = w.objs.find((o) => o.kind === "tree");
  if (tree) tree.fruit = 1 + Math.floor(Math.random() * 3);
  return w;
}

interface Sample { system: string; user: string; assistant: string; family: string }

function make(f: Family): Sample[] {
  const w = randomWorld();
  f.setup?.(w);
  if (!standable(w, w.blob.x, w.blob.y) || tileAt(w, w.blob.x, w.blob.y).kind !== "ground") return [];
  const say = pick(f.says) || null;
  let answer: Step[] | null = null;
  for (const plan of f.plans(w)) {
    if (!plan.length) continue;
    const sim = simulate(w, plan);
    if (sim.ok && f.goal(sim.world, w) && visible(w, plan)) { answer = plan; break; }
  }
  if (!answer) return [];
  const decision = (plan: Step[]): string =>
    canonical({ thought: f.thought(plan), plan, say: pick(typeof f.reply === "function" ? f.reply(plan) : f.reply) } satisfies Decision);
  const system = systemPrompt("Blobb");
  const out: Sample[] = [];
  const [clean] = observe(w, { ownerSaid: say, recent: [], lessons: [], skills: [], retry: null });
  out.push({ system, user: clean, assistant: decision(answer), family: f.name });

  // Retry example: the naive plan fails, the blob hears why, and fixes it.
  const naive = f.naive?.(w);
  if (naive && chance(0.6)) {
    const sim = simulate(w, naive);
    if (!sim.ok && sim.failed) {
      const after = sim.world;
      let fix: Step[] | null = null;
      for (const plan of f.plans(after)) {
        const s2 = simulate(after, plan);
        if (plan.length && s2.ok && f.goal(s2.world, w) && visible(after, plan)) { fix = plan; break; }
      }
      if (fix) {
        const retry = `${describePlan([sim.failed.step])} -> ${sim.failed.msg}`;
        const [clean2] = observe(after, { ownerSaid: say, recent: sim.notes.slice(-3), lessons: [], skills: [], retry });
        out.push({ system, user: clean2, assistant: decision(fix), family: `${f.name}-retry` });
      }
    }
  }
  return out;
}

const count = Number(process.argv[2] ?? 500);
const all = [...FAMILIES, ...SELF.map((f) => ({ ...f, weight: 1.5 }))];
const totalWeight = all.reduce((s, f) => s + f.weight, 0);
const samples: Sample[] = [];
const tally = new Map<string, number>();
let guard = 0;
while (samples.length < count && guard++ < count * 20) {
  let r = Math.random() * totalWeight;
  const f = all.find((x) => (r -= x.weight) < 0) ?? all[0];
  if (!f) break;
  for (const s of make(f)) {
    samples.push(s);
    tally.set(s.family, (tally.get(s.family) ?? 0) + 1);
  }
}
for (const s of samples.slice(0, count)) process.stdout.write(JSON.stringify(s) + "\n");
process.stderr.write([...tally.entries()].sort().map(([k, v]) => `${k}: ${String(v)}`).join(", ") + "\n");
