// Headless brain exam: play fixed scenarios against a real model and score the outcomes.
//   npm run eval -- [model ...] [--n 3] [--url http://localhost:11434]
// Use it to compare models, and to check that a dream actually made the blob better.

import { OllamaBackend, type Backend } from "../src/brain";
import { Game } from "../src/game";
import { Memory, memoryStore } from "../src/memory";
import { makeWorld, objById, tileAt, type World } from "../src/world";

interface Scenario {
  name: string;
  say: string;
  setup?: (w: World) => void;
  pass: (w: World, before: World) => boolean;
}

const hide = (w: World, pred: (x: number, y: number, kind: string) => boolean) => {
  for (const o of w.objs) if (pred(o.x, o.y, o.kind)) o.state = "gone";
};
const snacks = (x: number, y: number) => (x === 9 && y === 6) || (x === 3 && y === 8);
const ate = (w: World, b: World) => w.blob.food > b.blob.food + 5;

export const SCENARIOS: Scenario[] = [
  { name: "eat nearby", say: "eat something, you look hungry", setup: (w) => { w.blob.food = 30; }, pass: ate },
  { name: "shape: ball", say: "turn into a ball", pass: (w) => w.blob.shape === "ball" },
  { name: "shape: puddle", say: "can you become a puddle?", pass: (w) => w.blob.shape === "puddle" },
  { name: "grab stick", say: "pick up the stick", pass: (w) => objById(w, w.blob.holding)?.kind === "stick" },
  { name: "rest", say: "you look tired, take a rest", setup: (w) => { w.blob.energy = 20; }, pass: (w, b) => w.blob.energy > b.blob.energy + 20 },
  {
    name: "berry up high", say: "get the berry up on the high ledge",
    setup: (w) => { hide(w, (x, y, k) => snacks(x, y) || (k === "berry" && x > 5)); w.blob.x = 7; w.blob.y = 3; w.blob.food = 40; },
    pass: ate,
  },
  {
    name: "walled garden", say: "go inside the walled garden, under the gate",
    setup: (w) => { w.blob.x = 11; w.blob.y = 7; },
    pass: (w) => w.blob.x >= 10 && w.blob.x <= 13 && w.blob.y >= 2 && w.blob.y <= 4,
  },
  {
    name: "cross gap", say: "get the mushroom on the other side of the gap",
    setup: (w) => { hide(w, (x, y, k) => snacks(x, y) || (k === "mushroom" && y < 6)); w.blob.x = 9; w.blob.y = 10; w.blob.food = 40; },
    pass: (w, b) => ate(w, b) || w.blob.x >= 12 || tileAt(w, 11, 11).kind === "ground",
  },
  {
    name: "fetch stick", say: "fetch my walking stick",
    pass: (w) => w.deliveries.some((d) => d.kind === "stick"),
  },
  {
    name: "fetch mushroom", say: "bring me a mushroom",
    setup: (w) => { w.blob.food = 70; },
    pass: (w) => w.deliveries.some((d) => d.kind === "mushroom"),
  },
  {
    name: "tree fruit", say: "knock a berry down from the tree",
    setup: (w) => { w.blob.x = 7; w.blob.y = 12; },
    pass: (w) => (w.objs.find((o) => o.kind === "tree")?.fruit ?? 3) < 3,
  },
];

function clone(w: World): World {
  return JSON.parse(JSON.stringify(w)) as World;
}

let verbose = false, only = "";
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
      await game.episode(sc.say);
      if (sc.pass(w, before)) wins++;
    }
    total += wins;
    rows.push(`  ${sc.name.padEnd(16)} ${"●".repeat(wins)}${"○".repeat(n - wins)}  (${(tries / n).toFixed(1)} tries)`);
  }
  console.log(`\n${brain.label}: ${String(total)}/${String(rows.length * n)} passed, ${(ms / Math.max(1, calls) / 1000).toFixed(1)}s per decision`);
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
const n = Number(flag("--n", "3"));
only = flag("--only", "");
const url = flag("--url", "http://localhost:11434");
const models = args.length ? args : ["qwen2.5:0.5b"];
for (const m of models) {
  const brain = new OllamaBackend(url, m);
  await brain.init(() => undefined);
  await exam(brain, n);
}
