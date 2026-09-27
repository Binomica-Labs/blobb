// Edge cases that used to soft-lock or corrupt the game.
import { runStep } from "../src/actions";
import type { Backend } from "../src/brain";
import { BRAIN_TIMEOUT_S, Game, IDLE_S, type GameEvent } from "../src/game";
import { Memory, memoryStore, type Attempt } from "../src/memory";
import { catchUp, makeWorld, restoreWorld, tick, type World } from "../src/world";

class Scripted implements Backend {
  readonly label = "scripted";
  calls = 0;
  constructor(private reply: () => Promise<string>) {}
  async init(): Promise<void> { /* nothing to load */ }
  complete(): Promise<string> { this.calls++; return this.reply(); }
}
const json = (o: object) => () => Promise.resolve(JSON.stringify(o));
const plan = (...steps: [string, string][]) => ({ thought: "t", say: "s", plan: steps.map(([d, a]) => ({ do: d, arg: a })) });

function game(brain: Backend, w: World = makeWorld(), play: () => Promise<void> = () => Promise.resolve()) {
  const events: GameEvent[] = [];
  const g = new Game(w, new Memory(memoryStore()), brain, play, (e) => events.push(e));
  return { g, w, events };
}

describe("starving and exhausted", () => {
  it("resting while starving still gives enough energy to crawl to food", () => {
    const w = makeWorld();
    w.blob.food = 0; w.blob.energy = 0;
    expect(runStep(w, { do: "eat", arg: "berry" }).reason).toBe("too_tired");
    expect(runStep(w, { do: "rest", arg: "none" }).ok).toBe(true);
    expect(runStep(w, { do: "eat", arg: "berry" }).ok).toBe(true);
  });

  it("a long absence leaves the blob hungry, not dead on its feet", () => {
    const w = makeWorld();
    catchUp(w, 600);
    expect(w.blob.food).toBeGreaterThan(0);
    expect(w.blob.energy).toBeGreaterThan(50);
  });

  it("one big tick doesn't skip past food hitting zero", () => {
    const a = makeWorld(), b = makeWorld();
    tick(a, 400);
    for (let i = 0; i < 400; i++) tick(b, 1);
    expect(a.blob.energy).toBeCloseTo(b.blob.energy, 0);
  });
});

describe("world bookkeeping", () => {
  it("eaten fruit that never respawns is cleaned up", () => {
    const w = makeWorld();
    const tree = w.objs.find((o) => o.kind === "tree");
    if (!tree) throw new Error("no tree");
    for (let i = 0; i < 50; i++) {
      w.objs.push({ id: w.nextId++, kind: "berry", x: 1, y: 1, state: "gone", home: null, respawnAt: 0, fruit: 0, regrowAt: 0 });
    }
    const before = w.objs.length;
    tick(w, 1);
    expect(w.objs.length).toBe(before - 50);
  });

  it("garbage saves fall back to a fresh island", () => {
    for (const junk of [null, 42, "x", { w: 16, d: 16, objs: [] }, { ...makeWorld(), tiles: [] },
      { ...makeWorld(), blob: { x: "a" } }, { ...makeWorld(), blob: { ...makeWorld().blob, x: 99 } }]) {
      const w = restoreWorld(junk, 0);
      expect(w.tiles).toHaveLength(w.w * w.d);
      expect(w.blob.x).toBe(makeWorld().blob.x);
    }
  });

  it("a good save survives, with a gentle catch-up", () => {
    const saved = { ...makeWorld(), savedAt: 0 };
    saved.blob.x = 3;
    const w = restoreWorld(JSON.parse(JSON.stringify(saved)), 10 * 60 * 1000);
    expect(w.blob.x).toBe(3);
    expect(w.blob.food).toBeGreaterThan(0);
  });
});

describe("game loop robustness", () => {
  it("a brain that never answers times out instead of freezing the blob", async () => {
    vi.useFakeTimers();
    try {
      const { g, events } = game(new Scripted(() => new Promise<string>(() => undefined)));
      const run = g.episode("hello");
      await vi.advanceTimersByTimeAsync(BRAIN_TIMEOUT_S * 1000 + 10);
      await run;
      expect(g.busy).toBe(false);
      expect(events.some((e) => e.t === "system" && e.text.includes("Brain error"))).toBe(true);
    } finally { vi.useRealTimers(); }
  });

  it("a timeout also cancels the brain request, so it can't hold up the next one", async () => {
    vi.useFakeTimers();
    try {
      let aborted = false;
      const brain: Backend = { label: "x", init: () => Promise.resolve(),
        complete: (_s, _u, _f, signal) => new Promise<string>(() => { signal?.addEventListener("abort", () => { aborted = true; }); }) };
      const { g } = game(brain);
      const run = g.episode(null);
      await vi.advanceTimersByTimeAsync(BRAIN_TIMEOUT_S * 1000 + 10);
      await run;
      expect(aborted).toBe(true);
    } finally { vi.useRealTimers(); }
  });

  it("a new command ends the retries of the old one", async () => {
    const replies = [plan(["eat", "rock"]), plan(["eat", "rock"]), plan(["rest", "none"])];
    const asked: string[] = [];
    const brain: Backend = { label: "x", init: () => Promise.resolve(), complete: (_s, u) => {
      asked.push(u.slice(u.lastIndexOf("The wizard says")));
      return json(replies.shift() ?? plan(["rest", "none"]))();
    } };
    const { g } = game(brain, makeWorld(), () => { if (asked.length === 1) g.say("go rest"); return Promise.resolve(); });
    g.say("eat the rock");
    while (g.busy) await new Promise((r) => setTimeout(r, 0));
    expect(asked.filter((a) => a.includes("eat the rock"))).toHaveLength(1);
    expect(asked.some((a) => a.includes("go rest"))).toBe(true);
  });

  it("autonomy backs off when the brain keeps failing", async () => {
    const brain = new Scripted(() => Promise.reject(new Error("ollama is down")));
    const { g } = game(brain);
    for (let t = 0; t < 10 * 60; t++) { g.update(1); await Promise.resolve(); await Promise.resolve(); }
    // Without backoff this is ~24 calls (one per IDLE_S).
    expect(brain.calls).toBeLessThan(10 * 60 / IDLE_S / 2);
  });

  it("resetting the island mid-plan stops the plan and doesn't touch the new island", async () => {
    let release: () => void = () => undefined;
    const play = () => new Promise<void>((r) => { release = r; });
    const { g, w } = game(new Scripted(json(plan(["move_to", "rock"], ["push", "rock"]))), makeWorld(), play);
    const run = g.episode("push the rock");
    while (g.busy && w.blob.x === 7 && w.blob.y === 9) await new Promise((r) => setTimeout(r, 0));
    const fresh = makeWorld();
    g.world = fresh;
    release();
    await run;
    expect(g.busy).toBe(false);
    expect([fresh.blob.x, fresh.blob.y]).toEqual([7, 9]);
    expect(fresh.objs.find((o) => o.kind === "rock")?.x).toBe(10);
  });

  it("very long owner messages are trimmed before they reach the brain", async () => {
    let seen = "";
    const brain: Backend = { label: "x", init: () => Promise.resolve(), complete: (_s, u) => { seen = u; return json(plan(["rest", "none"]))(); } };
    const { g } = game(brain);
    g.say("berry ".repeat(500));
    while (g.busy) await new Promise((r) => setTimeout(r, 0));
    expect(seen.length).toBeLessThan(2000);
  });

  it("skill names that clean up to nothing, or clash with things, are refused", async () => {
    const { g } = game(new Scripted(json(plan(["rest", "none"]))));
    await g.episode("rest");
    expect(g.teachSkill("!!!")).toBe(false);
    expect(g.teachSkill("berry")).toBe(false);
    expect(g.teachSkill("none")).toBe(false);
    expect(g.memory.skills).toHaveLength(0);
    expect(g.teachSkill("nap time")).toBe(true);
  });
});

describe("memory robustness", () => {
  const attempt = (i: number, rated: boolean): Omit<Attempt, "id" | "rated" | "dreamed"> => ({
    t: i, command: "x", obsClean: "", decision: { thought: "", plan: [], say: "" },
    executed: [{ do: "rest", arg: "none" }], results: [], ok: true, reward: rated ? 1 : 0,
  });

  it("a corrupt save loads as empty instead of crashing later", () => {
    for (const raw of ['{"attempts":5,"skills":[{"name":3}],"insights":"no"}', '{"skills":[{"name":"hop","steps":"x"}]}', "[1,2]"]) {
      const s = memoryStore(); s.save(raw);
      const m = new Memory(s);
      expect(m.skills).toEqual([]);
      expect(() => m.recall("hello", new Set(["gap"]))).not.toThrow();
      expect(m.skillLines()).toEqual([]);
    }
  });

  it("forgetting old attempts never drops rated ones for unrated ones", () => {
    const m = new Memory(memoryStore());
    for (let i = 0; i < 500; i++) { const a = m.add(attempt(i, true)); a.rated = true; }
    for (let i = 500; i < 600; i++) m.add(attempt(i, false));
    expect(m.attempts).toHaveLength(500);
    expect(m.attempts.every((a) => a.rated)).toBe(true);
  });

  it("each memory is only exported for dreaming once", () => {
    const m = new Memory(memoryStore());
    m.add({ ...attempt(1, false), reward: 0.3 });
    const first = m.exportForDream("Blobb") as { samples: { id: string }[] };
    expect(first.samples).toHaveLength(1);
    m.markDreamed(first.samples.map((s) => s.id));
    m.add({ ...attempt(2, false), reward: 0.3 });
    const second = m.exportForDream("Blobb") as { samples: unknown[] };
    expect(second.samples).toHaveLength(1);
  });
});

describe("the wizard's verdict", () => {
  const base = (reward: number): Omit<Attempt, "id" | "rated" | "dreamed"> => ({
    t: 1, command: "fetch my stick", obsClean: "", decision: { thought: "", plan: [], say: "" },
    executed: [{ do: "deliver", arg: "none" }], results: [], ok: true, reward,
  });

  it("👎 means no, even on an attempt the world rewarded well (e.g. it finished an errand)", () => {
    const m = new Memory(memoryStore());
    const a = m.add(base(1.3));
    m.rate(a.id, false);
    expect(a.reward).toBeLessThan(0);
    expect((m.exportForDream("B") as { samples: unknown[] }).samples).toHaveLength(0);
    expect(m.recall("fetch my stick", new Set()).join()).toContain("did NOT like");
  });

  it("verdicts replace each other instead of stacking", () => {
    const m = new Memory(memoryStore());
    const a = m.add(base(0.3));
    m.rate(a.id, true); m.rate(a.id, true); m.rate(a.id, true);
    expect(a.reward).toBeCloseTo(1.3);
    m.rate(a.id, false);
    m.rate(a.id, true);
    expect(a.reward).toBeCloseTo(1.3);
  });
});
