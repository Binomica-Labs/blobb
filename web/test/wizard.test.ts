// The wizard's side of things: delivering to the tower, villagers' errands, and peeking into dreams.
import { runStep } from "../src/actions";
import { GLOW_HI, GLOW_LO, describe as dreamSays, glow, heat, sparkline } from "../src/dream";
import { Errands, STORY_LENGTH } from "../src/errands";
import { Game } from "../src/game";
import { Memory, memoryStore } from "../src/memory";
import { look } from "../src/perception";
import { makeWorld, restoreWorld, setTile, tick, type World } from "../src/world";

const at = (w: World, x: number, y: number) => { w.blob.x = x; w.blob.y = y; };
const run = (w: World, ...steps: [string, string][]) => steps.map(([d, a]) => runStep(w, { do: d as never, arg: a }));

describe("delivering to the tower", () => {
  it("grab the stick, deliver it: the wizard gets it and it comes back later", () => {
    const w = makeWorld();
    const [grab, give] = run(w, ["grab", "stick"], ["deliver", "none"]);
    expect(grab?.ok).toBe(true);
    expect(give?.ok).toBe(true);
    expect(w.blob.holding).toBeNull();
    expect(w.deliveries).toEqual([{ kind: "stick", home: [8, 13] }]);
    tick(w, 120);
    expect(w.objs.find((o) => o.kind === "stick")?.state).toBe("world");
  });

  it("delivering with empty hands fails and says why", () => {
    const out = runStep(makeWorld(), { do: "deliver", arg: "none" });
    expect(out.reason).toBe("not_holding");
  });

  it("the blob can see the tower and knows what it's for", () => {
    const seen = look(makeWorld()).find((s) => s.kind === "tower");
    expect(seen?.note).toContain("deliver");
  });

  it("the tower is solid and can't be pushed", () => {
    const w = makeWorld();
    expect(runStep(w, { do: "push", arg: "tower" }).ok).toBe(false);
    at(w, 7, 8);
    expect(runStep(w, { do: "move_to", arg: "tower" }).ok).toBe(true);
    expect([w.blob.x, w.blob.y]).not.toEqual([7, 7]);
  });

  it("only the last few unclaimed deliveries are kept", () => {
    const w = makeWorld();
    for (let i = 0; i < 15; i++) {
      w.blob.food = 50; w.blob.energy = 100;
      const stick = w.objs.find((o) => o.kind === "stick");
      if (stick) { stick.state = "world"; stick.x = 8; stick.y = 8; }
      run(w, ["grab", "stick"], ["deliver", "none"]);
    }
    expect(w.deliveries.length).toBeLessThanOrEqual(10);
  });

  it("old saves get a tower built and sticks that come back", () => {
    // A save from before deliveries existed.
    const old = JSON.parse(JSON.stringify({ ...makeWorld(), savedAt: 0 }), (k, v: unknown) => (k === "deliveries" ? undefined : v)) as World;
    old.objs = old.objs.filter((o) => o.kind !== "tower");
    for (const o of old.objs) if (o.kind === "stick") o.home = null;
    const w = restoreWorld(old, 0);
    expect(w.objs.some((o) => o.kind === "tower" && o.x === 7 && o.y === 7)).toBe(true);
    expect(w.objs.find((o) => o.kind === "stick")?.home).not.toBeNull();
    expect(w.deliveries).toEqual([]);
  });
});

describe("errands", () => {
  it("the first letter is the wizard's own: fetch the walking stick", () => {
    const e = new Errands(memoryStore());
    const w = makeWorld();
    expect(e.active().id).toBe("stick");
    expect(e.check(w)).toBeNull();
    run(w, ["grab", "stick"], ["deliver", "none"]);
    expect(e.check(w)?.id).toBe("stick");
    expect(e.active().id).toBe("soup");
    expect(w.deliveries).toEqual([]);
  });

  it("where a thing came from matters: a field berry isn't a meadow berry", () => {
    const e = new Errands(memoryStore());
    e.finished = 2; // Bram wants a berry from the high meadow
    const w = makeWorld();
    w.deliveries.push({ kind: "berry", home: [9, 6] });
    expect(e.check(w)).toBeNull();
    w.deliveries.push({ kind: "berry", home: [2, 2] });
    expect(e.check(w)?.id).toBe("meadow");
    expect(w.deliveries).toEqual([{ kind: "berry", home: [9, 6] }]);
  });

  it("the ferryman is happy once the gap is bridged", () => {
    const e = new Errands(memoryStore());
    e.finished = 5;
    const w = makeWorld();
    expect(e.check(w)).toBeNull();
    setTile(w, 11, 11, { kind: "ground", h: 0 });
    expect(e.check(w)?.id).toBe("bridge");
  });

  it("after the story, letters keep coming and progress survives reloads", () => {
    const store = memoryStore();
    const e = new Errands(store);
    e.finished = STORY_LENGTH + 3;
    const w = makeWorld();
    w.deliveries.push({ kind: "mushroom", home: [11, 3] }, { kind: "berry", home: [2, 2] }, { kind: "berry", home: null },
      { kind: "mushroom", home: [13, 11] });
    expect(e.check(w)).not.toBeNull();
    expect(new Errands(store).finished).toBe(STORY_LENGTH + 4);
    store.save("garbage");
    expect(new Errands(store).finished).toBe(0);
  });

  it("finishing an errand during a plan boosts that attempt's reward and announces the next letter", async () => {
    const w = makeWorld();
    const reply = { thought: "t", say: "s", plan: [{ do: "grab", arg: "stick" }, { do: "deliver", arg: "none" }] };
    const brain = { label: "x", init: () => Promise.resolve(), complete: () => Promise.resolve(JSON.stringify(reply)) };
    const events: { t: string }[] = [];
    const g = new Game(w, new Memory(memoryStore()), brain, () => Promise.resolve(), (e) => events.push(e));
    g.errands = new Errands(memoryStore());
    await g.episode("fetch my walking stick");
    expect(g.memory.last()?.reward).toBeGreaterThan(1);
    expect(events.some((e) => e.t === "errand")).toBe(true);
  });
});

describe("errands and interruptions", () => {
  it("a delivery still counts if the wizard interrupts right after it", async () => {
    const w = makeWorld();
    const reply = { thought: "t", say: "s", plan: [{ do: "grab", arg: "stick" }, { do: "deliver", arg: "none" }, { do: "rest", arg: "none" }] };
    const brain = { label: "x", init: () => Promise.resolve(), complete: () => Promise.resolve(JSON.stringify(reply)) };
    const events: { t: string }[] = [];
    let g: Game | null = null;
    const play = () => { if (w.deliveries.length && g && !g.busy) return Promise.resolve(); if (w.deliveries.length) g?.say("stop"); return Promise.resolve(); };
    g = new Game(w, new Memory(memoryStore()), brain, play, (e) => events.push(e));
    g.errands = new Errands(memoryStore());
    await g.episode("fetch my walking stick");
    expect(events.some((e) => e.t === "errand")).toBe(true);
  });
});

describe("dream peek", () => {
  const now = 1_000_000;
  it("says how far along a dream is", () => {
    const v = dreamSays({ phase: "dreaming", t: now - 10, gen: 2, step: 30, total: 300, eta: 5400 }, now, "Blobb");
    expect(v.kind).toBe("asleep");
    expect(v.progress).toBeCloseTo(0.1);
    expect(v.detail).toContain("1h 30m");
    expect(v.title).toContain("generation 2");
  });

  it("notices when the trainer has gone quiet", () => {
    expect(dreamSays({ phase: "dreaming", t: now - 3600, step: 3, total: 300 }, now, "B").kind).toBe("stalled");
    expect(dreamSays({ phase: "done", t: now - 86400, model: "blobb-gen1" }, now, "B").kind).toBe("done");
    expect(dreamSays({ phase: "error", t: now, error: "boom" }, now, "B").detail).toBe("boom");
  });

  it("says how much of its instinct its own brain kept, and warns when a dream wrecked it", () => {
    const ok = dreamSays({ phase: "done", t: now, model: "own:gen2", instinct: [270, 266, 300] }, now, "B");
    expect(ok.detail).toContain("266 of 300");
    expect(ok.detail).not.toContain("forgot");
    const bad = dreamSays({ phase: "done", t: now, model: "own:gen2", instinct: [270, 180, 300] }, now, "B");
    expect(bad.detail).toContain("forgot");
    expect(dreamSays({ phase: "done", t: now, model: "own:gen2" }, now, "B").detail).toBe("Its brain grew a little in the night.");
  });

  it("glow is per weight and on a fixed scale, so big matrices don't win and the map brightens over time", () => {
    expect(glow(0, 100)).toBe(0);
    expect(glow(1, 0)).toBe(0);
    expect(glow(GLOW_LO * 10, 100 * 100)).toBe(0); // spread over 10k weights, that is tiny per weight
    expect(glow(GLOW_HI * 100, 100 * 100)).toBe(1);
    // Same per-weight change, 5x the weights: same glow.
    expect(glow(3e-4 * 100, 100 ** 2)).toBeCloseTo(glow(3e-4 * Math.sqrt(5e4), 5e4));
    expect(glow(1e-4 * 100, 1e4)).toBeLessThan(glow(5e-4 * 100, 1e4));
  });

  it("colours and curves stay sane at the edges", () => {
    expect(heat(0)).toBe("rgb(28, 21, 64)");
    expect(heat(1)).toBe("rgb(255, 211, 107)");
    expect(heat(-1)).toBe(heat(0));
    expect(heat(7)).toBe(heat(1));
    expect(sparkline([1], 300, 60)).toBe("");
    expect(sparkline([2, 2, 2], 300, 60).split(" ")).toHaveLength(3);
  });
});
