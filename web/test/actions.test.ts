import { runStep, findPath, reachGoal } from "../src/actions";
import { look, observe } from "../src/perception";
import { makeWorld, objsAt, tileAt, type World } from "../src/world";

const at = (w: World, x: number, y: number) => { w.blob.x = x; w.blob.y = y; };
/** Hide the easy snacks near the start so tests exercise the puzzle food. */
const noSnacks = (w: World) => { for (const o of w.objs) if ((o.x === 9 && o.y === 6) || (o.x === 3 && o.y === 8)) o.state = "gone"; };
const count = (w: World, kind: string) => w.objs.filter((o) => o.kind === kind && o.state === "world").length;

describe("plateau berries (too high)", () => {
  it("blob fails on the plateau berry with too_high; spring succeeds", () => {
    const w = makeWorld();
    for (const o of w.objs) if (o.kind === "berry" && o.x > 5) o.state = "gone";
    at(w, 7, 3);
    const fail = runStep(w, { do: "eat", arg: "berry" });
    expect(fail.ok).toBe(false);
    expect(fail.reason).toBe("too_high");
    expect(runStep(w, { do: "morph", arg: "spring" }).ok).toBe(true);
    const food = w.blob.food;
    const ok = runStep(w, { do: "eat", arg: "berry" });
    expect(ok.ok).toBe(true);
    expect(w.blob.food).toBeGreaterThan(food);
  });
  it("a stick can hook a plateau berry down", () => {
    const w = makeWorld();
    for (const o of w.objs) if (o.kind === "berry" && o.x > 5) o.state = "gone";
    at(w, 7, 5);
    expect(runStep(w, { do: "grab", arg: "stick" }).ok).toBe(true);
    const pull = runStep(w, { do: "use", arg: "berry" });
    expect(pull.ok).toBe(true);
    expect(runStep(w, { do: "eat", arg: "berry" }).ok).toBe(true);
  });
});

describe("walled garden (gate)", () => {
  it("only a puddle gets under the gate", () => {
    const w = makeWorld();
    noSnacks(w);
    at(w, 11, 7);
    const fail = runStep(w, { do: "eat", arg: "mushroom" });
    // the SE mushroom is across the gap, the garden one behind the gate: either way, blocked
    expect(fail.ok).toBe(false);
    expect(["gate", "gap"]).toContain(fail.reason);
    runStep(w, { do: "morph", arg: "puddle" });
    const ok = runStep(w, { do: "eat", arg: "mushroom" });
    expect(ok.ok).toBe(true);
    expect(w.blob.y).toBeLessThan(5); // it's inside the garden
  });
  it("can't morph while squeezed under the gate", () => {
    const w = makeWorld();
    at(w, 11, 6);
    runStep(w, { do: "morph", arg: "puddle" });
    runStep(w, { do: "move_to", arg: "mushroom" });
    at(w, 11, 5);
    expect(runStep(w, { do: "morph", arg: "blob" }).ok).toBe(false);
  });
});

describe("gap pocket", () => {
  it("pushing the rock fills the gap and makes a bridge", () => {
    const w = makeWorld();
    noSnacks(w);
    for (const o of w.objs) if (o.kind === "mushroom" && o.y < 5) o.state = "gone";
    at(w, 7, 9);
    expect(runStep(w, { do: "eat", arg: "mushroom" }).reason).toBe("gap");
    const push = runStep(w, { do: "push", arg: "rock" });
    expect(push.ok).toBe(true);
    expect(push.msg).toMatch(/fills/);
    expect(tileAt(w, 11, 11).kind).toBe("ground");
    expect(runStep(w, { do: "eat", arg: "mushroom" }).ok).toBe(true);
  });
  it("a spring can jump the gap", () => {
    const w = makeWorld();
    noSnacks(w);
    for (const o of w.objs) if (o.kind === "mushroom" && o.y < 5) o.state = "gone";
    at(w, 9, 12);
    runStep(w, { do: "morph", arg: "spring" });
    expect(runStep(w, { do: "eat", arg: "mushroom" }).ok).toBe(true);
    expect(w.blob.x).toBeGreaterThan(11);
  });
});

describe("tree", () => {
  it("a blob push only wobbles it; a ball knocks fruit down", () => {
    const w = makeWorld();
    at(w, 7, 12);
    expect(runStep(w, { do: "push", arg: "tree" }).ok).toBe(false);
    const before = count(w, "berry");
    runStep(w, { do: "morph", arg: "ball" });
    expect(runStep(w, { do: "push", arg: "tree" }).ok).toBe(true);
    expect(count(w, "berry")).toBe(before + 1);
  });
  it("a stick knocks fruit down too", () => {
    const w = makeWorld();
    at(w, 8, 12);
    runStep(w, { do: "grab", arg: "stick" });
    const before = count(w, "berry");
    expect(runStep(w, { do: "use", arg: "tree" }).ok).toBe(true);
    expect(count(w, "berry")).toBe(before + 1);
  });
});

describe("misc", () => {
  it("drop puts the held thing down", () => {
    const w = makeWorld();
    at(w, 8, 13);
    runStep(w, { do: "grab", arg: "stick" });
    expect(runStep(w, { do: "drop", arg: "none" }).ok).toBe(true);
    expect(objsAt(w, w.blob.x, w.blob.y).some((o) => o.kind === "stick")).toBe(true);
  });
  it("walking costs energy and a tired blob refuses long trips", () => {
    const w = makeWorld();
    w.blob.energy = 1;
    at(w, 2, 13);
    const out = runStep(w, { do: "move_to", arg: "rock" });
    expect(out.reason).toBe("too_tired");
  });
  it("paths never go through water", () => {
    const w = makeWorld();
    expect(findPath(w, "blob", (x, y) => x === 0 && y === 0)).toBeNull();
    expect(findPath(w, "spring", reachGoal(w, 12, 2, "berry"))).toBeNull();
  });
  it("perception flags what can't be reached", () => {
    const w = makeWorld();
    at(w, 6, 3);
    const berry = look(w).find((s) => s.kind === "berry" && s.dh >= 2);
    expect(berry?.blocked).toBe("too_high");
    const [clean, full] = observe(w, { ownerSaid: "hi", recent: [], lessons: ["x"], skills: [], retry: null });
    expect(clean).not.toContain("learned");
    expect(full).toContain("Things you learned");
    expect(clean).toContain('The wizard says: "hi"');
  });
});

describe("tree with no room for fruit", () => {
  it("shaking fails honestly instead of claiming a berry fell", () => {
    const w = makeWorld();
    const tree = w.objs.find((o) => o.kind === "tree");
    if (!tree) throw new Error("no tree");
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      w.objs.push({ id: w.nextId++, kind: "berry", x: tree.x + dx, y: tree.y + dy, state: "world", home: null, respawnAt: 0, fruit: 0, regrowAt: 0 });
    }
    at(w, 7, 12);
    const before = w.objs.length;
    runStep(w, { do: "morph", arg: "ball" });
    const out = runStep(w, { do: "push", arg: "tree" });
    expect(out.ok).toBe(false);
    expect(out.msg).toContain("no room");
    expect(w.objs.length).toBe(before);
  });
});
