import type { Backend } from "../src/brain";
import { normalize, schema } from "../src/brain";
import { Game, type GameEvent } from "../src/game";
import { Memory, memoryStore } from "../src/memory";
import { makeWorld } from "../src/world";

class Scripted implements Backend {
  readonly label = "scripted";
  prompts: string[] = [];
  constructor(private replies: object[]) {}
  async init(): Promise<void> { /* nothing to load */ }
  complete(_s: string, user: string): Promise<string> {
    this.prompts.push(user);
    return Promise.resolve(JSON.stringify(this.replies.shift() ?? { thought: "", plan: [], say: "" }));
  }
}

function setup(replies: object[]) {
  const w = makeWorld();
  for (const o of w.objs) if (o.kind === "berry" && o.x > 5) o.state = "gone";
  w.blob.x = 7; w.blob.y = 3;
  const brain = new Scripted(replies);
  const events: GameEvent[] = [];
  const game = new Game(w, new Memory(memoryStore()), brain, () => Promise.resolve(), (e) => events.push(e));
  return { w, brain, events, game };
}

const plan = (...steps: [string, string][]) => ({ thought: "t", say: "s", plan: steps.map(([d, a]) => ({ do: d, arg: a })) });

describe("game loop", () => {
  it("retries after failure, feeds the reason back, and learns the fix", async () => {
    const { brain, events, game } = setup([plan(["eat", "berry"]), plan(["morph", "spring"], ["eat", "berry"])]);
    await game.episode("get the berry");
    expect(brain.prompts).toHaveLength(2);
    expect(brain.prompts[1]).toContain("Your last plan failed: eat berry -> ");
    expect(brain.prompts[1]).toContain("too high");
    expect(events.some((e) => e.t === "learned")).toBe(true);
    expect(game.memory.insights[0]?.reason).toBe("too_high");
    // next time something is too high, the lesson shows up in the prompt
    const lessons = game.memory.recall("eat", new Set(["too_high"]));
    expect(lessons.join("\n")).toContain("morph spring, eat berry");
  });

  it("thumbs up rewards the attempt; thumbs down unlearns its insight", async () => {
    const { game } = setup([plan(["eat", "berry"]), plan(["morph", "spring"], ["eat", "berry"])]);
    await game.episode("get the berry");
    const a = game.rate(false);
    expect(a?.reward).toBeLessThan(0.5);
    expect(game.memory.insights).toHaveLength(0);
  });

  it("skills expand into their steps and are recalled for similar commands", async () => {
    const { game, w } = setup([plan(["morph", "spring"], ["eat", "berry"]), plan(["skill", "berry hop"])]);
    await game.episode("berry please");
    expect(game.teachSkill("Berry Hop")).toBe(true);
    w.blob.shape = "blob"; w.blob.x = 7; w.blob.y = 3;
    for (const o of w.objs) if (o.kind === "berry" && o.x <= 5) { o.state = "world"; }
    await game.episode("berry hop!");
    const last = game.memory.last();
    expect(last?.ok).toBe(true);
    expect(last?.executed.map((s) => s.do)).toEqual(["morph", "eat"]);
  });

  it("gives up after MAX_TRIES", async () => {
    const { brain, game } = setup([plan(["eat", "berry"]), plan(["eat", "berry"]), plan(["eat", "berry"]), plan(["eat", "berry"])]);
    await game.episode("get the berry");
    expect(brain.prompts).toHaveLength(3);
    expect(game.memory.attempts.every((a) => !a.ok)).toBe(true);
  });
});

describe("brain output", () => {
  it("normalize survives garbage", () => {
    expect(normalize("not json").plan).toEqual([]);
    expect(normalize('{"plan":[{"do":"fly","arg":"moon"},{"do":"rest"}]}').plan).toEqual([{ do: "rest", arg: "none" }]);
  });
  it("schema lists skills as allowed arguments", () => {
    expect(JSON.stringify(schema(["berry hop"]))).toContain("berry hop");
  });
});
