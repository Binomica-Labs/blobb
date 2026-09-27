// The browser's copy of the blob's own brain must think exactly what the trainer's does: same tokens,
// same n-gram hashes, same decisions from the same fp16 weights (fixtures written by `brain.py export`).
import { readFileSync } from "node:fs";
import { crc32, ngrams, ScratchBrain, tokenize, type Manifest } from "../src/scratch";
import type { Step } from "../src/actions";

const dir = new URL("../public/brain/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("brain.json", dir), "utf8")) as Manifest;
const bin = readFileSync(new URL("brain.bin", dir));
const fixtures = JSON.parse(readFileSync(new URL("./fixtures/scratch.json", import.meta.url), "utf8")) as { user: string; decision: unknown }[];

describe("the blob's own brain", () => {
  it("tokenizes and hashes like the trainer", () => {
    expect(tokenize('Food 30 (hungry). The wizard says: "Get the BERRY, blob!"')).toEqual(
      ["food", "30", "hungry", "the", "wizard", "says", '"', "get", "the", "berry", "blob", "!", '"']);
    expect(crc32("<be")).toBe(0xb8235914); // zlib.crc32(b"<be")
    expect(ngrams("42", 8192, 24)).toEqual([]);
    expect(ngrams("berry", 8192, 24)).toHaveLength(12);
  });

  it("reproduces the trainer's decisions", { timeout: 60_000 }, () => {
    const brain = new ScratchBrain(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
    expect(fixtures.length).toBeGreaterThan(5);
    for (const f of fixtures) expect(brain.decide(f.user)).toEqual(f.decision);
  });

  const weights = () => bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  const brain = new ScratchBrain(manifest, weights());
  const legal = (plan: Step[], m: Manifest = manifest) => {
    expect(plan.length).toBeGreaterThan(0);
    expect(plan.length).toBeLessThanOrEqual(4);
    for (const s of plan) {
      const v = m.out.indexOf(`v:${s.do}`), a = m.out.indexOf(`a:${s.arg}`);
      expect(m.pairs[String(v)]).toContain(a);
    }
  };

  it("always answers with a legal plan, even to nothing or to a wall of text", { timeout: 60_000 }, () => {
    legal(brain.decide("").plan);
    legal(brain.decide("!!! ??? 12345 ✨✨✨").plan);
    const long = fixtures[0]?.user.repeat(40) ?? "";
    expect(tokenize(long).length).toBeGreaterThan(manifest.cfg.maxLen);
    legal(brain.decide(long).plan);
  });

  it("sampling stays legal", { timeout: 60_000 }, () => {
    let seed = 1;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (const f of fixtures.slice(0, 10)) {
      for (let i = 0; i < 3; i++) {
        const d = brain.decide(f.user, 1.5, rnd);
        legal(d.plan);
        expect(manifest.says).toContain(d.say);
      }
    }
  });

  it("never starts a step it knows no argument for", { timeout: 60_000 }, () => {
    const f = fixtures[0];
    if (!f) throw new Error("no fixtures");
    const first = brain.decide(f.user).plan[0];
    if (!first) throw new Error("empty plan");
    const v = String(manifest.out.indexOf(`v:${first.do}`));
    const pairs = Object.fromEntries(Object.entries(manifest.pairs).filter(([k]) => k !== v));
    const m = { ...manifest, pairs };
    const d = new ScratchBrain(m, weights()).decide(f.user);
    expect(d.plan[0]?.do).not.toBe(first.do);
    legal(d.plan, m);
  });

  it("refuses a damaged or foreign brain file with a clear message", () => {
    const w = weights();
    expect(() => new ScratchBrain(manifest, w.slice(0, w.byteLength / 2))).toThrow(/damaged/);
    expect(() => new ScratchBrain(manifest, w.slice(0, w.byteLength - 1))).toThrow(/damaged/);
    expect(() => new ScratchBrain({ ...manifest, version: 2 }, w)).toThrow(/damaged/);
    expect(() => new ScratchBrain(null as unknown as Manifest, w)).toThrow(/damaged/);
    const tensors = Object.fromEntries(Object.entries(manifest.tensors).filter(([k]) => k !== "plan_head.weight"));
    expect(() => new ScratchBrain({ ...manifest, tensors }, w)).toThrow(/plan_head/);
  });
});
