// Everything the blob remembers. Three kinds of learning live here:
//   attempts  - every plan it tried, how it went, and how the owner felt about it
//   insights  - "when X blocked me, Y got me past it", learned from failing then succeeding
//   skills    - plans the owner named, callable later as a single step
// Attempts with positive reward are also what dreams train into the weights.

import { REASON_TEXT, VERBS, type Reason, type Step } from "./actions";
import { canonical, describePlan, systemPrompt, type Decision } from "./brain";
import { SHAPES, THINGS } from "./world";

export interface StepResult { step: string; ok: boolean; msg: string; reason: Reason | null }

export interface Attempt {
  id: string;
  t: number;
  command: string | null;
  obsClean: string;
  decision: Decision;
  /** The plan as actually run, with skills expanded - the training target. */
  executed: Step[];
  results: StepResult[];
  ok: boolean;
  /** What dreams and recall go by: the world's reward, or the wizard's verdict once he's given one. */
  reward: number;
  /** The world's own reward (finished the plan, ate, did an errand), kept so a verdict can change. */
  worldReward?: number;
  rated: boolean;
  dreamed: boolean;
}

export interface Insight { reason: Reason; fix: string; count: number; t: number }
export interface Skill { name: string; steps: Step[]; uses: number }

interface Saved { attempts: Attempt[]; insights: Insight[]; skills: Skill[] }

export interface Store { load(): string | null; save(data: string): void }

export const localStore = (key = "blobb.memory.v1"): Store => ({
  load: () => { try { return localStorage.getItem(key); } catch { return null; } },
  save: (data) => { try { localStorage.setItem(key, data); } catch { /* full or blocked: keep going in memory */ } },
});

export const memoryStore = (): Store => {
  let data: string | null = null;
  return { load: () => data, save: (d) => { data = d; } };
};

const STOP = new Set(["a", "an", "the", "do", "please", "now", "can", "you", "me", "to", "go", "and", "on",
  "i", "want", "your", "my", "it", "of", "for", "blobb", "hey", "ok", "okay", "that", "this", "some", "get"]);

export function words(text: string | null): Set<string> {
  return new Set((text ?? "").toLowerCase().match(/[a-z']+/g)?.filter((w) => !STOP.has(w)) ?? []);
}

function similarity(a: Set<string>, b: Set<string>): number {
  if (!a.size || !b.size) return 0;
  let both = 0;
  for (const w of a) if (b.has(w)) both++;
  return both / (a.size + b.size - both);
}

const MAX_ATTEMPTS = 500;

/** Words a skill can't be called: they're already plan arguments, so "skill berry" would be ambiguous. */
const RESERVED = new Set<string>(["none", ...THINGS, ...SHAPES, ...VERBS]);

/** Lowercase, safe characters, at least one letter, and not a reserved word - or null. */
export function cleanSkillName(name: string): string | null {
  const clean = name.toLowerCase().replace(/[^a-z0-9 '-]/g, "").replace(/\s+/g, " ").trim().slice(0, 24).trim();
  return /[a-z]/.test(clean) && !RESERVED.has(clean) ? clean : null;
}

// ---------- loading old or hand-edited saves without trusting them ----------

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const isStep = (v: unknown): v is Step =>
  isRecord(v) && (VERBS as readonly unknown[]).includes(v["do"]) && typeof v["arg"] === "string";
const isAttempt = (v: unknown): v is Attempt =>
  isRecord(v) && typeof v["id"] === "string" && typeof v["t"] === "number" && typeof v["ok"] === "boolean"
  && typeof v["reward"] === "number" && typeof v["obsClean"] === "string"
  && (v["command"] === null || typeof v["command"] === "string") && isRecord(v["decision"])
  && Array.isArray(v["executed"]) && v["executed"].every(isStep) && Array.isArray(v["results"]);
const isInsight = (v: unknown): v is Insight =>
  isRecord(v) && typeof v["reason"] === "string" && v["reason"] in REASON_TEXT && typeof v["fix"] === "string"
  && typeof v["count"] === "number" && typeof v["t"] === "number";
const isSkill = (v: unknown): v is Skill =>
  isRecord(v) && typeof v["name"] === "string" && cleanSkillName(v["name"]) === v["name"]
  && Array.isArray(v["steps"]) && v["steps"].every(isStep) && typeof v["uses"] === "number";

export class Memory {
  attempts: Attempt[] = [];
  insights: Insight[] = [];
  skills: Skill[] = [];

  constructor(private store: Store) {
    const raw = store.load();
    if (!raw) return;
    try {
      const d: unknown = JSON.parse(raw);
      if (!isRecord(d)) return;
      this.attempts = list(d["attempts"]).filter(isAttempt);
      this.insights = list(d["insights"]).filter(isInsight);
      this.skills = list(d["skills"]).filter(isSkill);
      for (const a of this.attempts as (Partial<Attempt> & { obsFull?: unknown })[]) {
        a.rated = a.rated === true; a.dreamed = a.dreamed === true;
        delete a.obsFull; // older saves kept the prompt-with-memories too: 40% of each attempt, never read
        if (typeof a.worldReward !== "number" || !Number.isFinite(a.worldReward)) delete a.worldReward;
      }
    } catch { /* corrupt save: start fresh rather than crash */ }
  }

  save(): void {
    if (this.attempts.length > MAX_ATTEMPTS) {
      // Forget unremarkable attempts first; keep anything the owner rated.
      const keep = this.attempts.filter((a) => a.rated || a.reward > 0).slice(-MAX_ATTEMPTS);
      const rest = this.attempts.filter((a) => !(a.rated || a.reward > 0));
      const room = MAX_ATTEMPTS - keep.length;
      this.attempts = [...keep, ...rest.slice(rest.length - room)].sort((a, b) => a.t - b.t);
    }
    const data: Saved = { attempts: this.attempts, insights: this.insights, skills: this.skills };
    this.store.save(JSON.stringify(data));
  }

  add(a: Omit<Attempt, "id" | "rated" | "dreamed">): Attempt {
    const full: Attempt = { ...a, id: Math.random().toString(36).slice(2, 10), rated: false, dreamed: false };
    this.attempts.push(full);
    this.save();
    return full;
  }

  last(): Attempt | undefined {
    return this.attempts[this.attempts.length - 1];
  }

  /** The wizard's verdict replaces the last one rather than piling up: 👎 always means "don't",
   *  however well the world rewarded it; 👍 is one point better than the world's opinion. */
  rate(id: string, good: boolean): void {
    const a = this.attempts.find((x) => x.id === id);
    if (!a) return;
    a.worldReward ??= a.reward;
    a.reward = good ? Math.min(3, Math.max(0, a.worldReward) + 1) : -1;
    a.rated = true;
    this.save();
  }

  /** Remember what got the blob past an obstacle. */
  learn(reason: Reason, plan: readonly Step[]): void {
    const fix = describePlan(plan);
    const hit = this.insights.find((i) => i.reason === reason && i.fix === fix);
    if (hit) { hit.count++; hit.t = Date.now(); } else this.insights.push({ reason, fix, count: 1, t: Date.now() });
    this.save();
  }

  unlearn(reason: Reason, plan: readonly Step[]): void {
    const fix = describePlan(plan);
    this.insights = this.insights.filter((i) => !(i.reason === reason && i.fix === fix));
    this.save();
  }

  /** Null if the name is unusable (see cleanSkillName). */
  addSkill(name: string, steps: readonly Step[]): Skill | null {
    const clean = cleanSkillName(name);
    if (!clean) return null;
    const existing = this.skills.find((s) => s.name === clean);
    if (existing) { existing.steps = [...steps]; this.save(); return existing; }
    const s: Skill = { name: clean, steps: [...steps], uses: 0 };
    this.skills.push(s);
    this.save();
    return s;
  }

  removeSkill(name: string): void {
    this.skills = this.skills.filter((s) => s.name !== name);
    this.save();
  }

  skill(name: string): Skill | undefined {
    return this.skills.find((s) => s.name === name);
  }

  /** Lessons to put in the prompt: rated reactions to similar commands, and fixes for current obstacles. */
  recall(command: string | null, obstacles: ReadonlySet<Reason>, k = 4): string[] {
    const out: string[] = [];
    const seen = new Set<string>();
    const cmd = words(command);
    if (cmd.size) {
      const scored = this.attempts
        .filter((a) => a.command && (a.rated || !a.ok))
        .map((a) => ({ a, s: similarity(cmd, words(a.command)) }))
        .filter(({ s }) => s >= 0.34)
        .sort((p, q) => q.s - p.s || Math.abs(q.a.reward) - Math.abs(p.a.reward) || q.a.t - p.a.t);
      for (const { a } of scored) {
        const plan = describePlan(a.executed);
        const verdict = a.reward > 0 ? "the wizard LOVED it"
          : a.reward < 0 ? "the wizard did NOT like it"
          : `it failed (${a.results.find((r) => !r.ok)?.msg ?? "no luck"})`;
        const line = `When the wizard said "${a.command ?? ""}", you did: ${plan} - ${verdict}.`;
        if (seen.has(plan + verdict)) continue;
        seen.add(plan + verdict);
        out.push(line);
        if (out.length >= k - 1) break;
      }
    }
    for (const reason of obstacles) {
      const best = this.insights.filter((i) => i.reason === reason).sort((p, q) => q.count - p.count || q.t - p.t)[0];
      if (best) out.push(`When ${REASON_TEXT[reason]}, this worked before: ${best.fix}.`);
    }
    return out.slice(0, k + 1);
  }

  skillLines(): string[] {
    return this.skills.map((s) => `${s.name} = ${describePlan(s.steps)}`);
  }

  /** Training data for a dream: observation without recalled memories -> the plan that earned praise.
   *  Same shape as the curriculum, so the trainer treats them alike. */
  exportForDream(name: string): object {
    const system = systemPrompt(name);
    return {
      version: 1,
      name,
      exported: new Date().toISOString(),
      samples: this.attempts.filter((a) => a.reward > 0 && !a.dreamed).map((a) => ({
        id: a.id, system, user: a.obsClean, reward: a.reward, owner: a.rated,
        assistant: canonical({ thought: a.decision.thought, plan: a.executed, say: a.decision.say }),
      })),
    };
  }

  /** Exported memories are marked so the next dream doesn't train on them again. */
  markDreamed(ids: readonly string[]): void {
    const set = new Set(ids);
    for (const a of this.attempts) if (set.has(a.id)) a.dreamed = true;
    this.save();
  }

  forget(): void {
    this.attempts = []; this.insights = []; this.skills = [];
    this.save();
  }
}
