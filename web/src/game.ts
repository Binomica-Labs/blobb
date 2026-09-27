// The loop that ties it together: owner speaks -> blob thinks -> plan runs in the world ->
// results feed back into another try, into memory, and into what the blob will recall next time.

import { REASON_TEXT, planStep, type Anim, type Reason, type Step } from "./actions";
import { describePlan, normalize, schema, systemPrompt, type Backend, type Decision } from "./brain";
import type { Errand, Errands } from "./errands";
import type { Attempt, Memory, StepResult } from "./memory";
import { look, observe } from "./perception";
import type { World } from "./world";

export type GameEvent =
  | { t: "owner"; text: string }
  | { t: "thinking"; on: boolean }
  | { t: "decision"; d: Decision; expanded: string }
  | { t: "step"; text: string; ok: boolean; msg: string }
  | { t: "system"; text: string }
  | { t: "learned"; text: string }
  | { t: "attempt"; attempt: Attempt }
  | { t: "errand"; done: Errand; next: Errand };

export const MAX_TRIES = 3;
/** Failures worth learning a workaround for. */
const OBSTACLES: readonly Reason[] = ["too_high", "gap", "gate", "too_tired"];
export const IDLE_S = 25;
/** A brain that hasn't answered by now isn't going to (Ollama died, GPU hung...). */
export const BRAIN_TIMEOUT_S = 90;
/** Owner messages longer than this are cut - the brain is tiny and reads every token. */
const MAX_SAY = 200;

/** Ask the brain, giving up (and cancelling the request) after `s` seconds. */
function askWithTimeout(ask: (signal: AbortSignal) => Promise<string>, s: number): Promise<string> {
  const ctl = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const late = new Promise<never>((_, reject) => {
    timer = setTimeout(() => { ctl.abort(); reject(new Error(`no answer after ${String(s)}s`)); }, s * 1000);
  });
  return Promise.race([ask(ctl.signal), late]).finally(() => { clearTimeout(timer); });
}

export interface GameOptions { name: string; autonomy: boolean; useMemory: boolean }

export class Game {
  busy = false;
  idle = 0;
  opts: GameOptions = { name: "Blobb", autonomy: true, useMemory: true };
  /** The villagers' letters, if this game has them (tests and evals may not). */
  errands: Errands | null = null;
  private pending: string | null = null;
  private recent: string[] = [];
  /** Autonomous episodes that failed in a row; each one doubles the wait before trying again. */
  private streak = 0;
  /** Insights created by an attempt, so a thumbs-down can take them back. */
  private learnedBy = new Map<string, { reason: Reason; plan: Step[] }[]>();

  constructor(
    public world: World,
    public memory: Memory,
    public brain: Backend | null,
    private play: (a: Anim) => Promise<void>,
    private emit: (e: GameEvent) => void,
  ) {}

  /** Owner typed something. Interrupts whatever the blob is doing. */
  say(text: string): void {
    const t = text.trim().slice(0, MAX_SAY);
    if (!t) return;
    this.emit({ t: "owner", text: t });
    this.idle = 0;
    this.streak = 0;
    if (this.busy) { this.pending = t; return; }
    void this.episode(t);
  }

  /** Called every frame. Lets the blob act on its own when left alone (or hungry). */
  update(dt: number): void {
    if (this.busy || !this.brain || !this.opts.autonomy) { this.idle = 0; return; }
    this.idle += dt;
    const b = this.world.blob;
    const urgent = b.food < 25 || b.energy < 15;
    const wait = (urgent ? 8 : IDLE_S) * 2 ** Math.min(this.streak, 4);
    if (this.idle > wait) void this.episode(null);
  }

  private note(text: string): void {
    this.recent.push(text);
    if (this.recent.length > 3) this.recent.shift();
  }

  private expand(plan: readonly Step[]): Step[] {
    const out: Step[] = [];
    for (const s of plan) {
      if (s.do !== "skill") { out.push(s); continue; }
      const sk = this.memory.skill(s.arg);
      if (sk) { sk.uses++; out.push(...sk.steps.filter((x) => x.do !== "skill")); }
      else out.push(s); // unknown skill: let it fail visibly
    }
    return out.slice(0, 8);
  }

  async episode(command: string | null): Promise<void> {
    if (!this.brain) { this.emit({ t: "system", text: "No brain loaded yet." }); return; }
    this.busy = true;
    this.idle = 0;
    // If the island is reset mid-plan, stop: the plan was made for the old one.
    const world = this.world;
    let succeeded = false;
    const fails: { reason: Reason; target: string; text: string }[] = [];
    try {
      // A new command (pending) or a reset island ends the retries for this one.
      for (let attempt = 0; attempt < (command ? MAX_TRIES : 1) && this.world === world && !this.pending; attempt++) {
        const obstacles = new Set<Reason>(fails.map((f) => f.reason));
        for (const s of look(world)) if (s.blocked) obstacles.add(s.blocked);
        const lessons = this.opts.useMemory ? this.memory.recall(command, obstacles) : [];
        const skills = this.opts.useMemory ? this.memory.skillLines() : [];
        const [obsClean, obsFull] = observe(world, {
          ownerSaid: command, recent: this.recent, lessons, skills, retry: fails[fails.length - 1]?.text ?? null,
        });

        this.emit({ t: "thinking", on: true });
        let d: Decision;
        try {
          const brain = this.brain;
          const obs = brain.readsMemories === false ? obsClean : obsFull;
          const raw = await askWithTimeout((signal) => brain.complete(systemPrompt(this.opts.name), obs,
            schema(this.memory.skills.map((s) => s.name)), signal), BRAIN_TIMEOUT_S);
          d = normalize(raw);
        } finally {
          this.emit({ t: "thinking", on: false });
        }
        const executed = this.expand(d.plan);
        this.emit({ t: "decision", d, expanded: describePlan(executed) });

        const foodBefore = world.blob.food;
        const results: StepResult[] = [];
        let ok = executed.length > 0, ate = 0, interrupted = false;
        if (!ok) results.push({ step: "(nothing)", ok: false, msg: "You didn't plan anything.", reason: "unknown_step" });
        for (const step of executed) {
          if (this.pending || this.world !== world) { interrupted = true; break; }
          const out = planStep(world, step);
          for (const op of out.ops) {
            if (this.world !== world) break;
            op.apply(world);
            await this.play(op.anim);
          }
          if (this.world !== world) { interrupted = true; break; }
          const text = describePlan([step]);
          results.push({ step: text, ok: out.ok, msg: out.msg, reason: out.reason });
          this.emit({ t: "step", text, ok: out.ok, msg: out.msg });
          this.note(out.msg);
          if (!out.ok) { ok = false; break; }
          ate += out.ate;
        }
        if (interrupted) {
          // Something may have been delivered before the interruption - the villager still gets it.
          const errand = this.world === world ? this.errands?.check(world) : null;
          if (errand && this.errands) this.emit({ t: "errand", done: errand, next: this.errands.active() });
          break;
        }

        // World reward: finishing a plan is mildly good; eating when hungry is better.
        let reward = ok ? 0.3 : -0.3;
        if (ate > 0 && foodBefore < 50) reward += 0.5;
        // Finishing an errand is the best thing a blob can do - even if a later step flopped.
        const errand = this.errands?.check(world) ?? null;
        if (errand) reward += 1;
        const rec = this.memory.add({
          t: Date.now(), command, obsClean, decision: d, executed, results, ok,
          reward: Math.round(reward * 10) / 10,
        });
        this.emit({ t: "attempt", attempt: rec });
        if (errand && this.errands) this.emit({ t: "errand", done: errand, next: this.errands.active() });

        if (ok) {
          const learned: { reason: Reason; plan: Step[] }[] = [];
          // Only real obstacles, and only if this plan actually reached the thing that was blocked.
          for (const f of fails) {
            if (!OBSTACLES.includes(f.reason) || learned.some((l) => l.reason === f.reason)) continue;
            if (!executed.some((s) => s.arg === f.target)) continue;
            this.memory.learn(f.reason, executed);
            learned.push({ reason: f.reason, plan: executed });
            this.emit({ t: "learned", text: `Learned: when ${REASON_TEXT[f.reason]}, try "${describePlan(executed)}"` });
          }
          if (learned.length) this.learnedBy.set(rec.id, learned);
          succeeded = true;
          break;
        }
        const bad = results.find((r) => !r.ok);
        const badStep = executed[results.length - 1];
        fails.push({
          reason: bad?.reason ?? "cant_do", target: badStep?.arg ?? "none",
          text: `${bad?.step ?? "?"} -> ${bad?.msg ?? "it didn't work."}`,
        });
      }
    } catch (e) {
      this.emit({ t: "system", text: `Brain error: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      if (!command && !this.pending) this.streak = succeeded ? 0 : this.streak + 1;
      this.busy = false;
      this.idle = 0;
      const next = this.pending;
      this.pending = null;
      if (next) void this.episode(next);
    }
  }

  /** Owner's thumbs up/down on the latest attempt. */
  rate(good: boolean): Attempt | undefined {
    const a = this.memory.last();
    if (!a) { this.emit({ t: "system", text: "Nothing to rate yet." }); return undefined; }
    this.memory.rate(a.id, good);
    if (!good) {
      for (const l of this.learnedBy.get(a.id) ?? []) this.memory.unlearn(l.reason, l.plan);
      this.learnedBy.delete(a.id);
    }
    this.emit({ t: "system", text: good ? `${this.opts.name} feels proud! (+1)` : `${this.opts.name} looks sheepish. (-1)` });
    return a;
  }

  /** Name the last successful plan so the blob can reuse it as one step. */
  teachSkill(name: string): boolean {
    const a = [...this.memory.attempts].reverse().find((x) => x.ok && x.executed.length);
    if (!a) { this.emit({ t: "system", text: "Nothing to name yet - wait for a plan that works." }); return false; }
    const s = this.memory.addSkill(name, a.executed);
    if (!s) {
      this.emit({ t: "system", text: `"${name.trim()}" won't work as a skill name - use letters, and not the name of a thing or shape.` });
      return false;
    }
    this.emit({ t: "learned", text: `New skill "${s.name}": ${describePlan(s.steps)}` });
    return true;
  }
}

