// The blob's mind: a small LLM that turns what it sees into a short plan.
// Output is constrained to a JSON schema so even a 0.5B model always answers in a usable shape.

import { VERBS, type Step, type Verb } from "./actions";
import { SHAPES, THINGS } from "./world";

export interface Decision { thought: string; plan: Step[]; say: string }

export const MAX_STEPS = 4;

export function systemPrompt(name: string): string {
  return [
    `You are ${name}, a squishy blob a grumpy old wizard summoned from another dimension with his spellbook.`,
    "He hates people, so he sends you on errands for the villagers. You try to do what he asks, and learn to use your body and the things around you.",
    "Reply with a short thought, a plan of 1-4 steps, and something short and cute to say.",
    "Steps:",
    "- move_to <thing>: go next to a thing",
    "- morph <shape>: change your shape",
    "- eat <food>, grab <thing>, drop",
    "- push <thing>: shove a rock or tree",
    "- use <thing>: use what you are holding on a thing",
    "- deliver: bring what you are holding to the wizard's tower",
    "- rest: get energy back",
    "- skill <name>: do a skill you learned",
    "Shapes:",
    "- blob: walks, climbs small steps",
    "- ball: rolls fast and hits hard, can't climb",
    "- puddle: flat and slow, slides under low things",
    "- spring: bouncy, jumps over gaps and up high ledges",
    'Example: {"thought":"The berry is up high. Spring can jump up.","plan":[{"do":"morph","arg":"spring"},{"do":"eat","arg":"berry"}],"say":"Boing!"}',
  ].join("\n");
}

/** Schema with the current skill names allowed as arguments. */
export function schema(skills: readonly string[]): object {
  return {
    type: "object",
    properties: {
      thought: { type: "string" },
      plan: {
        type: "array",
        minItems: 1,
        maxItems: MAX_STEPS,
        items: {
          type: "object",
          properties: {
            do: { type: "string", enum: VERBS },
            arg: { type: "string", enum: ["none", ...THINGS, ...SHAPES, ...skills] },
          },
          required: ["do", "arg"],
        },
      },
      say: { type: "string" },
    },
    required: ["thought", "plan", "say"],
  };
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** Coerce whatever came back into a valid Decision. Never throws. */
export function normalize(raw: string): Decision {
  let v: unknown;
  try { v = JSON.parse(raw); } catch { v = null; }
  if (!isRecord(v)) return { thought: "...huh?", plan: [], say: "Blub?" };
  const plan: Step[] = [];
  if (Array.isArray(v["plan"])) {
    for (const s of v["plan"] as unknown[]) {
      if (!isRecord(s)) continue;
      const verb = s["do"], arg = s["arg"];
      if (typeof verb === "string" && (VERBS as readonly string[]).includes(verb)) {
        plan.push({ do: verb as Verb, arg: typeof arg === "string" ? arg : "none" });
      }
    }
  }
  return {
    thought: (typeof v["thought"] === "string" ? v["thought"] : "").slice(0, 160),
    plan: plan.slice(0, MAX_STEPS),
    say: (typeof v["say"] === "string" ? v["say"] : "").slice(0, 80),
  };
}

/** Canonical JSON for a decision - the exact string dreams train on. */
export function canonical(d: Decision): string {
  return JSON.stringify({ thought: d.thought, plan: d.plan.map((s) => ({ do: s.do, arg: s.arg })), say: d.say });
}

export function describePlan(plan: readonly Step[]): string {
  return plan.map((s) => (s.arg === "none" ? s.do : `${s.do} ${s.arg}`)).join(", ");
}

// ---------- backends ----------

export interface Backend {
  readonly label: string;
  /** Load the model if needed. Progress is 0..1. */
  init(onProgress: (p: number, text: string) => void): Promise<void>;
  /** `signal` aborts the request (on timeout), so an abandoned answer doesn't hold up the next one. */
  complete(system: string, user: string, schema: object, signal?: AbortSignal): Promise<string>;
}

/** Thread counts worth timing for Ollama: its default, plus common sweet spots for `cpus` logical CPUs.
 *  Physical cores usually win; SMT siblings hurt (i7-1260P: 17 ms/token default, 10 at 12, 35 at 16). */
export function threadCandidates(cpus: number): (number | undefined)[] {
  const out = new Set<number | undefined>([undefined]);
  if (cpus >= 4) { out.add(Math.floor(cpus / 2)); out.add(Math.floor(cpus * 3 / 4)); }
  return [...out];
}

const KEEP_ALIVE = "30m";
/** One tuning probe (24 tokens) that takes longer than this means something's wrong - stop tuning. */
const PROBE_TIMEOUT_MS = 60_000;

export class OllamaBackend implements Backend {
  readonly label: string;
  /** Fastest num_thread found by tune(); undefined = Ollama's default. */
  threads: number | undefined;
  constructor(private url: string, private model: string, private cache: { get(k: string): string | null; set(k: string, v: string): void } | null = null) {
    this.label = `Ollama · ${model}`;
  }

  async init(onProgress: (p: number, text: string) => void): Promise<void> {
    onProgress(0, `Connecting to Ollama at ${this.url}…`);
    const r = await fetch(`${this.url}/api/tags`, { signal: AbortSignal.timeout(15_000) });
    if (!r.ok) throw new Error(`Ollama answered ${String(r.status)}`);
    const tags = (await r.json()) as { models?: { name: string }[] };
    const names = (tags.models ?? []).map((m) => m.name);
    if (!names.some((n) => n === this.model || n === `${this.model}:latest`)) {
      throw new Error(`Ollama has no model "${this.model}". Try: ollama pull ${this.model}`);
    }
    // Load the model now (and find its fastest thread count, once), so the first command isn't the slow one.
    // Only an optimization: if it fails or Ollama is slow, play on with Ollama's defaults.
    onProgress(0.3, "Waking the brain up…");
    try {
      await this.tune();
    } catch {
      this.threads = undefined;
    }
    onProgress(1, "Ready");
  }

  /** ms per generated token for a short reply at a given thread count (Ollama's own timing, not wall clock). */
  private async speed(threads: number | undefined): Promise<number> {
    const r = await fetch(`${this.url}/api/generate`, {
      method: "POST",
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
      body: JSON.stringify({
        model: this.model, stream: false, keep_alive: KEEP_ALIVE, prompt: "Count from one to twenty in words.",
        options: { temperature: 0, num_predict: 24, ...(threads ? { num_thread: threads } : {}) },
      }),
    });
    if (!r.ok) throw new Error(`Ollama answered ${String(r.status)}`);
    const b = (await r.json()) as { eval_count?: number; eval_duration?: number };
    return b.eval_count ? (b.eval_duration ?? 0) / 1e6 / b.eval_count : Infinity;
  }

  private async tune(): Promise<void> {
    const cpus = typeof navigator === "undefined" ? 0 : navigator.hardwareConcurrency;
    const key = `blobb.ollamaThreads:${this.url}|${this.model}|${String(cpus)}`;
    const known = this.cache?.get(key);
    if (known !== null && known !== undefined) {
      this.threads = Number(known) || undefined;
      await this.speed(this.threads); // still warm it up
      return;
    }
    await this.speed(undefined); // first call pays the model load; don't let that skew the timing
    let best = Infinity;
    const candidates = threadCandidates(cpus);
    for (const t of candidates) {
      const ms = await this.speed(t);
      if (ms < best * 0.95) { best = ms; this.threads = t; } // ties go to the default
    }
    this.cache?.set(key, String(this.threads ?? 0));
    // Changing num_thread reloads Ollama's runner; leave it loaded with the winner, not the last one tried.
    if (this.threads !== candidates[candidates.length - 1]) await this.speed(this.threads);
  }

  async complete(system: string, user: string, format: object, signal?: AbortSignal): Promise<string> {
    const r = await fetch(`${this.url}/api/chat`, {
      method: "POST",
      signal: signal ?? null,
      body: JSON.stringify({
        model: this.model, stream: false, format, keep_alive: KEEP_ALIVE,
        options: { temperature: 0.7, num_predict: 200, ...(this.threads ? { num_thread: this.threads } : {}) },
        messages: [{ role: "system", content: system }, { role: "user", content: user }],
      }),
    });
    if (!r.ok) throw new Error(`Ollama answered ${String(r.status)}`);
    const body = (await r.json()) as { message?: { content?: string } };
    return body.message?.content ?? "";
  }
}

export const WEBLLM_MODEL = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";

interface WebLLMEngine {
  chat: { completions: { create(req: object): Promise<{ choices: { message: { content: string | null } }[] }> } };
  interruptGenerate(): void;
}

export class WebLLMBackend implements Backend {
  readonly label: string;
  private engine: WebLLMEngine | null = null;
  constructor(private model: string = WEBLLM_MODEL) {
    this.label = `In-browser · ${model.replace(/-MLC$/, "")}`;
  }

  async init(onProgress: (p: number, text: string) => void): Promise<void> {
    if (this.engine) return;
    if (!("gpu" in navigator)) throw new Error("This browser has no WebGPU, so it can't run the brain locally. Try Chrome, or use Ollama.");
    onProgress(0, "Loading the brain engine…");
    const webllm = await import("@mlc-ai/web-llm");
    this.engine = await webllm.CreateMLCEngine(this.model, {
      initProgressCallback: (r) => { onProgress(r.progress, r.text); },
    });
    // The first generation compiles GPU kernels; do it now rather than on the first command.
    onProgress(1, "Waking the brain up…");
    await this.engine.chat.completions.create({ messages: [{ role: "user", content: "hi" }], max_tokens: 1 });
  }

  async complete(system: string, user: string, format: object, signal?: AbortSignal): Promise<string> {
    const engine = this.engine;
    if (!engine) throw new Error("brain not loaded");
    const stop = () => { engine.interruptGenerate(); };
    signal?.addEventListener("abort", stop, { once: true });
    try {
      return await this.generate(engine, system, user, format);
    } finally {
      signal?.removeEventListener("abort", stop);
    }
  }

  private async generate(engine: WebLLMEngine, system: string, user: string, format: object): Promise<string> {
    const r = await engine.chat.completions.create({
      messages: [{ role: "system", content: system }, { role: "user", content: user }],
      temperature: 0.7,
      max_tokens: 200,
      response_format: { type: "json_object", schema: JSON.stringify(format) },
    });
    return r.choices[0]?.message.content ?? "";
  }
}
