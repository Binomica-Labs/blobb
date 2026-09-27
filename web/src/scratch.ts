// The blob's own brain: a few-million-parameter transformer trained from scratch on the simulator's
// curriculum (experiments/scratch-brain/brain.py). It runs right here in plain TypeScript - a ~6 MB
// download instead of a 350 MB LLM, and a decision in a blink instead of seconds.
//
// It reads the same observation text the LLM would, embeds each word as word-id + hashed character
// n-grams (so "shroom" lands near "mushroom"), and writes the plan as tokens: verb, arg, verb, arg, end.
// Only legal verb/arg pairs can come out. Thought and say are picked from lines it learned.
// This must stay in step with brain.py - test/scratch.test.ts checks it reproduces the trainer's answers.

import type { Backend, Decision } from "./brain";
import type { FromWorker, ToWorker } from "./scratch-worker";

export interface Manifest {
  version: number;
  cfg: { d: number; layers: number; heads: number; maxLen: number; maxPlan: number; buckets: number; maxNgrams: number };
  words: string[];
  out: string[];
  thoughts: string[];
  says: string[];
  pairs: Record<string, number[]>;
  tensors: Record<string, { offset: number; shape: number[] }>;
}

const BOS = 1, END = 2, MAX_STEPS = 4;
const TOKEN_RE = /[a-z]+|\d+|[^\sa-z\d,():.]/g;

export function tokenize(text: string): string[] {
  return text.toLowerCase().match(TOKEN_RE) ?? [];
}

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

/** zlib.crc32 of an ASCII string (n-grams are only ever [a-z<>]). */
export function crc32(s: string): number {
  let c = 0xffffffff;
  for (let i = 0; i < s.length; i++) c = (CRC_TABLE[(c ^ s.charCodeAt(i)) & 0xff] ?? 0) ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export function ngrams(word: string, buckets: number, max: number): number[] {
  if (!/^[a-z]+$/.test(word)) return [];
  const w = `<${word}>`;
  const out: number[] = [];
  for (const n of [3, 4, 5]) {
    for (let i = 0; i + n <= w.length; i++) out.push(1 + (crc32(w.slice(i, i + n)) % (buckets - 1)));
  }
  return out.slice(0, max);
}

/** IEEE half -> float, for the fp16 weights file. */
function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

/** erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7) - for PyTorch's exact GELU. */
function erf(x: number): number {
  const s = x < 0 ? -1 : 1, a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

interface Linear { w: Float32Array; b: Float32Array; inp: number; out: number }
interface Norm { w: Float32Array; b: Float32Array }
interface Layer { n1: Norm; n2: Norm; qkv: Linear; proj: Linear; ff1: Linear; ff2: Linear }

function linear(x: Float32Array, l: Linear, out = new Float32Array(l.out)): Float32Array {
  const { w, b, inp } = l;
  for (let o = 0; o < l.out; o++) {
    let s = b[o] ?? 0;
    const row = o * inp;
    for (let i = 0; i < inp; i++) s += (w[row + i] ?? 0) * (x[i] ?? 0);
    out[o] = s;
  }
  return out;
}

function layerNorm(x: Float32Array, n: Norm): Float32Array {
  const d = x.length;
  let mean = 0;
  for (let i = 0; i < d; i++) mean += x[i] ?? 0;
  mean /= d;
  let v = 0;
  for (let i = 0; i < d; i++) { const z = (x[i] ?? 0) - mean; v += z * z; }
  const inv = 1 / Math.sqrt(v / d + 1e-5);
  const out = new Float32Array(d);
  for (let i = 0; i < d; i++) out[i] = ((x[i] ?? 0) - mean) * inv * (n.w[i] ?? 0) + (n.b[i] ?? 0);
  return out;
}

/** Throws a clear error for a brain file this code can't run: wrong version, missing or out-of-range
 *  tensors (a truncated download), or plan tokens it doesn't know. */
export function checkBrain(m: Manifest, weights: ArrayBuffer): void {
  const bad = (why: string) => new Error(`This brain file is damaged or from another version (${why}). Reload to fetch it again.`);
  // Typed, but really whatever JSON came over the wire.
  if ((m as unknown as { version?: unknown } | null)?.version !== 1) throw bad("version");
  const c = m.cfg as Partial<Manifest["cfg"]> | undefined;
  if (!c || ![c.d, c.layers, c.heads, c.maxLen, c.buckets, c.maxNgrams].every((v) => Number.isInteger(v) && (v ?? 0) > 0) || (c.d ?? 0) % (c.heads ?? 1)) throw bad("shape");
  if (![m.words, m.out, m.thoughts, m.says].every((x) => Array.isArray(x) && x.length > 0)) throw bad("vocabulary");
  if (m.out[BOS] !== "<bos>" || m.out[END] !== "<end>") throw bad("plan tokens");
  if (weights.byteLength % 2) throw bad("weights");
  const n = weights.byteLength / 2;
  for (const [name, t] of Object.entries(m.tensors)) {
    const size = t.shape.reduce((a, b) => a * b, 1);
    if (!Number.isInteger(t.offset) || t.offset < 0 || t.offset + size > n) throw bad(`weights for ${name}`);
  }
  for (const need of ["word.weight", "gram.weight", "pos.weight", "out_emb.weight", "out_pos.weight", "norm.weight", "plan_head.weight", "thought_head.weight", "say_head.weight"]) {
    if (!(need in m.tensors)) throw bad(`no ${need}`);
  }
}

export class ScratchBrain {
  private words: Map<string, number>;
  private layers: Layer[];
  private get: (name: string) => Float32Array;
  private readonly d: number;
  private readonly heads: number;

  /** Verbs it may start a step with: only ones it learned an argument for. */
  private verbs: number[];

  constructor(readonly m: Manifest, weights: ArrayBuffer) {
    checkBrain(m, weights);
    const halves = new Uint16Array(weights);
    const cache = new Map<string, Float32Array>();
    this.get = (name) => {
      let t = cache.get(name);
      if (!t) {
        const spec = m.tensors[name];
        if (!spec) throw new Error(`brain file has no ${name}`);
        const n = spec.shape.reduce((a, b) => a * b, 1);
        t = new Float32Array(n);
        for (let i = 0; i < n; i++) t[i] = halfToFloat(halves[spec.offset + i] ?? 0);
        cache.set(name, t);
      }
      return t;
    };
    this.d = m.cfg.d;
    this.heads = m.cfg.heads;
    this.words = new Map(m.words.map((w, i) => [w, i]));
    this.verbs = m.out.flatMap((t, i) => (t.startsWith("v:") && m.pairs[String(i)]?.length ? [i] : []));
    const lin = (p: string, inp: number, out: number): Linear => ({ w: this.get(`${p}.weight`), b: this.get(`${p}.bias`), inp, out });
    const norm = (p: string): Norm => ({ w: this.get(`${p}.weight`), b: this.get(`${p}.bias`) });
    const d = this.d;
    this.layers = Array.from({ length: m.cfg.layers }, (_, i) => ({
      n1: norm(`blocks.${String(i)}.n1`), n2: norm(`blocks.${String(i)}.n2`),
      qkv: lin(`blocks.${String(i)}.qkv`, d, 3 * d), proj: lin(`blocks.${String(i)}.proj`, d, d),
      ff1: lin(`blocks.${String(i)}.ff.0`, d, 4 * d), ff2: lin(`blocks.${String(i)}.ff.2`, 4 * d, d),
    }));
  }

  private row(name: string, i: number): Float32Array {
    return this.get(name).subarray(i * this.d, (i + 1) * this.d);
  }

  /** Embedding of one observation token: word + mean of its n-grams + position. */
  private embed(tok: string, pos: number): Float32Array {
    const x = new Float32Array(this.d);
    const add = (v: Float32Array, k = 1) => { for (let i = 0; i < this.d; i++) x[i] = (x[i] ?? 0) + (v[i] ?? 0) * k; };
    add(this.row("word.weight", this.words.get(tok) ?? 1));
    const g = ngrams(tok, this.m.cfg.buckets, this.m.cfg.maxNgrams);
    for (const b of g) add(this.row("gram.weight", b), 1 / g.length);
    add(this.row("pos.weight", Math.min(pos, this.m.cfg.maxLen - 1)));
    return x;
  }

  /** Attention of one token's query (inside its qkv) over the first `n` cached keys/values. */
  private attend(qkv: Float32Array, keys: Float32Array[], vals: Float32Array[], n: number): Float32Array {
    const d = this.d, dh = d / this.heads, scale = 1 / Math.sqrt(dh);
    const att = new Float32Array(d), scores = new Float32Array(n);
    for (let h = 0; h < this.heads; h++) {
      const o = h * dh;
      let max = -Infinity;
      for (let j = 0; j < n; j++) {
        const k = keys[j] ?? att;
        let s = 0;
        for (let i = 0; i < dh; i++) s += (qkv[o + i] ?? 0) * (k[o + i] ?? 0);
        s *= scale;
        scores[j] = s;
        if (s > max) max = s;
      }
      let sum = 0;
      for (let j = 0; j < n; j++) { const e = Math.exp((scores[j] ?? 0) - max); scores[j] = e; sum += e; }
      for (let j = 0; j < n; j++) {
        const v = vals[j] ?? att, p = (scores[j] ?? 0) / sum;
        for (let i = 0; i < dh; i++) att[o + i] = (att[o + i] ?? 0) + p * (v[o + i] ?? 0);
      }
    }
    return att;
  }

  /** The rest of a block after attention: projection and residual, then the MLP and residual. */
  private rest(x: Float32Array, att: Float32Array, l: Layer): Float32Array {
    const d = this.d;
    const y = linear(att, l.proj);
    for (let i = 0; i < d; i++) y[i] = (y[i] ?? 0) + (x[i] ?? 0);
    const f = linear(layerNorm(y, l.n2), l.ff1);
    for (let i = 0; i < f.length; i++) { const z = f[i] ?? 0; f[i] = 0.5 * z * (1 + erf(z / Math.SQRT2)); }
    const g = linear(f, l.ff2);
    for (let i = 0; i < d; i++) g[i] = (g[i] ?? 0) + (y[i] ?? 0);
    return g;
  }

  /** Query, key and value of token x for layer l; its K/V go into the cache. */
  private qkv(x: Float32Array, l: Layer, keys: Float32Array[], vals: Float32Array[]): Float32Array {
    const q = linear(layerNorm(x, l.n1), l.qkv);
    keys.push(q.subarray(this.d, 2 * this.d));
    vals.push(q.subarray(2 * this.d, 3 * this.d));
    return q;
  }

  /** `rnd` in [0, 1) for sampling; temperature 0 = always the likeliest. */
  decide(user: string, temperature = 0, rnd: () => number = Math.random): Decision {
    const toks = tokenize(user).slice(0, this.m.cfg.maxLen);
    const L = toks.length;
    const keys: Float32Array[][] = this.layers.map(() => []);
    const vals: Float32Array[][] = this.layers.map(() => []);
    // The observation sees all of itself (not just what came before), so each layer needs every token's
    // K/V before any token attends: run it layer by layer. Plan tokens then only add to this cache.
    let xs = toks.map((t, i) => this.embed(t, i));
    this.layers.forEach((l, li) => {
      const ks = keys[li] ?? [], vs = vals[li] ?? [];
      const qs = xs.map((x) => this.qkv(x, l, ks, vs));
      xs = xs.map((x, i) => this.rest(x, this.attend(qs[i] ?? x, ks, vs, L), l));
    });

    const outNorm = { w: this.get("norm.weight"), b: this.get("norm.bias") };
    const head = (name: string, rows: number): Linear => ({ w: this.get(`${name}.weight`), b: this.get(`${name}.bias`), inp: this.d, out: rows });
    const planHead = head("plan_head", this.m.out.length);
    const plan = [BOS];
    let h: Float32Array | undefined; // final hidden state of the last plan token: thought and say read it
    for (;;) {
      let x: Float32Array = new Float32Array(this.d);
      const e = this.row("out_emb.weight", plan[plan.length - 1] ?? BOS), p = this.row("out_pos.weight", plan.length - 1);
      for (let i = 0; i < this.d; i++) x[i] = (e[i] ?? 0) + (p[i] ?? 0);
      this.layers.forEach((l, li) => {
        const ks = keys[li] ?? [], vs = vals[li] ?? [];
        x = this.rest(x, this.attend(this.qkv(x, l, ks, vs), ks, vs, ks.length), l);
      });
      h = layerNorm(x, outNorm);
      const logits = linear(h, planHead);
      const allowed = new Set<number>();
      if (plan.length % 2 === 1) { // verb position
        const steps = (plan.length - 1) / 2;
        if (steps === MAX_STEPS) break;
        for (const v of this.verbs) allowed.add(v);
        if (steps) allowed.add(END);
      } else {
        for (const a of this.m.pairs[String(plan[plan.length - 1])] ?? []) allowed.add(a);
      }
      const t = choose(logits, temperature, rnd, allowed);
      if (t === END) break;
      plan.push(t);
    }
    h ??= new Float32Array(this.d);
    const thought = this.m.thoughts[choose(linear(h, head("thought_head", this.m.thoughts.length)), 0, rnd)] ?? "";
    const say = this.m.says[choose(linear(h, head("say_head", this.m.says.length)), temperature, rnd)] ?? "";
    const steps = [];
    for (let i = 1; i + 1 < plan.length; i += 2) {
      steps.push({ do: (this.m.out[plan[i] ?? 0] ?? "").slice(2), arg: (this.m.out[plan[i + 1] ?? 0] ?? "").slice(2) });
    }
    return { thought, plan: steps as Decision["plan"], say };
  }
}

function choose(logits: Float32Array, temperature: number, rnd: () => number, allowed?: Set<number>): number {
  const ok = (i: number) => !allowed || allowed.has(i);
  let best = -1, max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (ok(i) && (logits[i] ?? -Infinity) > max) { max = logits[i] ?? -Infinity; best = i; }
  if (temperature <= 0) return best;
  let sum = 0;
  const p = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) if (ok(i)) { p[i] = Math.exp(((logits[i] ?? 0) - max) / temperature); sum += p[i] ?? 0; }
  let r = rnd() * sum;
  for (let i = 0; i < logits.length; i++) { r -= p[i] ?? 0; if (ok(i) && r <= 0) return i; }
  return best;
}

export type BrainFiles = () => Promise<[Manifest, ArrayBuffer]>;

/** Fetch brain.json + brain.bin from a URL prefix. */
export const fetchBrain = (base: string): BrainFiles => async () => {
  const get = async (f: string) => {
    const r = await fetch(`${base}${f}`);
    if (!r.ok) throw new Error(`no brain at ${base}${f} (${String(r.status)})`);
    return r;
  };
  const [m, w] = await Promise.all([get("brain.json").then((r) => r.json() as Promise<Manifest>), get("brain.bin").then((r) => r.arrayBuffer())]);
  return [m, w];
};

/** Plays the blob with its own brain - in a worker when given one (the browser), so thinking doesn't
 *  freeze the island; inline otherwise (evals, tests), or if the worker won't start. */
export class ScratchBackend implements Backend {
  /** It learns by dreaming (its weights), and never saw recalled memories in training. */
  readonly readsMemories = false;
  private brain: ScratchBrain | null = null;
  private worker: Worker | null = null;
  private pending = new Map<number, { resolve: (s: string) => void; reject: (e: Error) => void }>();
  private nextId = 0;

  constructor(readonly label: string, private files: BrainFiles, private temperature = 0.7, private makeWorker?: () => Worker) {}

  async init(onProgress: (p: number, text: string) => void): Promise<void> {
    if (this.brain || this.worker) return;
    onProgress(0, "Growing a brain…");
    const [m, w] = await this.files();
    onProgress(0.9, "Waking the brain up…");
    checkBrain(m, w); // a damaged file fails here, with a clear message, wherever it would run
    if (this.makeWorker) {
      try {
        this.worker = await this.startWorker(m, w.slice(0));
      } catch {
        this.worker = null; // no workers here (or it died): think inline instead
      }
    }
    if (!this.worker) this.brain = new ScratchBrain(m, w);
    onProgress(1, "Ready");
  }

  private startWorker(m: Manifest, w: ArrayBuffer): Promise<Worker> {
    const make = this.makeWorker;
    if (!make) return Promise.reject(new Error("no worker"));
    return new Promise((resolve, reject) => {
      const worker = make();
      const fail = (why: string) => {
        worker.terminate();
        this.worker = null;
        for (const p of this.pending.values()) p.reject(new Error(why));
        this.pending.clear();
        reject(new Error(why));
      };
      worker.onerror = (e) => { e.preventDefault(); fail(e.message || "the brain's worker crashed"); };
      worker.onmessage = (ev: MessageEvent<FromWorker>) => {
        const r = ev.data;
        if (r.t === "ready") { resolve(worker); return; }
        const p = r.id === undefined ? undefined : this.pending.get(r.id);
        if (r.id !== undefined) this.pending.delete(r.id);
        if (r.t === "decision") p?.resolve(JSON.stringify(r.d));
        else if (p) p.reject(new Error(r.error));
        else fail(r.error);
      };
      worker.postMessage({ t: "load", manifest: m, weights: w } satisfies ToWorker, [w]);
    });
  }

  complete(_system: string, user: string): Promise<string> {
    if (this.worker) {
      const id = this.nextId++;
      const worker = this.worker;
      return new Promise((resolve, reject) => {
        this.pending.set(id, { resolve, reject });
        worker.postMessage({ t: "decide", id, user, temperature: this.temperature } satisfies ToWorker);
      });
    }
    if (!this.brain) return Promise.reject(new Error("brain not loaded"));
    return Promise.resolve(JSON.stringify(this.brain.decide(user, this.temperature)));
  }
}
