// Peeking into the blob's dream: while the trainer rewrites its weights, the local server exposes
// the trainer's progress (see serve.mjs), and this draws it - how far along it is, what it's dreaming
// about, how wrong it still is, and a "weight map" of how much each part of its brain has changed.
// Only works when the game is served by serve.mjs next to the trainer; elsewhere the 🌙 never shows.

export interface DreamStatus {
  phase: string;
  t: number;
  gen?: number;
  step?: number;
  total?: number;
  loss?: number | null;
  eta?: number | null;
  about?: string;
  error?: string;
  model?: string;
  /** Its own brain's dreams: held-out instinct lessons it got right [before, after, out of]. */
  instinct?: [number, number, number];
}

export interface DreamMap {
  step: number;
  total: number;
  layers: number;
  modules: string[];
  /** ||ΔW|| per [layer][module]: how far the dream has moved each weight matrix. */
  norms: number[][];
  /** Weights in each module's matrix, to turn norms into change-per-weight. */
  numel?: number[];
  losses: number[];
}

export type DreamKind = "asleep" | "stalled" | "done" | "error";
export interface DreamView { kind: DreamKind; title: string; detail: string; progress: number | null }

const ASLEEP = ["falling asleep", "dreaming", "waking"];
/** No word from the trainer for this long means it died (steps take ~45 s; saving can take minutes). */
const QUIET_S: Record<string, number> = { "falling asleep": 900, dreaming: 300, waking: 900 };

export function fmtDuration(s: number): string {
  const m = Math.max(1, Math.round(s / 60));
  return m < 60 ? `${String(m)} min` : `${String(Math.floor(m / 60))}h ${String(m % 60)}m`;
}

/** What to say about a status, given the time now (seconds). */
export function describe(s: DreamStatus, now: number, name: string): DreamView {
  const gen = s.gen ? ` (generation ${String(s.gen)})` : "";
  const progress = s.total ? Math.min(1, (s.step ?? 0) / s.total) : null;
  if (ASLEEP.includes(s.phase) && now - s.t > (QUIET_S[s.phase] ?? 300)) {
    return { kind: "stalled", progress, title: `${name}'s dream went quiet`,
      detail: `No word from the dream for ${fmtDuration(now - s.t)}. The trainer probably stopped - check its terminal.` };
  }
  if (s.phase === "falling asleep") {
    return { kind: "asleep", progress: null, title: `${name} is drifting off${gen}`, detail: "Loading its brain into the dream…" };
  }
  if (s.phase === "dreaming") {
    const left = s.eta ? `about ${fmtDuration(s.eta)} left` : "working out how long this takes…";
    return { kind: "asleep", progress, title: `${name} is dreaming${gen}`,
      detail: `Step ${String(s.step ?? 0)} of ${String(s.total ?? "?")} - ${left}` };
  }
  if (s.phase === "waking") {
    return { kind: "asleep", progress: 1, title: `${name} is waking up${gen}`, detail: "Saving the new brain…" };
  }
  if (s.phase === "done") {
    let detail = s.model?.startsWith("own:") ? "Its brain grew a little in the night." : s.model ? `The new brain is "${s.model}".` : "The new brain is ready.";
    const [before, after, of] = s.instinct ?? [];
    if (before !== undefined && after !== undefined && of) {
      detail += ` It still gets ${String(after)} of ${String(of)} practice lessons right (${String(before)} before).`;
      if ((before - after) / of > 0.05) {
        detail += ` It forgot a lot of what it knew - too much to avoid, too little to do? You can keep the old brain: just don't wake it up with this one.`;
      }
    }
    return { kind: "done", progress: 1, title: `${name} woke up${gen}`, detail };
  }
  return { kind: "error", progress, title: "The dream turned into a nightmare", detail: s.error ?? `Unknown phase "${s.phase}".` };
}

/** Change per weight (RMS of ΔW) that counts as "fully rewritten" and "barely touched". A fixed
 *  scale, so the map starts dark and brightens as the dream goes on, instead of always looking done. */
export const GLOW_LO = 1e-6, GLOW_HI = 1e-3;

/** 0..1 glow for a matrix whose change has Frobenius norm `norm` spread over `numel` weights. */
export function glow(norm: number, numel: number): number {
  const rms = numel > 0 ? norm / Math.sqrt(numel) : 0;
  if (rms <= GLOW_LO) return 0;
  return Math.min(1, Math.log10(rms / GLOW_LO) / Math.log10(GLOW_HI / GLOW_LO));
}

/** Night-sky colour for a glow of 0..1: indigo (untouched) -> violet -> gold (rewritten a lot). */
export function heat(glowLevel: number): string {
  const k = Math.max(0, Math.min(1, glowLevel));
  const stops = [[28, 21, 64], [138, 92, 255], [255, 211, 107]] as const;
  const [a, b, f] = k < 0.5 ? [stops[0], stops[1], k * 2] : [stops[1], stops[2], (k - 0.5) * 2];
  const c = a.map((x, i) => Math.round(x + ((b[i] ?? x) - x) * f));
  return `rgb(${c.join(", ")})`;
}

/** SVG polyline points for a loss curve, scaled to fit w x h. */
export function sparkline(values: readonly number[], w: number, h: number): string {
  if (values.length < 2) return "";
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  return values.map((v, i) => `${(i / (values.length - 1) * w).toFixed(1)},${(h - 2 - (v - lo) / span * (h - 4)).toFixed(1)}`).join(" ");
}

// ---------- the panel ----------

const $ = (id: string): HTMLElement => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
};

async function getJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url, { cache: "no-store" });
    return r.ok ? await r.json() as T : null;
  } catch { return null; }
}

export class DreamPeek {
  private status: DreamStatus | null = null;
  private prev: number[][] | null = null;
  private timer: ReturnType<typeof setTimeout> | undefined;

  constructor(private name: () => string, private onUse: (model: string) => void) {
    $("dreamBtn").addEventListener("click", () => { this.open(); });
    $("closeDream").addEventListener("click", () => { $("dream").hidden = true; });
    $("dreamUse").addEventListener("click", () => {
      if (this.status?.model) this.onUse(this.status.model);
      $("dream").hidden = true;
    });
    void this.poll();
  }

  open(): void {
    $("dream").hidden = false;
    void this.poll();
  }

  private async poll(): Promise<void> {
    clearTimeout(this.timer);
    const open = !$("dream").hidden;
    this.status = await getJson<DreamStatus>("dream/status.json");
    const btn = $("dreamBtn");
    btn.hidden = !this.status;
    if (this.status) {
      const v = describe(this.status, Date.now() / 1000, this.name());
      btn.classList.toggle("asleep", v.kind === "asleep");
      btn.title = v.title;
      if (open) await this.render(v);
    }
    // Quick while you're watching; slow otherwise, just to keep the 🌙 honest.
    this.timer = setTimeout(() => { void this.poll(); }, open ? 5000 : 30000);
  }

  private async render(v: DreamView): Promise<void> {
    const s = this.status;
    $("dreamTitle").textContent = v.title;
    $("dreamDetail").textContent = v.detail;
    const bar = $("dreamBar") as HTMLProgressElement;
    bar.hidden = v.progress === null;
    bar.value = v.progress ?? 0;
    $("dreamAbout").textContent = v.kind === "asleep" && s?.about ? `Dreaming about: "${s.about}"` : "";
    $("dreamUse").hidden = !(v.kind === "done" && s?.model);

    const map = await getJson<DreamMap>("dream/map.json");
    $("dreamLossWrap").hidden = !map || map.losses.length < 2;
    $("dreamMapWrap").hidden = !map;
    if (!map) return;
    $("dreamLoss").querySelector("polyline")?.setAttribute("points", sparkline(map.losses, 300, 60));
    const last = map.losses[map.losses.length - 1];
    $("dreamLossNow").textContent = last === undefined ? "" : last.toFixed(3);

    const grid = $("dreamMap");
    grid.style.gridTemplateColumns = `2.2em repeat(${String(map.modules.length)}, 1fr)`;
    // Old maps (no sizes) fall back to "relative to the brightest cell".
    const max = Math.max(1e-12, ...map.norms.flat());
    const level = (v: number, i: number) => {
      const n = map.numel?.[i];
      return n ? glow(v, n) : Math.sqrt(v / max);
    };
    const levels = map.norms.map((row) => row.map(level));
    const cells: HTMLElement[] = [document.createElement("span")];
    for (const m of map.modules) cells.push(Object.assign(document.createElement("b"), { textContent: m }));
    levels.forEach((row, layer) => {
      cells.push(Object.assign(document.createElement("b"), { textContent: String(layer + 1) }));
      row.forEach((k, i) => {
        const c = document.createElement("i");
        c.style.background = heat(k);
        const n = map.numel?.[i];
        const v = map.norms[layer]?.[i] ?? 0;
        c.title = `layer ${String(layer + 1)} · ${map.modules[i] ?? "?"}: ` +
          (n ? `each weight moved ~${(v / Math.sqrt(n)).toExponential(1)}` : `changed ${v.toFixed(4)}`);
        // Cells that brightened noticeably since the last look shimmer.
        const before = this.prev?.[layer]?.[i];
        if (before !== undefined && k - before > 0.03) c.className = "fresh";
        cells.push(c);
      });
    });
    grid.replaceChildren(...cells);
    this.prev = levels;
  }
}
