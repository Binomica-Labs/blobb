// Runs the blob's own brain off the main thread, so the island keeps moving while it thinks.
// Messages: {t: "load", manifest, weights} -> {t: "ready"} | {t: "error", error};
//           {t: "decide", id, user, temperature} -> {t: "decision", id, d} | {t: "error", id, error}.

import { ScratchBrain, type Manifest } from "./scratch";

export type ToWorker =
  | { t: "load"; manifest: Manifest; weights: ArrayBuffer }
  | { t: "decide"; id: number; user: string; temperature: number };
export type FromWorker =
  | { t: "ready" }
  | { t: "decision"; id: number; d: unknown }
  | { t: "error"; id?: number; error: string };

let brain: ScratchBrain | null = null;
const reply = (m: FromWorker) => { postMessage(m); };

addEventListener("message", (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  try {
    if (m.t === "load") {
      brain = new ScratchBrain(m.manifest, m.weights);
      reply({ t: "ready" });
    } else {
      if (!brain) throw new Error("brain not loaded");
      reply({ t: "decision", id: m.id, d: brain.decide(m.user, m.temperature) });
    }
  } catch (e) {
    reply({ t: "error", ...(m.t === "decide" ? { id: m.id } : {}), error: e instanceof Error ? e.message : String(e) });
  }
});
