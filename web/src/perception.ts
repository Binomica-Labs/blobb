// What the blob "sees": a compact text description of its surroundings.
// Kept short on purpose - a 0.5B model reads every token of it on every decision.

import { REASON_TEXT, blockedReason, candidates, findPath, reachGoal, type Reason } from "./actions";
import { THINGS, objById, tileAt, type Thing, type World } from "./world";

export const SIGHT = 9;

export interface Seen { kind: Thing; dx: number; dy: number; dh: number; blocked: Reason | null; note: string }

function level(v: number, low: string, mid: string, high: string): string {
  return v < 20 ? low : v < 45 ? mid : high;
}

export function where(dx: number, dy: number): string {
  if (dx === 0 && dy === 0) return "right here";
  const parts: string[] = [];
  if (dy) parts.push(`${Math.abs(dy)} ${dy < 0 ? "north" : "south"}`);
  if (dx) parts.push(`${Math.abs(dx)} ${dx > 0 ? "east" : "west"}`);
  return parts.join(", ");
}

/** The nearest few things of each kind, with whether the blob can currently reach them. */
export function look(w: World): Seen[] {
  const b = w.blob;
  const here = tileAt(w, b.x, b.y).h;
  const out: Seen[] = [];
  for (const kind of THINGS) {
    const near = candidates(w, kind)
      .map((c) => ({ c, d: Math.abs(c.x - b.x) + Math.abs(c.y - b.y) }))
      .filter(({ d }) => d <= SIGHT)
      .sort((p, q) => p.d - q.d)
      .slice(0, kind === "gap" || kind === "gate" || kind === "tower" ? 1 : 2);
    for (const { c } of near) {
      const goal = reachGoal(w, c.x, c.y, kind);
      const blocked = findPath(w, b.shape, goal) ? null : blockedReason(w, goal);
      const notes: string[] = [];
      const dh = tileAt(w, c.x, c.y).h - here;
      if (dh >= 2) notes.push("up high");
      if (kind === "tree" && c.obj) notes.push(c.obj.fruit ? `${c.obj.fruit} fruit up in it` : "no fruit");
      if (kind === "gate") notes.push("very low, flat things fit under");
      if (kind === "tower") notes.push("the wizard's, deliver things here");
      if (blocked) notes.push(`can't reach as ${b.shape}: ${REASON_TEXT[blocked]}`);
      out.push({ kind, dx: c.x - b.x, dy: c.y - b.y, dh, blocked, note: notes.join("; ") });
    }
  }
  return out.sort((p, q) => Math.abs(p.dx) + Math.abs(p.dy) - (Math.abs(q.dx) + Math.abs(q.dy))).slice(0, 9);
}

export interface Context {
  ownerSaid: string | null;
  recent: readonly string[];
  lessons: readonly string[];
  skills: readonly string[];
  /** Feedback on the previous attempt at the same command, if it failed. */
  retry: string | null;
}

/** [clean, full]: `full` adds recalled lessons and skills; `clean` is what dreams train on. */
export function observe(w: World, ctx: Context): [string, string] {
  const b = w.blob;
  const held = objById(w, b.holding);
  const head = [
    `You are a ${b.shape}. Food ${b.food.toFixed(0)} (${level(b.food, "STARVING", "hungry", "full")}), ` +
      `energy ${b.energy.toFixed(0)} (${level(b.energy, "EXHAUSTED", "tired", "fine")}). ` +
      `Holding: ${held ? held.kind : "nothing"}.`,
    "You see:",
    ...look(w).map((s) => `- ${s.kind}: ${where(s.dx, s.dy)}${s.note ? ` (${s.note})` : ""}`),
  ];
  if (ctx.recent.length) head.push("Recently: " + ctx.recent.join(" "));
  const tail: string[] = [];
  if (ctx.retry) tail.push(`Your last plan failed: ${ctx.retry} Try something different.`);
  tail.push(ctx.ownerSaid ? `The wizard says: "${ctx.ownerSaid}"` : "The wizard says nothing.");
  const memory: string[] = [];
  if (ctx.lessons.length) memory.push("Things you learned:", ...ctx.lessons.map((l) => `- ${l}`));
  if (ctx.skills.length) memory.push("Your skills: " + ctx.skills.join("; "));
  return [[...head, ...tail].join("\n"), [...head, ...memory, ...tail].join("\n")];
}
