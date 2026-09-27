// What the blob can do, and whether it works. Each step is planned against the
// current world and returns a list of ops: small mutations, each paired with the
// animation that shows it. Tests apply them all at once; the game plays them.

import {
  CLIMB, FOODS, PORTABLE, RESPAWN_S, SHAPES, THINGS, clamp, inBounds, objById, objsAt,
  setTile, solidAt, standable, tileAt,
  type Obj, type Shape, type Thing, type World,
} from "./world";

export const VERBS = ["move_to", "morph", "push", "grab", "drop", "eat", "use", "deliver", "rest", "skill"] as const;
export type Verb = (typeof VERBS)[number];
export interface Step { do: Verb; arg: string }

export type Anim =
  | { t: "walk"; x: number; y: number; dur: number }
  | { t: "jump"; x: number; y: number; dur: number }
  | { t: "morph"; shape: Shape }
  | { t: "fx"; kind: "chomp" | "push" | "shake" | "grab" | "drop" | "rest" | "fail" | "splash"; dur: number };

export interface Op { anim: Anim; apply: (w: World) => void }

/** Why something didn't work. Keys are stable so lessons can be keyed on them. */
export type Reason =
  | "too_high" | "gap" | "gate" | "no_way" | "not_found" | "too_tired" | "not_holding"
  | "cant_do" | "unknown_step";

export interface Outcome {
  ok: boolean;
  msg: string;
  reason: Reason | null;
  ops: Op[];
  /** Food gained - used as a world reward. */
  ate: number;
}

const fail = (reason: Reason, msg: string, ops: Op[] = []): Outcome =>
  ({ ok: false, msg, reason, ate: 0, ops: [...ops, { anim: { t: "fx", kind: "fail", dur: 0.6 }, apply: () => undefined }] });
const done = (msg: string, ops: Op[], ate = 0): Outcome => ({ ok: true, msg, reason: null, ops, ate });

export const REASON_TEXT: Record<Reason, string> = {
  too_high: "it's up too high to climb",
  gap: "there's a gap in the way",
  gate: "a low gate is in the way",
  no_way: "there's no way to get there",
  not_found: "you can't see one",
  too_tired: "you're too tired",
  not_holding: "you're not holding anything",
  cant_do: "that doesn't work",
  unknown_step: "you don't know how to do that",
};

// ---------- movement ----------

interface Relax { height?: boolean; gaps?: boolean; gates?: boolean }
export interface PathStep { x: number; y: number; jump: boolean }

const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]] as const;

function canStep(w: World, shape: Shape, fx: number, fy: number, tx: number, ty: number, r: Relax): boolean {
  if (!inBounds(w, tx, ty) || solidAt(w, tx, ty)) return false;
  const t = tileAt(w, tx, ty);
  if (t.kind === "water") return false;
  if (t.kind === "gap" && !r.gaps) return false;
  if (t.kind === "gate" && shape !== "puddle" && !r.gates) return false;
  const up = t.h - tileAt(w, fx, fy).h;
  return r.height === true || up <= CLIMB[shape];
}

function canJump(w: World, fx: number, fy: number, dx: number, dy: number, dist: number): boolean {
  const tx = fx + dx * dist, ty = fy + dy * dist;
  if (!standable(w, tx, ty) || tileAt(w, tx, ty).kind === "gate") return false;
  const from = tileAt(w, fx, fy).h, to = tileAt(w, tx, ty).h;
  if (to - from > 1) return false;
  let overGap = false;
  for (let i = 1; i < dist; i++) {
    const m = tileAt(w, fx + dx * i, fy + dy * i);
    if (m.kind === "water") return false;
    if (m.kind === "gap") overGap = true;
    else if (m.h > Math.max(from, to) + 1) return false;
  }
  return overGap;
}

/** Cheapest path from the blob to any tile where `goal` holds. [] if already there, null if unreachable. */
export function findPath(w: World, shape: Shape, goal: (x: number, y: number) => boolean, r: Relax = {}): PathStep[] | null {
  const { x: sx, y: sy } = w.blob;
  if (goal(sx, sy)) return [];
  const n = w.w * w.d;
  const cost = new Array<number>(n).fill(Infinity);
  const prev = new Array<number>(n).fill(-1);
  const viaJump = new Array<boolean>(n).fill(false);
  const open: number[] = [sy * w.w + sx];
  cost[sy * w.w + sx] = 0;
  while (open.length) {
    open.sort((a, b) => (cost[a] ?? 0) - (cost[b] ?? 0));
    const cur = open.shift() ?? 0;
    const cx = cur % w.w, cy = Math.floor(cur / w.w);
    if (goal(cx, cy) && cur !== sy * w.w + sx) {
      const path: PathStep[] = [];
      for (let i = cur; i !== sy * w.w + sx; i = prev[i] ?? 0) {
        path.push({ x: i % w.w, y: Math.floor(i / w.w), jump: viaJump[i] ?? false });
      }
      return path.reverse();
    }
    const edges: [number, number, number, boolean][] = [];
    for (const [dx, dy] of DIRS) {
      if (canStep(w, shape, cx, cy, cx + dx, cy + dy, r)) edges.push([cx + dx, cy + dy, 1, false]);
      if (shape === "spring") {
        for (const dist of [2, 3]) {
          if (canJump(w, cx, cy, dx, dy, dist)) edges.push([cx + dx * dist, cy + dy * dist, dist + 1, true]);
        }
      }
    }
    for (const [nx, ny, c, jump] of edges) {
      const ni = ny * w.w + nx;
      const nc = (cost[cur] ?? 0) + c;
      if (nc < (cost[ni] ?? Infinity)) {
        cost[ni] = nc; prev[ni] = cur; viaJump[ni] = jump;
        if (!open.includes(ni)) open.push(ni);
      }
    }
  }
  return null;
}

/** Why can't the current shape get there? Tries again with each rule relaxed. */
export function blockedReason(w: World, goal: (x: number, y: number) => boolean): Reason {
  const s = w.blob.shape;
  if (findPath(w, s, goal, { gates: true })) return "gate";
  if (findPath(w, s, goal, { height: true })) return "too_high";
  if (findPath(w, s, goal, { gaps: true })) return "gap";
  return "no_way";
}

const SPEED: Record<Shape, number> = { blob: 0.32, ball: 0.16, puddle: 0.55, spring: 0.38 };
const MOVE_COST: Record<Shape, number> = { blob: 0.4, ball: 0.2, puddle: 0.5, spring: 0.7 };

function pathCost(shape: Shape, path: PathStep[]): number {
  return path.reduce((s, p) => s + (p.jump ? 3 : MOVE_COST[shape]), 0);
}

function walkOps(w: World, path: PathStep[]): Op[] {
  const shape = w.blob.shape;
  return path.map((p) => ({
    anim: p.jump ? { t: "jump", x: p.x, y: p.y, dur: 0.7 } : { t: "walk", x: p.x, y: p.y, dur: SPEED[shape] },
    apply: (world: World) => {
      const b = world.blob;
      const fx = Math.sign(p.x - b.x), fy = Math.sign(p.y - b.y);
      if (fx || fy) b.facing = [fx, fy];
      b.x = p.x; b.y = p.y;
      b.energy = clamp(b.energy - (p.jump ? 3 : MOVE_COST[shape]), 0, 100);
    },
  }));
}

// ---------- targets ----------

const dist = (ax: number, ay: number, bx: number, by: number) => Math.abs(ax - bx) + Math.abs(ay - by);

/** Tiles you can touch a thing from: its own tile for small things, next to it for big ones. */
export function reachGoal(w: World, x: number, y: number, kind: Thing): (px: number, py: number) => boolean {
  const big = kind === "rock" || kind === "tree" || kind === "tower" || kind === "gap" || kind === "gate";
  const th = tileAt(w, x, y).h;
  return (px, py) => {
    const d = dist(px, py, x, y);
    if (big) return d === 1;
    return d === 0 || (d === 1 && Math.abs(tileAt(w, px, py).h - th) <= 1);
  };
}

export interface Target { kind: Thing; x: number; y: number; obj: Obj | null }

export function candidates(w: World, kind: Thing): Target[] {
  if (kind === "gap") {
    const out: Target[] = [];
    for (let y = 0; y < w.d; y++) for (let x = 0; x < w.w; x++) {
      if (tileAt(w, x, y).kind === "gap") out.push({ kind, x, y, obj: null });
    }
    return out;
  }
  if (kind === "gate") {
    const out: Target[] = [];
    for (let y = 0; y < w.d; y++) for (let x = 0; x < w.w; x++) {
      if (tileAt(w, x, y).kind === "gate") out.push({ kind, x, y, obj: null });
    }
    return out;
  }
  return w.objs.filter((o) => o.state === "world" && o.kind === kind).map((o) => ({ kind, x: o.x, y: o.y, obj: o }));
}

/** The nearest thing of this kind you can actually get to (or the nearest at all, with a path of null).
 *  A spring goes for things up high: that's what you turn into one for, and without it no plan could
 *  ever say "the berry on the meadow, not the one by my feet". */
export function pickTarget(w: World, kind: Thing): { target: Target; path: PathStep[] | null } | null {
  const b = w.blob;
  const cands = candidates(w, kind).sort((p, q) => dist(b.x, b.y, p.x, p.y) - dist(b.x, b.y, q.x, q.y));
  const first = cands[0];
  if (!first) return null;
  const here = tileAt(w, b.x, b.y).h;
  const high = (c: Target) => tileAt(w, c.x, c.y).h - here >= 2;
  const pool = b.shape === "spring" ? [...new Set([...cands.filter(high).slice(0, 6), ...cands.slice(0, 6)])] : cands.slice(0, 6);
  let best: { target: Target; path: PathStep[]; high: boolean } | null = null;
  for (const c of pool) {
    const path = findPath(w, b.shape, reachGoal(w, c.x, c.y, c.kind));
    if (!path) continue;
    const up = b.shape === "spring" && high(c);
    if (!best || (up && !best.high) || (up === best.high && path.length < best.path.length)) best = { target: c, path, high: up };
  }
  return best ? { target: best.target, path: best.path } : { target: first, path: null };
}

/** Walk to a thing, or explain why not. */
function approach(w: World, kind: Thing): { target: Target; ops: Op[] } | Outcome {
  const pick = pickTarget(w, kind);
  if (!pick) return fail("not_found", `You look around for a ${kind} but ${REASON_TEXT.not_found}.`);
  const { target, path } = pick;
  if (!path) {
    const reason = blockedReason(w, reachGoal(w, target.x, target.y, kind));
    return fail(reason, `You can't get to the ${kind} as a ${w.blob.shape}: ${REASON_TEXT[reason]}.`);
  }
  if (pathCost(w.blob.shape, path) > w.blob.energy) {
    return fail("too_tired", `The ${kind} is too far - ${REASON_TEXT.too_tired}. Rest first.`);
  }
  return { target, ops: walkOps(w, path) };
}

const isOutcome = (x: { target: Target; ops: Op[] } | Outcome): x is Outcome => "ok" in x;

// ---------- the verbs ----------

function asThing(arg: string): Thing | null {
  return (THINGS as readonly string[]).includes(arg) ? (arg as Thing) : null;
}

function eatObj(o: Obj): Op {
  return {
    anim: { t: "fx", kind: "chomp", dur: 0.9 },
    apply: (w) => {
      const b = w.blob;
      b.food = clamp(b.food + (o.kind === "berry" ? 25 : 15), 0, 100);
      if (o.kind === "mushroom") b.energy = clamp(b.energy + 10, 0, 100);
      if (b.holding === o.id) b.holding = null;
      o.state = "gone";
      o.respawnAt = o.home ? w.time + RESPAWN_S : 0;
    },
  };
}

function morph(w: World, arg: string): Outcome {
  if (!(SHAPES as readonly string[]).includes(arg)) return fail("unknown_step", `"${arg}" isn't a shape you know.`);
  const shape = arg as Shape;
  if (w.blob.shape === shape) return done(`You're already a ${shape}.`, []);
  if (tileAt(w, w.blob.x, w.blob.y).kind === "gate" && shape !== "puddle") {
    return fail("cant_do", "There's no room to change shape under the gate.");
  }
  if (w.blob.energy < 4) return fail("too_tired", `You're too tired to change shape.`);
  return done(`You squish into a ${shape}.`, [{
    anim: { t: "morph", shape },
    apply: (world) => { world.blob.shape = shape; world.blob.energy -= 4; },
  }]);
}

function moveTo(w: World, arg: string): Outcome {
  const kind = asThing(arg);
  if (!kind) return fail("unknown_step", `You don't know what "${arg}" is.`);
  const a = approach(w, kind);
  if (isOutcome(a)) return a;
  return done(`You go to the ${kind}.`, a.ops);
}

function eat(w: World, arg: string): Outcome {
  const held = objById(w, w.blob.holding);
  if (held && FOODS.includes(held.kind) && (arg === held.kind || arg === "none")) {
    return done(`You eat the ${held.kind} you were holding. Yum!`, [eatObj(held)], held.kind === "berry" ? 25 : 15);
  }
  const kind = asThing(arg);
  if (!kind || !FOODS.includes(kind as Obj["kind"])) return fail("cant_do", `You can't eat ${arg === "none" ? "nothing" : `a ${arg}`}.`);
  const a = approach(w, kind);
  if (isOutcome(a)) return a;
  const o = a.target.obj;
  if (!o) return fail("not_found", `No ${kind} here.`);
  return done(`You eat the ${kind}. Yum!`, [...a.ops, eatObj(o)], kind === "berry" ? 25 : 15);
}

function grab(w: World, arg: string): Outcome {
  const kind = asThing(arg);
  if (!kind || !PORTABLE.includes(kind as Obj["kind"])) return fail("cant_do", `You can't pick up ${arg === "none" ? "nothing" : `a ${arg}`}.`);
  const held = objById(w, w.blob.holding);
  if (held) return fail("cant_do", `You're already holding a ${held.kind}. Drop it first.`);
  const a = approach(w, kind);
  if (isOutcome(a)) return a;
  const o = a.target.obj;
  if (!o) return fail("not_found", `No ${kind} here.`);
  return done(`You pick up the ${kind}.`, [...a.ops, {
    anim: { t: "fx", kind: "grab", dur: 0.4 },
    apply: (world) => { o.state = "held"; world.blob.holding = o.id; },
  }]);
}

function drop(w: World): Outcome {
  const held = objById(w, w.blob.holding);
  if (!held) return fail("not_holding", `Nothing to drop - ${REASON_TEXT.not_holding}.`);
  return done(`You drop the ${held.kind}.`, [{
    anim: { t: "fx", kind: "drop", dur: 0.3 },
    apply: (world) => { held.state = "world"; held.x = world.blob.x; held.y = world.blob.y; world.blob.holding = null; },
  }]);
}

const NO_ROOM = "Fruit has piled up all around the tree - there's no room for more to fall. Pick some up first.";

/** Free tile next to (x, y) for fruit to land on. */
function landingSpot(w: World, x: number, y: number): [number, number] | null {
  for (const [dx, dy] of DIRS) {
    const nx = x + dx, ny = y + dy;
    if (standable(w, nx, ny) && tileAt(w, nx, ny).kind === "ground" && !objsAt(w, nx, ny).length) return [nx, ny];
  }
  return null;
}

function dropFruit(tree: Obj): Op {
  return {
    anim: { t: "fx", kind: "shake", dur: 0.8 },
    apply: (world) => {
      const spot = landingSpot(world, tree.x, tree.y);
      if (!spot || tree.fruit <= 0) return;
      tree.fruit--;
      tree.regrowAt = Math.max(tree.regrowAt, world.time + 45);
      const berry: Obj = {
        id: world.nextId++, kind: "berry", x: spot[0], y: spot[1], state: "world",
        home: null, respawnAt: 0, fruit: 0, regrowAt: 0,
      };
      world.objs.push(berry);
    },
  };
}

function push(w: World, arg: string): Outcome {
  const kind = asThing(arg);
  if (!kind || !["rock", "tree", "gate"].includes(kind)) return fail("cant_do", `You can't push ${arg === "none" ? "nothing" : `a ${arg}`}.`);
  if (w.blob.energy < 3) return fail("too_tired", REASON_TEXT.too_tired + ".");
  if (kind === "rock") return pushRock(w);
  const a = approach(w, kind);
  if (isOutcome(a)) return a;
  const { target, ops } = a;
  if (kind === "gate") return fail("cant_do", "You push the gate. It won't budge.", ops);
  const tree = target.obj;
  if (kind === "tree" && tree) {
    if (w.blob.shape !== "ball") return fail("cant_do", "You push the tree. It barely wobbles - you'd need to hit it harder.", ops);
    if (tree.fruit <= 0) return fail("cant_do", "You ram the tree but it has no fruit left.", ops);
    if (!landingSpot(w, tree.x, tree.y)) return fail("cant_do", NO_ROOM, ops);
    return done("You ram the tree! A berry falls down.", [...ops, pushFx(), dropFruit(tree)]);
  }
  return fail("cant_do", "Nothing to push there.", ops);
}

/** Pick a rock and a side to push it from - preferring a push that fills a gap. */
function pushRock(w: World): Outcome {
  const b = w.blob;
  const rocks = candidates(w, "rock").sort((p, q) => dist(b.x, b.y, p.x, p.y) - dist(b.x, b.y, q.x, q.y)).slice(0, 4);
  if (!rocks.length) return fail("not_found", `You look around for a rock but ${REASON_TEXT.not_found}.`);
  let best: { rock: Obj; path: PathStep[]; roll: [number, number, boolean] } | null = null;
  for (const r of rocks) {
    const rock = r.obj;
    if (!rock) continue;
    for (const [dx, dy] of DIRS) {
      const sx = rock.x - dx, sy = rock.y - dy;
      if (!standable(w, sx, sy)) continue;
      const path = findPath(w, b.shape, (px, py) => px === sx && py === sy);
      const roll = path && rollRock(w, rock, dx, dy, sx, sy);
      if (!path || !roll) continue;
      const better = !best || (roll[2] && !best.roll[2]) || (roll[2] === best.roll[2] && path.length < best.path.length);
      if (better) best = { rock, path, roll };
    }
  }
  if (!best) {
    const first = rocks[0];
    const reason = first ? blockedReason(w, reachGoal(w, first.x, first.y, "rock")) : "no_way";
    return fail(reason === "no_way" ? "cant_do" : reason, reason === "no_way"
      ? "You can't find a side to push the rock from - it's stuck."
      : `You can't get to the rock as a ${b.shape}: ${REASON_TEXT[reason]}.`);
  }
  if (pathCost(b.shape, best.path) + 3 > b.energy) return fail("too_tired", `The rock is too far - ${REASON_TEXT.too_tired}.`);
  const { rock, path, roll: [rx, ry, fills] } = best;
  return done(fills ? "You push the rock... it rolls into the gap and fills it! Now there's a bridge." : "You push the rock and it rolls away.",
    [...walkOps(w, path), pushFx(), {
      anim: { t: "fx", kind: fills ? "splash" : "push", dur: 0.5 },
      apply: (world) => {
        rock.x = rx; rock.y = ry;
        if (fills) {
          setTile(world, rx, ry, { kind: "ground", h: 0 });
          rock.state = "gone";
        }
      },
    }]);
}

function pushFx(): Op {
  return { anim: { t: "fx", kind: "push", dur: 0.4 }, apply: (w) => { w.blob.energy = clamp(w.blob.energy - 3, 0, 100); } };
}

/** Roll until blocked; a gap swallows the rock and becomes ground. */
function rollRock(w: World, rock: Obj, dx: number, dy: number, bx: number, by: number): [number, number, boolean] | null {
  let x = rock.x, y = rock.y, moved = false;
  for (let i = 0; i < 6; i++) {
    const nx = x + dx, ny = y + dy;
    if (!inBounds(w, nx, ny)) break;
    const t = tileAt(w, nx, ny);
    if (t.kind === "gap") return [nx, ny, true];
    if (t.kind !== "ground" || t.h > tileAt(w, x, y).h || solidAt(w, nx, ny)) break;
    if (bx === nx && by === ny) break;
    x = nx; y = ny; moved = true;
  }
  return moved ? [x, y, false] : null;
}

function use(w: World, arg: string): Outcome {
  const held = objById(w, w.blob.holding);
  if (!held) return fail("not_holding", `Use what? ${REASON_TEXT.not_holding}.`);
  if (held.kind !== "stick") return fail("cant_do", `You wave the ${held.kind} around. Nothing happens.`);
  const kind = asThing(arg);
  if (!kind) return fail("unknown_step", `Use the stick on what?`);
  if (kind === "tree") {
    const a = approach(w, "tree");
    if (isOutcome(a)) return a;
    const tree = a.target.obj;
    if (!tree || tree.fruit <= 0) return fail("cant_do", "You poke the tree but it has no fruit left.", a.ops);
    if (!landingSpot(w, tree.x, tree.y)) return fail("cant_do", NO_ROOM, a.ops);
    return done("You whack the tree with the stick. A berry falls down!", [...a.ops, dropFruit(tree)]);
  }
  if (FOODS.includes(kind as Obj["kind"])) {
    // The stick lets you reach things 2 tiles away, at any height, even across gaps.
    const cands = candidates(w, kind);
    const near = (px: number, py: number) => cands.some((c) => {
      const d = dist(px, py, c.x, c.y);
      return d >= 1 && d <= 2 && (px === c.x || py === c.y) && !solidBetween(w, px, py, c.x, c.y);
    });
    const path = findPath(w, w.blob.shape, near);
    if (!path) return fail(blockedReason(w, near), `You can't get close enough to reach a ${kind} with the stick.`);
    const end = path.length ? path[path.length - 1] ?? { x: w.blob.x, y: w.blob.y } : { x: w.blob.x, y: w.blob.y };
    const item = cands.map((c) => c.obj).find((o) => o && dist(end.x, end.y, o.x, o.y) <= 2);
    if (!item) return fail("not_found", `No ${kind} in reach.`);
    return done(`You hook the ${kind} with the stick and pull it to you!`, [...walkOps(w, path), {
      anim: { t: "fx", kind: "grab", dur: 0.6 },
      apply: (world) => { item.x = world.blob.x; item.y = world.blob.y; },
    }]);
  }
  return fail("cant_do", `You poke the ${kind} with the stick. Nothing happens.`);
}

function solidBetween(w: World, ax: number, ay: number, bx: number, by: number): boolean {
  const dx = Math.sign(bx - ax), dy = Math.sign(by - ay);
  for (let x = ax + dx, y = ay + dy; x !== bx || y !== by; x += dx, y += dy) {
    if (solidAt(w, x, y) || tileAt(w, x, y).h > Math.max(tileAt(w, ax, ay).h, tileAt(w, bx, by).h) + 1) return true;
  }
  return false;
}

/** Only this many unclaimed deliveries are remembered - the wizard throws out the rest. */
const MAX_DELIVERIES = 10;

/** Carry what you're holding to the wizard's tower and hand it over. */
function deliver(w: World): Outcome {
  const held = objById(w, w.blob.holding);
  if (!held) return fail("not_holding", `Deliver what? ${REASON_TEXT.not_holding}. Grab something first.`);
  const a = approach(w, "tower");
  if (isOutcome(a)) return a;
  return done(`You bring the ${held.kind} to the tower. A wrinkly hand snatches it through the window.`, [...a.ops, {
    anim: { t: "fx", kind: "drop", dur: 0.5 },
    apply: (world) => {
      world.deliveries.push({ kind: held.kind, home: held.home });
      if (world.deliveries.length > MAX_DELIVERIES) world.deliveries.shift();
      world.blob.holding = null;
      held.state = "gone";
      held.respawnAt = held.home ? world.time + RESPAWN_S : 0;
    },
  }]);
}

function rest(w: World): Outcome {
  // Starving still gets a fitful nap - enough to crawl to food, so there's never a dead end.
  if (w.blob.food <= 0) {
    return done("Your tummy rumbles too much to really sleep, but you get a little energy back. Find food!", [{
      anim: { t: "fx", kind: "rest", dur: 1.5 },
      apply: (world) => { world.blob.energy = clamp(world.blob.energy + 12, 0, 100); },
    }]);
  }
  return done("You rest for a bit. Zzz...", [{
    anim: { t: "fx", kind: "rest", dur: 2.5 },
    apply: (world) => { world.blob.energy = clamp(world.blob.energy + 35, 0, 100); },
  }]);
}

/** Plan one step against the current world. Nothing changes until the ops are applied. */
export function planStep(w: World, step: Step): Outcome {
  switch (step.do) {
    case "move_to": return moveTo(w, step.arg);
    case "morph": return morph(w, step.arg);
    case "eat": return eat(w, step.arg);
    case "grab": return grab(w, step.arg);
    case "drop": return drop(w);
    case "push": return push(w, step.arg);
    case "use": return use(w, step.arg);
    case "deliver": return deliver(w);
    case "rest": return rest(w);
    case "skill": return fail("unknown_step", "Skills are expanded before they run.");
  }
}

/** Plan and apply immediately (tests, and headless runs). */
export function runStep(w: World, step: Step): Outcome {
  const out = planStep(w, step);
  for (const op of out.ops) op.apply(w);
  return out;
}
