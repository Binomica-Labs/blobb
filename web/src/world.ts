// The island: terrain, things on it, and the blob. Pure data + rules, no rendering.
// An old wizard lives in the tower here. He summoned the blob out of another dimension with his
// spellbook so it can run the villagers' errands and he never has to talk to anyone again.

export const SHAPES = ["blob", "ball", "puddle", "spring"] as const;
export type Shape = (typeof SHAPES)[number];

export const THINGS = ["berry", "mushroom", "stick", "rock", "tree", "tower", "gate", "gap"] as const;
export type Thing = (typeof THINGS)[number];
export type ObjKind = Exclude<Thing, "gap">;

export const FOODS: readonly ObjKind[] = ["berry", "mushroom"];
export const PORTABLE: readonly ObjKind[] = ["berry", "mushroom", "stick"];
/** Things that occupy their tile so nothing can stand there. */
export const SOLID: readonly ObjKind[] = ["rock", "tree", "tower"];
/** Things that come back after being eaten or handed to the wizard. */
const RESPAWNS: readonly ObjKind[] = ["berry", "mushroom", "stick"];

export type TileKind = "ground" | "gap" | "water" | "gate";
export interface Tile { kind: TileKind; h: number }

export interface Obj {
  id: number;
  kind: ObjKind;
  x: number;
  y: number;
  state: "world" | "held" | "gone";
  /** Where it respawns (food and sticks). */
  home: readonly [number, number] | null;
  respawnAt: number;
  /** Trees: fruit left to shake down. */
  fruit: number;
  regrowAt: number;
}

export interface Blob {
  x: number;
  y: number;
  shape: Shape;
  food: number;
  energy: number;
  holding: number | null;
  facing: readonly [number, number];
}

/** Something handed in at the tower. `home` says where it came from (null: shaken from a tree). */
export interface Delivery { kind: ObjKind; home: readonly [number, number] | null }

export interface World {
  w: number;
  d: number;
  tiles: Tile[];
  objs: Obj[];
  blob: Blob;
  time: number;
  nextId: number;
  /** Bumped whenever terrain changes, so the renderer knows to rebuild. */
  terrainVersion: number;
  /** Handed to the wizard and not yet matched to an errand (newest last, capped). */
  deliveries: Delivery[];
}

/** How high each shape can climb in one step. */
export const CLIMB: Record<Shape, number> = { blob: 1, ball: 0, puddle: 0, spring: 2 };

export const RESPAWN_S = 90;
export const REGROW_S = 45;

// Tutorial island. North is up (y = 0). Digits are ground height, '#' a gap,
// 'G' a low gate (only a puddle fits under), '~' water.
//   NW: a high plateau with berries (spring up, or reach with a stick)
//   NE: a walled garden behind a low gate (puddle under it)
//   SE: a pocket across a gap (spring over, or roll the rock in to bridge it)
//   middle: the wizard's tower, where errands get delivered
export const TUTORIAL_MAP = [
  "~~~~~~~~~~~~~~~~",
  "~22220000333333~",
  "~22220000300003~",
  "~22220000300003~",
  "~22220000300003~",
  "~0000000033G333~",
  "~00000000000000~",
  "~00000000000000~",
  "~0000000000####~",
  "~0000000000#000~",
  "~0001100000#000~",
  "~0001100000#000~",
  "~0000000000#000~",
  "~0000000000#000~",
  "~0000000000#000~",
  "~~~~~~~~~~~~~~~~",
];

type Placement = readonly [ObjKind, number, number];
export const TUTORIAL_OBJS: readonly Placement[] = [
  ["berry", 2, 2], ["berry", 4, 3],
  ["mushroom", 11, 3], ["berry", 12, 2],
  ["mushroom", 13, 11], ["berry", 12, 13],
  ["berry", 9, 6], ["mushroom", 3, 8],
  ["tree", 5, 12], ["stick", 8, 13], ["rock", 10, 11], ["tower", 7, 7],
];
export const TOWER_AT = [7, 7] as const;
export const TUTORIAL_START = [7, 9] as const;

export function parseMap(rows: readonly string[]): { w: number; d: number; tiles: Tile[] } {
  const d = rows.length;
  const w = rows[0]?.length ?? 0;
  const tiles: Tile[] = [];
  for (const row of rows) {
    if (row.length !== w) throw new Error("map rows must all be the same width");
    for (const c of row) {
      if (c === "~") tiles.push({ kind: "water", h: 0 });
      else if (c === "#") tiles.push({ kind: "gap", h: 0 });
      else if (c === "G") tiles.push({ kind: "gate", h: 0 });
      else if (c >= "0" && c <= "9") tiles.push({ kind: "ground", h: Number(c) });
      else throw new Error(`unknown map char ${c}`);
    }
  }
  return { w, d, tiles };
}

export function makeWorld(
  rows: readonly string[] = TUTORIAL_MAP,
  objs: readonly Placement[] = TUTORIAL_OBJS,
  start: readonly [number, number] = TUTORIAL_START,
): World {
  const { w, d, tiles } = parseMap(rows);
  const world: World = {
    w, d, tiles, objs: [], time: 0, nextId: 1, terrainVersion: 0, deliveries: [],
    blob: { x: start[0], y: start[1], shape: "blob", food: 70, energy: 90, holding: null, facing: [0, -1] },
  };
  for (const [kind, x, y] of objs) addObj(world, kind, x, y);
  return world;
}

export function addObj(world: World, kind: ObjKind, x: number, y: number): Obj {
  const o: Obj = {
    id: world.nextId++, kind, x, y, state: "world",
    home: RESPAWNS.includes(kind) ? [x, y] : null, respawnAt: 0,
    fruit: kind === "tree" ? 3 : 0, regrowAt: 0,
  };
  world.objs.push(o);
  return o;
}

export function inBounds(world: World, x: number, y: number): boolean {
  return x >= 0 && y >= 0 && x < world.w && y < world.d;
}

export function tileAt(world: World, x: number, y: number): Tile {
  return world.tiles[y * world.w + x] ?? { kind: "water", h: 0 };
}

export function setTile(world: World, x: number, y: number, tile: Tile): void {
  world.tiles[y * world.w + x] = tile;
  world.terrainVersion++;
}

export function objsAt(world: World, x: number, y: number): Obj[] {
  return world.objs.filter((o) => o.state === "world" && o.x === x && o.y === y);
}

export function solidAt(world: World, x: number, y: number): Obj | undefined {
  return objsAt(world, x, y).find((o) => SOLID.includes(o.kind));
}

export function objById(world: World, id: number | null): Obj | undefined {
  return id === null ? undefined : world.objs.find((o) => o.id === id);
}

/** Can anything stand here at all (ignoring shape)? */
export function standable(world: World, x: number, y: number): boolean {
  if (!inBounds(world, x, y)) return false;
  const t = tileAt(world, x, y);
  return (t.kind === "ground" || t.kind === "gate") && !solidAt(world, x, y);
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/** Advance the clock: hunger, tiredness, respawns and regrowth. Big steps are split up so
 *  e.g. food running out halfway through still counts the first half as fed. */
export function tick(world: World, dt: number): void {
  for (let left = dt; left > 0; left -= 1) step(world, Math.min(1, left));
  // Fruit shaken from trees doesn't respawn; once eaten it's just clutter in the save.
  if (world.objs.some((o) => o.state === "gone" && !o.home)) world.objs = world.objs.filter((o) => o.state !== "gone" || o.home);
}

/** Hunger never drops below this while you're away - coming back to a starving blob isn't fun. */
export const AWAY_FOOD_FLOOR = 20;

/** Time passing while the game was closed: capped, and the blob waits for you hungry, not starving. */
export function catchUp(world: World, seconds: number): void {
  const floor = Math.min(AWAY_FOOD_FLOOR, world.blob.food);
  for (let left = Math.min(600, Math.max(0, seconds)); left > 0; left -= 1) {
    step(world, Math.min(1, left));
    world.blob.food = Math.max(world.blob.food, floor);
  }
  tick(world, 0);
}

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isRec = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;
const isTile = (t: unknown): boolean => isRec(t) && isNum(t["h"]) && ["ground", "gap", "water", "gate"].includes(String(t["kind"]));
const isObj = (o: unknown): boolean => isRec(o) && isNum(o["id"]) && isNum(o["x"]) && isNum(o["y"])
  && (THINGS as readonly string[]).includes(String(o["kind"])) && ["world", "held", "gone"].includes(String(o["state"]));

/** A saved island, if it's sane and matches the current map size; otherwise a fresh one. */
export function restoreWorld(raw: unknown, now: number): World {
  const fresh = makeWorld();
  if (typeof raw !== "object" || raw === null) return fresh;
  const s = raw as Partial<World> & { savedAt?: unknown };
  const b = s.blob as Partial<Blob> | undefined;
  const ok = s.w === fresh.w && s.d === fresh.d && isNum(s.time) && isNum(s.nextId)
    && Array.isArray(s.tiles) && s.tiles.length === fresh.w * fresh.d
    && (s.tiles as unknown[]).every(isTile) && Array.isArray(s.objs) && (s.objs as unknown[]).every(isObj)
    && typeof b === "object" && isNum(b.x) && isNum(b.y) && isNum(b.food) && isNum(b.energy)
    && inBounds(fresh, b.x, b.y) && (SHAPES as readonly string[]).includes(b.shape ?? "");
  if (!ok) return fresh;
  const w = s as World;
  // A held thing that no longer exists (or a thing marked held that nobody holds) - drop it at the blob's feet.
  const held = w.objs.find((o) => o.id === w.blob.holding);
  if (!held) w.blob.holding = null;
  for (const o of w.objs) if (o.state === "held" && o !== held) { o.state = "world"; o.x = w.blob.x; o.y = w.blob.y; }
  if (held) held.state = "held";
  if (!Array.isArray(w.blob.facing)) w.blob.facing = [0, -1];
  // Saves from before the wizard moved in: build his tower, and let sticks come back.
  if (!Array.isArray(w.deliveries)) w.deliveries = [];
  if (!w.objs.some((o) => o.kind === "tower")) {
    const [tx, ty] = TOWER_AT;
    if ((w.blob.x === tx && w.blob.y === ty) || w.objs.some((o) => o.state === "world" && o.x === tx && o.y === ty && SOLID.includes(o.kind))) return fresh;
    addObj(w, "tower", tx, ty);
  }
  for (const o of w.objs) if (o.kind === "stick" && !o.home) o.home = [o.x, o.y];
  w.terrainVersion = isNum(w.terrainVersion) ? w.terrainVersion : 0;
  if (isNum(s.savedAt)) catchUp(w, (now - s.savedAt) / 1000);
  return w;
}

function step(world: World, dt: number): void {
  world.time += dt;
  const b = world.blob;
  b.food = clamp(b.food - dt / 5, 0, 100);
  b.energy = clamp(b.energy + (b.food > 0 ? dt * 0.15 : -dt * 0.5), 0, 100);
  for (const o of world.objs) {
    if (o.state === "gone" && o.home && o.respawnAt > 0 && world.time >= o.respawnAt) {
      const [hx, hy] = o.home;
      if (standable(world, hx, hy) && !(b.x === hx && b.y === hy)) {
        o.x = hx; o.y = hy; o.state = "world"; o.respawnAt = 0;
      }
    }
    if (o.kind === "tree" && o.fruit < 3 && world.time >= o.regrowAt) {
      o.fruit++;
      o.regrowAt = world.time + REGROW_S;
    }
  }
}
