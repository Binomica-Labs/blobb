// Three.js view of the world. Reads world state every frame and smooths toward it;
// the blob itself follows an animation queue so multi-tile moves play out step by step.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Anim } from "./actions";
import { objById, tileAt, type Obj, type Shape, type World } from "./world";

const topY = (h: number) => 0.4 + h * 0.5;

const COLORS = {
  grass: [0x7cc86b, 0x74c063], hill: [0x96d47c, 0x8ccc72], high: [0xb3dd8b, 0xa9d682],
  wall: [0xa39e8f, 0x999485], side: 0x8a6f4e, water: 0x4aa3df, sky: 0xbfe6ff, gapDark: 0x1d1a24,
};

const SHAPE_SCALE: Record<Shape, THREE.Vector3> = {
  blob: new THREE.Vector3(1, 0.82, 1),
  ball: new THREE.Vector3(0.78, 0.78, 0.78),
  puddle: new THREE.Vector3(1.5, 0.22, 1.5),
  spring: new THREE.Vector3(0.62, 1.5, 0.62),
};
const SHAPE_TINT: Record<Shape, number> = { blob: 0x5fd3b0, ball: 0x4cc2a8, puddle: 0x6fdcc0, spring: 0x8fe0a0 };
const R = 0.38;

interface Running { anim: Anim; t: number; from: THREE.Vector3; resolve: () => void }

export class View {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private terrain = new THREE.Group();
  private terrainVersion = -1;
  private objMeshes = new Map<number, { g: THREE.Object3D; seen: boolean; fade: number }>();
  private blob: THREE.Group;
  private body: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>;
  private basePos: Float32Array;
  private face: THREE.Group;
  private shadow: THREE.Mesh;
  private heldSlot = new THREE.Group();
  private scale = SHAPE_SCALE.blob.clone();
  private queue: Running[] = [];
  private running: Running | null = null;
  /** Displayed blob position (tile-space x, y, and world height). */
  private pos = new THREE.Vector3();
  private squash = 0;
  private fxKind: string | null = null;
  private fxT = 0;
  private yaw = 0;
  private roll = new THREE.Quaternion();
  private time = 0;
  private primed = false;
  /** Summoning ring the blob drops out of on load and reset. */
  private portal = new THREE.Mesh(new THREE.TorusGeometry(0.6, 0.07, 12, 48),
    new THREE.MeshBasicMaterial({ color: 0xb48cff, transparent: true, opacity: 0.9, depthWrite: false }));
  private portalT = 0;

  constructor(private canvas: HTMLCanvasElement, private world: World) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene.background = new THREE.Color(COLORS.sky);
    this.scene.fog = new THREE.Fog(COLORS.sky, 22, 45);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    this.camera.position.set(0, 9, 10);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enablePan = false;
    this.controls.enableDamping = true;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 22;
    this.controls.maxPolarAngle = Math.PI * 0.45;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x6b8f5e, 1.6));
    const sun = new THREE.DirectionalLight(0xfff2d6, 1.8);
    sun.position.set(-6, 12, 4);
    this.scene.add(sun);

    const water = new THREE.Mesh(new THREE.PlaneGeometry(200, 200),
      new THREE.MeshStandardMaterial({ color: COLORS.water, roughness: 0.25, transparent: true, opacity: 0.9 }));
    water.rotation.x = -Math.PI / 2;
    water.position.y = 0.15;
    this.scene.add(water, this.terrain);

    // The blob: a wobbly sphere with a face.
    const geo = new THREE.SphereGeometry(R, 40, 28);
    this.basePos = Float32Array.from(geo.attributes["position"]?.array ?? []);
    this.body = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color: SHAPE_TINT.blob, roughness: 0.25, metalness: 0.05, transparent: true, opacity: 0.93,
      emissive: 0x1a5c4c, emissiveIntensity: 0.25,
    }));
    this.face = makeFace();
    this.blob = new THREE.Group();
    this.blob.add(this.body, this.face, this.heldSlot);
    this.shadow = new THREE.Mesh(new THREE.CircleGeometry(R * 1.1, 24),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.22, depthWrite: false }));
    this.shadow.rotation.x = -Math.PI / 2;
    this.scene.add(this.blob, this.shadow, this.portal);

    const b = world.blob;
    this.pos.set(b.x, b.y, topY(tileAt(world, b.x, b.y).h));
    this.syncTerrain();
  }

  setWorld(world: World): void {
    this.world = world;
    for (const { g } of this.objMeshes.values()) g.removeFromParent(); // held things hang off the blob, not the scene
    this.objMeshes.clear();
    this.terrainVersion = -1;
    this.primed = false;
    // Anything still queued belonged to the old island. Resolve it so the plan awaiting it can notice and stop.
    for (const r of [this.running, ...this.queue]) r?.resolve();
    this.queue = []; this.running = null;
    const b = world.blob;
    this.pos.set(b.x, b.y, topY(tileAt(world, b.x, b.y).h));
    this.portalT = 0;
  }

  resize(w: number, h: number): void {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Tile coords -> scene position. */
  private at(x: number, y: number, height: number): THREE.Vector3 {
    return new THREE.Vector3(x - this.world.w / 2 + 0.5, height, y - this.world.d / 2 + 0.5);
  }

  play(anim: Anim): Promise<void> {
    return new Promise((resolve) => { this.queue.push({ anim, t: 0, from: this.pos.clone(), resolve }); });
  }

  /** Blob head position on screen, for speech bubbles. */
  bubbleAnchor(): { x: number; y: number } {
    const p = this.blob.position.clone().add(new THREE.Vector3(0, 0.55 + R * this.scale.y, 0)).project(this.camera);
    const r = this.canvas.getBoundingClientRect();
    return { x: r.left + (p.x + 1) / 2 * r.width, y: r.top + (1 - p.y) / 2 * r.height };
  }

  private syncTerrain(): void {
    if (this.terrainVersion === this.world.terrainVersion) return;
    this.terrainVersion = this.world.terrainVersion;
    this.terrain.clear();
    const w = this.world;
    const box = new THREE.BoxGeometry(1, 1, 1);
    const count = w.w * w.d;
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.9 });
    const tops = new THREE.InstancedMesh(box, mat, count);
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    let i = 0;
    for (let y = 0; y < w.d; y++) for (let x = 0; x < w.w; x++) {
      const t = tileAt(w, x, y);
      if (t.kind === "water") continue;
      const checker = (x + y) % 2;
      if (t.kind === "gap") {
        m.compose(this.at(x, y, -2.6), new THREE.Quaternion(), new THREE.Vector3(1, 0.2, 1));
        c.set(COLORS.gapDark);
      } else {
        const top = topY(t.h);
        m.compose(this.at(x, y, (top - 1.2) / 2), new THREE.Quaternion(), new THREE.Vector3(1, top + 1.2, 1));
        const pal = t.h >= 3 ? COLORS.wall : t.h === 2 ? COLORS.high : t.h === 1 ? COLORS.hill : COLORS.grass;
        c.set(pal[checker] ?? COLORS.side);
      }
      tops.setMatrixAt(i, m);
      tops.setColorAt(i, c);
      i++;
      if (t.kind === "gate") this.terrain.add(makeGate(this.at(x, y, topY(t.h))));
    }
    tops.count = i;
    this.terrain.add(tops);
    // Island skirt so the edges read as cliffs, not floating tiles.
    const skirt = new THREE.Mesh(new THREE.BoxGeometry(w.w - 1.6, 1.2, w.d - 1.6),
      new THREE.MeshStandardMaterial({ color: COLORS.side, roughness: 1 }));
    skirt.position.y = -0.5;
    this.terrain.add(skirt);
  }

  private syncObjects(dt: number): void {
    const w = this.world;
    if (this.objMeshes.size > w.objs.length) {
      const live = new Set(w.objs.map((o) => o.id));
      for (const [id, { g }] of this.objMeshes) if (!live.has(id)) { g.removeFromParent(); this.objMeshes.delete(id); }
    }
    for (const o of w.objs) {
      let entry = this.objMeshes.get(o.id);
      if (!entry) {
        const g = makeObj(o);
        entry = { g, seen: false, fade: o.state === "world" ? 1 : 0 };
        this.objMeshes.set(o.id, entry);
        this.scene.add(g);
        const p = this.at(o.x, o.y, topY(tileAt(w, o.x, o.y).h));
        // New fruit drops from above.
        g.position.copy(p).add(new THREE.Vector3(0, this.primed ? 1.6 : 0, 0));
      }
      const { g } = entry;
      if (o.state === "held") {
        if (g.parent !== this.heldSlot) { this.heldSlot.add(g); g.position.set(0, 0, 0); }
        g.scale.setScalar(0.8);
        continue;
      }
      if (g.parent !== this.scene) { this.scene.add(g); g.position.copy(this.blob.position); }
      const t = tileAt(w, o.x, o.y);
      const target = this.at(o.x, o.y, t.kind === "gap" ? -0.3 : topY(t.h));
      g.position.lerp(target, 1 - Math.exp(-dt * (o.kind === "rock" ? 5 : 8)));
      entry.fade += (o.state === "world" ? 1 : -1) * dt * 2.5;
      entry.fade = Math.max(0, Math.min(1, entry.fade));
      g.visible = entry.fade > 0.01;
      g.scale.setScalar(entry.fade);
      if (o.kind === "tree") {
        g.children.forEach((ch) => { if (ch.name === "fruit") ch.visible = Number(ch.userData["n"]) < o.fruit; });
        if (this.fxKind === "shake" && this.fxT < 0.8) g.rotation.z = Math.sin(this.fxT * 40) * 0.06 * (1 - this.fxT / 0.8);
        else g.rotation.z = 0;
      }
      if (o.kind === "rock" && entry.g.userData["lastX"] !== o.x) {
        g.rotation.x += dt * 4;
        if (g.position.distanceTo(target) < 0.05) g.userData["lastX"] = o.x;
      }
    }
    this.primed = true;
  }

  private stepAnim(dt: number): void {
    if (!this.running) {
      const next = this.queue.shift();
      if (!next) return;
      next.from = this.pos.clone();
      this.running = next;
      if (next.anim.t === "morph") this.squash = 1;
      if (next.anim.t === "fx") { this.fxKind = next.anim.kind; this.fxT = 0; }
    }
    const r = this.running;
    r.t += dt;
    const a = r.anim;
    const dur = a.t === "morph" ? 0.45 : a.dur;
    const k = Math.min(1, r.t / dur);
    if (a.t === "walk" || a.t === "jump") {
      const toH = topY(tileAt(this.world, a.x, a.y).h);
      const e = a.t === "jump" ? k : k * k * (3 - 2 * k);
      this.pos.set(r.from.x + (a.x - r.from.x) * e, r.from.y + (a.y - r.from.y) * e, r.from.z + (toH - r.from.z) * e);
      const arc = a.t === "jump" ? 1.4 : this.world.blob.shape === "spring" ? 0.35 : this.world.blob.shape === "blob" ? 0.12 : 0;
      this.pos.z += Math.sin(Math.PI * k) * arc;
      const dx = a.x - r.from.x, dy = a.y - r.from.y;
      if (dx || dy) {
        const targetYaw = Math.atan2(dx, dy);
        let d = targetYaw - this.yaw;
        while (d > Math.PI) d -= 2 * Math.PI;
        while (d < -Math.PI) d += 2 * Math.PI;
        this.yaw += d * Math.min(1, dt * 14);
        if (this.world.blob.shape === "ball") {
          const axis = new THREE.Vector3(dy, 0, -dx).normalize();
          this.roll.premultiply(new THREE.Quaternion().setFromAxisAngle(axis, -dt * 9));
        }
      }
    }
    if (k >= 1) {
      this.running = null;
      r.resolve();
    }
  }

  frame(dt: number): void {
    this.time += dt;
    this.fxT += dt;
    this.portalT += dt;
    this.syncTerrain();
    this.stepAnim(dt);
    this.syncObjects(dt);
    const b = this.world.blob;
    if (!this.running && !this.queue.length) {
      // Idle: settle onto the logical tile (e.g. after a reset).
      const target = new THREE.Vector3(b.x, b.y, topY(tileAt(this.world, b.x, b.y).h));
      this.pos.lerp(target, 1 - Math.exp(-dt * 10));
    }

    // Shape: ease toward the current shape, with a squash on morph and fx.
    const want = SHAPE_SCALE[b.shape];
    this.scale.lerp(want, 1 - Math.exp(-dt * 8));
    this.squash = Math.max(0, this.squash - dt * 2.5);
    let sx = this.scale.x, sy = this.scale.y;
    const breathe = Math.sin(this.time * 2.2) * 0.03;
    sy *= 1 + breathe - this.squash * 0.35;
    sx *= 1 - breathe * 0.5 + this.squash * 0.25;
    let lunge = 0, headShake = 0;
    const f = this.fxKind;
    if (f && this.fxT < 1.2) {
      const p = this.fxT;
      if (f === "chomp") sy *= 1 + Math.abs(Math.sin(p * 18)) * 0.18 * Math.max(0, 1 - p);
      if (f === "push") lunge = Math.sin(Math.min(1, p / 0.4) * Math.PI) * 0.3;
      if (f === "fail") headShake = Math.sin(p * 30) * 0.35 * Math.max(0, 1 - p / 0.6);
      if (f === "grab" || f === "drop") sy *= 1 - Math.sin(Math.min(1, p / 0.4) * Math.PI) * 0.25;
    }
    const resting = f === "rest" && this.fxT < 2.5;
    if (resting) { sy *= 0.75; sx *= 1.12; }
    if (b.shape === "spring") sy *= 1 + Math.abs(Math.sin(this.time * 5)) * 0.08;
    // Summoning: the blob squeezes out of the ring, then the ring closes.
    const arrive = Math.min(1, Math.max(0, (this.portalT - 0.3) / 0.6));
    sx *= arrive; sy *= arrive * (1 + Math.sin(arrive * Math.PI) * 0.4);
    this.body.scale.set(sx, sy, sx);
    this.body.material.color.lerp(new THREE.Color(SHAPE_TINT[b.shape]), 1 - Math.exp(-dt * 6));
    if (f === "fail" && this.fxT < 0.6) this.body.material.color.lerp(new THREE.Color(0x7a9ad6), 0.15);

    // Wobble the surface.
    const pos = this.body.geometry.attributes["position"];
    if (pos) {
      const arr = pos.array as Float32Array;
      const amp = b.shape === "ball" ? 0.015 : 0.05;
      for (let i = 0; i < arr.length; i += 3) {
        const x = this.basePos[i] ?? 0, y = this.basePos[i + 1] ?? 0, z = this.basePos[i + 2] ?? 0;
        const n = 1 + amp * Math.sin(this.time * 3 + x * 9 + y * 5) * Math.sin(this.time * 2.3 + z * 8);
        arr[i] = x * n; arr[i + 1] = y * n; arr[i + 2] = z * n;
      }
      pos.needsUpdate = true;
      this.body.geometry.computeVertexNormals();
    }

    const fwd = new THREE.Vector3(Math.sin(this.yaw), 0, Math.cos(this.yaw));
    const ground = this.at(this.pos.x, this.pos.y, this.pos.z);
    this.blob.position.copy(ground).add(new THREE.Vector3(0, R * sy, 0)).addScaledVector(fwd, lunge);
    this.blob.rotation.set(0, this.yaw + headShake, 0);
    this.body.quaternion.copy(b.shape === "ball" ? this.roll : new THREE.Quaternion());
    // Face sits on the front, a bit above centre; on a puddle it peeks out the top.
    this.face.position.set(0, R * sy * (b.shape === "puddle" ? 0.9 : 0.25), R * sx * 0.92);
    this.face.scale.setScalar(b.shape === "puddle" ? 0.8 : 1);
    this.face.children.forEach((eye) => { eye.scale.y = resting ? 0.15 : Math.sin(this.time * 0.9) > 0.985 ? 0.1 : 1; });
    this.heldSlot.position.set(0, R * sy * 2 + 0.12, 0);
    const shadowH = topY(tileAt(this.world, Math.round(this.pos.x), Math.round(this.pos.y)).h);
    this.shadow.position.copy(this.at(this.pos.x, this.pos.y, shadowH + 0.01));
    this.shadow.scale.setScalar(sx * (1 - Math.min(0.5, (this.pos.z - shadowH) * 0.3)));
    const open = this.portalT < 0.3 ? this.portalT / 0.3 : Math.max(0, 1 - (this.portalT - 1) / 0.6);
    this.portal.visible = open > 0.01;
    if (this.portal.visible) {
      this.portal.position.copy(this.at(this.pos.x, this.pos.y, shadowH + 0.05));
      this.portal.rotation.set(-Math.PI / 2, 0, this.time * 4);
      this.portal.scale.setScalar(open);
    }

    // Camera follows the blob.
    this.controls.target.lerp(this.blob.position, 1 - Math.exp(-dt * 3));
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Recenter the camera behind the blob (after load or reset). */
  snapCamera(): void {
    this.frame(0);
    this.controls.target.copy(this.blob.position);
    this.camera.position.copy(this.blob.position).add(new THREE.Vector3(0, 9, 11));
  }

  heldKind(): string | null {
    return objById(this.world, this.world.blob.holding)?.kind ?? null;
  }
}

function makeFace(): THREE.Group {
  const face = new THREE.Group();
  const white = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 });
  const black = new THREE.MeshBasicMaterial({ color: 0x111122 });
  for (const side of [-1, 1]) {
    const eye = new THREE.Group();
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 12), white);
    const pupil = new THREE.Mesh(new THREE.SphereGeometry(0.042, 12, 10), black);
    pupil.position.z = 0.05;
    eye.add(ball, pupil);
    eye.position.set(side * 0.12, 0, 0);
    face.add(eye);
  }
  return face;
}

function makeGate(p: THREE.Vector3): THREE.Group {
  const g = new THREE.Group();
  const wood = new THREE.MeshStandardMaterial({ color: 0x9b6a3c, roughness: 0.8 });
  for (const side of [-0.45, 0.45]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.12, 1.4, 0.12), wood);
    post.position.set(side, 0.7, 0);
    g.add(post);
  }
  // Solid panel from shoulder height down to a gap only a puddle fits through.
  const panel = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.1, 0.08), wood);
  panel.position.y = 0.28 + 0.55;
  g.add(panel);
  g.position.copy(p);
  return g;
}

function makeObj(o: Obj): THREE.Group {
  const g = new THREE.Group();
  const std = (color: number, roughness = 0.6) => new THREE.MeshStandardMaterial({ color, roughness });
  switch (o.kind) {
    case "berry": {
      const b = new THREE.Mesh(new THREE.SphereGeometry(0.16, 16, 12), std(0xe0304a, 0.35));
      b.position.y = 0.16;
      const leaf = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.12, 6), std(0x3f9a3a));
      leaf.position.y = 0.34;
      g.add(b, leaf);
      break;
    }
    case "mushroom": {
      const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 0.22, 10), std(0xf3ead8));
      stem.position.y = 0.11;
      const cap = new THREE.Mesh(new THREE.SphereGeometry(0.2, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2), std(0xd9542b, 0.5));
      cap.position.y = 0.2;
      g.add(stem, cap);
      break;
    }
    case "stick": {
      const s = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.045, 0.9, 8), std(0x8a5a2b, 0.9));
      s.rotation.z = Math.PI / 2;
      s.rotation.y = 0.5;
      s.position.y = 0.05;
      g.add(s);
      break;
    }
    case "rock": {
      const r = new THREE.Mesh(new THREE.DodecahedronGeometry(0.4, 0), std(0x8d8d96, 0.95));
      r.position.y = 0.36;
      g.add(r);
      break;
    }
    case "tree": {
      const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.18, 1.2, 8), std(0x7a5230, 0.9));
      trunk.position.y = 0.6;
      const crown = new THREE.Mesh(new THREE.IcosahedronGeometry(0.75, 0), std(0x3f9f4a, 0.8));
      crown.position.y = 1.6;
      g.add(trunk, crown);
      const spots = [[0.5, 1.4, 0.35], [-0.45, 1.55, 0.4], [0.1, 1.3, -0.6]] as const;
      spots.forEach(([x, y, z], n) => {
        const f = new THREE.Mesh(new THREE.SphereGeometry(0.12, 12, 10), std(0xe0304a, 0.35));
        f.position.set(x, y, z);
        f.name = "fruit";
        f.userData["n"] = n;
        g.add(f);
      });
      break;
    }
    case "tower": {
      const stone = std(0x9c95a8, 0.95);
      const base = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.48, 2.2, 12), stone);
      base.position.y = 1.1;
      const roof = new THREE.Mesh(new THREE.ConeGeometry(0.58, 1.1, 12), std(0x5b3fa6, 0.5));
      roof.position.y = 2.75;
      const star = new THREE.Mesh(new THREE.OctahedronGeometry(0.1), new THREE.MeshBasicMaterial({ color: 0xffe27a }));
      star.position.y = 3.4;
      // A lit window facing the camera's usual side, where the wrinkly hand comes out.
      const glass = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.3, 0.08), new THREE.MeshBasicMaterial({ color: 0xffd36b }));
      glass.position.set(0, 1.5, 0.44);
      const door = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.5, 0.08), std(0x6b4526, 0.9));
      door.position.set(0, 0.25, 0.47);
      g.add(base, roof, star, glass, door);
      break;
    }
    case "gate":
      break;
  }
  return g;
}
