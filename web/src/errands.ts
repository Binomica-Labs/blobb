// Errands: letters from villagers the wizard would rather not talk to. Each is checked against the
// world (usually something handed in at the tower), so the wizard has to teach the blob how - there's
// no button that does it. The first run walks through every puzzle on the island once; after that the
// villagers keep writing.

import type { Store } from "./memory";
import { TUTORIAL_MAP, parseMap, tileAt, type Delivery, type World } from "./world";

export interface Errand {
  id: string;
  from: string;
  /** What the villager wrote. */
  letter: string;
  /** What the wizard mutters on reading it. */
  grumble: string;
  /** What he mutters when it's done. */
  thanks: string;
  /** A command the player might start with - shown as a suggestion chip. */
  hint: string;
  /** Done yet? Consumes the delivery that satisfied it. */
  check: (w: World) => boolean;
}

/** Take the first delivery that matches, if any. */
function take(w: World, match: (d: Delivery) => boolean): boolean {
  const i = w.deliveries.findIndex(match);
  if (i < 0) return false;
  w.deliveries.splice(i, 1);
  return true;
}

const inBox = (d: Delivery, x0: number, y0: number, x1: number, y1: number) =>
  d.home !== null && d.home[0] >= x0 && d.home[0] <= x1 && d.home[1] >= y0 && d.home[1] <= y1;
const GAPS = parseMap(TUTORIAL_MAP).tiles.flatMap((t, i) => (t.kind === "gap" ? [i] : []));

const STORY: Errand[] = [
  {
    id: "stick", from: "yourself",
    letter: "(A note pinned inside your spellbook, in your own handwriting:) FIND WALKING STICK. IT IS OUTSIDE. YOU ARE NOT GOING OUTSIDE.",
    grumble: "Blob. My stick is out there somewhere. Pick it up and bring it to the tower.",
    thanks: "My stick! ...Don't expect a thank-you. That was it.",
    hint: "fetch my walking stick",
    check: (w) => take(w, (d) => d.kind === "stick"),
  },
  {
    id: "soup", from: "Mrs. Pennywhistle",
    letter: "Dear Wizard, my soup lacks a mushroom and my knees lack the strength to pick one. Would you be a dear? Warmly, Mrs. P.",
    grumble: "If she gets a mushroom she stops writing. Probably. Get a mushroom, blob.",
    thanks: "Mushroom sent. Pigeon grumbled about the weight. Everyone grumbles.",
    hint: "bring me a mushroom",
    check: (w) => take(w, (d) => d.kind === "mushroom"),
  },
  {
    id: "meadow", from: "Bram the Baker",
    letter: "To the Wizard: the only berries worth baking grow on the high meadow and I cannot climb. Two loaves for one berry? - Bram",
    grumble: "Bread. Fine. The meadow is up high, blob. Work it out.",
    thanks: "A meadow berry. Bram will send bread. Bread means Bram will visit. Hm.",
    hint: "bring me a berry from up high",
    check: (w) => take(w, (d) => d.kind === "berry" && d.home !== null && tileAt(w, d.home[0], d.home[1]).h >= 2),
  },
  {
    id: "garden", from: "Hazel the herbalist",
    letter: "Wizard, only the mushroom in the walled garden is fit for my tonic. Mind the gate, it's very low. - H.",
    grumble: "The gate is low. The blob is squishy. I shouldn't have to explain this.",
    thanks: "Garden mushroom delivered. Hazel says her tonic 'fixes grumpiness'. The nerve.",
    hint: "bring me the mushroom from the walled garden",
    check: (w) => take(w, (d) => d.kind === "mushroom" && inBox(d, 10, 1, 13, 4)),
  },
  {
    id: "tree", from: "the village children",
    letter: "deer wizzard can we hav a frute from the big tree. we cant reech. from the kids (and Tobble)",
    grumble: "Children. And a Tobble, whatever that is. Knock one out of the tree and make them go away.",
    thanks: "Fruit sent. They've drawn me a picture. It's... a blob. It's quite good, actually.",
    hint: "knock a fruit from the tree and bring it",
    check: (w) => take(w, (d) => d.kind === "berry" && d.home === null),
  },
  {
    id: "bridge", from: "Olm the ferryman",
    letter: "Wizard. The path east has a hole in it. Fix it. - Olm",
    grumble: "Rude. Correct, but rude. There's a rock near the hole, blob.",
    thanks: "Bridged. Olm replied 'k'. I've been more thanked by the pigeon.",
    hint: "push the rock into the gap",
    check: (w) => GAPS.some((i) => w.tiles[i]?.kind === "ground"),
  },
  {
    id: "festival", from: "Mayor Quillby",
    letter: "Esteemed Wizard, the council humbly requests the rare mushroom from beyond the chasm, for the Harvest Festival. You are of course invited!",
    grumble: "A festival. Noise. People. I'd sooner eat the mushroom myself. ...Fetch it, blob. I am NOT going.",
    thanks: "Delivered, with a note saying I have a cold. I do not have a cold.",
    hint: "bring me the mushroom from across the gap",
    check: (w) => take(w, (d) => d.kind === "mushroom" && inBox(d, 12, 9, 14, 14)),
  },
];

/** Once the story is done, villagers keep asking for the ones that can happen again. */
const AGAIN = STORY.filter((e) => e.id !== "stick" && e.id !== "bridge").map((e): Errand => ({
  ...e, letter: `(Another one.) ${e.letter}`,
}));

export class Errands {
  /** How many errands have been finished, ever. */
  finished = 0;

  constructor(private store: Store) {
    try {
      const n: unknown = JSON.parse(store.load() ?? "0");
      if (typeof n === "number" && Number.isInteger(n) && n >= 0) this.finished = n;
    } catch { /* corrupt: start from the first letter */ }
  }

  active(): Errand {
    const n = this.finished;
    const e = n < STORY.length ? STORY[n] : AGAIN[(n - STORY.length) % AGAIN.length];
    if (!e) throw new Error("no errands defined");
    return e;
  }

  get storyDone(): boolean {
    return this.finished >= STORY.length;
  }

  /** If the current errand is now done, move on and return it. */
  check(w: World): Errand | null {
    const e = this.active();
    if (!e.check(w)) return null;
    this.finished++;
    this.store.save(JSON.stringify(this.finished));
    return e;
  }
}

export const STORY_LENGTH = STORY.length;
