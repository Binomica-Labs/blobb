// Ways the wizard might say each thing. A brain trained from scratch only understands wordings it has
// seen (or ones spelled like them), so the curriculum draws from these plus the families' own `says`,
// then roughs them up the way a grumpy old man types (see `wizardly`).
// The eval's `--novel` exam uses wordings that are deliberately NOT in here.

import type { Shape } from "../src/world";

/** Other words for each shape, so "go flat" means puddle and "get bouncy" means spring. */
export const SHAPE_WORDS: Record<Shape, readonly string[]> = {
  blob: ["blob", "normal", "your usual self", "regular blob", "your normal shape", "normal blob"],
  ball: ["ball", "round ball", "hard ball", "sphere", "bouncy ball", "rolling ball"],
  puddle: ["puddle", "flat puddle", "flat", "splat", "goo puddle", "flat as a pancake"],
  spring: ["spring", "bouncy spring", "coil", "springy", "jumpy spring", "boing shape"],
};

const SHAPE_TEMPLATES = [
  "turn into a {s}", "become a {s}", "be a {s}", "change into a {s}", "morph into a {s}", "shift into a {s}",
  "go {s}", "{s} form", "{s} shape now", "make yourself a {s}", "i want you as a {s}", "switch to {s}",
  "transform into a {s}", "can you turn {s}?", "time to be a {s}", "be {s}", "shape: {s}", "{s}!",
];

export function shapePhrasings(shape: Shape): string[] {
  return SHAPE_TEMPLATES.flatMap((t) => SHAPE_WORDS[shape].map((s) => t.replace("{s}", s)));
}

export const PARAPHRASES: Record<string, readonly string[]> = {
  eat: [
    "eat", "eat something", "go get food", "you're starving, eat", "feed yourself", "find yourself a snack",
    "eat anything", "go have lunch", "get some food in you", "you need food", "fill your belly", "snack time",
    "go and eat", "eat up", "hungry? eat", "have something to eat", "don't starve on me, eat", "munch something",
    "go forage", "find food and eat it", "breakfast", "supper", "eat whatever's close", "grab some grub",
  ],
  "eat-berry": [
    "eat a berry", "eat the berry", "have a berry", "go eat a berry", "berry snack", "munch a berry",
    "find a berry and eat it", "eat berries", "go get a berry to eat", "you should eat a berry", "try a berry",
  ],
  "eat-mushroom": [
    "eat the mushroom", "eat a mushroom", "have a mushroom", "go eat a mushroom", "mushroom snack",
    "munch a mushroom", "eat mushrooms", "find a mushroom and eat it", "try a mushroom", "eat a shroom",
  ],
  "grab-stick": [
    "grab the stick", "pick up the stick", "take the stick", "get the stick", "hold the stick", "fetch the stick",
    "pick that stick up", "go take the stick", "the stick, pick it up", "carry the stick", "get hold of the stick",
    "lift the stick", "take hold of that stick", "stick. grab it.", "you'll want the stick, get it",
  ],
  drop: [
    "drop it", "put it down", "let go", "drop that", "put that down", "let it go", "release it", "drop what you're holding",
    "stop carrying that", "unhand it", "set it down", "leave it", "drop the thing", "empty your hands", "let go of it",
  ],
  rest: [
    "rest", "take a nap", "sleep", "have a rest", "lie down", "take a break", "get some sleep", "go to sleep",
    "recover your energy", "you're exhausted, rest", "catch your breath", "put your feet up", "doze off",
    "rest up", "have a lie down", "recharge", "sit still for a bit", "go rest", "take five", "nap",
  ],
  "high-berry": [
    "get the high berry", "eat the berry up on the ledge", "reach the berry up high", "the berry up there, eat it",
    "climb to the berry on the ledge", "that berry is up high, go eat it", "jump up and eat the berry",
    "get up to the high berry", "eat the berry on top of the ledge", "the berry on the cliff, eat it",
    "berry up top, go get it", "get the berry that's up high", "go up the ledge and eat the berry",
    "eat the berry up on the cliff", "reach up and eat that berry", "the high one, eat the berry",
  ],
  garden: [
    "get in the garden", "go in the walled garden", "slip under the gate", "go under the low gate",
    "get past the gate into the garden", "the garden, get inside", "sneak into the garden", "enter the garden",
    "go through under the gate", "get into the walled garden", "the mushroom in the garden, eat it",
    "eat the garden mushroom", "squeeze into the garden", "get behind the gate", "crawl under the gate",
    "go eat the mushroom in the garden", "slide under the gate into the garden", "get the mushroom behind the gate and eat it",
  ],
  gap: [
    "cross the gap", "get over the gap", "jump over the gap", "get across the gap", "go over the gap",
    "get the mushroom past the gap", "eat the mushroom across the gap", "the mushroom across the gap, eat it",
    "get to the far side of the gap", "go across", "get over to the other side", "bridge the gap",
    "get past the gap", "the mushroom on the far side, eat it", "leap the gap", "get to the other side and eat",
    "eat the mushroom on the other side", "over the gap and eat the mushroom",
  ],
  tree: [
    "knock fruit off the tree", "get the fruit out of the tree", "get a berry down from the tree",
    "make the tree drop a berry", "bash the tree", "get fruit down", "the tree has berries, knock one down",
    "whack the tree", "hit the tree", "get the berries off the tree", "knock the tree", "shake some fruit loose",
    "get fruit from that tree", "make a berry fall from the tree", "knock the fruit down", "ram the tree",
  ],
  "fetch-stick": [
    "bring me my stick", "fetch my stick", "get my walking stick", "my walking stick, bring it", "bring the stick to me",
    "bring the stick to the tower", "deliver my stick", "i need my walking stick", "stick. tower. now.",
    "fetch the walking stick", "bring me the stick", "where's my stick? get it here", "go get my stick for me",
    "bring back my stick", "my cane, fetch it", "get the stick and bring it to me", "deliver the stick to the tower",
    "fetch me that stick", "i want my stick back", "i dropped my stick, bring it",
  ],
  "fetch-food": [
    "bring me some food", "fetch me food", "bring food to the tower", "bring me something to eat", "deliver some food",
    "a villager is hungry, bring food", "fetch something edible", "get food and bring it here",
    "bring me a snack", "i'm hungry, bring me something", "bring some food to the tower", "fetch a snack for the tower",
  ],
  "fetch-mushroom": [
    "bring me a mushroom", "fetch me a mushroom", "get me a mushroom", "deliver a mushroom", "a mushroom for the tower",
    "bring a mushroom to the tower", "the cook wants a mushroom", "a villager needs a mushroom",
    "mushroom. tower. go.", "go get a mushroom and bring it back", "i need a mushroom", "fetch a shroom",
    "bring back a mushroom", "get a mushroom for the soup", "someone wants a mushroom, fetch it",
    "don't eat it, bring me a mushroom", "carry a mushroom here", "fetch a mushroom for mrs pennywhistle",
  ],
  "fetch-berry": [
    "bring me a berry", "fetch me a berry", "get me a berry", "deliver a berry", "a berry for the tower",
    "bring a berry to the tower", "the baker wants a berry", "a villager needs a berry",
    "berry. tower. go.", "go get a berry and bring it back", "i need a berry", "bring back a berry",
    "get a berry for the pie", "someone wants a berry, fetch it", "don't eat it, bring me a berry",
    "carry a berry here", "fetch a berry for the jam",
  ],
  "fetch-meadow-berry": [
    "bring me a berry from the ledge", "fetch the high berry", "get the berry up high and bring it",
    "the berry on the ledge, deliver it", "bring me the berry from up top", "a high meadow berry for the baker",
    "fetch a berry from the cliff", "deliver the high berry", "get me the berry up on the ledge",
    "go up, grab a berry, bring it to me", "the meadow berry, fetch it", "bring the ledge berry to the tower",
  ],
  deliver: [
    "deliver it", "take it to the tower", "bring that to me", "bring it here", "carry it to the tower", "hand it to me",
    "give it to me", "bring what you've got to the tower", "drop it off at the tower", "deliver that", "tower. now.",
    "bring it back to the tower", "i'll take that", "hand that over", "deliver what you're holding",
  ],
  bridge: [
    "push the rock in the gap", "shove the rock into the hole", "make a bridge with the rock", "fill in the gap",
    "roll the rock into the gap", "fix the hole in the path", "plug the gap with that rock", "bridge the hole",
    "push that rock into the gap", "use the rock to fill the gap", "mend the path, fill the gap", "rock. gap. push.",
  ],
  "fetch-gap-mushroom": [
    "bring me the mushroom across the gap", "fetch the mushroom beyond the gap", "the mushroom on the far side of the gap, bring it",
    "get the mushroom from past the gap and bring it here", "deliver the mushroom from across the gap",
    "fetch the festival mushroom", "bring the mushroom from the other side", "get me the mushroom over the gap",
    "the mushroom across the gap, deliver it", "cross the gap and bring back the mushroom",
  ],
  "fetch-tree-fruit": [
    "knock a fruit down and bring it", "get fruit from the tree and bring it to me", "bring me a fruit from the tree",
    "knock a berry out of the tree and deliver it", "fetch a fruit from the big tree", "get the kids a fruit from the tree",
    "knock down a fruit for the children", "bring me a berry from the tree", "shake a fruit loose and bring it here",
    "get a fruit off the tree and take it to the tower",
  ],
  "fetch-garden-mushroom": [
    "bring me the garden mushroom", "fetch the mushroom in the garden", "the mushroom behind the gate, bring it",
    "get the walled garden mushroom and bring it here", "deliver the garden mushroom", "the herbalist wants the garden mushroom",
    "go under the gate and fetch the mushroom", "bring the mushroom from the walled garden",
    "i want the mushroom from the garden", "get me the mushroom behind the gate", "fetch the garden shroom",
    "bring me the mushroom from the walled garden", "bring the garden mushroom to the tower",
  ],
};

// ---------- the wizard's manner ----------

const PREFIXES = ["", "", "", "", "blob, ", "blobb, ", "hey, ", "listen, ", "oi! ", "right. ", "ugh. ", "you there, ",
  "now then, ", "hmph. ", "go on, ", "blob! ", "look, ", "for goodness sake, ", "alright, ", "quickly, "];
const SUFFIXES = ["", "", "", "", " please", " now", ", quickly", " and be quick about it", "!", ".", " - and don't dawdle",
  ", you useless jelly", " before i turn you into soup", ", blob", " already", ". chop chop", "!!", ", would you",
  " if you can manage it", " right now"];

/** A typo in a longer word, now and then: dropped, doubled, or swapped letters. */
function typo(word: string, rnd: () => number): string {
  if (word.length < 5 || !/^[a-z]+$/.test(word)) return word;
  const i = 1 + Math.floor(rnd() * (word.length - 2));
  const r = rnd();
  if (r < 0.4) return word.slice(0, i) + word.slice(i + 1);
  if (r < 0.7) return word.slice(0, i) + (word[i] ?? "") + word.slice(i);
  return word.slice(0, i) + (word[i + 1] ?? "") + (word[i] ?? "") + word.slice(i + 2);
}

/** Rough up a phrasing: grumpy openers and closers, typos, SHOUTING, Sentence case. */
export function wizardly(text: string, rnd: () => number = Math.random): string {
  const pick = <T>(xs: readonly T[]): T => xs[Math.floor(rnd() * xs.length)] as T;
  let t = pick(PREFIXES) + text + pick(SUFFIXES);
  if (rnd() < 0.3) t = t.split(" ").map((w) => (rnd() < 0.12 ? typo(w, rnd) : w)).join(" ");
  const c = rnd();
  if (c < 0.08) t = t.toUpperCase();
  else if (c < 0.5) t = t.charAt(0).toUpperCase() + t.slice(1);
  return t;
}
