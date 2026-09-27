"""The blob's own brain: tokens and hashes the browser must match, memory loading, and that a dream really
learns - repeats what was praised, stops doing what failed.   python -m unittest discover tests"""
import json
import random
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from blobb import scratch
except ImportError:  # the LLM trainer's tests don't need torch; these do
    torch = None


def obs(said: str) -> str:
    return f'You are a blob. Food 50 (hungry).\nYou see:\n- berry: 2 east\n- stick: 1 north\nThe wizard says: "{said}"'


def steps(*pairs):
    return [{"do": d, "arg": a} for d, a in pairs]


@unittest.skipUnless(torch, "needs torch")
class Tokens(unittest.TestCase):
    def test_same_tokens_and_hashes_as_the_browser(self):
        # web/test/scratch.test.ts checks the same values on the TypeScript side.
        self.assertEqual(scratch.tokenize('Food 30 (hungry). The wizard says: "Get the BERRY, blob!"'),
                         ["food", "30", "hungry", "the", "wizard", "says", '"', "get", "the", "berry", "blob", "!", '"'])
        self.assertEqual(scratch.ngrams("42"), [])
        self.assertEqual(len(scratch.ngrams("berry")), 12)
        self.assertEqual(scratch.ngrams("berry")[0], 1 + 0xb8235914 % (scratch.BUCKETS - 1))  # zlib.crc32(b"<be")

    def test_plans_round_trip_and_skills_are_refused(self):
        p = steps(("morph", "spring"), ("eat", "berry"))
        self.assertEqual(scratch.plan_steps(scratch.plan_tokens(p)), p)
        self.assertIsNone(scratch.plan_tokens(steps(("skill", "ledge hop"))))
        self.assertIsNone(scratch.plan_tokens([]))
        # Skills expand to up to 8 steps; cutting one to 4 would teach it half a plan.
        self.assertIsNone(scratch.plan_tokens(steps(*[("rest", "none")] * 5)))
        self.assertIsNone(scratch.plan_tokens([{"do": "eat"}, "junk"]))


@unittest.skipUnless(torch, "needs torch")
class Memories(unittest.TestCase):
    def test_praise_counts_more_and_failures_become_things_to_avoid(self):
        a = json.dumps({"thought": "t", "plan": steps(("eat", "berry")), "say": "s"})
        with tempfile.TemporaryDirectory() as d:
            f = Path(d, "m.json")
            f.write_text(json.dumps({"samples": [
                {"user": "u", "assistant": a, "reward": 2.4, "owner": True},   # praised
                {"user": "u", "assistant": a, "reward": 0.3},                  # just worked
                {"user": "u", "assistant": a, "reward": -1, "owner": True},    # 👎
                {"user": "u", "assistant": a, "reward": -0.3, "failedStep": 0},  # failed at its first step
                {"user": "u", "assistant": a, "reward": 0},                    # nothing to learn
                {"user": 1}, 5,
            ]}))
            got = [(m["weight"], m["negative"]) for m in scratch.load_memories([f])]
            self.assertEqual(got, [(3, False), (1, False), (1.0, True), (0.5, True)])
            # 👎: every choice in the plan (not the END); a failure: just the step that failed.
            self.assertEqual([m["avoid"] for m in scratch.load_memories([f])][2:], [(0, 2), (0, 2)])

            Path(d, "bad.json").write_text("{nope")
            with self.assertRaises(SystemExit):
                scratch.load_memories([Path(d, "bad.json")])

    def test_the_failed_step_is_what_gets_avoided(self):
        a = json.dumps({"thought": "", "plan": steps(("morph", "ball"), ("eat", "berry"), ("deliver", "none")), "say": ""})
        self.assertEqual(scratch.sample("u", a, negative=True, failed_step=1)["avoid"], (2, 4))
        self.assertEqual(scratch.sample("u", a, negative=True)["avoid"], (0, 6))
        self.assertEqual(scratch.sample("u", a, negative=True, failed_step=7)["avoid"], (0, 6))  # nonsense: whole plan


@unittest.skipUnless(torch, "needs torch")
class Life(unittest.TestCase):
    def mem(self, id, said="get the berry"):
        return scratch.sample(obs(said), json.dumps({"thought": "", "plan": steps(("eat", "berry")), "say": ""}), negative=True, id=id)

    def test_every_dream_keeps_what_earlier_dreams_learned(self):
        with tempfile.TemporaryDirectory() as d:
            brain = Path(d, "gen1")
            brain.mkdir()
            old = [self.mem("a"), self.mem("b")]
            (brain / scratch.MEMORIES).write_text("".join(json.dumps(m) + "\n" for m in old) + "junk\n{}\n")
            back = scratch.life(brain)
            self.assertEqual([m["id"] for m in back], ["a", "b"])
            self.assertEqual(back[0]["avoid"], (0, 2))  # JSON made it a list; it's a range again
            self.assertEqual(scratch.life(Path(d, "nowhere")), [])
            # A re-exported memory counts once (the newer copy), in the order it was lived.
            merged = scratch.merge_life(back, [self.mem("b"), self.mem("c"), self.mem(None), self.mem(None)])
            self.assertEqual([m["id"] for m in merged], ["a", "b", "c", None, None])

    def test_lessons_teaching_what_the_wizard_hates_stop_being_replayed(self):
        def x(plan, negative=False, owner=False):
            return scratch.sample(obs("get the berry"), json.dumps({"thought": "", "plan": plan, "say": ""}), negative=negative, owner=owner)
        spring, stick = steps(("morph", "spring"), ("eat", "berry")), steps(("grab", "stick"), ("use", "berry"), ("eat", "berry"))
        replay = [x(spring), x(stick), x(steps(("rest", "none")))]
        self.assertEqual(len(scratch.to_taste(replay, [x(spring, negative=True, owner=True)])), 2)
        self.assertEqual(len(scratch.to_taste(replay, [x(spring, negative=True)])), 3)  # the world failing it isn't taste
        both = [x(spring, negative=True, owner=True), x(spring, owner=True)]  # he's of two minds: keep it
        self.assertEqual(len(scratch.to_taste(replay, both)), 3)

    def test_the_oldest_memories_fade_first(self):
        many = [self.mem(str(i)) for i in range(scratch.MAX_LIFE + 5)]
        kept = scratch.merge_life(many[:10], many[10:])
        self.assertEqual(len(kept), scratch.MAX_LIFE)
        self.assertEqual(kept[0]["id"], "5")


@unittest.skipUnless(torch, "needs torch")
class Learning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)
        texts = [obs("get the berry"), obs("grab the stick"), obs("rest")]
        words = ["<pad>", "<unk>"] + sorted({w for t in texts for w in scratch.tokenize(t)})
        allowed = {scratch.OUT_ID[f"v:{v}"]: [scratch.OUT_ID[f"a:{a}"] for a in scratch.ARGS] for v in scratch.VERBS}
        cfg = {"d": 32, "layers": 1, "heads": 2}
        self.model = scratch.Model(scratch.Brain(len(words), 2, 2, **cfg), words, ["hm", "ok"], ["boing", "yay"], allowed, cfg)

    def item(self, said, plan, negative=False, failed_step=None):
        return scratch.sample(obs(said), json.dumps({"thought": "ok", "plan": plan, "say": "yay"}), negative=negative, failed_step=failed_step)

    def test_learns_what_it_is_shown(self):
        spring = steps(("morph", "spring"), ("eat", "berry"))
        stick = steps(("grab", "stick"))
        scratch.fit(self.model, [self.item("get the berry", spring), self.item("grab the stick", stick)] * 16, 30, 3e-3, 8, 0.0)
        self.assertEqual(self.model.decide(obs("get the berry"))["plan"], spring)
        self.assertEqual(self.model.decide(obs("grab the stick"))["plan"], stick)
        self.assertEqual(self.model.decide(obs("get the berry"))["say"], "yay")

    def test_a_bad_memory_makes_it_stop_and_try_something_else(self):
        spring = steps(("morph", "spring"), ("eat", "berry"))
        stick = steps(("grab", "stick"), ("use", "berry"), ("eat", "berry"))
        rest = steps(("rest", "none"))
        # Taught mostly spring, some stick: it goes with spring...
        taught = [self.item("get the berry", spring)] * 12 + [self.item("get the berry", stick)] * 8 + [self.item("rest", rest)] * 8
        scratch.fit(self.model, taught, 30, 3e-3, 8, 0.0)
        self.assertEqual(self.model.decide(obs("get the berry"))["plan"], spring)
        # ...until it dreams of spring failing at its first step, and switches to the other thing it knows.
        # Gently and among other lessons, as a real dream does: pushing hard on failures alone makes a
        # small brain forget which step comes first (2 in 6 seeds pass that way, 5 in 6 this way).
        dream = [self.item("get the berry", spring, negative=True, failed_step=0)] * 16 + [self.item("rest", rest)] * 8
        scratch.fit(self.model, dream, 5, 1e-3, 8, 0.0)
        self.assertEqual(self.model.decide(obs("get the berry"))["plan"], stick)
        self.assertEqual(self.model.decide(obs("rest"))["plan"], rest)

    def test_a_thumbs_down_puts_it_off_the_whole_plan(self):
        spring = steps(("morph", "spring"), ("eat", "berry"))
        stick = steps(("grab", "stick"), ("use", "berry"), ("eat", "berry"))
        rest = steps(("rest", "none"))
        taught = [self.item("get the berry", spring)] * 12 + [self.item("get the berry", stick)] * 4 + [self.item("rest", rest)] * 8
        scratch.fit(self.model, taught, 20, 3e-3, 8, 0.0)
        self.assertEqual(self.model.decide(obs("get the berry"))["plan"], spring)
        # A dream mixes the 👎 with other lessons it already knows, as the real one does.
        scratch.fit(self.model, [self.item("get the berry", spring, negative=True)] * 16 + [self.item("rest", rest)] * 8, 5, 1e-3, 8, 0.0)
        self.assertNotEqual(self.model.decide(obs("get the berry"))["plan"][:1], spring[:1])
        self.assertEqual(self.model.decide(obs("rest"))["plan"], rest)

    def test_never_starts_a_step_it_knows_no_argument_for(self):
        # move_to is a verb, but no lesson uses it: with nothing it could go to, it's never chosen.
        self.model.pairs = {k: v for k, v in self.model.pairs.items() if k != scratch.OUT_ID["v:move_to"]}
        with torch.no_grad():
            self.model.net.plan_head.bias[scratch.OUT_ID["v:move_to"]] = 100.0  # as tempting as can be
        for said in ("get the berry", "rest", ""):
            self.assertNotIn("move_to", [s["do"] for s in self.model.decide(obs(said))["plan"]])

    def test_saved_brain_answers_the_same(self):
        spring = steps(("morph", "spring"), ("eat", "berry"))
        scratch.fit(self.model, [self.item("get the berry", spring)] * 16, 20, 3e-3, 8, 0.0)
        with tempfile.TemporaryDirectory() as d:
            self.model.save(Path(d, "gen1"))
            again = scratch.Model.open(Path(d, "gen1"))
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["gen1"])  # no .partial/.old left behind
        self.assertEqual(again.decide(obs("get the berry")), self.model.decide(obs("get the berry")))


if __name__ == "__main__":
    unittest.main()
