"""Fast checks for the trainer's pure logic (no model is loaded).   python -m unittest discover tests"""
import json
import tempfile
import unittest
from pathlib import Path

from blobb import dream


def sample(prefix, tail_prompt, target):
    """(ids, labels, prompt_len) the way train() encodes a sample."""
    p = prefix + tail_prompt
    return p + target, [-100] * len(p) + target, len(p)


class Pack(unittest.TestCase):
    def setUp(self):
        self.sys = [1, 2, 3, 4, 5]  # stands in for the shared system prompt
        self.batch = [sample(self.sys, [10, 11], [90, 91]), sample(self.sys, [20], [92]), sample(self.sys, [30, 31, 32], [93, 94, 95])]

    def test_shared_prefix_appears_once(self):
        ids, labels, pos, seg = dream.pack(self.batch)
        self.assertEqual(ids[:5], self.sys)
        self.assertEqual(len(ids), 5 + (2 + 2) + (1 + 1) + (3 + 3))
        self.assertEqual(seg, [-1] * 5 + [0] * 4 + [1] * 2 + [2] * 6)

    def test_each_tail_is_positioned_as_if_alone(self):
        ids, labels, pos, seg = dream.pack(self.batch)
        for i, (x, y, _) in enumerate(self.batch):
            tail = [k for k, s in enumerate(seg) if s == i]
            prefix = [k for k, s in enumerate(seg) if s == -1]
            self.assertEqual([ids[k] for k in prefix + tail], x)
            self.assertEqual([labels[k] for k in prefix + tail], y)
            self.assertEqual([pos[k] for k in prefix + tail], list(range(len(x))))

    def test_no_target_is_predicted_across_a_seam(self):
        ids, labels, pos, seg = dream.pack(self.batch)
        for k in range(1, len(ids)):
            if labels[k] != -100:
                self.assertIn(seg[k - 1], (seg[k], -1) if seg[k] == -1 else (seg[k],))

    def test_identical_prompts_still_keep_a_prompt_token_per_tail(self):
        same = [sample(self.sys, [7], [80]), sample(self.sys, [7], [81])]
        ids, labels, pos, seg = dream.pack(same)
        self.assertEqual(seg.count(-1), 5)  # lcp capped at prompt_len - 1
        self.assertEqual(labels, [-100] * 5 + [-100, 80, -100, 81])

    def test_different_system_prompts_share_less(self):
        a, b = sample([1, 2, 3], [4], [9]), sample([1, 7, 3], [4], [9])
        ids, labels, pos, seg = dream.pack([a, b])
        self.assertEqual(seg.count(-1), 1)

    def test_single_sample(self):
        x = sample(self.sys, [10], [90, 91])
        ids, labels, pos, seg = dream.pack([x])
        self.assertEqual(ids, x[0])
        self.assertEqual(labels, x[1])


class Checkpointing(unittest.TestCase):
    def test_small_steps_need_none(self):
        self.assertEqual(dream.layers_to_checkpoint(200, 24, 6.0), 0)

    def test_more_tokens_or_less_ram_need_more(self):
        a = dream.layers_to_checkpoint(1000, 24, 6.0)
        self.assertGreater(a, 0)
        self.assertGreaterEqual(dream.layers_to_checkpoint(1400, 24, 6.0), a)
        self.assertGreaterEqual(dream.layers_to_checkpoint(1000, 24, 5.0), a)

    def test_never_more_than_all_layers(self):
        self.assertEqual(dream.layers_to_checkpoint(100_000, 24, 4.5), 24)

    def test_fits_the_budget_when_possible(self):
        for tokens, scored in ((600, 100), (900, 200), (1200, 250)):
            m = dream.layers_to_checkpoint(tokens, 24, 6.0, scored)
            self.assertLessEqual(dream.step_gb(tokens, scored, m, 24), 6.0 + 1e-9)

    def test_scored_tokens_cost_memory_too(self):
        self.assertGreater(dream.layers_to_checkpoint(900, 24, 6.0, 400), dream.layers_to_checkpoint(900, 24, 6.0, 0))


class MicroBatches(unittest.TestCase):
    """A batch that can't fit even fully checkpointed is split, and gradients accumulate across parts."""
    def long(self, name_token, tail=600):
        return sample([name_token] + list(range(100, 390)), list(range(1000, 1000 + tail)), [7] * 60)

    def test_normal_batches_stay_whole(self):
        batch = [sample(list(range(100, 390)), [i, i + 1, i + 2] * 50, [9] * 50) for i in range(4)]
        self.assertEqual(dream.micro_batches(batch, 24, 6.0), [batch])

    def test_samples_sharing_little_prompt_get_split_and_each_part_fits(self):
        # Different first token = a different blob name: almost nothing shared, so ~4 x 950 tokens.
        batch = [self.long(1), self.long(2), self.long(3), self.long(4)]
        parts = dream.micro_batches(batch, 24, 4.6)
        self.assertGreater(len(parts), 1)
        self.assertEqual([x for p in parts for x in p], batch)  # nothing lost or reordered
        for p in parts:
            ids, labels, _, _ = dream.pack(p)
            self.assertLessEqual(dream.step_gb(len(ids), dream.scored(labels), 24, 24), 4.6)

    def test_scored_counts_targets(self):
        self.assertEqual(dream.scored([-100, -100, 5, 6, -100]), 2)


class Cpus(unittest.TestCase):
    def test_cpu_list(self):
        self.assertEqual(dream._cpu_list("0-3,8,10-11\n"), [0, 1, 2, 3, 8, 10, 11])
        self.assertEqual(dream._cpu_list(""), [])

    def test_performance_cores_is_sane(self):
        self.assertGreaterEqual(dream.performance_cores(), 1)


class Samples(unittest.TestCase):
    def test_about(self):
        self.assertEqual(dream.about({"user": 'x\nThe wizard says: "fetch my stick"'}), "fetch my stick")
        self.assertEqual(dream.about({"user": "The wizard says nothing."}), "(looking after itself)")

    def test_bad_lines_and_files_are_skipped_or_explained(self):
        with tempfile.TemporaryDirectory() as d:
            cur = Path(d, "c.jsonl")
            cur.write_text('{"system":"s","user":"u","assistant":"a"}\nnope\n{"system":"s"}\n\n')
            mem = Path(d, "m.json")
            mem.write_text(json.dumps({"samples": [{"system": "s", "user": "u", "assistant": "a", "reward": 2.4, "owner": True}, {"user": 1}, 5]}))
            self.assertEqual(len(dream.load_samples(cur, [mem], 600)), 1 + 3)
            bad = Path(d, "bad.json")
            bad.write_text("{nope")
            with self.assertRaises(SystemExit):
                dream.load_samples(cur, [bad], 600)
            # The game exports failures and 👎s too (for its own brain); the LLM only learns the good ones.
            mem.write_text(json.dumps({"samples": [{"system": "s", "user": "u", "assistant": "a", "reward": -1, "owner": True},
                                                   {"system": "s", "user": "u", "assistant": "a", "reward": 0.3}]}))
            self.assertEqual(len(dream.load_samples(None, [mem], 600)), 1)
            Path(d, "list.json").write_text("[1, 2]")
            self.assertEqual(len(dream.load_samples(None, [Path(d, "list.json")], 600)), 0)


if __name__ == "__main__":
    unittest.main()
