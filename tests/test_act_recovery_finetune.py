import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_act_recovery_finetune import sample_plans


class RecoverySamplingTest(unittest.TestCase):
    def test_exact_fraction_balancing_and_source_identity(self):
        original = [(ep, frame) for ep in range(4) for frame in range(20)]
        # IDs deliberately overlap between datasets. Prefix frames 0..6 are forbidden.
        expert = [(ep, frame) for ep in range(3) for frame in range(7, 12 + ep)]
        plans = sample_plans(original, expert)
        self.assertEqual(len(plans['control']), 6400)
        selected = [r for r in plans['recovery'] if r['source'] == 'recovery']
        self.assertEqual(len(selected), 640)
        self.assertTrue(all((r['episode_index'], r['frame_index']) in expert for r in selected))
        self.assertEqual(sum(a == b for a, b in zip(plans['control'], plans['recovery'])), 5760)
        counts = [sum(r['episode_index'] == ep for r in selected) for ep in range(3)]
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertTrue(all(r['source'] == 'original' for r in plans['control']))
        self.assertEqual(plans, sample_plans(original, expert))

    def test_rejects_duplicate_source_observations(self):
        with self.assertRaises(ValueError):
            sample_plans([(0, 0), (0, 0)], [(1, 3)])


if __name__ == '__main__':
    unittest.main()
