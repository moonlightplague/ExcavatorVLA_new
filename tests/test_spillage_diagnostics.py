import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from spillage_diagnostics import observe_after_rollout


class PostRolloutTests(unittest.TestCase):
    def reply(self, truck, outside):
        return dict(particle_transfer_metrics=dict(available=True, payload_particles=truck,
            retained_particles=0, spillage_particles=outside, scooped_particles=100),
            truck_bed_load_metrics=dict(scene_seed=4), observation_state_28d=[0.] * 28,
            joint_command_positions=[0.] * 4)

    def test_airborne_particles_can_enter_truck_without_rewriting_terminal_counts(self):
        original = self.reply(60, 40); before = copy.deepcopy(original); commands = []
        following = iter([self.reply(80, 20), self.reply(95, 5)])
        def exchange(command):
            commands.append(command)
            return next(following)
        with tempfile.TemporaryDirectory() as directory:
            result = observe_after_rollout(Path(directory), original, exchange, 2, 4)
            self.assertEqual(result['terminal']['payload_particles'], 60)
            self.assertEqual(result['final']['payload_particles'], 95)
            rows = [json.loads(s) for s in (Path(directory)/'post_rollout_steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['observation'] for r in rows], [0, 1, 2])
        self.assertEqual(original, before)
        self.assertEqual(commands, [{'joint_velocities': [0., 0., 0., 0.]}] * 2)

    def test_rejects_observations_from_a_different_scene(self):
        original = self.reply(60, 40)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Invalid post-rollout'):
                observe_after_rollout(Path(directory), original, lambda _: original, 1, 13)


if __name__ == '__main__':
    unittest.main()
