"""Saved traces must independently substantiate the reported score."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'scripts'))
from audit_smolvla_success import audit_episode, TRUCK_COUNT_METHOD
from success_eval_runtime import atomic_json


class TraceAuditTests(unittest.TestCase):
    def audit(self, mutation=None):
        status = dict(scene_seed=2, min_particles=100, policy_steps_executed=3,
                      success=True, task_completed_success_and_bucket_open=True,
                      simulated_rollout_time_seconds=.3)
        rows = [dict(step=i, deposited_particles=100, particle_transfer_metrics={'payload_particles': 100},
                     truck_bed_load_metrics=dict(method=TRUCK_COUNT_METHOD, scene_seed=2,
                         available=True, source_tracking_valid=True, count=100, local_box_count=100, candidate_count=100),
                     state=[0.] * 28, joint_command_positions=[0.] * 4,
                     camera_sha256={str(k): 'a' * 64 for k in range(3)},
                     bridge_timing=dict(simulated_seconds=.1, physics_ticks=6,
                                        physics_time_advanced_seconds=.1, camera_render_updates=5),
                     bucket_position=0., completion_observations=i + 1) for i in range(3)]
        if mutation:
            mutation(rows, status)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            atomic_json(folder / 'scene_context.json', dict(scene_seed=2,
                truck_counting_volume={'scene_seed': 2}, canonical_joint_limits_rad=[[-1, 1]] * 4))
            atomic_json(folder / 'initial_observation.json', {'truck_bed_load_metrics': {'scene_seed': 2}})
            with (folder / 'steps.jsonl').open('w') as stream:
                for row in rows:
                    stream.write(json.dumps(row) + '\n')
            return audit_episode(folder, status)

    def test_complete_trace_substantiates_success(self):
        self.assertTrue(self.audit()['success'])

    def test_payload_drop_cannot_claim_an_unbroken_completion_window(self):
        def mutate(rows, status):
            rows[1]['deposited_particles'] = rows[1]['particle_transfer_metrics']['payload_particles'] = 99
        with self.assertRaisesRegex(ValueError, 'completion observation count'):
            self.audit(mutate)

    def test_stale_scene_geometry_and_unbounded_commands_are_detected(self):
        with self.assertRaisesRegex(ValueError, 'stale scene'):
            self.audit(lambda rows, status: rows[1]['truck_bed_load_metrics'].update(scene_seed=3))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.audit(lambda rows, status: rows[1]['joint_command_positions'].__setitem__(0, 2.))


if __name__ == '__main__':
    unittest.main()
