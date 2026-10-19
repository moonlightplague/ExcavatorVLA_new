"""Regressions for the final evaluation protocol and invalid-result handling."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'ExcavatorVLA-smolvla'),
                str(BASE / 'ExcavatorVLA-smolvla/scripts/evaluation'), str(BASE / 'scripts')]
from excavator_common.evaluation_standard import (
    EVALUATION_STANDARD, TRUCK_COUNT_METHOD, CompletionTracker, execution_profile, canonical_joint_limits,
)
from excavator_common.evaluation_metrics import ParticleTransferTracker, failure_mode
from summarize_isaacsim_success_rate import build_summary, load_episode_rows
from success_eval_runtime import atomic_json, file_manifest, content_id


class CompletionTests(unittest.TestCase):
    def test_joint_limit_compatibility_preserves_canonical_order(self):
        raw = np.array([[-3, 3], [-2, 2], [-1, 1], [-.5, .5]], dtype=np.float32)
        legacy = SimpleNamespace(get_dof_limits=lambda: raw[None])
        properties = np.zeros(4, dtype=[('lower', 'f4'), ('upper', 'f4')])
        properties['lower'], properties['upper'] = raw[:, 0], raw[:, 1]
        modern = SimpleNamespace(dof_properties=properties)
        for articulation in (legacy, modern):
            limits = canonical_joint_limits(articulation, [3, 2, 1, 0], np)
            np.testing.assert_array_equal(limits, raw[::-1])
            np.testing.assert_array_equal(np.clip([20] * 4, limits[:, 0], limits[:, 1]), raw[::-1, 1])
        properties['upper'][0] = float('nan')
        with self.assertRaises(ValueError):
            canonical_joint_limits(modern, [0, 1, 2, 3], np)

    def test_growing_payload_qualifies_and_closed_bucket_resets_the_window(self):
        completion = CompletionTracker(100)
        self.assertFalse(completion.observe(100, -.8))
        self.assertFalse(completion.observe(120, -.8))
        self.assertTrue(completion.observe(150, -.8))
        self.assertFalse(completion.observe(200, -.81))
        self.assertFalse(completion.observe(200, -.8))
        self.assertFalse(completion.observe(200, -.8))
        self.assertTrue(completion.observe(200, -.8))

    def test_payload_drop_resets_completion_and_invalid_values_are_rejected(self):
        completion = CompletionTracker(100)
        completion.observe(100, 0.)
        self.assertFalse(completion.observe(99, 0.))
        self.assertEqual(completion.consecutive, 0)
        for payload, angle in [(True, 0.), (-1, 0.), (100, float('nan'))]:
            with self.assertRaises(ValueError):
                completion.observe(payload, angle)

    def test_identity_exclusion_handles_departing_initial_deposits(self):
        tracker = ParticleTransferTracker()
        tracker.observe(200, [], range(100))
        result = tracker.observe(200, [], range(100, 200))
        self.assertEqual(result['payload_particles'], 100)

    def test_nonstandard_simulation_modes_are_explicitly_diagnostic(self):
        self.assertEqual(execution_profile(True, True, True), 'standard')
        for fresh, render, headless, reference in [(False, True, True, False),
                (True, False, True, False), (True, True, False, False), (True, True, True, True)]:
            self.assertEqual(execution_profile(fresh, render, headless, reference), 'diagnostic')

    def test_current_scene_metric_validation_rejects_stale_geometry(self):
        namespace = {'TRUCK_COUNT_METHOD': TRUCK_COUNT_METHOD}
        tree = ast.parse((BASE / 'scripts/eval_smolvla_success.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('deposited_count', 'validate_scene_metric')]
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<metric>', 'exec'), namespace)
        reply = {'truck_bed_load_metrics': dict(available=True, source_tracking_valid=True,
                    method=TRUCK_COUNT_METHOD, scene_seed=2, count=100)}
        namespace['validate_scene_metric'](reply, 2)
        with self.assertRaisesRegex(ValueError, 'different scene'):
            namespace['validate_scene_metric'](reply, 3)
        reply['truck_bed_load_metrics']['method'] = 'world_aabb_excluding_bucket_v2'
        with self.assertRaisesRegex(ValueError, 'Invalid truck'):
            namespace['validate_scene_metric'](reply, 2)


class ReportValidityTests(unittest.TestCase):
    def row(self, **updates):
        result = dict(episode_index=1, status='completed', success=True,
                      task_completed_success_and_bucket_open=True, early_terminated=True,
                      evaluation_standard=EVALUATION_STANDARD, evaluation_profile='standard',
                      evaluation_id='id', checkpoint_sha256='weights', execution_source='model',
                      completion_observations=3, final_bucket_open=True, payload_particles=150,
                      min_particles=100, schema_version='excavator_success_rate_episode_v1')
        for key in ('success_time_from_policy_start_seconds', 'success_time_from_simulation_launch_seconds',
                    'task_completion_time_from_policy_start_seconds', 'task_completion_time_from_simulation_launch_seconds'):
            result[key] = 1.
        result.update(updates)
        return result

    def test_only_complete_standard_model_batches_have_an_official_rate(self):
        self.assertEqual(build_summary([self.row()], 1)['official_success_rate'], 1.)
        self.assertIsNone(build_summary([self.row()], 2)['official_success_rate'])
        for updates in ({'status': 'error'}, {'evaluation_profile': 'diagnostic'},
                        {'execution_source': 'expert_replay'}, {'evaluation_standard': None}):
            self.assertFalse(build_summary([self.row(**updates)], 1)['official_result_valid'])

    def test_mixed_rules_checkpoints_sources_and_duplicate_indices_cannot_be_aggregated(self):
        for updates in ({'evaluation_standard': 'old'}, {'evaluation_id': 'different-checkpoint'},
                        {'checkpoint_sha256': 'different-weights'},
                        {'execution_source': 'expert_replay'}, {'evaluation_profile': 'diagnostic'}):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, 'Cannot aggregate'):
                build_summary([self.row(), self.row(episode_index=2, **updates)], 2)
        with self.assertRaisesRegex(ValueError, 'episode indices'):
            build_summary([self.row(), self.row()], 2)

    def test_inconsistent_success_is_a_validation_error(self):
        for updates in ({'task_completed_success_and_bucket_open': False},
                        {'completion_observations': 2}, {'payload_particles': 99}):
            with tempfile.TemporaryDirectory() as folder:
                episode = Path(folder) / 'episode_0001'
                episode.mkdir()
                atomic_json(episode / 'episode_status.json', self.row(**updates))
                atomic_json(episode / 'rollout_validation.json', {'status': 'passed'})
                rows = load_episode_rows(Path(folder))
                self.assertEqual(rows[0]['status'], 'validation_error')
                self.assertFalse(build_summary(rows, 1)['official_result_valid'])

    def test_new_failure_labels_distinguish_retained_load_from_no_scoop(self):
        base = self.row(success=False, task_completed_success_and_bucket_open=False,
                        payload_particles=0, retained_particles=0, scooped_particles=0)
        self.assertEqual(failure_mode(base), 'no_scoop')
        self.assertEqual(failure_mode({**base, 'retained_particles': 100}), 'retained_payload')
        self.assertEqual(failure_mode({**base, 'payload_particles': 100,
                                      'final_bucket_open': False}), 'bucket_not_open')

    def test_provenance_hash_detects_weight_or_source_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'model.safetensors'
            path.write_bytes(b'checkpoint-a')
            first = content_id(file_manifest([path], root))
            path.write_bytes(b'checkpoint-b')
            self.assertNotEqual(first, content_id(file_manifest([path], root)))


if __name__ == '__main__':
    unittest.main()
