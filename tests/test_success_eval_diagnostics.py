"""Regressions for hidden physics changes and failed loaded-bucket rollouts."""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'scripts'), str(BASE / 'ExcavatorVLA-smolvla')]
from excavator_common.simulation_settings import resolve_simulation_settings, apply_simulation_settings
from success_eval_diagnostics import RolloutDiagnostics
from success_eval_runtime import content_id, evaluation_provenance


class SimulationSettingsTests(unittest.TestCase):
    def test_defaults_describe_newtons_free_pile_instead_of_physx_walls(self):
        settings = resolve_simulation_settings(environ={})
        self.assertEqual(settings['soil_backend'], 'newton_mpm')
        self.assertEqual(settings['sand_initialization_method'], 'newton_mpm_free_pile_settle_v1')

    def test_explicit_options_override_inherited_physics(self):
        settings = resolve_simulation_settings('physx', False, {
            'EXCAVATOR_SOIL_BACKEND': 'newton_mpm', 'EXCAVATOR_SAND_WALLS_DURING_SETTLE': '1',
            'EXCAVATOR_MPM_DENSITY': '1600',
        })
        self.assertEqual(settings['soil_backend'], 'physx')
        self.assertFalse(settings['sand_walls_during_settle'])
        self.assertEqual(settings['mpm_environment'], {})
        self.assertEqual(settings['sand_initialization_method'], 'free_pile_settle_without_retaining_walls_v1')
        with_walls = resolve_simulation_settings('physx', True, {})
        self.assertEqual(with_walls['sand_initialization_method'], 'temporary_rectangular_walls_during_settle_v1')

    def test_workers_reproduce_saved_settings_despite_a_different_environment(self):
        saved = resolve_simulation_settings(environ={
            'EXCAVATOR_MPM_DENSITY': '1200', 'EXCAVATOR_SAND_WALLS_DURING_SETTLE': 'off',
        })
        worker = {'EXCAVATOR_SOIL_BACKEND': 'physx', 'EXCAVATOR_MPM_VOXEL_SIZE': '.2', 'UNRELATED': 'keep'}
        apply_simulation_settings(saved, worker)
        self.assertEqual(resolve_simulation_settings(environ=worker), saved)
        self.assertEqual(worker['UNRELATED'], 'keep')
        self.assertNotIn('EXCAVATOR_MPM_VOXEL_SIZE', worker)

    def test_changed_backend_wall_timing_and_material_change_the_identity(self):
        baseline = content_id(resolve_simulation_settings('physx', True, {}))
        for backend, walls, env in [('physx', False, {}), ('newton_mpm', True, {}),
                                     ('newton_mpm', True, {'EXCAVATOR_MPM_DENSITY': '1200'})]:
            self.assertNotEqual(baseline, content_id(resolve_simulation_settings(backend, walls, env)))
        self.assertNotEqual(content_id(resolve_simulation_settings(environ={})), content_id(
            resolve_simulation_settings(environ={'EXCAVATOR_MPM_DENSITY': '1200'})))

    def test_invalid_settings_fail_before_starting_a_simulator(self):
        for env in ({'EXCAVATOR_SOIL_BACKEND': 'unknown'},
                    {'EXCAVATOR_SAND_WALLS_DURING_SETTLE': 'sometimes'}):
            with self.assertRaises(ValueError):
                resolve_simulation_settings(environ=env)

    def test_mpm_source_change_invalidates_provenance_with_unchanged_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [
                'ExcavatorVLA-smolvla/run_simulation.py',
                'ExcavatorVLA-smolvla/scripts/excavator_app/sand_site_runtime.py',
                'ExcavatorVLA-smolvla/scripts/excavator_app/newton_mpm_soil.py',
                'ExcavatorVLA-smolvla/scripts/evaluation/summarize_isaacsim_success_rate.py',
                'ExcavatorVLA-isaac/assets/usd/excavator_scene.usd', 'run_isaac_python.sh',
                'scripts/eval_smolvla_success.py', 'scripts/success_eval_runtime.py',
                'scripts/eval_video.py', 'scripts/excavator_diagnostics.py',
                'scripts/success_eval_diagnostics.py', 'checkpoint/model.safetensors',
                'dataset/meta/info.json', 'isaacsim/VERSION',
            ]
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            with patch.dict('os.environ', {'EXCAVATOR_ISAAC_DIR': str(root / 'isaacsim')}), \
                    patch('success_eval_runtime.subprocess.run', return_value=SimpleNamespace(stdout='GPU-a')):
                first = evaluation_provenance(root, root / 'checkpoint', root / 'dataset', {}, {})
                (root / files[2]).write_text('changed contact solver')
                second = evaluation_provenance(root, root / 'checkpoint', root / 'dataset', {}, {})
            self.assertEqual(first['checkpoint_sha256'], second['checkpoint_sha256'])
            self.assertNotEqual(first['evaluation_id'], second['evaluation_id'])


class RolloutDiagnosticsTests(unittest.TestCase):
    limits = [[-math.pi, math.pi], [-1.3, 1.3], [-1.7, 1.7], [-2.1, 1.6]]

    def tracker(self, window=3):
        return RolloutDiagnostics(self.limits, stall_window_steps=window)

    def observe(self, tracker, step, *, positions=None, commands=None, payload=0, retained=100):
        positions = list(positions if positions is not None else [0., 0., 0., -1.])
        state = positions + [0.] * 24
        tracker.observe(step, state, commands if commands is not None else positions,
                        {'payload_particles': payload, 'retained_particles': retained})

    def test_out_of_limit_measurements_are_distinct_from_valid_commands(self):
        tracker = self.tracker()
        for step in range(2):
            self.observe(tracker, step, positions=[0., 0., 0., -math.pi], commands=[0., 0., 0., -1.3])
        result = tracker.summary()
        bucket = result['measured_joint_limit_violations']['bucket']
        self.assertEqual(bucket['steps'], 2)
        self.assertEqual(bucket['first_step'], 0)
        self.assertAlmostEqual(bucket['max_excess_rad'], math.pi - 2.1)
        self.assertAlmostEqual(result['max_tracking_error_rad']['bucket'], math.pi - 1.3)
        self.assertIn('bucket_outside_joint_limits', tracker.warnings())

    def test_small_solver_overshoot_does_not_produce_a_limit_warning(self):
        tracker = self.tracker()
        self.observe(tracker, 0, positions=[0., 0., 0., -2.11])
        self.assertEqual(tracker.summary()['measured_joint_limit_violations'], {})

    def test_loaded_stall_is_reported_without_modifying_the_inputs(self):
        tracker = self.tracker()
        state = [0., 0., 0., -1.] + [0.] * 24
        commands = [0., 0., 0., -1.]
        transfer = {'payload_particles': 0, 'retained_particles': 100}
        original = json.dumps([state, commands, transfer])
        for step in range(3):
            tracker.observe(step, state, commands, transfer)
        stall = tracker.summary()['retained_payload_stall']
        self.assertTrue(stall['detected'])
        self.assertTrue(stall['active'])
        self.assertEqual(stall['first_step'], 2)
        self.assertEqual(json.dumps([state, commands, transfer]), original)

    def test_swing_motion_deposit_or_empty_bucket_prevents_a_stall_warning(self):
        for mode in ('motion', 'deposit', 'empty'):
            tracker = self.tracker()
            for step in range(4):
                self.observe(tracker, step,
                    positions=[step * .1 if mode == 'motion' else 0., 0., 0., -1.],
                    payload=step * 100 if mode == 'deposit' else 0,
                    retained=0 if mode == 'empty' else 100)
            self.assertFalse(tracker.summary()['retained_payload_stall']['detected'], mode)

    def test_transport_recovery_clears_active_warning_and_preserves_history(self):
        tracker = self.tracker()
        for step in range(3):
            self.observe(tracker, step)
        self.observe(tracker, 3, positions=[.5, 0., 0., -1.])
        stall = tracker.summary()['retained_payload_stall']
        self.assertTrue(stall['detected'])
        self.assertFalse(stall['active'])
        self.assertNotIn('retained_payload_stall', tracker.warnings())


if __name__ == '__main__':
    unittest.main()
