import json
import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from act_stall_recovery import PostDigStallDetector, StallSettings, SecureLiftRecovery, SecureAlignRecovery, mode_settings, make_stall_guard_class


class StallRecoveryTests(unittest.TestCase):
    def state(self):
        s = np.zeros(28); s[26] = .003; s[20] = -1.; s[21] = 12.
        return s

    def test_timeout_needs_actual_loaded_stationary_observations(self):
        detector = PostDigStallDetector()
        state = self.state()
        for i in range(250):
            self.assertFalse(detector.observe(state, np.zeros(4))[0])
        self.assertTrue(detector.observe(state, np.zeros(4))[0])
        state[26] = 0
        self.assertFalse(detector.observe(state, np.zeros(4))[0])

    def test_does_not_interrupt_unloading_or_active_motion(self):
        for field, value, action in [(21, 2., np.zeros(4)), (21, 12., np.array([0., .2, 0., 0.]))]:
            detector = PostDigStallDetector(StallSettings(earliest_trigger_step=0))
            state = self.state(); state[field] = value
            for _ in range(60):
                self.assertFalse(detector.observe(state, action)[0])

    def test_early_mode_requires_low_fill_not_just_a_short_pause(self):
        for fill, expected in [(.02, True), (.07, False)]:
            detector = PostDigStallDetector(mode_settings('secure_align'))
            state = self.state(); state[26] = fill
            for _ in range(40):
                triggered, _ = detector.observe(state, np.zeros(4))
            self.assertEqual(triggered, expected)

    def test_alignment_hands_back_with_closed_bucket_before_executing_open_command(self):
        model_path = ROOT / 'outputs/experiments/act_recovery_collection_progressive_20261105_02/robot_model.json'
        base = ROOT / 'outputs/experiments/act_transport_validation40_20261104_01/evaluation/episodes/episode_0009'
        if not model_path.exists():
            self.skipTest('Authored geometry unavailable')
        row = json.loads((base / 'steps.jsonl').read_text().splitlines()[73])
        state = np.asarray(row['input_state'], dtype=float); state[26] = .02
        recovery = SecureAlignRecovery(json.loads(model_path.read_text()),
                                      json.loads((base / 'scene_context.json').read_text()), state)
        command = state[:4].copy()
        for _ in range(180):
            landmarks = {name: recovery.secure_lift.chain.point(state[:4], name).tolist()
                         for name in ('tip', 'load', 'pour')}
            action, details = recovery.action(state, command, landmarks)
            if action is None:
                break
            command = np.clip(command + .1 * action, recovery.secure_lift.bounds[:, 0], recovery.secure_lift.bounds[:, 1])
            state[:4] = command
        self.assertEqual(details['phase'], 'completed_secure_align')
        self.assertLess(state[3], -1.3)
        self.assertLessEqual(details['contained_opening_arc_residual_m'], .015)

    def test_carry_guard_does_not_block_intermediate_closure_targets(self):
        from act_carry_guard import CarryAttitudeGuard, CarryGuardSettings
        from excavator_reachability import AuthoredExcavatorChain
        model_path = ROOT / 'outputs/experiments/act_recovery_collection_progressive_20261105_02/robot_model.json'
        folder = ROOT / 'outputs/experiments/act_transport_validation40_20261104_01/evaluation/episodes/episode_0002'
        if not model_path.exists():
            self.skipTest('Authored geometry unavailable')
        model = json.loads(model_path.read_text()); scene = json.loads((folder / 'scene_context.json').read_text())
        chain = AuthoredExcavatorChain(model)
        row = json.loads((folder / 'steps.jsonl').read_text().splitlines()[69])
        state = np.asarray(row['input_state'], dtype=float); state[26] = .02
        command = state[:4].copy()
        guard = make_stall_guard_class(CarryAttitudeGuard, model, 'secure_align')(
            scene, settings=CarryGuardSettings(parent_projection='arm_first', release_geometry='opening_pose'))
        suspended = 0
        for _ in range(230):
            landmarks = {name: chain.point(state[:4], name).tolist() for name in ('tip', 'load', 'pour')}
            heading = state[0]; c, s = np.cos(heading), np.sin(heading)
            for offset, point, target in [(18, landmarks['tip'], scene['observation_context']['dig_target_xyz']),
                                          (21, landmarks['load'], scene['observation_context']['unload_landing_xyz'])]:
                delta = np.asarray(target) - point
                state[offset:offset+3] = [c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1], delta[2]]
            pitch = chain.carry_pitch(state[:4]); state[16:18] = [np.cos(pitch), np.sin(pitch)]
            applied, trace = guard.update(state, np.zeros(4), command, landmarks)
            suspended += int(trace.get('suspended_for_progressive_closure', False))
            recovery = trace['stall_recovery']
            if recovery.get('handback'):
                break
            command = np.clip(command + .1 * applied, guard.joint_limits[:, 0], guard.joint_limits[:, 1])
            state[:4] = command
        self.assertGreater(suspended, 0)
        self.assertEqual(recovery['phase'], 'completed_secure_align')
        self.assertLess(state[3], -1.3)
        self.assertLessEqual(recovery['contained_opening_arc_residual_m'], .015)

    def test_recovery_geometry_is_bounded_and_leaves_swing_for_act(self):
        model_path = ROOT / 'outputs/experiments/act_recovery_collection_progressive_20261105_02/robot_model.json'
        base = ROOT / 'outputs/experiments/act_transport_validation40_20261104_01/evaluation/episodes'
        if not model_path.exists():
            self.skipTest('Authored geometry unavailable')
        model = json.loads(model_path.read_text())
        for scene in [1, 9, 24]:
            folder = base / f'episode_{scene:04d}'
            rows = [json.loads(s) for s in (folder / 'steps.jsonl').read_text().splitlines()]
            initial = rows[250]['input_state']
            recovery = SecureLiftRecovery(model, json.loads((folder / 'scene_context.json').read_text()), initial)
            self.assertLess(abs(recovery.lift[0] - initial[0]), .1)
            self.assertTrue(np.all(recovery.lift >= recovery.bounds[:, 0]))
            self.assertTrue(np.all(recovery.lift <= recovery.bounds[:, 1]))
            self.assertGreater(recovery.chain.point(recovery.lift, 'tip')[2], recovery.initial_tip[2] + 1.)
            self.assertLessEqual(recovery.chain.carry_pitch(recovery.lift), np.deg2rad(-130))

    def test_scoop_dip_adds_real_insertion_without_crossing_the_ground(self):
        model_path = ROOT / 'outputs/experiments/act_recovery_collection_progressive_20261105_02/robot_model.json'
        folder = ROOT / 'outputs/experiments/act_transport_validation40_20261104_01/evaluation/episodes/episode_0002'
        if not model_path.exists():
            self.skipTest('Authored geometry unavailable')
        model = json.loads(model_path.read_text()); scene = json.loads((folder / 'scene_context.json').read_text())
        state = json.loads((folder / 'steps.jsonl').read_text().splitlines()[69])['input_state']
        recovery = SecureLiftRecovery(model, scene, state, dip_m=.35)
        z = [recovery.chain.point(q, 'tip')[2] for q in recovery.path]
        self.assertLess(min(z), recovery.initial_tip[2] - .05)
        previous = np.asarray(state[:4])
        for goal in recovery.path:
            for q in previous + (goal - previous) * np.linspace(0, 1, 7)[:, None]:
                self.assertGreaterEqual(recovery.minimum_bucket_z(q), .005)
            previous = goal


if __name__ == '__main__':
    unittest.main()
