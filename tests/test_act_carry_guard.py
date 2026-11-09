import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from act_carry_guard import CarryAttitudeGuard, CarryGuardSettings, opening_pour_point


def scene():
    return dict(canonical_joint_limits_rad=[[-math.pi, math.pi], [-1.309, 1.309],
                                            [-1.658, 1.658], [-2.094, 1.571]],
        truck_counting_volume=dict(local_from_world=np.eye(4).tolist(),
                                   local_min=[-1., -2., 0.], local_max=[1., 2., 2.]),
        observation_context=dict(unload_landing_xyz=[0., 0., 0.]),
        scene_profile=dict(actual_truck_yaw_deg=0.))


def observation(q=None, fill=.1, load=(5., 0., 3.)):
    q = list(q or [0., .3, -.7, -1.9])
    state = np.zeros(28)
    state[:4] = q
    state[16:18] = [math.cos(sum(q[1:])), math.sin(sum(q[1:]))]
    state[20] = -1.
    state[21:24] = -np.array(load)
    state[25] = 1.
    state[26] = fill
    landmarks = dict(load=list(load), pour=list(np.array(load) + [.1, 0., .1]))
    return state, landmarks


class CarryGuardTests(unittest.TestCase):
    def test_no_load_does_not_change_digging_actions(self):
        guard = CarryAttitudeGuard(scene())
        state, landmarks = observation(fill=0.)
        action = np.array([.3, .5, .8, 1.5], dtype=np.float32)
        adjusted, details = guard.update(state, action, state[:4], landmarks)
        np.testing.assert_array_equal(adjusted, action)
        self.assertFalse(details['active'])

    def test_recorded_profile_without_randomized_yaw_fields_is_supported(self):
        data = scene()
        data['scene_profile'] = {'truck_yaw_deg': -108.}
        guard = CarryAttitudeGuard(data)
        state, landmarks = observation()
        state[0] = 1.
        world_delta = -np.array(landmarks['load'])
        c, s = math.cos(1.), math.sin(1.)
        state[21:23] = [c * world_delta[0] + s * world_delta[1],
                        -s * world_delta[0] + c * world_delta[1]]
        guard.update(state, [.2, 0., 0., 0.], state[:4], landmarks)

    def test_compensates_parent_motion_with_bucket_travel_first(self):
        guard = CarryAttitudeGuard(scene())
        state, landmarks = observation()
        action = np.array([.3, .2, .4, .1], dtype=np.float32)
        adjusted, details = guard.update(state, action, state[:4], landmarks)
        self.assertTrue(details['modified'])
        np.testing.assert_array_equal(adjusted[:3], action[:3])
        self.assertLess(adjusted[3], action[3])
        self.assertLessEqual(details['corrected_pitch_rad'], details['limit_rad'] + 1e-7)

    def test_at_bucket_limit_constrains_parent_joints_and_keeps_swing(self):
        guard = CarryAttitudeGuard(scene())
        state, landmarks = observation(q=[0., .5, -.7, -2.094])
        action = np.array([.3, .6, .6, 0.], dtype=np.float32)
        adjusted, details = guard.update(state, action, state[:4], landmarks)
        self.assertTrue(details['modified'])
        self.assertEqual(adjusted[0], action[0])
        self.assertAlmostEqual(adjusted[3], 0., places=5)
        self.assertLess(adjusted[1] + adjusted[2], action[1] + action[2])
        self.assertLessEqual(details['corrected_pitch_rad'], details['limit_rad'] + 1e-7)
        target = state[:4] + .1 * adjusted
        self.assertTrue(np.all(target >= guard.joint_limits[:, 0] - 1e-6))
        self.assertTrue(np.all(target <= guard.joint_limits[:, 1] + 1e-6))

    def test_uses_integrated_target_instead_of_assuming_zero_tracking_error(self):
        guard = CarryAttitudeGuard(scene())
        state, landmarks = observation()
        command = state[:4].copy()
        command[1:3] += .1
        adjusted, details = guard.update(state, [.2, .2, .2, 0.], command, landmarks)
        next_target = command + .1 * adjusted
        self.assertAlmostEqual(sum(next_target[1:]), details['corrected_pitch_rad'], places=6)
        self.assertTrue(details['modified'])

    def test_arm_first_correction_preserves_boom_lift_when_bucket_is_at_limit(self):
        guard = CarryAttitudeGuard(scene(), settings=CarryGuardSettings(parent_projection='arm_first'))
        state, landmarks = observation(q=[0., .5, -.7, -2.094])
        action = np.array([.3, .6, .6, 0.], dtype=np.float32)
        adjusted, details = guard.update(state, action, state[:4], landmarks)
        self.assertEqual(adjusted[1], action[1])
        self.assertLess(adjusted[2], 0.)
        self.assertLessEqual(details['corrected_pitch_rad'], details['limit_rad'] + 1e-7)

    def test_releases_only_after_current_landmarks_are_above_bed_for_three_steps(self):
        guard = CarryAttitudeGuard(scene())
        state, landmarks = observation(load=(0., 0., 3.))
        action = np.array([.2, .2, .2, 3.], dtype=np.float32)
        for _ in range(2):
            _, details = guard.update(state, action, state[:4], landmarks)
            self.assertFalse(details['released'])
        adjusted, details = guard.update(state, action, state[:4], landmarks)
        self.assertTrue(details['released'])
        np.testing.assert_array_equal(adjusted, action)
        self.assertFalse(CarryAttitudeGuard(scene()).released)

    def test_near_but_outside_rotated_bed_does_not_release(self):
        data = scene()
        # Gf/USD uses row-vector matrices, including translation in row 3.
        matrix = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]])
        data['truck_counting_volume']['local_from_world'] = matrix.tolist()
        guard = CarryAttitudeGuard(data)
        state, landmarks = observation(load=(0., 1.5, 3.))
        for _ in range(4):
            _, details = guard.update(state, [.2, 0., 0., 3.], state[:4], landmarks)
        self.assertFalse(details['over_bed'])
        self.assertFalse(details['released'])

    def test_load_point_does_not_release_just_because_pour_marker_enters_bed(self):
        state, landmarks = observation(load=(1.05, 0., 3.))
        landmarks['pour'] = [0., 0., 3.]
        current = CarryAttitudeGuard(scene())
        guarded = CarryAttitudeGuard(scene(), settings=CarryGuardSettings(release_geometry='load_point'))
        self.assertTrue(current.release_geometry(state, landmarks)[0])
        self.assertFalse(guarded.release_geometry(state, landmarks)[0])

    def test_loaded_departure_rearms_transport_protection(self):
        guard = CarryAttitudeGuard(scene(), settings=CarryGuardSettings(release_geometry='load_point'))
        state, landmarks = observation(load=(0., 0., 3.))
        for _ in range(3):
            guard.update(state, [.2, .2, .2, 0.], state[:4], landmarks)
        self.assertTrue(guard.released)
        state, landmarks = observation(fill=.005, load=(2., 0., 3.))
        _, detail = guard.update(state, [.2, .2, .2, 0.], state[:4], landmarks)
        self.assertFalse(guard.released)
        self.assertTrue(detail['active'])
        self.assertTrue(detail['rearmed'])

    def test_release_margin_does_not_chatter_and_empty_departure_does_not_rearm(self):
        guard = CarryAttitudeGuard(scene(), settings=CarryGuardSettings(release_geometry='load_point'))
        state, landmarks = observation(load=(0., 0., 3.))
        for _ in range(3):
            guard.update(state, [.2, .2, .2, 0.], state[:4], landmarks)
        state, landmarks = observation(load=(.95, 0., 3.))
        _, detail = guard.update(state, [.2, .2, .2, 0.], state[:4], landmarks)
        self.assertTrue(guard.released)
        self.assertFalse(detail['rearmed'])
        state, landmarks = observation(fill=0., load=(2., 0., 3.))
        _, detail = guard.update(state, [.2, .2, .2, 0.], state[:4], landmarks)
        self.assertTrue(guard.released)
        self.assertFalse(detail['rearmed'])

    def test_infeasible_attitude_is_reported_without_exceeding_velocity_limits(self):
        guard = CarryAttitudeGuard(scene())
        state, landmarks = observation(q=[0., 1.2, 1.5, -1.5])
        adjusted, details = guard.update(state, [.2, 1., 1., 1.], state[:4], landmarks)
        self.assertTrue(details['infeasible'])
        self.assertTrue(np.all(np.abs(adjusted) <= guard.action_limits + 1e-6))

    def test_mismatched_geometry_fails_instead_of_silently_allowing_dump(self):
        guard = CarryAttitudeGuard(scene())
        state, landmarks = observation()
        landmarks['load'][0] += 1
        with self.assertRaisesRegex(ValueError, 'disagree'):
            guard.update(state, [.2, 0., 0., 0.], state[:4], landmarks)

    def rigid_landmarks(self, pivot_x):
        phi = math.radians(-130.)
        pivot = np.array([pivot_x, 0., 4.])
        x = np.array([math.cos(phi), 0., math.sin(phi)])
        z = np.array([0., -1., 0.])
        landmarks = {key: (pivot + a*x + b*z).tolist()
                     for key,a,b in (('tip',.75,-.18),('load',.35,.08),('pour',.85,.30))}
        state,_ = observation(q=[0., .3, -.7, phi+.4], load=landmarks['load'])
        return state, landmarks, pivot

    def test_opening_pose_rotates_about_actual_hinge_and_reproduces_zero_delta(self):
        state, landmarks, pivot = self.rigid_landmarks(-.825)
        np.testing.assert_allclose(opening_pour_point(state, landmarks, state[3]), landmarks['pour'], atol=1e-12)
        angle = sum(state[1:3]) - .8
        expected = pivot + .85*np.array([math.cos(angle),0.,math.sin(angle)]) + [0.,-.30,0.]
        np.testing.assert_allclose(opening_pour_point(state, landmarks, -.8), expected, atol=1e-12)

    def test_near_edge_can_release_if_opening_moves_pour_point_inside_bed(self):
        state, landmarks,_ = self.rigid_landmarks(-.825)
        current = CarryAttitudeGuard(scene())
        projected = CarryAttitudeGuard(scene(), settings=CarryGuardSettings(release_geometry='opening_pose'))
        self.assertFalse(current.release_geometry(state, landmarks)[0])
        self.assertTrue(projected.release_geometry(state, landmarks)[0])

    def test_opening_prediction_never_releases_a_bucket_far_from_bed(self):
        state, landmarks,_ = self.rigid_landmarks(-2.)
        projected = CarryAttitudeGuard(scene(), settings=CarryGuardSettings(release_geometry='opening_pose'))
        self.assertFalse(projected.release_geometry(state, landmarks)[0])


if __name__ == '__main__':
    unittest.main()
