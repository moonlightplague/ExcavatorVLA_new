"""Guard against falsely excluding scenes on distance heuristics or IK misses."""
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from excavator_reachability import (AuthoredExcavatorChain, TruckBedColumn, find_endpoint,
                                    restored_truck_geometry, workspace_bound)


def transform(x=0., y=0., z=0., pitch_frame=False):
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    if pitch_frame:
        matrix[:3, :3] = [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
    return matrix.tolist()


def model():
    frames = [transform(z=.6), transform(x=.53, z=.77, pitch_frame=True),
              transform(x=5.715), transform(x=2.95)]
    return dict(schema='excavator_authored_chain_v1', base_world=transform(z=1.2),
        joints=[dict(name=name, axis='Z', parent_from_joint=frame, child_from_joint=transform())
                for name, frame in zip(('swing', 'boom', 'arm', 'bucket'), frames)],
        landmarks=dict(tip=[.75, 0., -.18], load=[.35, 0., .08], pour=[.85, 0., .30]),
        truck_template=dict(root_xyz=[0., 0., 0.], root_yaw_deg=0., bed_world_row_matrix=np.eye(4).tolist()))


def scene(x0=5., x1=9., target_x=7.):
    return dict(scene_profile=dict(truck_translation_xyz=[0., 0., 0.], truck_yaw_deg=0.),
        canonical_joint_limits_rad=np.radians([[-180., 180.], [-75., 75.], [-95., 95.], [-120., 90.]]).tolist(),
        observation_context=dict(unload_landing_xyz=[target_x, 0., 4.]),
        truck_counting_volume=dict(local_from_world=np.eye(4).tolist(),
                                   local_min=[x0, -1., 1.], local_max=[x1, 1., 4.]))


class ForwardGeometryTests(unittest.TestCase):
    def test_zero_and_raised_chain_match_independent_closed_form(self):
        chain = AuthoredExcavatorChain(model())
        np.testing.assert_allclose(chain.point([0., 0., 0., 0.]), [9.795, -.19, 2.57], atol=1e-9)
        np.testing.assert_allclose(chain.point([0., math.pi/2, 0., 0.]), [.53, -.19, 11.835], atol=1e-9)
        np.testing.assert_allclose(chain.point([math.pi/2, 0., 0., 0.]), [.19, 9.795, 2.57], atol=1e-9)

    def test_world_attitude_tracks_all_three_pitch_joints(self):
        chain = AuthoredExcavatorChain(model())
        q = np.array([[.8, .3, -.7, -1.9], [-2., -.5, -.6, -2.], [1., 1., 1., 1.]])
        np.testing.assert_allclose(chain.carry_pitch(q), q[:, 1:].sum(1), atol=1e-9)

    def test_outer_sphere_contains_actual_points_across_joint_rotations(self):
        chain = AuthoredExcavatorChain(model())
        rng = np.random.default_rng(17)
        q = rng.uniform(-math.pi, math.pi, (500, 4))
        for name in ('tip', 'load', 'pour'):
            distances = np.linalg.norm(chain.point(q, name)-chain.swing_anchor, axis=1)
            self.assertLessEqual(float(distances.max()), chain.release_radius_upper_bound()+1e-9)

    def test_scaled_frames_cannot_create_an_invalid_reach_certificate(self):
        data = model()
        data['base_world'][0][0] = 2.
        with self.assertRaisesRegex(ValueError, 'unscaled'):
            AuthoredExcavatorChain(data)


class ReachBoundsTests(unittest.TestCase):
    def test_far_target_does_not_exclude_reachable_near_side_of_bed(self):
        chain, value = AuthoredExcavatorChain(model()), scene(x0=5., x1=13., target_x=12.)
        bound = workspace_bound(chain, TruckBedColumn(value), value['observation_context']['unload_landing_xyz'])
        self.assertTrue(bound['target_column_certified_outside'])
        self.assertFalse(bound['bed_column_certified_outside'])

    def test_entire_far_bed_is_certified_outside_without_ik(self):
        chain, value = AuthoredExcavatorChain(model()), scene(x0=30., x1=34., target_x=32.)
        bound = workspace_bound(chain, TruckBedColumn(value), value['observation_context']['unload_landing_xyz'])
        self.assertTrue(bound['bed_column_certified_outside'])

    def test_release_height_can_exclude_target_inside_simple_horizontal_radius(self):
        chain, value = AuthoredExcavatorChain(model()), scene(x0=5., x1=13., target_x=10.2)
        value['truck_counting_volume']['local_max'][2] = 5.
        bound = workspace_bound(chain, TruckBedColumn(value), value['observation_context']['unload_landing_xyz'])
        self.assertLess(bound['target_xy_distance_m'], bound['radius_upper_bound_m'])
        self.assertTrue(bound['target_column_certified_outside'])
        self.assertFalse(bound['bed_column_certified_outside'])

    def test_rotated_scaled_bed_uses_exact_local_geometry(self):
        value = scene()
        yaw = math.pi/3
        world = np.array([[2*np.cos(yaw), 2*np.sin(yaw), 0., 0.],
                          [-3*np.sin(yaw), 3*np.cos(yaw), 0., 0.],
                          [0., 0., 1.5, 0.], [10., -4., 0., 1.]])
        value['truck_counting_volume']['local_from_world'] = np.linalg.inv(world).tolist()
        bed = TruckBedColumn(value)
        local = np.array([7., 0., 5., 1.])
        point = (local@world)[:3]
        np.testing.assert_allclose(bed.local(point), local[:3], atol=1e-9)
        np.testing.assert_allclose(bed.residual(point), [0., 0., 0.], atol=1e-9)
        self.assertGreater(bed.distance_lower_bound(np.zeros(3)), 0.)


class EndpointAndRestorationTests(unittest.TestCase):
    def test_positive_endpoint_keeps_joint_bounds_and_explicit_path_limitations(self):
        chain, value = AuthoredExcavatorChain(model()), scene()
        for kind in ('dump', 'carry'):
            with self.subTest(kind=kind):
                result = find_endpoint(chain, value, kind)
                self.assertEqual(result['status'], 'endpoint_found_path_unverified')
                q = np.asarray(result['q_rad'])
                bounds = np.asarray(value['canonical_joint_limits_rad'])
                self.assertTrue(np.all(q >= bounds[:, 0]-1e-8))
                self.assertTrue(np.all(q <= bounds[:, 1]+1e-8))
                self.assertFalse(result['collision_checked'])
                self.assertFalse(result['path_checked'])
                if kind == 'carry':
                    self.assertLessEqual(chain.carry_pitch(q), math.radians(-130)+1e-4)

    def test_numerical_search_failure_is_unresolved(self):
        result = find_endpoint(AuthoredExcavatorChain(model()), scene(x0=30., x1=34., target_x=32.),
                               'dump', max_starts=1)
        self.assertEqual(result['status'], 'unresolved')
        self.assertNotIn('unreachable', result['status'])

    def test_restores_authored_box_from_pose_without_mutating_record(self):
        value = scene()
        value['truck_counting_volume'].pop('local_from_world')
        value['scene_profile'] = dict(truck_translation_xyz=[-2., 8., 0.], truck_yaw_deg=90.)
        original = copy.deepcopy(value)
        restored = restored_truck_geometry(value, model())
        bed = TruckBedColumn(restored)
        np.testing.assert_allclose(bed.local([-2., 15., 5.]), [7., 0., 5.], atol=1e-9)
        self.assertEqual(value, original)

    def test_reconstruction_must_agree_with_existing_exact_matrix(self):
        value = scene()
        value['scene_profile']['truck_yaw_deg'] = 45.
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            restored_truck_geometry(value, model())


if __name__ == '__main__':
    unittest.main()
