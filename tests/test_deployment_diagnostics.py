import importlib.util
import ast
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(BASE / 'ExcavatorVLA-smolvla'))
from excavator_diagnostics import dataset_initial_pose, exporter_module, reference_episode
from excavator_common.deployment_scene_contract import (
    relative_task_text, truck_clears_robot_body, sample_random_scene_profile,
)


class DeploymentDiagnosticsTests(unittest.TestCase):
    def test_long_truck_is_rejected_even_if_its_center_is_far_away(self):
        self.assertFalse(truck_clears_robot_body([0, 0], [-1, 1, 0], [1, 11, 5]))
        self.assertTrue(truck_clears_robot_body([0, 0], [-1, 3, 0], [1, 13, 5]))

    def test_collision_rejection_resamples_reproducibly(self):
        kwargs = dict(seed=2, robot_xy=[-6.95, 2.23], truck_center_xy=[-5, -9],
                      truck_dump_center_xy=[-6.8, -7.5], truck_yaw_deg=-93)
        def validate(candidate):
            return candidate['sample_try'] >= 3
        first = sample_random_scene_profile(**kwargs, placement_validator=validate)
        second = sample_random_scene_profile(**kwargs, placement_validator=validate)
        self.assertGreaterEqual(first['sample_try'], 3)
        self.assertEqual(first, second)

    def test_render_updates_do_not_move_restored_articulation(self):
        tree = ast.parse((BASE / 'ExcavatorVLA-smolvla/run_simulation.py').read_text())
        restore = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                       and node.name == 'restore_excavator_pose_after_sand_init')
        timeline = SimpleNamespace(playing=True)
        timeline.pause = lambda: setattr(timeline, 'playing', False)
        robot = SimpleNamespace(position=np.zeros(3), orientation=np.array([1., 0, 0, 0]), q=np.zeros(4))
        robot.set_world_pose = lambda position, orientation: (
            setattr(robot, 'position', position.copy()), setattr(robot, 'orientation', orientation.copy()))
        robot.set_joint_positions = lambda values: setattr(robot, 'q', values.copy())
        robot.set_joint_velocities = lambda values: None
        robot.apply_action = lambda action: None
        robot.get_world_pose = lambda: (robot.position.copy(), robot.orientation.copy())
        robot.get_joint_positions = lambda: robot.q.copy()

        def render_update():
            if timeline.playing:
                robot.q += .004  # The startup drift that used to fail parity checks.

        namespace = dict(np=np, math=math, robot=robot, timeline=timeline,
                         simulation_app=SimpleNamespace(update=render_update),
                         ArticulationAction=lambda **kwargs: kwargs, print=lambda *args, **kwargs: None)
        exec(compile(ast.Module(body=[restore], type_ignores=[]), '<restore>', 'exec'), namespace)
        snapshot = dict(position=np.array([1., 2, 3]), orientation=np.array([1., 0, 0, 0]),
                        joint_positions=np.array([0., .8, -1., -.3]), joint_velocities=np.zeros(4))
        namespace['restore_excavator_pose_after_sand_init'](snapshot)
        self.assertFalse(timeline.playing)
        np.testing.assert_array_equal(robot.q, snapshot['joint_positions'])

    def test_reference_replay_uses_the_actual_exported_action_alignment(self):
        reference = reference_episode(BASE / 'data/raw/run_20261012_133855/episode_000002', 10)
        data = pd.read_parquet(BASE / 'data/export/data/chunk-000/file-000.parquet')
        np.testing.assert_allclose(reference['actions'], np.stack(data.action), atol=1e-6)
        np.testing.assert_allclose(reference['expected_initial_state'], data.iloc[0]['observation.state'], atol=1e-6)
        np.testing.assert_allclose(reference['initial_joints_deg'], dataset_initial_pose(BASE / 'data/export'), atol=1e-6)

    def test_rollout_language_matches_exporter_for_all_direction_pairs(self):
        exporter = exporter_module()
        origin = [2, -3]
        heading = .3
        for source in range(8):
            for destination in range(8):
                def point(direction):
                    angle = heading + direction * math.pi / 4
                    return [origin[0] + 7 * math.cos(angle), origin[1] + 7 * math.sin(angle), .2]
                dig, unload = point(source), point(destination)
                meta = {'target_xyz': dig, 'unload_landing_xyz': unload,
                        'scene_randomization': {'applied': {'robot_body_yaw_deg': math.degrees(heading)}}}
                sample = {'obs.state_legacy_14d': [*origin, 0] + [0] * 11}
                expected = exporter.build_episode_task_text(sample, meta)
                self.assertEqual(relative_task_text(dig, unload, origin, heading), expected)


if __name__ == '__main__':
    unittest.main()
