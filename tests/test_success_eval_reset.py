"""Exercise scene/reset transitions with Isaac stand-ins; live runs cover PhysX."""
import ast
from pathlib import Path
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

BASE = Path(__file__).resolve().parents[1] / 'ExcavatorVLA-smolvla'


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    # The real function closes over main's episode state. Expose that state to
    # the test so we can check the transition without constructing a Kit app.
    function.body = [ast.copy_location(ast.Global(node.names), node)
                     if isinstance(node, ast.Nonlocal) else node for node in function.body]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace[name]


class SandRelocationTests(unittest.TestCase):
    def test_repeated_relocation_preserves_footprint_and_absent_wall_topology(self):
        hull = np.array([[1, 2], [4, 2], [4, 5], [1, 5]], dtype=np.float32)
        namespace = dict(np=np, SAND_CENTER_X=2.5, SAND_CENTER_Y=3.5,
                         PILE_CENTER_X=2.5, PILE_CENTER_Y=3.5,
                         WALL_CENTER_X=2.5, WALL_CENTER_Y=3.5,
                         SAND_SOURCE_SELECTED_HULL_XY=hull.copy(),
                         SAND_SOURCE_POLYGON_XY=hull.copy(),
                         SAND_SOURCE_RAW_FACE_POLYGONS_XY=None, SAND_SOURCE_FACE_POLYGONS_XY=None,
                         root_path=lambda: '/World/SandSite',
                         get_stage=lambda: SimpleNamespace(GetPrimAtPath=lambda path:
                             SimpleNamespace(IsValid=lambda: False)),
                         rebuild_sand_reference_grid=Mock(), set_sand_generation_range_box=Mock(),
                         store_runtime_api=Mock())
        relocate = extract(BASE / 'scripts/excavator_app/sand_site_runtime.py',
                           'relocate_sand_source', namespace)
        for index in range(100):
            relocate([2.5 + index, 3.5 - index])
        np.testing.assert_allclose(namespace['SAND_SOURCE_SELECTED_HULL_XY'], hull + [99, -99])
        self.assertEqual((namespace['PILE_CENTER_X'], namespace['PILE_CENTER_Y']), (101.5, -95.5))
        self.assertIsNone(namespace['SAND_SOURCE_FACE_POLYGONS_XY'])
        namespace['store_runtime_api'].assert_called()


class SceneResetTests(unittest.TestCase):
    def setUp(self):
        self.initial = {'name': 'initial'}
        self.lifted = {'name': 'lifted'}
        self.particles = Mock()
        self.sand = SimpleNamespace(SAND_FLOOR_Z=0, SANDBOX_FILL_HEIGHT=3,
                                    PARTICLE_POINTS_PATH_SUFFIX='RealSandPoints', STATE={},
                                    relocate_sand_source=Mock(), sand_particle_points=Mock(
                                        return_value=([1, 2], [0, 0], [.1, .1], {})))
        self.namespace = dict(
            np=np, sand_enabled=True, args=SimpleNamespace(sand_wall=False, scene_seed=2, sand_settle_frames=3),
            timeline=Mock(), robot=Mock(), simulation_app=Mock(), world=Mock(),
            episode_initial_pose=self.initial, restore_excavator_pose_after_sand_init=Mock(),
            set_prim_translation_and_yaw=Mock(), stage=Mock(), TRUCK_ROOT_PATH='/World/truck',
            authored_truck_translation=[0, 0, 0], authored_truck_yaw=0,
            apply_randomized_dataset_scene=lambda: {'sand_xy': [5, 6]},
            elevate_excavator_for_sand_init=lambda *args: self.lifted,
            sand_module=self.sand, sand_api={}, builtins=SimpleNamespace(), sand_root_path='/World/SandSite',
            UsdGeom=SimpleNamespace(Points=lambda prim: self.particles),
            resolve_state27_dig_target=lambda: ([5, 6, 1], 'new-dig'),
            resolve_state27_unload_target=lambda: ([2, 3, 4], 'new-unload'),
            read_truck_yaw_rad=lambda: .4, hide_sand_source_guides=Mock(),
            truck_bed_counting_volume={'available': True, 'scene_seed': 2},
            authored_truck_bed_counting_volume=Mock(return_value={'available': True, 'scene_seed': 9}),
        )
        self.reset = extract(BASE / 'run_simulation.py', 'reset_evaluation_scene', self.namespace)

    def test_reset_reuses_particle_prim_and_restores_robot_and_new_targets(self):
        result = self.reset(9)
        self.assertEqual(result['scene_seed'], 9)
        self.assertEqual(result['particle_count'], 2)
        self.sand.relocate_sand_source.assert_called_once_with([5, 6])
        self.particles.GetPointsAttr().Set.assert_called_once_with([1, 2])
        self.particles.GetVelocitiesAttr().Set.assert_called_once_with([0, 0])
        self.assertEqual(self.namespace['world'].step.call_count, 3)
        self.assertEqual(self.namespace['state27_dig_target_world'], [5, 6, 1])
        self.assertEqual(self.namespace['truck_bed_counting_volume']['scene_seed'], 9)
        self.namespace['authored_truck_bed_counting_volume'].assert_called_once()
        restores = self.namespace['restore_excavator_pose_after_sand_init'].call_args_list
        self.assertEqual([call.args[0] for call in restores], [self.initial, self.lifted, self.initial])

    def test_each_reset_replaces_truck_bounds_and_rejects_invalid_geometry(self):
        self.namespace['authored_truck_bed_counting_volume'].side_effect = [
            {'available': True, 'scene_seed': 9, 'world_min': [10, 0, 0]},
            {'available': True, 'scene_seed': 12, 'world_min': [-10, 0, 0]},
            {'available': False, 'reason': 'missing_bed'},
        ]
        self.reset(9)
        self.assertEqual(self.namespace['truck_bed_counting_volume']['world_min'], [10, 0, 0])
        self.reset(12)
        self.assertEqual(self.namespace['truck_bed_counting_volume']['world_min'], [-10, 0, 0])
        with self.assertRaisesRegex(RuntimeError, 'missing_bed'):
            self.reset(13)

    def test_settling_error_restores_robot_before_propagating_failure(self):
        self.namespace['world'].step.side_effect = RuntimeError('physics failed')
        with self.assertRaisesRegex(RuntimeError, 'physics failed'):
            self.reset(9)
        self.namespace['restore_excavator_pose_after_sand_init'].assert_called_with(self.initial)

    def test_reset_invalidates_contract_history_and_source_mask_even_on_error(self):
        tree = ast.parse((BASE / 'run_simulation.py').read_text())
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "cmd.get('type') == 'reset_episode'")
        wrapper = ast.parse('for cmd in commands:\n    pass')
        wrapper.body[0].body = [branch]
        compiled = compile(ast.fix_missing_locations(wrapper), '<reset-command>', 'exec')
        for fail in (False, True):
            with self.subTest(fail=fail):
                old_fields = ('active_contract', 'active_connection_id', 'tick_scheduler', 'active_phase_estimator',
                              'deployment_previous_q', 'deployment_previous_load', 'deployment_previous_fill_fraction',
                              'deployment_previous_fill_rate', 'q_target')
                namespace = {name: object() for name in old_fields}
                namespace.update(commands=[{'type': 'reset_episode', 'scene_seed': 8}],
                                 time=SimpleNamespace(perf_counter=lambda: 0),
                                 response_queue=Mock(), bucket_source_tracker={'mask': 'old'},
                                 reset_evaluation_scene=Mock(side_effect=RuntimeError('reset failed')) if fail
                                     else Mock(return_value={'scene_seed': 8}))
                exec(compiled, namespace)
                for name in old_fields:
                    self.assertIsNone(namespace[name], name)
                self.assertEqual(namespace['bucket_source_tracker'], {})
                self.assertEqual(namespace['active_observation_context'], {})
                self.assertEqual(namespace['deployment_elapsed_seconds'], 0)
                reply = namespace['response_queue'].put.call_args.args[0]
                self.assertEqual(reply['type'], 'error' if fail else 'reset_episode_ack')


class LockstepPhysicsTests(unittest.TestCase):
    def test_headless_policy_wait_never_changes_renderer_history(self):
        tree = ast.parse((BASE / 'run_simulation.py').read_text())
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and 'last_idle_ui_update' in ast.unparse(node.test))
        namespace = dict(args=SimpleNamespace(headless=True), now=1e9, last_idle_ui_update=0,
                         idle_ui_interval=10., lock_visible_viewport_to_overview=Mock(),
                         simulation_app=Mock(), time=SimpleNamespace(sleep=Mock()))
        code = compile(ast.Module(body=[branch], type_ignores=[]), '<idle-render>', 'exec')
        exec(code, namespace)
        namespace['simulation_app'].update.assert_not_called()
        namespace['time'].sleep.assert_called_once()
        namespace['args'].headless = False
        exec(code, namespace)
        namespace['simulation_app'].update.assert_called_once()

    def execute_ticks(self, fail=False):
        source = (BASE / 'run_simulation.py').read_text()
        start = source.index('            physics_start = time.perf_counter()')
        end = source.index('            physics_done = time.perf_counter()', start)
        code = compile(textwrap.dedent(source[start:end]), '<lockstep-physics>', 'exec')

        class Timeline:
            playing = False
            requested = False

            def play(self):
                self.requested = True

            def pause(self):
                self.requested = False

            def commit(self):
                self.playing = self.requested

        timeline = Timeline()
        steps = []

        def step(**kwargs):
            self.assertTrue(timeline.playing, 'Physics must be committed before a physics-only tick')
            steps.append(kwargs)
            if fail:
                raise RuntimeError('tick failed')

        namespace = dict(time=SimpleNamespace(perf_counter=lambda: 0), timeline=timeline,
                         bridge_step_ticks=6, q_target=np.zeros(4), vel=np.ones(4), PHYSICS_DT=1 / 60,
                         canonical_joint_limits=None, canonical_joint_indices=[0, 1, 2, 3],
                         robot=Mock(), ArticulationAction=lambda **kwargs: kwargs,
                         args=SimpleNamespace(render_every_physics_step=False),
                         get_physx_simulation_interface=lambda: SimpleNamespace(fetch_results=lambda: None),
                         sync_physics_to_usd=Mock(),
                         world=SimpleNamespace(step=step, current_time=0), np=np)
        if fail:
            with self.assertRaisesRegex(RuntimeError, 'tick failed'):
                exec(code, namespace)
        else:
            exec(code, namespace)
            self.assertEqual(len(steps), 6)
            np.testing.assert_allclose(namespace['q_target'], [.1] * 4)
            self.assertEqual(steps, [{'render': False}] * 5 + [{'render': True}])
            namespace['sync_physics_to_usd'].assert_called_once()
        self.assertFalse(timeline.playing, 'Capturing images must not advance physics')

    def test_physics_only_ticks_commit_play_and_pause_without_extra_ticks(self):
        self.execute_ticks()

    def test_physics_error_still_commits_pause(self):
        self.execute_ticks(fail=True)


if __name__ == '__main__':
    unittest.main()
