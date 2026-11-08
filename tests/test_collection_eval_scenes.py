"""Prevent changed targets or skipped collection checks in matched evaluations."""
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collection_eval_scenes import load_scene_manifest, simulator_command, SELECTOR, SCHEMA
from baseline.prepare_eval_scenes import install_capture


def manifest():
    return dict(schema_version=SCHEMA, selector=SELECTOR, soil_backend="physx", scenes=[dict(
        episode_index=1, scene_seed=123, selector=SELECTOR, soil_backend="physx",
        expert_actions_executed=False, sand_amount=1.0, task="Load the truck.",
        initial_joints_deg=[30., 46., -62., -20.], dig_target_xyz=[0., 6., .1],
        profile=dict(sand_xy=[0., 6.], truck_translation_xyz=[-8., -4., 0.],
                     truck_yaw_deg=80., unload_landing_xyz=[-7., -4., 4.2]),
        selection=dict(cell_id=7, radius_ok=True, dump_preflight_ok=True,
                       dump_release_error_m=.01, landing=[-7., -4., 4.2]))])


class ManifestTests(unittest.TestCase):
    def load(self, data, count=1, backend="physx"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenes.json"
            path.write_text(json.dumps(data))
            return load_scene_manifest(path, count, backend)

    def test_requires_both_reach_and_ik_checks_and_matching_selected_target(self):
        for key, value in [("radius_ok", False), ("dump_preflight_ok", False),
                           ("dump_release_error_m", .11), ("landing", [-8., -4., 4.2])]:
            data = manifest()
            data["scenes"][0]["selection"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(data)

    def test_rejects_different_backend_nonfinite_inputs_and_expert_execution(self):
        for field, value in [("expert_actions_executed", True), ("sand_amount", 2.),
                             ("initial_joints_deg", [0., 1., 2., float("nan")])]:
            data = manifest()
            data["scenes"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(data)
        with self.assertRaises(ValueError):
            self.load(manifest(), backend="newton_mpm")

    def test_rejects_missing_and_duplicate_trials(self):
        with self.assertRaises(ValueError):
            self.load(manifest(), count=2)
        data = manifest()
        other = copy.deepcopy(data["scenes"][0]); other["episode_index"] = 2
        data["scenes"].append(other)
        with self.assertRaisesRegex(ValueError, "unique"):
            self.load(data, count=2)

    def test_manifest_hash_changes_with_the_chosen_scene(self):
        data = manifest()
        first = self.load(data)[1]
        data["scenes"][0]["scene_seed"] += 1
        self.assertNotEqual(first, self.load(data)[1])

    def test_commands_use_each_scene_pose_and_selected_target_without_mutating_base(self):
        scene = manifest()["scenes"][0]
        base = ["isaac", "run.py", "--sand-amount", "1"]
        with tempfile.TemporaryDirectory() as directory:
            command = simulator_command(base, scene, directory)
            self.assertEqual(base, ["isaac", "run.py", "--sand-amount", "1"])
            start = command.index("--state27-unload-target-world")
            self.assertEqual(list(map(float, command[start+1:start+4])), scene["selection"]["landing"])
            profile = json.loads((Path(directory) / "scene_profile.json").read_text())
            self.assertEqual(profile["collection_selection"], scene["selection"])
            self.assertEqual(profile["target_selection_method"], SELECTOR)


class PreparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_feasible_plan_becomes_a_valid_scene_without_an_executable_sequence(self):
        @dataclass
        class Config:
            scoops: int = 5

        scene = manifest()["scenes"][0]
        profile = scene["profile"]
        rt = SimpleNamespace(
            STATE=dict(auto_collect_attempts=1, auto_scene_last_randomization=dict(applied=dict(
                sand_center=[*profile["sand_xy"], 0.],
                truck_translation_xyz=profile["truck_translation_xyz"],
                truck_yaw_deg=profile["truck_yaw_deg"], sand_amount_multiplier=1.0))),
            auto_collect_find_plan=AsyncMock(return_value=(np.array(scene["dig_target_xyz"]),
                [("pre_dig", np.zeros(4), 1.), ("dump", np.ones(4), 1.)], [])),
            auto_collect_record_planning_diagnostic=Mock(),
            get_real_joint_positions=lambda: np.radians(scene["initial_joints_deg"]),
            json_sanitize=lambda value: value)
        controller = SimpleNamespace(selected=scene["selection"], config=Config())
        with tempfile.TemporaryDirectory() as directory:
            install_capture(rt, controller, directory, count=1, seed_start=123, seed_stride=2)
            _target, executable, _details = await rt.auto_collect_find_plan(1)
            self.assertIsNone(executable, "The normal collector must never execute the prepared plan")
            self.assertTrue(rt.STATE["auto_collect_stop_requested"])
            path = Path(directory) / "scenes.json"
            data = json.loads(path.read_text())
            data["scenes"][0]["episode_index"] = 1
            path.write_text(json.dumps(data))
            loaded, _ = load_scene_manifest(path, 1, "physx")
            self.assertEqual(loaded[0]["selection"]["landing"], scene["selection"]["landing"])
            self.assertFalse(loaded[0]["expert_actions_executed"])


if __name__ == "__main__":
    unittest.main()
