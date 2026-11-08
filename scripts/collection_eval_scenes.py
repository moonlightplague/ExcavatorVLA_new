"""Validation and simulator commands for collection-prepared evaluation scenes."""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "collection_eval_scenes_v1"
SELECTOR = "repeat_flat_fill_v2"


def finite_vector(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Invalid {name}")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f"Nonfinite/nonnumeric {name}")
    return value


def load_scene_manifest(path, episodes, soil_backend):
    path = Path(path)
    payload = path.read_bytes()
    data = json.loads(payload)
    if data.get("schema_version") != SCHEMA or data.get("selector") != SELECTOR:
        raise ValueError("Expected collection-prepared scenes and the repeat_flat_fill_v2 selector")
    if soil_backend != "physx" or data.get("soil_backend") != soil_backend:
        raise ValueError("Collection manifest and evaluation must both use PhysX")
    scenes = data["scenes"]
    if len(scenes) != episodes:
        raise ValueError("Scene manifest count differs from --episodes")
    seeds = set()
    for index, scene in enumerate(scenes, 1):
        seed = scene.get("scene_seed")
        if type(seed) is not int or not 0 <= seed < 2**63 or seed in seeds:
            raise ValueError("Scene seeds must be unique nonnegative signed 64-bit integers")
        seeds.add(seed)
        if scene.get("episode_index") != index:
            raise ValueError("Scene manifest must have consecutive episode indices starting at 1")
        if scene.get("selector") != SELECTOR or scene.get("soil_backend") != soil_backend:
            raise ValueError("Scene selector/backend differs from the batch")
        if scene.get("expert_actions_executed") is not False:
            raise ValueError("Evaluation preparation must not execute expert digging actions")
        selection = scene["selection"]
        if selection.get("radius_ok") is not True or selection.get("dump_preflight_ok") is not True:
            raise ValueError("Scene lacks successful collection reach/IK preflight")
        if not math.isfinite(selection.get("dump_release_error_m", float("inf"))) or selection["dump_release_error_m"] > .10:
            raise ValueError("Scene dump preflight error exceeds the collection threshold")
        profile = scene["profile"]
        for key, size in (("sand_xy", 2), ("truck_translation_xyz", 3), ("unload_landing_xyz", 3)):
            finite_vector(profile[key], size, key)
        finite_vector(scene["dig_target_xyz"], 3, "dig_target_xyz")
        finite_vector(scene["initial_joints_deg"], 4, "initial_joints_deg")
        finite_vector(selection["landing"], 3, "selection.landing")
        if max(abs(a-b) for a,b in zip(profile["unload_landing_xyz"], selection["landing"])) > 1e-6:
            raise ValueError("Scene target differs from the selected collection landing")
        if scene.get("sand_amount") != 1.0 or not scene.get("task"):
            raise ValueError("Expected fixed sand amount 1.0 and a task instruction")
        finite_vector([profile["truck_yaw_deg"]], 1, "truck_yaw_deg")
    return scenes, hashlib.sha256(payload).hexdigest()


def simulator_command(base_command, scene, folder):
    """Keep a persistent policy worker while replacing each fresh scene's setup."""
    folder = Path(folder)
    profile = dict(scene["profile"], collection_selection=scene["selection"],
                   target_selection_method=SELECTOR,
                   initial_pose_source="collection_repeat_pile_facing")
    path = folder / "scene_profile.json"
    path.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    (folder / "collection_setup.json").write_text(json.dumps(scene, indent=2, allow_nan=False) + "\n")
    return [*base_command, "--robot-initial-joints-deg", *map(str, scene["initial_joints_deg"]),
            "--scene-profile", str(path), "--state27-dig-target-world", *map(str, scene["dig_target_xyz"]),
            "--state27-unload-target-world", *map(str, profile["unload_landing_xyz"]),
            "--task-text", scene["task"]]
