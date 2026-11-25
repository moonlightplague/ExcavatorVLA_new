"""Prepare and run a small training-only Newton ACT recovery collection pilot."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from act_evaluation_storage import release_completed_caches, require_space
from eval_smolvla_repeat40 import atomic_json, digest, evaluation_environment, now, read

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs/experiments"
HDR = EXPERIMENTS / "act_hdr_baseline_repeatability_20261120_02"
SOURCE = EXPERIMENTS / "act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model"
DATA = EXPERIMENTS / "act_repeat_control_repair_data_20261118_01/dataset"
ORIGINAL = ROOT / "data/repeat/export_repeat"
RUNTIME = EXPERIMENTS / "act_training_alternatives_runtime_20261119_01"
GEOMETRY = ROOT / "outputs/analysis/act_reachability_20261103_01/robot_and_truck_model.json"
SCIPY_SITE = ROOT / ".venv-smolvla-history/lib/python3.12/site-packages"
COVERAGE = EXPERIMENTS / "act_timeout_research_followup_20261123_01/coverage_and_trajectories.json"
PLAN_ENV = "ACT_LOADED_RECOVERY_COLLECTION_PLAN"
DEPENDENCIES = ("act_loaded_recovery_collection.py", "act_recovery_expert.py",
                "excavator_reachability.py", "repeat_eval.py", "repeat_mass_physical.py",
                "success_eval_runtime.py", "eval_video.py")


def private_evaluator(output):
    path = HDR / "evaluator_single_hdr.py"
    text = path.read_text()

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f"Ambiguous frozen evaluator anchor: {old}")
        text = text.replace(old, new, 1)

    replace("CAMERA_IDS = ('0', '1', '2')", """
sys.path.append(str(Path(__file__).with_name('expert_dependencies')))
from scipy.optimize import least_squares
from act_loaded_recovery_collection import LoadedRecoveryCollection
if not os.environ.get('ACT_LOADED_RECOVERY_COLLECTION_PLAN'):
    raise RuntimeError('Training-collection plan required')

CAMERA_IDS = ('0', '1', '2')""".strip())
    replace("                report_progress('rollout', force=True)", """
                collection = LoadedRecoveryCollection(
                    Path(os.environ['ACT_LOADED_RECOVERY_COLLECTION_PLAN']), folder, scene, reference, repeated)
                report_progress('rollout', force=True)""".strip("\n"))
    replace("        started = time.monotonic()\n        try:\n            report_progress",
            "        started = time.monotonic()\n        collection = None\n        try:\n            report_progress")
    replace("execution_source='expert_replay' if args.replay_reference else 'model')",
            "execution_source='training_collection_only')")
    replace("    if min(args.episodes, args.steps, args.min_particles, args.replan_interval) < 1:",
            """    if (args.policy_type != 'act' or args.expert_prefix_steps or args.replay_reference
            or args.act_carry_guard or args.act_temporal_window or not args.save_trajectory
            or args.repeat_cycles != 10):
        parser.error('Invalid dedicated recovery-collection configuration')
    if min(args.episodes, args.steps, args.min_particles, args.replan_interval) < 1:""")
    replace("                    if args.replay_reference or prefix_action:",
            """                    collection_action, source = collection.action(step, state_pre, reply)
                    if source == 'collection_finished':
                        status['termination_reason'] = source
                        break
                    if collection_action is not None:
                        actions.clear()
                        actions = [collection_action]
                    elif args.replay_reference or prefix_action:""")
    replace("                                            'action_source': 'expert_prefix' if prefix_action else config['execution_source'],",
            "                                            'action_source': source,")
    replace("                    report_progress('rollout')\n                    frames = post_action_frames",
            """                    collection.observe(reply, source, clipped_dimensions)
                    report_progress('rollout')
                    frames = post_action_frames
                    if collection.complete:
                        status['termination_reason'] = 'recovery_collection_finished'
                        break""")
    replace("                    try:\n                        atomic_json(folder / 'repeat_audit.json', audit_repeat_trace(folder, repeat_spec, status))",
            """                    try:
                        status.update(success=False, task_completed_success_and_bucket_open=False,
                                      execution_source='training_collection_only', model_evaluation=False)
                        collection.save(status)""")
    replace("'status': 'passed', 'pure_model_execution_validated': config['execution_source'] == 'model',",
            "'status': 'training_collection_only', 'pure_model_execution_validated': False,")
    target = output / "collect_private.py"
    compile(text, str(target), "exec")
    target.write_text(text)
    return target


def prepare(output):
    import numpy as np
    import pyarrow.parquet as pq
    from baseline.coarse_acceptance import validate_mass_report
    from excavator_common.vla_observation_contract import ACTION_NAMES_4D, STATE_NAMES_28D
    from repeat_eval import load_repeat_spec

    output.mkdir(parents=True, exist_ok=False)
    manifest = read(HDR / "source_manifest.json")
    protected = {str(HDR / name): sha for name, sha in manifest["isolated"].items()}
    for path in (GEOMETRY, COVERAGE, DATA / "meta/info.json", DATA / "audit.json",
                 DATA / "control_replay_gate.json", Path(__file__),
                 *[ROOT / "scripts" / name for name in DEPENDENCIES],
                 *[path for path in SOURCE.iterdir() if path.is_file()]):
        protected[str(path)] = digest(path)
    assert_protected(protected)
    overlay = output / "expert_dependencies"
    overlay.mkdir()
    for dependency in ("scipy", "scipy.libs"):
        target = SCIPY_SITE / dependency
        if not target.is_dir():
            raise ValueError(f"Missing CPU geometry solver dependency: {target}")
        (overlay / dependency).symlink_to(target, target_is_directory=True)
    for path in (SCIPY_SITE / "scipy/__init__.py", *sorted((SCIPY_SITE / "scipy.libs").glob("*.so*"))):
        protected[str(path)] = digest(path)
    evaluator = private_evaluator(output)
    protected[str(evaluator)] = digest(evaluator)
    refs = output / "references"
    refs.mkdir()
    split_path = SOURCE / "benchmark_provenance.json"
    split = read(split_path)["split_manifest"]
    candidates = read(COVERAGE)["training_data_coverage"]["training_parent_candidates_for_recovery_collection"]
    parents = [row["source_episode"] for row in candidates[:3]]
    if len(set(parents)) != 3 or any(i not in split["train"] or i in split["validation"] + split["test"] for i in parents):
        raise ValueError("Recovery pilot parents must be disjoint from validation and test")
    tasks = pq.read_table(DATA / "meta/tasks.parquet").to_pandas()
    metadata = read(DATA / "meta/info.json")
    receipts = {}
    for path in (ROOT / "data/repeat/raw").glob("run_*/episode_*/export.json"):
        receipt = read(path)
        parent = receipt["export_episode_index"]
        if parent in parents and receipt.get("validated") and Path(receipt["export_dir"]).resolve() == ORIGINAL.resolve():
            if parent in receipts:
                raise ValueError(f"Duplicate training receipt for parent {parent}")
            receipts[parent] = path
    jobs = []
    for index, parent in enumerate(parents):
        provenance = DATA / f"provenance/episode_{parent:04d}"
        receipt = receipts[parent]
        source_meta = receipt.with_name("meta.json")
        if digest(source_meta) != digest(provenance / "meta.json"):
            raise ValueError(f"Training parent {parent} metadata changed")
        meta = read(source_meta)
        metrics = source_meta.with_name("repeat_metrics.json")
        quality = source_meta.parent.parent / "quality_gate_config.json"
        validate_mass_report(meta, read(metrics))
        frame = DATA / metadata["data_path"].format(chunk_index=parent, file_index=0)
        first = pq.read_table(frame, columns=["frame_index", "episode_index", "task_index", "observation.state"],
                              filters=[("frame_index", "=", 0), ("episode_index", "=", parent)]).to_pylist()
        if len(first) != 1 or not read(receipt).get("validated") or Path(read(receipt)["export_dir"]).resolve() != ORIGINAL.resolve():
            raise ValueError(f"Invalid original training parent {parent}")
        sample = first[0]
        state = sample["observation.state"]
        scene = meta["scene_randomization"]["applied"]
        goal = meta["scoop_results"][0]
        evidence = {str(path): digest(path) for path in (source_meta, receipt, quality, metrics, split_path)}
        protected.update(evidence)
        reference_path = refs / f"parent_{parent:04d}.json"
        atomic_json(reference_path, {
            "schema_version": "excavator_scene_setup_v2", "fps": metadata["fps"], "action_units": "rad/s",
            "state_names": STATE_NAMES_28D, "action_names": ACTION_NAMES_4D,
            "profile": {"sand_xy": scene["sand_center"][:2], "truck_translation_xyz": scene["truck_translation_xyz"],
                        "truck_yaw_deg": scene["truck_yaw_deg"], "unload_landing_xyz": goal["unload_landing_xyz"]},
            "task": str(tasks.index[int(sample["task_index"])]), "initial_joints_deg": np.degrees(state[:4]).tolist(),
            "expected_initial_state": state, "dig_target_xyz": goal["target_xyz"],
            "sand_amount": scene["sand_amount_multiplier"],
            "provenance": {"source_export_episode": parent, "source_parent": meta["episode_id"],
                           "source_meta": str(source_meta), "source_receipt": str(receipt),
                           "source_quality_gate": str(quality), "source_evidence": evidence,
                           "source_scene_seed": meta["repeat_baseline"]["scene_seed"]},
        })
        spec = load_repeat_spec(DATA, reference_path, 10, 0, 100)
        if spec["minimum_per_cycle_kg"] != 50:
            raise ValueError("Unexpected Newton per-cycle mass threshold")
        plan = refs / f"parent_{parent:04d}_collection.json"
        atomic_json(plan, {"parent": parent, "split": str(split_path), "robot_geometry": str(GEOMETRY),
                           "controlled_overshoot": index == len(parents) - 1})
        protected[str(plan)] = digest(plan)
        protected[str(reference_path)] = digest(reference_path)
        jobs.append({"parent": parent, "reference": str(reference_path), "plan": str(plan),
                     "seed": int(meta["repeat_baseline"]["scene_seed"]), "controlled_overshoot": index == len(parents) - 1})
    atomic_json(output / "manifest.json", {
        "prepared_utc": now(), "jobs": jobs, "protected_sha256": protected,
        "baseline_checkpoint": str(SOURCE), "source_data": str(ORIGINAL),
        "scope": "Training-only physical pilot. Prefix and controlled deviations have zero action supervision.",
    })
    atomic_json(output / "status.json", {"status": "preflight_passed", "planned_training_parents": parents})
    return jobs, protected


def assert_protected(files):
    for path, sha in files.items():
        if digest(path) != sha:
            raise ValueError(f"Protected input changed: {path}")


def idle_gpu(output, parent):
    while True:
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
        occupancy = {}
        for row in inventory.strip().splitlines():
            uuid, used = (entry.strip() for entry in row.split(","))
            occupancy[uuid] = int(used)
        for uuid, used in occupancy.items():
            if used >= 1500:
                continue
            processes = subprocess.check_output(
                ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
            if not processes.strip():
                return uuid
        atomic_json(output / "status.json", {"status": "waiting_for_idle_gpu", "parent": parent,
                                             "gpu_occupied_mib": occupancy, "updated_utc": now()})
        time.sleep(30)


def run(output, jobs, protected):
    results = []
    for job in jobs:
        gpu = idle_gpu(output, job["parent"])
        assert_protected(protected)
        require_space(output, gib=18)
        folder = output / f"scene_{job['parent']:04d}"
        folder.mkdir()
        env = evaluation_environment(folder)
        env.update({PLAN_ENV: job["plan"], "CUDA_VISIBLE_DEVICES": gpu,
                    "EXCAVATOR_EVAL_MATERIAL_SNAPSHOTS": "0"})
        command = [str(RUNTIME / "python"), "-B", "-u", str(output / "collect_private.py"), "--policy-type", "act",
                   "--checkpoint", str(SOURCE), "--dataset", str(DATA), "--output", str(folder / "evaluation"),
                   "--reference-episode", job["reference"], "--episodes", "1", "--repeat-cycles", "10",
                   "--repeat-cycle-steps", "0", "--min-particles", "100", "--steps", "1600",
                   "--replan-interval", "3", "--seed", str(job["seed"]), "--gpus", gpu, "--workers", "1",
                   "--soil-backend", "newton_mpm", "--cpu-threads", "8", "--headless", "--video",
                   "--save-trajectory", "--fresh-simulator", "--render-every-physics-step"]
        atomic_json(folder / "launch.json", {"command": command, "job": job, "gpu": gpu})
        with (folder / "launch.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            atomic_json(output / "status.json", {"status": "collecting", "parent": job["parent"],
                                                 "gpu": gpu, "child_pid": child.pid, "started_utc": now()})
            code = child.wait()
        atomic_json(folder / "launch_result.json", {"exit_code": code, "finished_utc": now()})
        episode = folder / "evaluation/episodes/episode_0001"
        proof = episode / "recovery_supervision.json"
        if code or not proof.is_file():
            raise RuntimeError(f"Collection infrastructure failed for parent {job['parent']}; see {folder / 'launch.log'}")
        result = read(proof)
        results.append({"parent": job["parent"], **result})
        atomic_json(output / "results.json", results)
        release_completed_caches(output)
    atomic_json(output / "status.json", {"status": "pilot_completed", "results": str(output / "results.json"),
                                         "accepted_parents": [x["parent"] for x in results
                                                              if x["accepted_for_recovery_dataset"]],
                                         "finished_utc": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.run_prepared:
        manifest = read(output / "manifest.json")
        jobs, protected = manifest["jobs"], manifest["protected_sha256"]
    else:
        jobs, protected = prepare(output)
    if not args.prepare_only:
        try:
            run(output, jobs, protected)
        except Exception as error:
            atomic_json(output / "status.json", {"status": "collection_error", "error": repr(error),
                                                 "updated_utc": now()})
            raise
