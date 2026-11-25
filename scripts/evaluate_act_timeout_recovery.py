"""Pair a recovery checkpoint with velocity 8k on the five timeouts and scene 112."""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from act_evaluation_storage import release_completed_caches, require_space
from audit_act_model_only import audit_run
from diagnose_act_timeouts import analyze
from eval_smolvla_repeat40 import atomic_json, digest, evaluation_environment, now, read
from train_act_swing_recovery import ROOT, SOURCE

EXPERIMENTS = ROOT / "outputs/experiments"
HISTORY = EXPERIMENTS / "act_velocity_holdout8_20261119_01"
HDR = EXPERIMENTS / "act_hdr_baseline_repeatability_20261120_02"
RUNTIME = EXPERIMENTS / "act_training_alternatives_runtime_20261119_01"
LANES = (
    ("GPU-36fd8a97-912c-376e-4db4-a99c0c782d59", (162, 80, 152)),
    ("GPU-337b7e31-96ff-1982-3593-c5e72801d4fd", (143, 112, 50)),
)


def command(scene, gpu, checkpoint, folder):
    original = read(HISTORY / f"scene_{scene:04d}/launch.json")["command"]
    result = original.copy()
    evaluator = str(ROOT / "scripts/eval_smolvla_success.py")
    if result.count(evaluator) != 1:
        raise ValueError("Recorded evaluator is ambiguous")
    result[result.index(evaluator)] = str(HDR / "evaluator_single_hdr.py")
    for flag, value in {"--checkpoint": checkpoint, "--output": folder / "evaluation", "--gpus": gpu}.items():
        if result.count(flag) != 1:
            raise ValueError(f"Expected one {flag}")
        result[result.index(flag) + 1] = str(value)
    if result[result.index("--steps") + 1] != "5000":
        raise ValueError("Unexpected historical budget")
    return result


def prepare(output, checkpoint):
    from repeat_eval import load_repeat_spec
    manifest = read(HDR / "source_manifest.json")
    protected = {HDR / key: sha for key, sha in manifest["isolated"].items()}
    for name in ("repeat_eval.py", "repeat_mass_physical.py", "success_eval_runtime.py",
                 "eval_video.py", "evaluation_action_transfer.py", "audit_act_model_only.py",
                 "diagnose_act_timeouts.py"):
        path = ROOT / "scripts" / name
        protected[path] = digest(path)
    for model in (SOURCE, checkpoint):
        for file in model.iterdir():
            if file.is_file():
                protected[file] = digest(file)
    split = read(SOURCE / "benchmark_provenance.json")["split_manifest"]
    jobs = []
    for gpu, scenes in LANES:
        for scene in scenes:
            if scene not in split["test"] or scene in split["train"] + split["validation"]:
                raise ValueError(f"Unexpected split assignment: {scene}")
            arms = [("baseline", SOURCE), ("candidate", checkpoint)]
            if scene in (143, 80, 50):
                arms.reverse()
            for arm, model in arms:
                folder = output / f"scene_{scene:04d}" / arm / "scene_run"
                argv = command(scene, gpu, model, folder)
                pick = lambda flag: argv[argv.index(flag) + 1]
                ref = Path(pick("--reference-episode"))
                protected[ref] = digest(ref)
                jobs.append({"scene": scene, "gpu": gpu, "arm": arm, "command": argv, "folder": str(folder)})
            spec = load_repeat_spec(pick("--dataset"), str(ref), 10, 0, int(pick("--min-particles")))
            if spec["cycles"] != 10 or spec["min_payload_particles"] <= 0:
                raise ValueError(f"Invalid historical task contract: {scene}")
    return {"jobs": jobs, "protected_sha256": {str(key): sha for key, sha in protected.items()},
            "same_budget_steps": 5000, "lighting_profile": manifest["profile"],
            "scope": "Paired development runs on the five diagnosed timeouts with scene 112 as a regression check."}


def assert_protected(manifest):
    for path, sha in manifest["protected_sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"Frozen experiment dependency changed: {path}")


def wait_for_gpu(gpu, folder):
    while True:
        memory = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        active = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        if int(memory) < 2048 and not active:
            return
        atomic_json(folder / "status.json", {"status": "waiting_for_idle_gpu", "gpu": gpu,
                                             "occupied_mib": int(memory), "updated_utc": now()})
        time.sleep(30)


def run_lane(gpu, manifest):
    for job in manifest["jobs"]:
        if job["gpu"] != gpu:
            continue
        folder = Path(job["folder"])
        folder.mkdir(parents=True, exist_ok=False)
        try:
            wait_for_gpu(gpu, folder)
            assert_protected(manifest)
            require_space(folder, gib=18)
            env = evaluation_environment(folder)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONPATH"] = str(RUNTIME) + ":" + str(ROOT / "ExcavatorVLA-vlm")
            cudnn = Path(job["command"][0]).parents[1] / "lib/python3.12/site-packages/nvidia/cudnn/lib"
            env["LD_LIBRARY_PATH"] = str(cudnn) + ":" + env.get("LD_LIBRARY_PATH", "")
            atomic_json(folder / "launch.json", job)
            with (folder / "launch.log").open("w") as log:
                process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                atomic_json(folder / "status.json", {"status": "evaluating", "pid": process.pid})
                code = process.wait()
            atomic_json(folder / "launch_result.json", {"exit_code": code})
            if code:
                raise RuntimeError(f"Evaluator exited with status {code}")
            episode = folder / "evaluation/episodes/episode_0001"
            if read(episode / "episode_status.json")["gpu_id"] != gpu:
                raise ValueError("Unexpected physical GPU")
            audit = audit_run(folder)
            atomic_json(folder / "model_actions.json", audit)
            if audit["status"] != "passed" or read(episode / "repeat_audit.json")["status"] != "passed":
                raise ValueError("Action or physical-cycle audit failed")
            atomic_json(folder / "diagnosis.json", analyze(episode))
            release_completed_caches(folder.parent)
            atomic_json(folder / "status.json", {"status": "completed", "completed_utc": now()})
        except Exception as error:
            atomic_json(folder / "status.json", {"status": "evaluation_error", "error": repr(error)})
            raise


def summarize(output, manifest):
    scenes = {}
    for job in manifest["jobs"]:
        folder = Path(job["folder"])
        status = folder / "status.json"
        record = read(status) if status.is_file() else {"status": "pending"}
        if record["status"] == "completed":
            analysis = read(folder / "diagnosis.json")
            record.update({key: analysis[key] for key in ("steps", "completed", "successful", "task_success",
                                                           "terminal_phase", "cycles")})
        scenes.setdefault(str(job["scene"]), {})[job["arm"]] = record
    completed = sum(record["status"] == "completed" for pair in scenes.values() for record in pair.values())
    paired = {}
    for scene, pair in scenes.items():
        if all(item["status"] == "completed" for item in pair.values()):
            baseline, candidate = pair["baseline"], pair["candidate"]
            paired[scene] = {"completed_cycle_change": candidate["completed"] - baseline["completed"],
                             "successful_cycle_change": candidate["successful"] - baseline["successful"],
                             "ten_cycle_task_success": [baseline["task_success"], candidate["task_success"]]}
    report = {"completed_runs": completed, "total_runs": len(manifest["jobs"]),
              "scenes": scenes, "paired_changes": paired, "baseline_replaced": False}
    atomic_json(output / "summary.json", report)
    return report


def main(output, checkpoint, check_only):
    manifest = prepare(output, checkpoint)
    assert_protected(manifest)
    if check_only:
        print(f"Validated {len(manifest['jobs'])} paired full-budget jobs")
        return
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "status.json", {"status": "queued_or_evaluating", "pid": os.getpid()})
    errors = []
    with ThreadPoolExecutor(max_workers=len(LANES)) as pool:
        jobs = {pool.submit(run_lane, gpu, manifest): gpu for gpu, _ in LANES}
        while jobs:
            summarize(output, manifest)
            for future in [future for future in jobs if future.done()]:
                gpu = jobs.pop(future)
                error = future.exception()
                if error:
                    errors.append({"gpu": gpu, "error": repr(error)})
            if jobs:
                time.sleep(15)
    report = summarize(output, manifest)
    atomic_json(output / "status.json", {"status": "evaluation_error" if errors else "completed",
                                         "completed_runs": report["completed_runs"],
                                         "errors": errors, "updated_utc": now(), "baseline_replaced": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    main(args.output.resolve(), args.checkpoint.resolve(), args.check_only)
