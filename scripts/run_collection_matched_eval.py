"""Prepare new collection-distribution scenes, then evaluate a frozen policy on two GPUs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collection_eval_scenes import load_scene_manifest, SCHEMA, SELECTOR
from success_eval_runtime import atomic_json, file_manifest, resolve_gpu_ids, stop_simulator


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def freeze(job):
    frozen = job / "frozen_sources"
    frozen.mkdir()
    ignore = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache",
                                    ".mpm_warp_cache", ".cache", "*.log")
    for name in ("ExcavatorVLA-smolvla", "ExcavatorVLA-vlm", "baseline"):
        shutil.copytree(ROOT / name, frozen / name, ignore=ignore)
    isaac = frozen / "ExcavatorVLA-isaac"
    isaac.mkdir()
    for name in ("assets", "scripts", "excavator_common"):
        shutil.copytree(ROOT / "ExcavatorVLA-isaac" / name, isaac / name, ignore=ignore)
    for pattern in ("*.py", "*.json"):
        for path in (ROOT / "ExcavatorVLA-isaac").glob(pattern):
            shutil.copy2(path, isaac / path.name)
    (frozen / "scripts").mkdir()
    for path in (ROOT / "scripts").glob("*.py"):
        shutil.copy2(path, frozen / "scripts" / path.name)
    for name in ("run_isaac_python.sh", "training_config.yaml"):
        shutil.copy2(ROOT / name, frozen / name)
    (frozen / "isaacsim").symlink_to((ROOT / "isaacsim").resolve(), target_is_directory=True)
    sources = file_manifest([p for p in frozen.rglob("*") if p.is_file() and not p.is_symlink()], frozen)
    for name, digest in sources.items():
        if file_manifest([ROOT / name], ROOT)[name] != digest:
            raise ValueError(f"Source changed while freezing: {name}")
    atomic_json(job / "source_snapshot.json", dict(root=str(frozen), files=sources))
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026110300)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-preparation-attempts", type=int, default=500)
    args = parser.parse_args()
    if min(args.episodes, args.max_preparation_attempts) < 1 or args.seed < 0:
        parser.error("Episode/attempt counts must be positive and seed nonnegative")
    args.job = args.job.resolve(); args.output = args.output.resolve()
    args.dataset = args.dataset.resolve(); args.checkpoint = args.checkpoint.resolve()
    if args.job.exists() or args.output.exists():
        parser.error("Job/evaluation output already exists")
    args.job.mkdir(parents=True)
    gpus = resolve_gpu_ids(args.gpus)
    workers = min(len(gpus), args.episodes)
    state = dict(status="running", phase="preflight", pid=os.getpid(), created_utc=now(),
                 checkpoint=str(args.checkpoint), checkpoint_step=6000, dataset=str(args.dataset),
                 episodes=args.episodes, gpu_ids=gpus[:workers], output=str(args.output),
                 selector=SELECTOR, scene_sampling="collection_area_uniform_polar_v2",
                 initial_pose="collection_repeat_pile_facing", soil_backend="physx",
                 task="same single-cycle instruction as training", action_chunk_size=10,
                 execute_steps=3, rollout_step_limit=350, min_particles=100,
                 scope="New single-cycle scenes; collection preparation and planner feasibility only. "
                       "No expert digging actions or policy-outcome selection. Not repeated digging.")
    processes = []
    logs = []

    def progress():
        state["updated_utc"] = now()
        preparation = {}
        for index in range(workers):
            path = args.job / f"preparation/worker_{index:02d}/progress.json"
            if path.exists(): preparation[f"worker_{index:02d}"] = read(path)
        state["preparation"] = preparation
        state["prepared_scenes"] = sum(p["prepared_scenes"] for p in preparation.values())
        summary_path = args.output / "success_rate_summary.json"
        if summary_path.exists():
            summary = read(summary_path)
            state["evaluation_progress"] = {k: summary.get(k) for k in (
                "completed_episodes", "successes", "failures", "infrastructure_error_episodes",
                "official_result_valid", "official_success_rate", "success_rate_over_completed_episodes")}
        atomic_json(args.job / "status.json", state)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    progress()
    try:
        sys.path.insert(0, str(ROOT / "ExcavatorVLA-vlm"))
        from excavator_vlm.data.contract import audit_dataset
        audit, _ = audit_dataset(args.dataset, 10)
        provenance = read(args.checkpoint / "benchmark_provenance.json")
        if (audit["fingerprint"] != provenance["dataset_fingerprint"]
                or audit["collection_backend"]["soil_backend"] != "physx"):
            raise ValueError("Dataset differs from checkpoint or lacks verified PhysX collection")
        checkpoint = read(args.checkpoint / "checkpoint.json")
        digest = file_manifest([args.checkpoint / "model.safetensors"], args.checkpoint)["model.safetensors"]
        if checkpoint["step"] != 6000 or digest != checkpoint["delta_sha256"]:
            raise ValueError("6k checkpoint step/hash differs")
        state.update(dataset_fingerprint=audit["fingerprint"], checkpoint_delta_sha256=digest,
                     phase="freezing_sources")
        progress()
        frozen = freeze(args.job)
        state["phase"] = "preparing_collection_scenes"
        progress()
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4",
                       "OPENBLAS_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        counts = [len(range(index, args.episodes, workers)) for index in range(workers)]
        for index, gpu in enumerate(gpus[:workers]):
            folder = args.job / f"preparation/worker_{index:02d}"
            folder.parent.mkdir(exist_ok=True)
            log = (folder.parent / f"worker_{index:02d}.log").open("x"); logs.append(log)
            command = [str(frozen / "run_isaac_python.sh"), str(frozen / "baseline/prepare_eval_scenes.py"),
                       "--output", str(folder), "--count", str(counts[index]),
                       "--seed-start", str(args.seed + index), "--seed-stride", str(workers),
                       "--max-attempts", str(args.max_preparation_attempts)]
            processes.append(subprocess.Popen(command, cwd=frozen, env={**environment, "CUDA_VISIBLE_DEVICES": gpu},
                                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        state["preparation_pids"] = [p.pid for p in processes]
        while any(p.poll() is None for p in processes):
            progress()
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("Collection scene preparation worker failed; inspect preparation logs")
            time.sleep(10)
        scenes = []
        per_worker = []
        for index in range(workers):
            data = read(args.job / f"preparation/worker_{index:02d}/scenes.json")
            if len(data["scenes"]) != counts[index]:
                raise ValueError("Preparation did not produce the requested number of feasible scenes")
            per_worker.append(data["scenes"])
        for index in range(args.episodes):
            scene = dict(per_worker[index % workers][index // workers], episode_index=index + 1)
            scenes.append(scene)
        manifest = args.job / "scene_manifest.json"
        atomic_json(manifest, dict(schema_version=SCHEMA, selector=SELECTOR, soil_backend="physx",
                    dataset_fingerprint=audit["fingerprint"], scope=state["scope"], scenes=scenes))
        _, manifest_hash = load_scene_manifest(manifest, args.episodes, "physx")
        state.update(phase="evaluating", scene_manifest=str(manifest), scene_manifest_sha256=manifest_hash)
        progress()
        command = [sys.executable, "-u", str(frozen / "ExcavatorVLA-vlm/scripts/eval_pi05_success.py"),
                   "--checkpoint", str(args.checkpoint), "--dataset", str(args.dataset),
                   "--scene-manifest", str(manifest), "--episodes", str(args.episodes),
                   "--steps", "350", "--min-particles", "100", "--seed", str(args.seed),
                   "--replan-interval", "3", "--gpus", args.gpus, "--workers", str(workers),
                   "--cpu-threads", "8", "--headless", "--fresh-simulator", "--render-every-physics-step",
                   "--video", "--soil-backend", "physx", "--sand-walls-during-settle", "--output", str(args.output)]
        state["command"] = command
        log = (args.job / "evaluation.log").open("x"); logs.append(log)
        process = subprocess.Popen(command, cwd=frozen, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process); state["evaluation_pid"] = process.pid
        while process.poll() is None:
            progress(); time.sleep(10)
        summary = read(args.output / "success_rate_summary.json")
        if process.returncode or not summary["official_result_valid"]:
            raise RuntimeError("Evaluation incomplete; inspect logs and retain all failed/error trials")
        for scene in scenes:
            episode = args.output / f"episodes/episode_{scene['episode_index']:04d}"
            result = read(episode / "episode_status.json")
            setup = read(episode / "collection_setup.json")
            if (result["scene_seed"] != scene["scene_seed"] or setup != scene
                    or result["scene_manifest_sha256"] != manifest_hash
                    or not (episode / "evaluation.mp4").stat().st_size):
                raise ValueError("Episode setup/seed/manifest/video differs from the prepared scene")
        state.update(status="completed", phase="completed", completed_utc=now(),
                     success_rate=summary["official_success_rate"])
        atomic_json(args.job / "results.json", summary)
        progress()
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        progress()
        raise
    finally:
        for process in processes:
            if process.poll() is None: stop_simulator(process)
        for log in logs: log.close()


if __name__ == "__main__":
    main()
