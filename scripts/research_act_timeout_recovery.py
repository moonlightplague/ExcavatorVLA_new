"""Continue ACT timeout research after both requested pi0.5 rollouts end."""

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs/experiments"
DEPENDENCY = EXPERIMENTS / "pi05_old_physx10_new_test1_20261123_01"
BASELINE = EXPERIMENTS / "act_velocity_holdout8_20261119_01"
INTERRUPTED = EXPERIMENTS / "act_timeout_recovery_20261123_01/paired_hdr"
SOURCE = EXPERIMENTS / "act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model"
DATA = EXPERIMENTS / "act_repeat_control_repair_data_20261118_01/dataset"
TERMINAL = {"completed", "error", "stopped_by_user"}


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def trace_evidence(folder, completed_cycles):
    episode = folder / "evaluation/episodes/episode_0001"
    lower, upper = read(episode / "scene_context.json")["canonical_joint_limits_rad"][0]
    with (episode / "steps.jsonl").open() as trace:
        rows = [json.loads(line) for line in trace]
    final_goal = [row for row in rows if row["repeat_cycle_index"] == completed_cycles]
    if not final_goal:
        raise ValueError(f"No unfinished cycle in {episode}")
    outward = [
        row for row in final_goal
        if row["repeat_bucket_particles"] > 0 and (
            (row["input_state"][0] <= lower + .035 and row["applied"][0] < -.025)
            or (row["input_state"][0] >= upper - .035 and row["applied"][0] > .025)
        )
    ]
    before_limit = [row for row in final_goal if not outward or row["step"] < outward[0]["step"]]
    loaded_approach = [row for row in before_limit if row["repeat_bucket_particles"] >= 10]

    def sample(row):
        return {
            "step": row["step"], "swing_position_rad": row["input_state"][0],
            "swing_velocity_command_rad_s": row["applied"][0],
            "unload_delta_xy_m": row["input_state"][22:24],
            "bucket_particles": row["repeat_bucket_particles"],
        }

    closest = min(loaded_approach, key=lambda row: np.linalg.norm(row["input_state"][22:24]), default=None)
    tail = final_goal[-100:]
    return {
        "trace": str(episode / "steps.jsonl"), "recorded_steps": len(rows),
        "unfinished_cycle": completed_cycles + 1,
        "first_loaded_outward_limit_step": sample(outward[0]) if outward else None,
        "closest_loaded_approach_before_limit": sample(closest) if closest else None,
        "loaded_outward_limit_steps": len(outward),
        "tail_mean_swing_command_rad_s": statistics.fmean(row["applied"][0] for row in tail),
        "tail_mean_bucket_particles": statistics.fmean(row["repeat_bucket_particles"] for row in tail),
    }


def training_coverage(limits):
    split = read(SOURCE / "benchmark_provenance.json")["split_manifest"]
    training_ids = set(split["train"])
    if training_ids & set(split["validation"] + split["test"]):
        raise ValueError("Training parent overlaps another split")
    lower, upper = limits
    packed = []
    parents = []
    columns = [
        "episode_index", "observation.state", "observation.stage_current_id",
        "observation.return_to_pile", "observation.recovery_active",
        "observation.recovery_type_id", "observation.scoop_index",
    ]
    for path in sorted((DATA / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index"])
        episode = table["episode_index"][0].as_py()
        if episode not in training_ids:
            continue
        table = pq.read_table(path, columns=columns[1:])
        q = np.asarray(table["observation.state"].to_pylist())
        stage, returning, active, kind, scoop = (
            np.asarray(table[name].to_pylist()).reshape(-1) for name in columns[2:]
        )
        loaded = (stage >= 8) & ~returning.astype(bool) & (q[:, 26] > .002)
        near_limit = (q[:, 0] <= lower + .035) | (q[:, 0] >= upper - .035)
        redig = active.astype(bool) & (kind == 4)
        lifted_to_dig = ((stage[:-1] >= 5) & (stage[1:] <= 4) & (scoop[1:] == scoop[:-1])
                         & ~returning[:-1].astype(bool) & ~returning[1:].astype(bool))
        packed.append(q[loaded, 0])
        parents.append({
            "source_episode": episode, "loaded_transport_frames": int(loaded.sum()),
            "loaded_transport_near_limit_frames": int((loaded & near_limit).sum()),
            "labeled_underfill_redig_frames": int(redig.sum()),
            "same_scoop_stage_regressions": int(lifted_to_dig.sum()),
        })
    if {row["source_episode"] for row in parents} != training_ids or len(parents) != len(training_ids):
        raise ValueError("Expected one source parquet per training parent")
    yaw = np.concatenate(packed)
    return {
        "parent_count": len(parents), "loaded_transport_frames": len(yaw),
        "loaded_transport_yaw_rad_quantiles": dict(zip(
            ("min", "p01", "median", "p99", "max"), np.quantile(yaw, [0, .01, .5, .99, 1]).tolist(), strict=True)),
        "loaded_transport_near_limit_frames": sum(row["loaded_transport_near_limit_frames"] for row in parents),
        "labeled_underfill_redig_frames": sum(row["labeled_underfill_redig_frames"] for row in parents),
        "same_scoop_stage_regressions": sum(row["same_scoop_stage_regressions"] for row in parents),
        "training_parent_candidates_for_recovery_collection": sorted(
            parents, key=lambda row: (-row["loaded_transport_frames"], row["source_episode"]))[:12],
    }


def run(output):
    while True:
        state = read(DEPENDENCY / "status.json")
        jobs = state.get("job_states", {})
        if len(jobs) == 2 and set(jobs.values()) <= TERMINAL:
            break
        write(output / "status.json", {"status": "waiting_for_both_pi05_evaluations", "pid": os.getpid(),
                                      "pi05_job_states": jobs, "updated_unix": time.time()})
        time.sleep(30)

    write(output / "status.json", {"status": "analyzing", "pi05_job_states": jobs, "pid": os.getpid()})
    old_143 = BASELINE / "scene_0143"
    episode = old_143 / "evaluation/episodes/episode_0001/scene_context.json"
    limits = read(episode)["canonical_joint_limits_rad"][0]
    write(output / "coverage_and_trajectories.json", {
        "pi05_terminal_job_states": jobs,
        "training_data_coverage": training_coverage(limits),
        "evaluation_traces_for_diagnosis_only": {
            "historical_143_baseline": trace_evidence(old_143, 3),
            "interrupted_143_candidate": trace_evidence(INTERRUPTED / "scene_0143/candidate/scene_run", 3),
            "historical_162_baseline": trace_evidence(BASELINE / "scene_0162", 1),
            "interrupted_162_baseline": trace_evidence(INTERRUPTED / "scene_0162/baseline/scene_run", 1),
        },
        "scope": "Descriptive checkpoint trajectories under different launches. Interrupted traces are not scored or used as training data.",
    })
    write(output / "status.json", {
        "status": "research_audit_completed", "pi05_job_states": jobs, "updated_unix": time.time(),
        "evidence": str(output / "coverage_and_trajectories.json"),
        "next_experiment": "Collect physically realized off-trajectory recovery demonstrations in training scenes. Prioritize loaded swing recovery, then empty return and low-load redig.",
        "stopped_act_comparison_restarted": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run(args.output)
    except Exception as error:
        write(args.output / "status.json", {"status": "research_error", "error": repr(error)})
        raise
