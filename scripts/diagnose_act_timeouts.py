"""Measure physical progress and persistent swing saturation from saved ACT rollouts."""
import argparse
import json
import statistics
from pathlib import Path


def spans(rows, matches):
    ranges = []
    for row in rows:
        step = row["step"]
        if matches(row):
            if ranges and ranges[-1][1] == step - 1:
                ranges[-1][1] = step
            else:
                ranges.append([step, step])
    return ranges


def analyze(episode):
    status = json.loads((episode / "episode_status.json").read_text())
    context = json.loads((episode / "scene_context.json").read_text())
    limits = context["canonical_joint_limits_rad"][0]
    rows = []
    with (episode / "steps.jsonl").open() as trace:
        for line in trace:
            record = json.loads(line)
            rows.append({key: record[key] for key in (
                "step", "input_state", "applied", "repeat_bucket_particles",
                "joint_command_positions", "input_bucket_landmarks_world",
            )})
    if len(rows) != status["policy_steps_executed"]:
        raise ValueError(f"Incomplete trace: {episode}")
    cycles = []
    for event in status["repeat_cycles"]:
        if "start_step" not in event:
            continue
        start, end = event["start_step"], event["end_step"]
        window = rows[start:end + 1]
        saturated = spans(window, lambda row: any(
            abs(row["input_state"][0] - bound) <= .035
            and direction * row["applied"][0] >= .025
            for bound, direction in zip(limits, (-1, 1), strict=True)
        ))
        longest = max(saturated, key=lambda pair: pair[1] - pair[0], default=None)
        tail = window[-100:]
        keys = ("cycle_index", "start_step", "end_step", "dig_step", "carry_departure_step",
                "dump_step", "returned", "success", "failure_mode", "delivered_mass_kg")
        cycles.append({
            **{key: event.get(key) for key in keys},
            "peak_bucket_particles": max(row["repeat_bucket_particles"] for row in window),
            "longest_swing_limit_outward_span": longest,
            "swing_limit_outward_steps": sum(last - first + 1 for first, last in saturated),
            "tail_mean_bucket_particles": statistics.fmean(row["repeat_bucket_particles"] for row in tail),
            "tail_mean_swing_command": statistics.fmean(row["applied"][0] for row in tail),
            "tail_mean_swing_position": statistics.fmean(row["input_state"][0] for row in tail),
            "tail_swing_range": max(row["input_state"][0] for row in tail) - min(row["input_state"][0] for row in tail),
            "tail_mean_target_local_xy": [statistics.fmean(row["input_state"][i] for row in tail)
                                          for i in (19, 20, 22, 23)],
        })
    return {
        "steps": len(rows), "completed": status["repeat_cycles_completed"],
        "successful": status["repeat_cycles_successful"],
        "task_success": status["repeat_collection_success"],
        "terminal_phase": status["repeat_phase"], "cycles": cycles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.baseline / "summary.json").read_text())
    timed_out = [row["source_export_episode"] for row in summary["episodes"]
                 if row["repeat_failure_mode"] == "episode_step_limit"]
    scenes = {scene: analyze(args.baseline / f"scene_{scene:04d}/evaluation/episodes/episode_0001")
              for scene in [*timed_out, 112]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"timeouts": timed_out, "scenes": scenes}, indent=2) + "\n")


if __name__ == "__main__":
    main()
