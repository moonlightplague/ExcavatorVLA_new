"""Independently audit a complete v2 batch against its saved action observations."""
import argparse
import json
import math
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'ExcavatorVLA-smolvla'),
                str(BASE / 'ExcavatorVLA-smolvla/scripts/evaluation')]
from excavator_common.evaluation_standard import EVALUATION_STANDARD, TRUCK_COUNT_METHOD, CompletionTracker
from summarize_isaacsim_success_rate import build_summary, load_episode_rows
from success_eval_runtime import atomic_json, content_id, file_manifest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_episode(folder, status):
    scene = json.loads((folder / 'scene_context.json').read_text())
    initial = json.loads((folder / 'initial_observation.json').read_text())
    with (folder / 'steps.jsonl').open() as stream:
        trace = [json.loads(line) for line in stream]
    seed = status['scene_seed']
    require(scene['scene_seed'] == seed, 'Scene seed differs from episode status')
    require(scene['truck_counting_volume']['scene_seed'] == seed, 'Stale initial truck geometry')
    require(len(trace) == status['policy_steps_executed'] and trace, 'Executed action count differs from trace')
    require(len(scene['canonical_joint_limits_rad']) == 4, 'Missing canonical joint limits')
    completion = CompletionTracker(status['min_particles'])
    duration = 0.
    for index, row in enumerate(trace):
        require(row['step'] == index, 'Nonconsecutive action indices')
        metric = row['truck_bed_load_metrics']
        transfer = row['particle_transfer_metrics']
        require(metric['method'] == TRUCK_COUNT_METHOD and metric['scene_seed'] == seed,
                'Wrong truck method or stale scene geometry')
        require(metric['available'] and metric['source_tracking_valid'], 'Invalid source tracking')
        require(0 <= row['deposited_particles'] == transfer['payload_particles'] <= metric['count']
                <= metric['local_box_count'] <= metric['candidate_count'], 'Inconsistent deposit accounting')
        require(all(math.isfinite(v) for v in row['state']), 'Nonfinite state')
        for command, (lower, upper) in zip(row['joint_command_positions'], scene['canonical_joint_limits_rad']):
            require(math.isfinite(command) and lower - 1e-5 <= command <= upper + 1e-5,
                    'Integrated command exceeds the articulation limits')
        hashes = row['camera_sha256']
        require(set(hashes) == {'0', '1', '2'} and all(len(h) == 64 for h in hashes.values()),
                'Missing policy camera hashes')
        timing = row['bridge_timing']
        seconds = timing['simulated_seconds']
        require(seconds > 0 and math.isclose(seconds, timing['physics_ticks'] / 60, abs_tol=1e-8)
                and math.isclose(seconds, timing['physics_time_advanced_seconds'], abs_tol=1e-5),
                'Physics cadence differs from the action duration')
        require(timing['camera_render_updates'] >= 5, 'Observation did not complete the render window')
        duration += seconds
        succeeded = completion.observe(row['deposited_particles'], row['bucket_position'])
        require(completion.consecutive == row['completion_observations'], 'Wrong completion observation count')
        require(not succeeded or index == len(trace) - 1, 'Rollout continued after completion')
    require(status['success'] == succeeded == status['task_completed_success_and_bucket_open'],
            'Final result differs from the action observations')
    require(math.isclose(duration, status['simulated_rollout_time_seconds'], abs_tol=1e-8),
            'Rollout duration differs from its physics actions')
    require(initial['truck_bed_load_metrics']['scene_seed'] == seed, 'Stale initial count')
    return dict(seed=seed, steps=len(trace), success=succeeded, simulated_seconds=duration)


def audit_batch(folder, verify_sources=False):
    config = json.loads((folder / 'config.json').read_text())
    provenance = json.loads((folder / 'provenance.json').read_text())
    require(config['evaluation_standard'] == EVALUATION_STANDARD, 'Not a v2 evaluation')
    require(provenance['evaluation_id'] == config['evaluation_id'] == content_id(
        {key: value for key, value in provenance.items() if key != 'evaluation_id'}),
        'Evaluation manifest identity differs from its contents')
    if verify_sources:
        paths = [BASE / path for path in provenance['source_files']]
        require(file_manifest(paths, BASE) == provenance['source_files'], 'Current sources differ from this evaluation')
    rows = load_episode_rows(folder / 'episodes')
    summary = build_summary(rows, config['episodes'])
    require(summary['official_result_valid'], 'Batch is incomplete, diagnostic, or has invalid episodes')
    require(all(row['evaluation_id'] == config['evaluation_id'] for row in rows), 'Episodes use another evaluation identity')
    episodes = []
    for row in rows:
        episode = folder / 'episodes' / f"episode_{row['episode_index']:04d}"
        episodes.append(audit_episode(episode, json.loads((episode / 'episode_status.json').read_text())))
    return dict(status='passed', evaluation_id=config['evaluation_id'],
                official_success_rate=summary['official_success_rate'], episodes=episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--verify-sources', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit_batch(args.batch_dir.resolve(), args.verify_sources)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
