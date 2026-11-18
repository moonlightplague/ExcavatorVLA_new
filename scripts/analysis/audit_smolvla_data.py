"""Audit the exported training contract without running Isaac or a model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(root, chunk_size=50):
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    data = pd.concat([pd.read_parquet(p) for p in sorted(root.glob('data/**/*.parquet'))], ignore_index=True)
    data = data.sort_values(['episode_index', 'frame_index'])
    states = np.stack(data['observation.state']).astype(np.float64)
    actions = np.stack(data.action).astype(np.float64)
    failures = []
    if len(data) != info['total_frames'] or data.episode_index.nunique() != info['total_episodes']:
        failures.append('Episode/frame totals do not match metadata')
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        failures.append('Nonfinite state or action')
    limits = np.asarray(info['action_limits_rad_s'])
    if np.any(np.abs(actions) > limits + 1e-5):
        failures.append('Actions exceed the deployment velocity limits')
    max_history_error = 0.
    lengths = []
    first_states = []
    for _, episode in data.groupby('episode_index', sort=True):
        length = len(episode)
        lengths.append(length)
        s = np.stack(episode['observation.state'])
        a = np.stack(episode.action)
        first_states.append(s[0])
        if episode.frame_index.tolist() != list(range(length)):
            failures.append('Noncontiguous frame indices')
        if not np.allclose(episode.timestamp, np.arange(length) / info['fps'], atol=1e-5):
            failures.append('Timestamps differ from the deployment cadence')
        if length > 1:
            max_history_error = max(max_history_error, float(np.abs(s[1:, 12:16] - a[:-1]).max()))
    if max_history_error > 1e-5:
        failures.append('Previous actions are not causally aligned with action targets')
    stats = json.loads((root / 'meta/stats.json').read_text())
    stat_errors = {}
    for key, values in (('action', actions), ('observation.state', states)):
        stat_errors[key] = {name: float(np.abs(compute(values, axis=0) - np.asarray(stats[key][name])).max())
                            for name, compute in (('mean', np.mean), ('std', np.std))}
        if max(stat_errors[key].values()) > 1e-5:
            failures.append(f'{key} normalization statistics differ from data')
    valid = sum(sum(min(chunk_size, length - i) for i in range(length)) for length in lengths)
    first_states = np.asarray(first_states)
    return {
        'dataset': str(root.resolve()), 'status': 'failed' if failures else 'passed',
        'failures': sorted(set(failures)), 'episodes': len(lengths), 'frames': len(data), 'fps': info['fps'],
        'state_dimensions': states.shape[1], 'action_dimensions': actions.shape[1],
        'causal_action_history_max_error': max_history_error, 'normalization_max_errors': stat_errors,
        'action_loss_weights': {str(k): int(v) for k, v in data.action_loss_weight.value_counts().items()},
        'stage_frame_counts': {str(k): int(v) for k, v in data['observation.stage_current_id'].value_counts().sort_index().items()},
        'initial_joint_degrees': {
            'minimum': np.degrees(first_states[:, :4].min(0)).tolist(),
            'median': np.degrees(np.median(first_states[:, :4], axis=0)).tolist(),
            'maximum': np.degrees(first_states[:, :4].max(0)).tolist(),
        },
        'action_absolute_maximum': np.abs(actions).max(0).tolist(),
        'chunk_size': chunk_size, 'padded_timestep_fraction': 1 - valid / (len(data) * chunk_size),
        'note': 'Checks schema/timing/labels/statistics. A passing audit does not establish closed-loop success.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error('--chunk-size must be positive')
    report = audit(args.dataset, args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
