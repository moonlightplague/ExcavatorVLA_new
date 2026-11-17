"""Audit an immutable repeated publication and prepare an episode-disjoint SmolVLA experiment."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'ExcavatorVLA-vlm'))
from excavator_vlm.data.contract import CAMERAS, audit_dataset, file_hash, make_split, validate_split
from excavator_vlm.policies.pi05.dataset import training_stats
from smolvla_history import history_spec


def resolve_dataset(source):
    source = Path(source).resolve()
    if (source / 'meta/info.json').is_file():
        return source
    if (source / 'export_repeat/meta/info.json').is_file():
        return source / 'export_repeat'
    raise FileNotFoundError(f'No LeRobot publication at {source} or its export_repeat child')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inventory(root):
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
            for folder in ('meta', 'data', 'videos') for p in sorted((root / folder).rglob('*')) if p.is_file()}


def phase_subsets(frames, episodes, per_episode=8):
    result = {name: [] for name in ('excavation', 'loaded_transit', 'unloading', 'return_to_pile')}
    for episode in episodes:
        rows = frames[frames.episode_index == episode]
        stage = rows['observation.stage_current_id'].to_numpy()
        returning = rows['observation.return_to_pile'].to_numpy().astype(bool)
        state = np.stack(rows['observation.state'])
        masks = {'excavation': (stage <= 6) & ~returning,
                 'loaded_transit': np.isin(stage, [7, 8]) & ~returning & (state[:, 26] > 0),
                 'unloading': (stage == 9) & ~returning,
                 'return_to_pile': returning}
        for name, mask in masks.items():
            ids = rows.loc[mask, 'index'].to_numpy(dtype=int)
            if len(ids):
                chosen = ids[np.unique(np.linspace(0, len(ids)-1, min(per_episode, len(ids)), dtype=int))]
                result[name].extend(map(int, chosen))
    return result


def prepare(source, output, seed=1000):
    root, output = resolve_dataset(source), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = inventory(root)
    write(output / 'source_inventory.json', before)
    report, frames = audit_dataset(root, 10)
    import pandas as pd
    metadata = pd.concat([pd.read_parquet(p) for p in sorted(root.glob('meta/episodes/**/*.parquet'))],ignore_index=True)
    # Equal per-camera container offsets are the publication's synchronization contract.
    for field in ('from_timestamp', 'to_timestamp'):
        values = metadata[[f'videos/{camera}/{field}' for camera in CAMERAS]].to_numpy()
        if not np.allclose(values, values[:, :1], atol=1e-6):
            raise ValueError('Camera episode timestamps are not synchronized')
    split = make_split(frames, report['fingerprint'], seed)
    # Keep complete episodes together; reserve 20% for the final, sealed test.
    extra_count = len(frames.episode_index.unique()) // 10
    extra = set(map(int, np.random.default_rng(seed + 1).choice(split['train'], extra_count, replace=False)))
    split['train'] = [episode for episode in split['train'] if episode not in extra]
    split['test'] = sorted(split['test'] + list(extra))
    split['method'] = 'episode_disjoint_70_10_20_seeded'
    validate_split(split, frames.episode_index.unique(), report['fingerprint'])
    stats = training_stats(frames, split['train'])
    info = json.loads((root / 'meta/info.json').read_text())
    subsets = phase_subsets(frames, split['validation'])
    write(output / 'split_manifest.json', split)
    write(output / 'normalization_stats.json', stats)
    write(output / 'phase_subsets.json', {'scope': 'evaluation only; never policy inputs',
        'split': 'validation', 'selection': 'up to 8 uniformly spaced frames per phase per episode',
        'criteria': {'excavation': 'stage <= 6 and not return',
                     'loaded_transit': 'stage in [7,8], positive bucket fill, not return',
                     'unloading': 'stage 9 and not return', 'return_to_pile': 'recorded return flag'},
        'absolute_indices': subsets})
    report.update(source_directory=str(Path(source).resolve()), dataset=str(root),
        original_split=info['splits'], split_counts={k:len(split[k]) for k in ('train','validation','test')},
        split_frame_counts={k:int(metadata[metadata.episode_index.isin(split[k])].length.sum())
                            for k in ('train','validation','test')},
        camera_sync_episodes_verified=len(metadata),
        source_action_policy=info.get('action_policy_version'),
        normalization='train-only MEAN_STD for state/action; identity images',
        camera_timestamp_contract='same episode/frame indices and video start/end times; not independent exposure proof',
        history=history_spec(), original_dataset_modified=False,
        history_example={'decision_frame':20,'observation_frames':[14,17,20],
            'observation_times_seconds':[1.4,1.7,2.0], 'action_frames':list(range(20,30)),
            'episode_start_frames':[0,0,0], 'episode_start_mask':[True,True,False]},
        phase_counts={k:len(v) for k,v in subsets.items()},
        input_allowlist=['task','observation.state',*CAMERAS],
        length_range=[int(frames.groupby('episode_index').size().min()),int(frames.groupby('episode_index').size().max())])
    if inventory(root) != before:
        raise ValueError('Source publication changed during preparation')
    write(output / 'dataset_audit.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=BASE / 'data/repeat')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset, args.output)
