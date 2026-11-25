"""Compare recorded return/transition actions with a frozen ACT checkpoint.

Stage labels select offline diagnostics only; they never enter policy inputs.
All sampling proposals use training parents exclusively.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'ExcavatorVLA-vlm'), str(BASE / 'scripts')]


def masks(frame):
    import numpy as np
    action = np.stack(frame.action)
    returning = frame['observation.return_to_pile'].to_numpy().astype(bool)
    stage = frame['observation.stage_current_id'].to_numpy()
    scoop = frame['observation.scoop_index'].to_numpy()
    start = returning & ~np.r_[False, returning[:-1]]
    start_indices = np.flatnonzero(start)
    early = np.zeros(len(frame), dtype=bool)
    for index in start_indices:
        early[index:min(len(frame), index + 10)] = True
    early &= returning
    cut = np.isin(stage, [2, 3, 4, 5]) & ~returning
    active = np.max(np.abs(action), axis=1) > .05
    quiet = np.max(np.abs(action), axis=1) <= .02
    return dict(return_start=early, return_active=returning & active,
                return_quiet=returning & quiet,
                unload=(stage == 9) & ~returning,
                first_cut=cut & (scoop == 0), later_cut=cut & (scoop > 0))


def control_target_audit(frames):
    """Check only consecutive observations inside each original parent."""
    import numpy as np
    errors, returning, boundaries, required = [], [], [], []
    examples = []
    for episode, frame in frames.groupby('episode_index', sort=True):
        frame = frame.sort_values('frame_index').reset_index(drop=True)
        if len(frame) < 2:
            continue
        if not (np.diff(frame.frame_index.to_numpy()) == 1).all():
            raise ValueError('Control audit requires contiguous original frame identities')
        state, action = np.stack(frame['observation.state']), np.stack(frame.action)
        target = state[:, :4] + state[:, 8:12]
        slew = np.diff(target.astype(float), axis=0) * 10
        error = slew - action[:-1]
        errors.append(error)
        required.append(slew)
        returning.append(frame['observation.return_to_pile'].to_numpy()[:-1] > 0)
        boundaries.append(np.diff(frame['observation.scoop_index'].to_numpy()) != 0)
        for i in np.flatnonzero(np.max(abs(error), axis=1) > .1):
            examples.append(dict(episode_index=int(episode), frame_index=int(frame.iloc[i].frame_index),
                scoop_index=int(frame.iloc[i]['observation.scoop_index']),
                label=action[i].tolist(), required_target_slew=slew[i].tolist(),
                max_error_rad_s=float(abs(error[i]).max())))
    if not errors:
        raise ValueError('No within-parent transitions to audit')
    error = np.concatenate(errors)
    slew = np.concatenate(required)
    returning, boundaries = np.concatenate(returning), np.concatenate(boundaries)
    bad = np.max(abs(error), axis=1) > .1
    return dict(definition='10 * adjacent command-target difference minus current label; '
                'command target = measured joint + tracking error; no cross-episode differences',
                transitions=len(error), mismatched_transitions=int(bad.sum()),
                fraction_over_0_1_rad_s=float(bad.mean()),
                return_fraction_over_0_1_rad_s=float(bad[returning].mean()) if returning.any() else None,
                cycle_boundary_fraction_over_0_1_rad_s=float(bad[boundaries].mean()) if boundaries.any() else None,
                max_required_abs_rad_s=np.max(abs(slew), axis=0).tolist(),
                max_abs_label_error_rad_s=np.max(abs(error), axis=0).tolist(),
                examples=sorted(examples, key=lambda r: r['max_error_rad_s'], reverse=True)[:20])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=BASE / 'data/repeat/export_repeat')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-episode-group', type=int, default=8)
    args = parser.parse_args()
    if args.per_episode_group < 1:
        parser.error('per-episode-group must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('HF_HOME', str(BASE / 'models/hf-cache'))
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import DataLoader, Subset
    from excavator_vlm.policies.act.dataset import make_act_dataset
    from excavator_vlm.policies.act.inference import load_act
    from excavator_vlm.evaluation.offline_metrics import evaluate_loader, error_metrics
    from excavator_vlm.data.contract import audit_dataset, file_hash

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    provenance = json.loads((args.checkpoint / 'benchmark_provenance.json').read_text())
    split = provenance['split_manifest']
    report, _ = audit_dataset(args.dataset, 10)
    if report['fingerprint'] != provenance['dataset_fingerprint']:
        raise ValueError('Source dataset differs from the frozen checkpoint')
    columns = ['episode_index', 'frame_index', 'action', 'observation.state',
               'observation.stage_current_id', 'observation.scoop_index',
               'observation.return_to_pile']
    frames = pd.concat([pq.read_table(p, columns=columns).to_pandas()
                        for p in sorted((args.dataset / 'data').rglob('*.parquet'))], ignore_index=True)
    save('action_target_audit.json', control_target_audit(frames))
    groups = {}
    training = {}
    train_windows = []
    train_count = 0
    for episode, frame in frames.groupby('episode_index', sort=True):
        frame = frame.sort_values('frame_index').reset_index(drop=True)
        selections = masks(frame)
        if episode in split['train']:
            train_count += len(frame)
            for name, mask in selections.items():
                selected = frame.loc[mask]
                action = np.stack(selected.action) if len(selected) else np.empty((0, 4))
                entry = training.setdefault(name, {'frames': 0, 'sum_abs_action': np.zeros(4)})
                entry['frames'] += len(selected)
                entry['sum_abs_action'] += np.abs(action).sum(0)
            # A proposal, not a change to the dataset or an automatic training decision.
            selected = selections['return_active'] | selections['return_start']
            train_windows += [dict(episode_index=int(episode), observation_frame=int(f))
                              for f in frame.loc[selected, 'frame_index']]
        if episode not in split['validation']:
            continue
        for name, mask in selections.items():
            indices = np.flatnonzero(mask)
            chosen = indices[np.unique(np.linspace(0, len(indices) - 1,
                               min(args.per_episode_group, len(indices))).astype(int))] if len(indices) else []
            groups.setdefault(name, set()).update((int(episode), int(frame.iloc[i].frame_index))
                                                  for i in chosen)
    for name, entry in training.items():
        entry['fraction_of_training_frames'] = entry['frames'] / train_count
        entry['mean_abs_action_rad_s'] = (entry.pop('sum_abs_action') /
                                         max(1, entry['frames'])).tolist()
    save('training_coverage.json', dict(training_frames=train_count, groups=training,
                                      policy_inputs_exclude_stage_and_return_labels=True))
    save('return_sampling_proposal.json', dict(labels_changed=False,
         training_episodes=split['train'], windows=train_windows,
         scope='Training-only active return/return-start observations; not enabled by this diagnostic'))
    identities = sorted(set.union(*groups.values()))
    save('selection.json', dict(split='validation', per_episode_group=args.per_episode_group,
         identities=identities, groups={k: sorted(v) for k, v in groups.items()},
         partial_evaluation=True, selection='Evenly spaced within each episode/group; not a full validation score'))
    print('Selected', len(identities), 'validation observations', flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    policy, pre, post = load_act(args.checkpoint, 'cuda')
    dataset = make_act_dataset('local/excavator', args.dataset, split['validation'],
                               policy.config.chunk_size)
    index_frame = dataset.hf_dataset.select_columns(['episode_index', 'frame_index']).to_pandas()
    lookup = {(int(ep), int(fr)): i for i, (ep, fr) in
              enumerate(zip(index_frame.episode_index, index_frame.frame_index))}
    loader = DataLoader(Subset(dataset, [lookup[key] for key in identities]), batch_size=8)
    _, arrays = evaluate_loader(policy, pre, post, loader, [6., 2.5, 2.5, 6.])
    np.savez_compressed(args.output / 'predictions.npz', **arrays)
    keys = list(zip(arrays['episode_index'].tolist(), arrays['frame_index'].tolist()))
    metrics = {}
    for name, selected in groups.items():
        select = np.array([key in selected for key in keys])
        pred, target = arrays['predicted'][select, 0], arrays['target'][select, 0]
        if not len(pred):
            continue
        result = error_metrics(pred[:, None], target[:, None],
                 np.ones((len(pred), 1), bool), [6., 2.5, 2.5, 6.])
        moving = np.abs(target[:, 0]) > .2
        result.update(observations=len(pred),
            target_swing_active_observations=int(moving.sum()),
            swing_predicted_near_zero_given_target_active=float(
                (np.abs(pred[moving, 0]) <= .02).mean()) if moving.any() else None,
            swing_direction_accuracy_active=float(
                (np.sign(pred[moving, 0]) == np.sign(target[moving, 0])).mean()) if moving.any() else None)
        metrics[name] = result
    save('diagnosis.json', dict(checkpoint=str(args.checkpoint.resolve()),
         checkpoint_sha256=file_hash(args.checkpoint / 'model.safetensors'),
         dataset_fingerprint=report['fingerprint'], partial_evaluation=True,
         policy_inputs='Recorded RGB/state only; labels select offline groups', groups=metrics))
    print(json.dumps({k: {n: v.get(n) for n in ['observations', 'mae',
          'direction_accuracy_moving', 'swing_predicted_near_zero_given_target_active']}
          for k, v in metrics.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
