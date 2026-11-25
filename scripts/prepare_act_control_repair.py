"""Prepare a separate Newton control-label candidate; never modify source data."""
from __future__ import annotations

import copy
import errno
import json
import os
from pathlib import Path
import shutil
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'ExcavatorVLA-vlm'), str(BASE / 'scripts')]


def build(source, output, split_source):
    import numpy as np
    import pandas as pd
    from excavator_vlm.data.action_limits import BUCKET20_PROFILE, PROFILES
    from excavator_vlm.data.contract import audit_dataset, file_hash, validate_split
    from excavator_vlm.data.relabel import relabel_episode
    from excavator_vlm.policies.act.dataset import training_stats
    from run_act_original_experiment import inventory, write, now

    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'status.json', dict(phase='hashing_original_source', updated_at=now()))
    before = inventory(source)
    write(output / 'original_source_inventory.json', before)
    report, frames = audit_dataset(source, 10)
    split = json.loads(Path(split_source).read_text())
    validate_split(split, frames.episode_index.unique(), report['fingerprint'])
    info = json.loads((source / 'meta/info.json').read_text())
    if len(frames.episode_index.unique()) != 200 or len(frames) != 493456:
        raise ValueError('Expected the original 200 complete repeated parents')
    staging = output / 'dataset.building'
    staging.mkdir()
    shutil.copytree(source / 'meta', staging / 'meta')
    write(output / 'status.json', dict(phase='reconstructing_controls', updated_at=now(), episodes=0))
    revised_frames, repairs = [], []
    for eid, frame in frames.groupby('episode_index', sort=True):
        frame = frame.sort_values('frame_index').copy()
        states, actions = np.stack(frame['observation.state']), np.stack(frame.action)
        revised, candidate, repair = relabel_episode(states, actions, PROFILES[BUCKET20_PROFILE])
        # The helper uses float32 control targets; all other measured columns retain source precision.
        preserved = states.copy()
        preserved[:, 12:16] = revised[:, 12:16]
        if not np.array_equal(preserved[:, :12], states[:, :12]) or not np.array_equal(preserved[:, 16:], states[:, 16:]):
            raise ValueError('A measured state field was changed')
        frame['observation.state'], frame['action'] = list(preserved), list(candidate)
        revised_frames.append(frame)
        repair['episode_index'] = int(eid)
        repairs.append(repair)
        if (len(repairs) % 10) == 0:
            write(output / 'status.json', dict(phase='reconstructing_controls', updated_at=now(), episodes=len(repairs)))
    merged = pd.concat(revised_frames, ignore_index=True)
    metadata = pd.concat([pd.read_parquet(p) for p in sorted(source.glob('meta/episodes/**/*.parquet'))])
    # Write new tables in exactly the source partitions. Never modify hard-linked tabular data.
    for (chunk, file), entries in metadata.groupby(['data/chunk_index', 'data/file_index']):
        path = staging / info['data_path'].format(chunk_index=int(chunk), file_index=int(file))
        path.parent.mkdir(parents=True, exist_ok=True)
        merged[merged.episode_index.isin(entries.episode_index)].to_parquet(path, index=False)
    linked, copied = 0, 0
    for source_video in sorted((source / 'videos').rglob('*')):
        if not source_video.is_file():
            continue
        dest = staging / source_video.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_video, dest)
            linked += 1
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(source_video, dest)
            copied += 1
    info = copy.deepcopy(info)
    info.update(action_limit_profile=BUCKET20_PROFILE, action_limits_rad_s=PROFILES[BUCKET20_PROFILE],
        action_policy_version='sampled_command_target_difference_v1',
        relabel=dict(kind='newton_repeat_control_reconstruction_v1', source=str(source),
            source_fingerprint=report['fingerprint'],
            state_change='causal previous actions only; measured joints/velocities/errors/geometry/fill preserved',
            training_gate='control_replay_gate.json must pass three complete ten-cycle replays'))
    write(staging / 'meta/info.json', info)
    # The only fitted publication statistics are based on the 140 frozen training parents.
    stats = json.loads((source / 'meta/stats.json').read_text())
    fitted = training_stats(merged, split['train'])
    stats.update(fitted)
    write(staging / 'meta/stats.json', stats)
    write(output / 'train_normalization_stats.json', fitted)
    evidence = []
    for path in (BASE / 'data/repeat/raw').glob('run_*/episode_*/export.json'):
        receipt = json.loads(path.read_text())
        if not receipt.get('validated') or Path(receipt['export_dir']).resolve() != source:
            continue
        eid = int(receipt['export_episode_index'])
        meta_path = path.with_name('meta.json')
        meta = json.loads(meta_path.read_text())
        if (not meta.get('success') or meta['scoops_completed'] != 10
                or meta['repeat_baseline']['soil_backend'] != 'newton_mpm'):
            raise ValueError('Invalid complete Newton source parent')
        folder = staging / f'provenance/episode_{eid:04d}'
        folder.mkdir(parents=True, exist_ok=False)
        shutil.copy2(meta_path, folder / 'meta.json')
        shutil.copy2(path, folder / 'export.json')
        evidence.append(dict(episode_index=eid, soil_backend='newton_mpm',
            parent_id=meta['episode_id'], source_export_episode=eid,
            evidence=dict(path=f'provenance/episode_{eid:04d}/meta.json',
                          sha256=file_hash(meta_path), field='repeat_baseline.soil_backend')))
    if sorted(e['episode_index'] for e in evidence) != list(range(200)):
        raise ValueError('Source receipts do not cover all 200 parents exactly once')
    write(staging / 'backend_provenance.json', dict(schema_version='excavator_backend_provenance_v1',
                                                  soil_backend='newton_mpm', episodes=evidence))
    write(output / 'status.json', dict(phase='auditing_candidate', updated_at=now()))
    audit, _ = audit_dataset(staging, 10)
    split = copy.deepcopy(split)
    split['dataset_fingerprint'] = audit['fingerprint']
    validate_split(split, merged.episode_index.unique(), audit['fingerprint'])
    write(staging / 'split_manifest.json', split)
    write(staging / 'audit.json', audit)
    write(staging / 'repair_report.json', dict(status='numerical_and_dataset_audit_passed_replay_gate_pending',
        training_ready=False, source_fingerprint=report['fingerprint'], dataset_fingerprint=audit['fingerprint'],
        episodes=repairs, preserved_parent_split=True, labels_clipped=False,
        video_files_linked=linked, video_files_copied=copied,
        statistics_scope='140 frozen training parents only'))
    if inventory(source) != before:
        raise ValueError('Original source bytes changed while preparing candidate')
    staging.rename(output / 'dataset')
    write(output / 'status.json', dict(phase='candidate_prepared_waiting_replays', updated_at=now(),
        source_unchanged=True, dataset_fingerprint=audit['fingerprint'], episodes=200,
        frames=len(merged), split_counts={k: len(split[k]) for k in ('train', 'validation', 'test')}))
    return output / 'dataset'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=BASE / 'data/repeat/export_repeat')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-source', type=Path, default=BASE /
                        'outputs/experiments/act_shortchunk_repeat40k_20261117_01/prepared/split_manifest.json')
    args = parser.parse_args()
    print(build(args.source, args.output, args.split_source), flush=True)
