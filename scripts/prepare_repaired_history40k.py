"""Freeze a shared, repaired 200-parent publication and two sequential training configs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'ExcavatorVLA-vlm'), str(BASE / 'scripts')]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def prepare(output):
    from excavator_vlm.data.contract import audit_dataset, read_frames, file_hash, validate_split
    from excavator_vlm.data.control_repair import require_control_repair_gate
    from excavator_vlm.policies.pi05.dataset import training_stats
    from smolvla_history import history_spec

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / 'prepared'
    prepared.mkdir()
    root = BASE / 'outputs/experiments/act_repeat_control_repair_data_20261118_01/dataset'
    info = json.loads((root / 'meta/info.json').read_text())
    original = Path(info['relabel']['source']).resolve()
    raw_split_path = BASE / 'outputs/experiments/smolvla_history_repeat20k_20261116_01/prepared/split_manifest.json'
    raw_split = json.loads(raw_split_path.read_text())
    report, frames = audit_dataset(root, 10)
    if (report['episodes'], report['frames']) != (200, 493456):
        raise ValueError('Expected the selected complete 200-parent Newton publication')
    if (report['action_contract']['limits_rad_s'] != [6., 2.5, 2.5, 20.]
            or report['collection_backend']['soil_backend'] != 'newton_mpm'
            or info['relabel']['kind'] != 'newton_repeat_control_reconstruction_v1'):
        raise ValueError('Require reconstructed labels and the bucket 20 rad/s contract')
    gate = require_control_repair_gate(root, report['fingerprint'])
    split = json.loads((root / 'split_manifest.json').read_text())
    validate_split(split, frames.episode_index.unique(), report['fingerprint'])
    for name, count in [('train', 140), ('validation', 20), ('test', 40)]:
        if split[name] != raw_split[name] or len(split[name]) != count:
            raise ValueError('The original SmolVLA partition must be preserved exactly')
    if info['relabel']['source_fingerprint'] != raw_split['dataset_fingerprint']:
        raise ValueError('Reconstruction source differs from the original publication')
    if set(e['episode_index'] for e in gate['episodes']) & set(split['test']):
        raise ValueError('Control-repair evidence must not use the sealed test partition')
    raw = read_frames(original)
    for key in ('episode_index', 'frame_index', 'timestamp', 'index'):
        np.testing.assert_array_equal(raw[key].to_numpy(), frames[key].to_numpy())
    verified = []
    for eid, group in frames.groupby('episode_index', sort=True):
        source = raw[raw.episode_index == eid]
        state = np.stack(source['observation.state'])
        actual = np.stack(group['observation.state'])
        # Match the original float32 target reconstruction, with no clipping.
        q = state.astype(np.float32)
        targets = q[:, :4] + q[:, 8:12]
        expected = np.zeros((len(q), 4), dtype=np.float32)
        expected[:-1] = np.diff(targets.astype(np.float64), axis=0) * 10
        np.testing.assert_array_equal(np.stack(group.action), expected)
        previous = np.zeros_like(expected)
        previous[1:] = expected[:-1]
        np.testing.assert_array_equal(actual[:, 12:16], previous)
        np.testing.assert_array_equal(actual[:, :12], state[:, :12])
        np.testing.assert_array_equal(actual[:, 16:], state[:, 16:])
        verified.append(int(eid))
    video_identity = []
    for path in sorted((root / 'videos').rglob('*.mp4')):
        source = original / path.relative_to(root)
        a, b = path.stat(), source.stat()
        if (a.st_dev, a.st_ino, a.st_size) != (b.st_dev, b.st_ino, b.st_size):
            if file_hash(path) != file_hash(source):
                raise ValueError('A camera video changed during label reconstruction')
        video_identity.append({'path': str(path.relative_to(root)), 'bytes': a.st_size,
                               'device': a.st_dev, 'inode': a.st_ino, 'mtime_ns': a.st_mtime_ns})
    stats = training_stats(frames, split['train'])
    write(prepared / 'normalization_stats.json', stats)
    write(prepared / 'split_manifest.json', split)
    write(prepared / 'control_replay_gate_verified.json', gate)
    write(prepared / 'video_identity.json', video_identity)
    report.update(dataset=str(root.resolve()), normalization='Train-only per-feature moments and quantiles',
                  history=history_spec(state_history=True), original_dataset_modified=False,
                  split_counts={k: len(split[k]) for k in ('train', 'validation', 'test')},
                  action_and_previous_action_reconstruction_verified_episodes=verified,
                  measured_state_and_camera_preserved=True)
    write(prepared / 'dataset_audit.json', report)
    provenance = dict(adapter='excavator_smolvla_visual_state_history_v2', dataset=str(root.resolve()),
        dataset_fingerprint=report['fingerprint'], source_fingerprint=info['relabel']['source_fingerprint'],
        action_contract=report['action_contract'], collection_backend=report['collection_backend'],
        split_manifest=split, training_episodes=split['train'],
        normalization_stats_sha256=file_hash(prepared / 'normalization_stats.json'),
        split_sha256=file_hash(prepared / 'split_manifest.json'),
        observation_history=history_spec(state_history=True),
        batch_contract=dict(world_size=2, batch_size=4, accumulation=1, global_effective_batch=8),
        labels='sampled_command_target_difference_v1; causal previous actions rebuilt',
        normalization='MEAN_STD state/action fitted on the 140 training parents; identity RGB')
    write(prepared / 'training_provenance.json', provenance)
    smol = yaml.safe_load((BASE / 'training_config_smolvla_history.yaml').read_text())
    smol.update(prepared_data=str(prepared), steps=40000, batch_size=4, save_freq=10000,
                output_dir=str(output / 'smolvla/train'), job_name='smolvla_repaired_image_state_history40k')
    smol['observation_history']['state_history'] = True
    smol['dataset']['root'] = str(root.resolve())
    smol['policy'].update(path=str((BASE / 'models/smolvla_base').resolve()), scheduler_decay_steps=40000)
    smol['launcher'].update(output_root=str(output), hf_home=str((BASE / 'models/hf-cache').resolve()),
                            gpu_ids=[0, 1], micro_batch_size=None)
    pi = yaml.safe_load((BASE / 'training_config_pi05_history.yaml').read_text())
    pi.update(steps=40000, save_freq=10000, output_dir=str(output / 'pi05/train'),
              job_name='pi05_repaired_image_state_history40k')
    pi['dataset']['root'] = str(root.resolve())
    pi['training'].update(keep_checkpoints=4, recovery_save_freq=10000)
    pi['training'].pop('backend_provenance_root', None)
    pi['benchmark'].update(split_file=str(prepared / 'split_manifest.json'), validation_freq=10000)
    pi['launcher']['output_root'] = str(output)
    for name, config in [('smolvla', smol), ('pi05', pi)]:
        folder = output / name
        folder.mkdir()
        (folder / 'training_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write(output / 'training_contract.json', dict(status='prepared', order=['smolvla', 'pi05'],
        gpus=[0, 1], steps_per_model=40000, checkpoint_steps=[10000, 20000, 30000, 40000],
        initialization='fresh pretrained base for each model', dataset=str(root.resolve()),
        dataset_fingerprint=report['fingerprint'], action_contract=report['action_contract'],
        split_counts=report['split_counts'], global_effective_batch=8,
        batch_reference='ACT velocity 008000: 8 sequence fragments per optimizer update; unroll=64',
        batch_scope='SmolVLA/pi05 have 8 historical decision windows; sequence supervision is model-specific',
        smolvla_batch=dict(per_gpu=4, accumulation=1), pi05_batch=dict(per_gpu=1, accumulation=4),
        history_offsets=[-6, -3, 0], image_history_per_model=9, state_history_per_model=[3, 28],
        no_evaluation_trajectory_training=True))
    print(json.dumps({'status':'prepared', 'output':str(output), 'audit':report}, indent=2), flush=True)


def freeze(output):
    """Keep queued training independent of edits to the working repository."""
    from excavator_vlm.data.contract import file_hash
    output = Path(output).resolve()
    frozen = output / 'source'
    frozen.mkdir(exist_ok=False)
    patterns = [('scripts', '*.py'), ('ExcavatorVLA-vlm/excavator_vlm', '**/*.py'),
                ('ExcavatorVLA-vlm/scripts', '*.py'), ('ExcavatorVLA-smolvla/excavator_common', '**/*.py')]
    hashes = {}
    for folder, pattern in patterns:
        for path in sorted((BASE / folder).glob(pattern)):
            relative = path.relative_to(BASE)
            target = frozen / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(relative)] = file_hash(target)
    write(output / 'source_manifest.json', hashes)
    return frozen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    freeze(args.output) if args.freeze_only else prepare(args.output)
