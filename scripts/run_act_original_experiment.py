"""Prepare and supervise a source-pinned, stock-objective ACT training experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return Path(path).read_text()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for folder in ('meta', 'data', 'videos') for p in sorted((root / folder).rglob('*'))
            if p.is_file()}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def prepare(config, split_source):
    sys.path.insert(0, str(ROOT / 'ExcavatorVLA-vlm'))
    from excavator_vlm.common.config import setup_cache
    from excavator_vlm.policies.act.config import resolve_settings, build_train_config
    from excavator_vlm.policies.act.dataset import training_stats
    from excavator_vlm.policies.act.train import prepare as prepare_data

    settings = resolve_settings(config)
    setup_cache(settings)
    build_train_config(settings)
    if settings['objective'] != 'original':
        raise ValueError('This experiment requires the original ACT objective')
    out = Path(settings['output_dir']).parent
    out.mkdir(parents=True, exist_ok=False)
    prepared = out / 'prepared'
    prepared.mkdir()
    source = Path(settings['dataset']['root'])
    before = inventory(source)
    write(prepared / 'source_inventory.json', before)
    split_path = Path(settings['benchmark']['split_file'])
    if split_path != prepared / 'split_manifest.json':
        raise ValueError('The frozen split must belong to the new experiment')
    shutil.copyfile(split_source, split_path)
    report, split, frames, episodes = prepare_data(settings)
    if not (frames.action_loss_weight == 1).all() or not (frames.action_is_expert == 1).all():
        raise ValueError('Original ACT run requires expert labels with unit loss weights')
    report.update(split_counts={k: len(split[k]) for k in ('train', 'validation', 'test')},
                  split_frame_counts={k: int(frames.episode_index.isin(split[k]).sum())
                                      for k in ('train', 'validation', 'test')},
                  split_source=str(split_source), split_source_sha256=digest(split_source),
                  source_immutable=True, action_loss_weights='all one', action_is_expert='all one')
    write(prepared / 'dataset_audit.json', report)
    write(prepared / 'normalization_stats.json', training_stats(frames, episodes))
    snapshot = out / 'frozen_sources' / 'ExcavatorVLA-vlm'
    snapshot.mkdir(parents=True)
    shutil.copytree(ROOT / 'ExcavatorVLA-vlm/excavator_vlm', snapshot / 'excavator_vlm',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (snapshot / 'scripts').mkdir()
    shutil.copyfile(ROOT / 'ExcavatorVLA-vlm/scripts/train_act.py', snapshot / 'scripts/train_act.py')
    shutil.copyfile(__file__, out / Path(__file__).name)
    shutil.copyfile(config, out / 'source_training_config.yaml')
    (out / 'resolved_config.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
    write(out / 'frozen_sources_sha256.json', {str(p.relative_to(out)): digest(p)
          for p in sorted(snapshot.rglob('*.py'))} |
          {'resolved_config.yaml': digest(out / 'resolved_config.yaml')})
    write(out / 'status.json', {'phase': 'prepared', 'updated_at': now(),
          'steps_target': settings['steps'], 'step': 0, 'objective': 'original',
          'global_batch_size': settings['batch_size'] * len(settings['launcher']['gpu_ids']),
          'split_counts': report['split_counts'], 'output': str(out)})
    print(json.dumps({'output': str(out), 'audit': report}, indent=2), flush=True)
    return out


def supervise(out):
    out = Path(out).resolve()
    settings = yaml.safe_load(read(out / 'resolved_config.yaml'))
    state = json.loads(read(out / 'status.json'))
    lock = (out / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if state['phase'] != 'prepared' or Path(settings['output_dir']).exists():
        raise ValueError('This experiment has already started; inspect its state before resuming')
    state['controller_pid'] = os.getpid()
    child = None

    def update(phase, **values):
        if phase != state.get('phase'):
            state.pop('progress', None)
        state.update(phase=phase, updated_at=now(), **values)
        write(out / 'status.json', state)

    def stop(_signal, _frame):
        raise KeyboardInterrupt('ACT controller interrupted')

    def launch(phase, config, output):
        nonlocal child
        command = [sys.executable, '-B', '-u',
                   str(out / 'frozen_sources/ExcavatorVLA-vlm/scripts/train_act.py'),
                   '--config', str(config)]
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        with (out / f'{phase}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=out, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            update(phase, worker_pid=child.pid, command=command)
            while child.poll() is None:
                values = {}
                loss_log = output / 'train.log'
                if loss_log.exists():
                    lines = loss_log.read_text().splitlines()
                    if lines:
                        try:
                            values['progress'] = json.loads(lines[-1])
                            if phase == 'training':
                                progress = values['progress']
                                values['step'] = progress['step']
                                values['estimated_training_seconds_remaining'] = (
                                    progress['elapsed_seconds'] / progress['step'] *
                                    (settings['steps'] - progress['step']))
                        except json.JSONDecodeError:
                            pass
                update(phase, **values)
                time.sleep(5)
            code = child.returncode
            child = None
        if code:
            raise RuntimeError(f'{phase} exited with code {code}; see {phase}.log')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        update('verifying_inputs')
        for name, expected in json.loads(read(out / 'frozen_sources_sha256.json')).items():
            if digest(out / name) != expected:
                raise ValueError(f'Frozen source/config changed: {name}')
        source = Path(settings['dataset']['root'])
        expected_inventory = json.loads(read(out / 'prepared/source_inventory.json'))
        if inventory(source) != expected_inventory:
            raise ValueError('Source dataset changed since preparation')
        probe = json.loads(json.dumps(settings))
        probe.update(steps=2, save_freq=2, log_freq=1, num_workers=2,
                     output_dir=str(out / 'probe'))
        probe['benchmark']['validation_freq'] = 0
        probe['launcher']['gpu_ids'] = [settings['launcher']['gpu_ids'][-1]]
        probe_config = out / 'probe_config.yaml'
        probe_config.write_text(yaml.safe_dump(probe, sort_keys=False))
        launch('preflight', probe_config, out / 'probe')
        check = json.loads(read(out / 'probe/reload_check.json'))
        provenance = json.loads(read(out / 'probe/provenance.json'))
        losses = [json.loads(line) for line in read(out / 'probe/train.log').splitlines()]
        if (check != {'status': 'passed', 'shape': [1, settings['policy']['chunk_size'], 4]} or
                provenance['policy_class'] != 'lerobot.policies.act.modeling_act.ACTPolicy'):
            raise RuntimeError('Stock ACT training/checkpoint preflight failed')
        if max(row['cuda_peak_allocated_mib'] for row in losses) > 10 * 1024:
            raise RuntimeError('ACT probe exceeds the 10 GiB memory budget on these shared GPUs')
        write(out / 'preflight.json', {'status': 'passed', 'reload_check': check,
              'policy_class': provenance['policy_class'], 'parameters': provenance['parameters'],
              'policy_source_sha256': provenance['policy_source_sha256'],
              'batch_size': settings['batch_size'], 'steps': len(losses),
              'peak_cuda_allocated_mib': max(row['cuda_peak_allocated_mib'] for row in losses),
              'temporary_checkpoints_removed': True})
        shutil.rmtree(out / 'probe')
        launch('training', out / 'resolved_config.yaml', Path(settings['output_dir']))
        update('verifying_final')
        train = Path(settings['output_dir'])
        check = json.loads(read(train / 'reload_check.json'))
        checkpoint = (train / 'checkpoints/last').resolve()
        saved_step = json.loads(read(checkpoint / 'training_state/training_step.json'))['step']
        if check['status'] != 'passed' or saved_step != settings['steps']:
            raise RuntimeError('Final checkpoint/reload validation is incomplete')
        if inventory(source) != expected_inventory:
            raise ValueError('Source dataset changed during training')
        removed = []
        for candidate in sorted((train / 'checkpoints').iterdir()):
            if candidate.name.isdecimal() and candidate.is_dir() and candidate.resolve() != checkpoint:
                removed.append(candidate.name)
                shutil.rmtree(candidate)
        update('completed', step=saved_step, checkpoint=str(checkpoint / 'pretrained_model'),
               reload_check=check, source_unchanged=True, removed_intermediate_checkpoints=removed,
               validation=json.loads(read(checkpoint / 'validation.json')))
    except BaseException as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'training_config_act_original.yaml')
    parser.add_argument('--split-source', type=Path, default=ROOT /
                        'outputs/experiments/smolvla_history_repeat20k_20261116_01/prepared/split_manifest.json')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', type=Path)
    args = parser.parse_args()
    if args.run_prepared:
        supervise(args.run_prepared)
    else:
        out = prepare(args.config.resolve(), args.split_source.resolve())
        if not args.prepare_only:
            supervise(out)


if __name__ == '__main__':
    main()
