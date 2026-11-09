"""Audit PHW transition learning and run a queued, source-pinned evaluation.

This is a selection experiment, not a held-out benchmark or a training job.
The expert's actions are used only in explicitly labelled replay cases.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PHASES = ('pre_dig', 'approach_contact', 'insert_cut', 'pull_mid_cut',
          'curl_to_hold_material', 'pull_exit_cut', 'secure_load', 'lift_carry',
          'loaded_transit', 'unload_to_bin')
MPM = {'EXCAVATOR_MPM_VOXEL_SIZE': '0.1',
       'EXCAVATOR_MPM_PARTICLE_SPACING': '0.07500000000000001',
       'EXCAVATOR_MPM_MAX_PARTICLES': '40000', 'EXCAVATOR_MPM_SEED': '4107',
       'EXCAVATOR_MPM_RENDER_DIAMETER': '0.012', 'EXCAVATOR_MPM_SLEEP_SPEED': '0.01'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest(paths, base):
    return {str(p.relative_to(base)): sha(p) for p in sorted(paths) if p.is_file()}


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def offline_audit(out, dataset, checkpoint):
    import numpy as np
    import pyarrow.parquet as pq

    split = read(checkpoint / 'benchmark_provenance.json')['split_manifest']
    groups = {name: set(split[name]) for name in ('train', 'validation', 'test')}
    assert all(not groups[a] & groups[b] for a, b in
               [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    rows = {}
    frame_files = sorted((dataset / 'data').rglob('*.parquet'))
    for path in frame_files:
        table = pq.read_table(path, columns=['episode_index', 'frame_index', 'action',
            'observation.state', 'observation.stage_current_id', 'action_loss_weight'])
        for row in table.to_pylist():
            key = (int(row['episode_index']), int(row['frame_index']))
            if key in rows:
                raise ValueError(f'Duplicate dataset frame: {key}')
            rows[key] = row
    counts = {}
    for group, episodes in groups.items():
        selected = [r for (ep, _), r in rows.items() if ep in episodes]
        counts[group] = {'episodes': len(episodes), 'frames': len(selected), 'phases': {
            name: sum(r['observation.stage_current_id'] == i for r in selected)
            for i, name in enumerate(PHASES)}}
    metrics = {}
    for label, ladder in [('control', 'act_full_ladder'), ('phw', 'act_phw_ladder'),
                          ('armu', 'act_armu_ladder')]:
        path = ROOT / 'outputs/eval' / ladder / 'val_040000/actions.npz'
        arrays = np.load(path, allow_pickle=False)
        keys = list(zip(arrays['episode_index'].tolist(), arrays['frame_index'].tolist()))
        if set(ep for ep, _ in keys) != groups['validation']:
            raise ValueError(f'{label} offline result does not cover the validation split')
        if not np.allclose(arrays['target'][:, 0], [rows[k]['action'] for k in keys], atol=1e-5):
            raise ValueError(f'{label} labels differ from the pinned dataset')
        stages = np.array([rows[k]['observation.stage_current_id'] for k in keys])
        p, t = arrays['predicted'][:, 0], arrays['target'][:, 0]
        metrics[label] = {'actions_sha256': sha(path), 'phases': {}}
        for i, name in enumerate(PHASES):
            mask = stages == i
            if not mask.any():
                continue
            pm, tm = p[mask, 3], t[mask, 3]
            metrics[label]['phases'][name] = {
                'frames': int(mask.sum()), 'bucket_mae_rad_s': float(np.abs(pm - tm).mean()),
                'bucket_bias_rad_s': float((pm - tm).mean()),
                'bucket_predicted_mean_abs_rad_s': float(np.abs(pm).mean()),
                'bucket_target_mean_abs_rad_s': float(np.abs(tm).mean())}
    # Candidate windows are a manifest, not newly generated expert labels. They
    # include context on both sides of transitions and never cross an episode.
    windows = []
    for episode in sorted(groups['train']):
        episode_rows = sorted((r for (ep, _), r in rows.items() if ep == episode),
                              key=lambda r: r['frame_index'])
        indices = [r['frame_index'] for r in episode_rows]
        assert indices == list(range(len(indices))), 'Nonconsecutive episode frames'
        selected = set()
        for r in episode_rows:
            if r['observation.stage_current_id'] in (5, 6):
                frame = r['frame_index']
                selected.update(range(max(0, frame - 10), min(len(indices), frame + 11)))
        for frame in sorted(selected):
            windows.append({'episode_index': episode, 'observation_frame': frame,
                            'valid_action_chunk_end_exclusive': min(len(indices), frame + 10)})
    assert all(w['episode_index'] in groups['train'] for w in windows)
    write(out / 'transition_windows.json', {'purpose': 'candidate training-only observation windows',
          'context_frames': 10, 'action_chunk_size': 10, 'windows': windows,
          'labels_changed': False, 'training_started': False})
    result = {'created_utc': now(), 'counts': counts, 'validation_first_action': metrics,
              'training_window_count': len(windows), 'split': split,
              'dataset_files': manifest([dataset / 'meta/info.json', *frame_files], dataset),
              'limitation': 'Offline recorded observations; small transition-phase validation samples. '
                            'Window manifest is not evidence of successful recovery learning.'}
    write(out / 'offline_audit.json', result)
    return result


def snapshot(out):
    frozen = out / 'frozen_sources'
    frozen.mkdir()
    ignore = shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '.pytest_cache',
                                    '.mpm_warp_cache', '.cache', '*.log')
    # Separate real copies keep other workspace sessions from changing a case
    # halfway through evaluation. Assets are copied; only Isaac itself is shared.
    for name in ('ExcavatorVLA-smolvla', 'ExcavatorVLA-vlm'):
        shutil.copytree(ROOT / name, frozen / name, ignore=ignore)
    shutil.copytree(ROOT / 'ExcavatorVLA-isaac/assets',
                    frozen / 'ExcavatorVLA-isaac/assets', ignore=ignore)
    (frozen / 'scripts').mkdir()
    for path in (ROOT / 'scripts').glob('*.py'):
        shutil.copy2(path, frozen / 'scripts' / path.name)
    for name in ('run_isaac_python.sh', 'training_config.yaml'):
        shutil.copy2(ROOT / name, frozen / name)
    (frozen / 'isaacsim').symlink_to((ROOT / 'isaacsim').resolve(), target_is_directory=True)
    files = [p for p in frozen.rglob('*') if p.is_file() and not p.is_symlink()]
    pinned = manifest(files, frozen)
    changed = [name for name, digest in pinned.items() if not (ROOT / name).is_file()
               or sha(ROOT / name) != digest]
    if changed:
        raise RuntimeError(f'Workspace changed during snapshot: {changed[:10]}')
    lerobot = Path(importlib.util.find_spec('lerobot').origin).parent
    installed = manifest(lerobot.rglob('*.py'), lerobot)
    write(out / 'source_snapshot.json', {'root': str(frozen), 'files': pinned,
          'lerobot_root': str(lerobot), 'lerobot_files': installed})


class Experiment:
    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        if (self.out / 'status.json').exists():
            if not args.resume:
                raise FileExistsError('Use --resume for a prepared experiment')
            self.state = read(self.out / 'status.json')
            if self.state['status'] != 'prepared':
                raise ValueError('Resume is supported only before the first evaluation')
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            checkpoint = args.checkpoint.resolve()
            reference = args.reference.resolve()
            if read(checkpoint / 'config.json')['type'] != 'act':
                raise ValueError('Expected an ACT checkpoint')
            if read(checkpoint.parent / 'training_state/training_step.json')['step'] != 40000:
                raise ValueError('This experiment pins the 40k PHW checkpoint')
            self.state = {'status': 'preparing', 'created_utc': now(), 'cases': [],
                'checkpoint': str(checkpoint), 'reference': str(self.out / 'reference_setup.json'),
                'dataset': str(args.dataset.resolve()), 'gpu': args.gpu,
                'selection_seed_range': [12, 31], 'debug_seed_range': [2, 6],
                'steps': 350, 'min_particles': 100,
                'checkpoint_files': manifest(checkpoint.rglob('*'), checkpoint),
                'reference_origin': str(reference), 'reference_sha256': sha(reference),
                'scope': 'expert alignment, reference fitting diagnosis and 1-vs-3-step selection; '
                         'no held-out test or training, unchanged actions and checkpoint'}
            self.save()
            shutil.copy2(reference, self.out / 'reference_setup.json')
            offline_audit(self.out, args.dataset.resolve(), checkpoint)
            snapshot(self.out)
            self.state['status'] = 'prepared'
            self.save()
        self.frozen = self.out / 'frozen_sources'
        self.child = None

    def save(self):
        self.state['updated_utc'] = now()
        write(self.out / 'status.json', self.state)

    def verify(self):
        pinned = read(self.out / 'source_snapshot.json')
        for root, files in [(self.frozen, pinned['files']),
                            (Path(pinned['lerobot_root']), pinned['lerobot_files']),
                            (Path(self.state['checkpoint']), self.state['checkpoint_files']),
                            (Path(self.state['dataset']), read(self.out / 'offline_audit.json')['dataset_files'])]:
            changed = [name for name, digest in files.items() if sha(root / name) != digest]
            if changed:
                raise RuntimeError(f'Pinned files changed in {root}: {changed[:10]}')
        if sha(Path(self.state['reference'])) != self.state['reference_sha256']:
            raise RuntimeError('Recorded reference changed')

    def wait_gpu(self):
        stable = 0
        while stable < 3:
            reservations = [pid for pid in self.args.wait_pid if alive(pid)]
            inventory = subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True)
            gpu = next(line.split(',') for line in inventory.splitlines()
                       if int(line.split(',')[0]) == self.state['gpu'])
            free = int(gpu[1]) < 1500 and int(gpu[2]) < 15 and not reservations
            stable = stable + 1 if free else 0
            self.state.update(status='waiting_gpu', waiting_for_pids=reservations,
                              gpu_memory_mib=int(gpu[1]), gpu_utilization_pct=int(gpu[2]))
            self.save()
            if stable < 3:
                time.sleep(10)

    def case(self, name, backend, walls, replan=3, replay=False, reference=False,
             episodes=1, seed=2, video=True, trajectories=False):
        self.wait_gpu()
        self.verify()
        output = self.out / name
        command = [sys.executable, '-u', str(self.frozen / 'scripts/eval_smolvla_success.py'),
            '--policy-type', 'act', '--checkpoint', self.state['checkpoint'],
            '--dataset', self.state['dataset'], '--output', str(output),
            '--episodes', str(episodes), '--steps', '350', '--min-particles', '100',
            '--seed', str(seed), '--initial-episode', '0', '--replan-interval', str(replan),
            '--workers', '1', '--gpus', str(self.state['gpu']), '--cpu-threads', '8',
            '--headless', '--fresh-simulator', '--render-every-physics-step',
            '--video' if video else '--no-video',
            '--save-trajectory' if trajectories else '--no-save-trajectory',
            '--soil-backend', backend,
            '--sand-walls-during-settle' if walls else '--no-sand-walls-during-settle']
        if reference:
            command += ['--reference-episode', self.state['reference']]
        if replay:
            if not reference:
                raise ValueError('Expert replay requires a recorded reference')
            command.append('--replay-reference')
        env = {k: v for k, v in os.environ.items() if k != 'CUDA_VISIBLE_DEVICES'
               and not k.startswith(('EXCAVATOR_MPM_', 'EXCAVATOR_SOIL_',
                                     'EXCAVATOR_SAND_WALLS_'))}
        env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        if backend == 'newton_mpm':
            env.update(MPM)
        record = {'name': name, 'status': 'running', 'backend': backend, 'walls': walls,
                  'replan_interval': replan, 'replay': replay, 'reference': reference,
                  'episodes': episodes, 'seed': seed, 'command': command,
                  'output': str(output), 'started_utc': now()}
        self.state['cases'].append(record)
        self.state.update(status='running', phase=name)
        with (self.out / f'{name}.log').open('x') as log:
            self.child = subprocess.Popen(command, cwd=self.frozen, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            record['pid'] = self.child.pid
            self.save()
            while self.child.poll() is None:
                progress = output / 'workers/worker_00/progress.json'
                if progress.exists():
                    self.state['progress'] = read(progress)
                self.save()
                time.sleep(5)
            record['exit_code'] = self.child.returncode
        self.child = None
        if record['exit_code']:
            raise RuntimeError(f'{name} evaluator exited {record["exit_code"]}; see case log')
        self.verify()
        config = read(output / 'config.json')
        if (config['execution_source'] != ('expert_replay' if replay else 'model')
                or config['replan_interval'] != replan
                or config['simulation_settings']['soil_backend'] != backend):
            raise RuntimeError('Executed protocol differs from requested case')
        # Trace audit is also meaningful for diagnostic reference scenes, whereas
        # audit_batch deliberately rejects a reference scene as an official rate.
        sys.path[:0] = [str(self.frozen / 'scripts'), str(self.frozen / 'ExcavatorVLA-smolvla')]
        from audit_smolvla_success import audit_episode
        audits, statuses = [], []
        for folder in sorted((output / 'episodes').glob('episode_*')):
            status = read(folder / 'episode_status.json')
            if status['status'] != 'completed' or status.get('error'):
                raise RuntimeError(f'{name} episode incomplete: {status.get("error")}')
            if replay and status['policy_inference_calls'] != 0:
                raise RuntimeError('Expert gate invoked a model')
            if not replay and status['policy_inference_calls'] <= 0:
                raise RuntimeError('Model case did not invoke the policy')
            audits.append(audit_episode(folder, status))
            statuses.append(status)
        if len(statuses) != episodes or sorted(s['scene_seed'] for s in statuses) != list(range(seed, seed + episodes)):
            raise RuntimeError('Missing or mismatched scene seeds')
        write(output / 'trace_audit.json', {'status': 'passed', 'episodes': audits})
        record.update(status='completed', finished_utc=now(), successes=sum(s['success'] for s in statuses),
            mean_payload=sum(s['payload_particles'] for s in statuses) / episodes,
            mean_scooped=sum(s['scooped_particles'] for s in statuses) / episodes,
            failure_modes=dict(Counter(s['failure_mode'] for s in statuses)))
        self.save()
        self.report()
        return record

    def report(self):
        complete = [c for c in self.state['cases'] if c['status'] == 'completed']
        write(self.out / 'results.json', {'status': self.state['status'], 'cases': complete,
              'selection_environment': self.state.get('selection_environment'),
              'scope': self.state['scope']})
        lines = ['# PHW 第一轮诊断', '', f'状态：{self.state["status"]}', '',
                 '固定 PHW 40k，10 Hz、350 步、100 粒子门槛；每场景新进程并逐物理步渲染。',
                 '专家回放与模型结果分别标记；调参场景不作为独立测试集。', '',
                 '| 实验 | 专家回放 | 后端 | 成功/完成 | 平均入车粒子 |',
                 '| --- | --- | --- | ---: | ---: |']
        for c in complete:
            lines.append(f'| {c["name"]} | {c["replay"]} | {c["backend"]} | '
                         f'{c["successes"]}/{c["episodes"]} | {c["mean_payload"]:.1f} |')
        (self.out / 'REPORT.md').write_text('\n'.join(lines) + '\n')

    def run(self):
        self.state.update(pid=os.getpid(), status='queued', started_utc=now())
        self.save()
        passed = []
        for name, backend, walls in [('expert_physx_early', 'physx', False),
                                     ('expert_physx_settle', 'physx', True),
                                     ('expert_newton', 'newton_mpm', True)]:
            result = self.case(name, backend, walls, replay=True, reference=True)
            if result['successes']:
                passed.append((backend, walls, name))
        if not passed:
            self.state.update(status='alignment_failed', phase='expert_replay',
                              reason='No reference expert succeeded; model selection and training were not started')
            self.save()
            self.report()
            return
        backend, walls, gate = passed[0]
        self.state['selection_environment'] = {'backend': backend, 'walls': walls, 'expert_gate': gate}
        self.save()
        for interval in (3, 1):
            self.case(f'reference_replan{interval}', backend, walls, interval, reference=True,
                      trajectories=True)
        for interval in (3, 1):
            self.case(f'debug_replan{interval}', backend, walls, interval,
                      episodes=5, seed=2, trajectories=True)
        for interval in (3, 1):
            self.case(f'selection_replan{interval}', backend, walls, interval,
                      episodes=20, seed=12, video=False)
        self.state.update(status='completed', phase='completed', completed_utc=now())
        self.save()
        self.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/train/'
        'excavator_act_phw_20261028_150222_336724_1423224/checkpoints/040000/pretrained_model')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/export_phw')
    parser.add_argument('--reference', type=Path, default=ROOT / 'outputs/experiments/'
        'pi05_lora_compare3000_5000_20261102_01/reference_setup.json')
    parser.add_argument('--gpu', type=int, choices=(0, 1), default=1)
    parser.add_argument('--wait-pid', type=int, nargs='*', default=[])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    experiment = Experiment(args)
    if args.prepare_only:
        print(f'Prepared: {experiment.out}', flush=True)
        return

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    try:
        experiment.run()
    except BaseException as exc:
        experiment.state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if experiment.state['cases'] and experiment.state['cases'][-1]['status'] == 'running':
            experiment.state['cases'][-1].update(status='failed', error=str(exc))
        experiment.save()
        experiment.report()
        raise
    finally:
        if experiment.child is not None and experiment.child.poll() is None:
            os.killpg(experiment.child.pid, signal.SIGTERM)


if __name__ == '__main__':
    main()
