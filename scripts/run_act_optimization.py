"""Run the approved PHW fine-tuning and execution ablations from pinned sources.

All labels remain expert labels. Development and selection scenes are explicitly
separate from the predeclared final scenes. No failed rollout is relabelled.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(os.environ.get('ACT_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
PHASES = ('pre_dig', 'approach_contact', 'insert_cut', 'pull_mid_cut',
          'curl_to_hold_material', 'pull_exit_cut', 'secure_load', 'lift_carry',
          'loaded_transit', 'unload_to_bin')
PREVIOUS = ROOT / 'outputs/experiments/act_phw_diagnostics_20261102_01'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def prepare_windows(frames, episodes, chunk_size=10):
    """Include phase 5/6 context and the onset of phase 9, never another episode."""
    windows = []
    for episode in sorted(episodes):
        rows = frames[frames.episode_index == episode].sort_values('frame_index')
        indices = rows.frame_index.tolist()
        if indices != list(range(len(indices))):
            raise ValueError('Training episode has nonconsecutive frames')
        stages = rows['observation.stage_current_id'].astype(int).tolist()
        selected = set()
        for frame, stage in enumerate(stages):
            transition = stage in (5, 6) or (stage == 9 and (frame == 0 or stages[frame - 1] != 9))
            if transition:
                selected.update(range(max(0, frame - 10), min(len(indices), frame + 11)))
        for frame in sorted(selected):
            windows.append({'episode_index': int(episode), 'observation_frame': frame,
                            'stage_id': stages[frame],
                            'valid_action_chunk_end_exclusive': min(len(indices), frame + chunk_size)})
    return {'schema_version': 'act_transition_windows_v2', 'labels_changed': False,
            'training_episodes': list(map(int, episodes)), 'context_frames': 10,
            'action_chunk_size': chunk_size, 'selection': 'phase5/6 context plus unload onset context',
            'windows': windows}


def sampling_audit(frames, episodes, manifest):
    import numpy as np
    from excavator_vlm.policies.act.sampling import observation_probabilities

    selected = frames[frames.episode_index.isin(episodes)].sort_values(['episode_index', 'frame_index'])
    keys = list(zip(selected.episode_index.astype(int), selected.frame_index.astype(int)))
    stages = selected['observation.stage_current_id'].to_numpy(dtype=int)
    supervision = np.asarray(selected.action_loss_weight.tolist()).reshape(-1)
    lookup = {key: i for i, key in enumerate(keys)}
    length = Counter(ep for ep, _ in keys)
    future_indices = np.array([[lookup[(ep, min(frame + h, length[ep] - 1))]
                                for h in range(10)] for ep, frame in keys])
    valid = np.array([[frame + h < length[ep] for h in range(10)] for ep, frame in keys])
    result = {}
    for label, probability in (('uniform', 0.), ('transition', .3)):
        p, report = observation_probabilities(keys, episodes, manifest, probability)
        masses = p[:, None] * supervision[future_indices] * valid
        report['expected_observation_phase_fraction'] = {
            name: float(p[stages == i].sum()) for i, name in enumerate(PHASES)}
        report['expected_supervised_chunk_phase_fraction'] = {
            name: float(masses[stages[future_indices] == i].sum() / masses.sum())
            for i, name in enumerate(PHASES)}
        result[label] = report
    return result


def diagnose_previous(previous):
    """Describe event order and tracking errors; do not infer expert actions."""
    result = []
    for name in ('reference_replan3', 'reference_replan1', 'debug_replan3', 'debug_replan1',
                 'selection_replan3', 'selection_replan1'):
        for folder in sorted((previous / name / 'episodes').glob('episode_*')):
            status = read(folder / 'episode_status.json')
            steps = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
            retained = [r['particle_transfer_metrics']['retained_particles'] for r in steps]
            deposited = [r['particle_transfer_metrics']['payload_particles'] for r in steps]
            peak = max(retained, default=0)
            first_scoop = next((r['step'] for r, n in zip(steps, retained) if n >= 10), None)
            first_deposit = next((r['step'] for r, n in zip(steps, deposited) if n >= 100), None)
            peak_step = retained.index(peak) if retained else None
            loss = next((i for i in range((peak_step or 0) + 1, len(steps))
                         if peak >= 100 and retained[i] <= peak - 100
                         and deposited[i] < 100), None)
            diagnostics = read(folder / 'diagnostics.json')
            result.append({'case': name, 'scene_seed': status['scene_seed'],
                'reference_scene': name.startswith('reference'), 'success': status['success'],
                'failure_mode': status['failure_mode'], 'first_10_retained_step': first_scoop,
                'peak_retained': peak, 'peak_retained_step': peak_step,
                'first_drop_100_without_100_deposited_step': loss,
                'first_100_deposited_step': first_deposit,
                'bucket_angle_at_drop_rad': steps[loss]['bucket_position'] if loss is not None else None,
                'bucket_action_at_drop_rad_s': steps[loss]['applied'][3] if loss is not None else None,
                'max_tracking_error_rad': diagnostics['max_tracking_error_rad'],
                'retained_payload_stall': diagnostics['retained_payload_stall'],
                'source': str(folder)})
    return {'scope': 'Descriptive event timing only; not a causal stage diagnosis or new expert labels',
            'episodes': result,
            'failure_counts': dict(Counter(r['failure_mode'] for r in result if not r['success']))}


def heldout_seeds():
    """Exclude seed IDs in archived collection metadata and prior evaluation configs."""
    used = set(range(2, 32))
    files = [*(ROOT / 'data/raw').rglob('meta.json'),
             *(ROOT / 'outputs/eval').glob('*/config.json')]
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if 'seed' in key.lower() and type(child) is int:
                    used.add(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for path in files:
        payload = read(path)
        visit(payload)
        if type(payload.get('seed')) is int and type(payload.get('episodes')) is int:
            used.update(range(payload['seed'], payload['seed'] + payload['episodes']))
    start = 1_000_000
    while set(range(start, start + 100)) & used:
        start += 100
    return {'seed': start, 'episodes': 100, 'source_metadata_files_inspected': len(files),
            'excluded_seed_ids': sorted(used),
            'limitation': 'Disjoint scene RNG seeds; cannot prove distinct physical layouts from seed IDs alone'}


def prepare(args):
    import yaml
    sys.path.insert(0, str(ROOT / 'ExcavatorVLA-vlm'))
    from excavator_vlm.policies.act.config import resolve_settings, build_train_config
    from excavator_vlm.policies.act.train import prepare as prepare_data
    from excavator_vlm.policies.act.dataset import training_stats
    from excavator_vlm.policies.act.continuation import validate_continuation
    from run_act_phw_diagnostics import snapshot, manifest

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(read(PREVIOUS / 'status.json')['checkpoint'])
    base_config = ROOT / 'ExcavatorVLA-vlm/configs/act/full_phw.yaml'
    base = resolve_settings(base_config, {'policy': {'optimizer_lr': 3e-6,
        'optimizer_lr_backbone': 3e-6}, 'batch_size': 16, 'num_workers': 4,
        'steps': args.train_steps, 'save_freq': args.train_steps, 'log_freq': 100,
        'benchmark': {'validation_freq': args.train_steps},
        'launcher': {'gpu_ids': [args.gpus[0]], 'cpu_threads': 4}})
    report, split, frames, episodes = prepare_data(base)
    validate_continuation(checkpoint, build_train_config(base).policy, report, split, episodes,
                          training_stats(frames, episodes))
    windows = prepare_windows(frames, episodes)
    write(out / 'transition_windows.json', windows)
    write(out / 'sampling_audit.json', sampling_audit(frames, episodes, windows))
    write(out / 'failure_diagnosis.json', diagnose_previous(PREVIOUS))
    write(out / 'heldout_protocol.json', heldout_seeds())
    shutil.copy2(PREVIOUS / 'reference_setup.json', out / 'reference_setup.json')
    snapshot(out)
    frozen = out / 'frozen_sources'
    (out / 'configs').mkdir()
    configurations = {}
    for name, probability in (('uniform', 0.), ('transition', .3)):
        config = json.loads(json.dumps(base))
        config['benchmark']['split_file'] = str(frozen / 'ExcavatorVLA-vlm/configs/splits' /
                                                Path(base['benchmark']['split_file']).name)
        config['continuation'] = {'checkpoint': str(checkpoint)}
        config['sampling'] = {'manifest': str(out / 'transition_windows.json'),
                              'transition_probability': probability}
        config['output_dir'] = str(ROOT / 'outputs/train' / f'{out.name}_{name}')
        config['job_name'] = f'act_phw_{name}'
        path = out / 'configs' / f'{name}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        configurations[name] = str(path)
    dataset = Path(base['dataset']['root'])
    state = {'status': 'prepared', 'created_utc': now(), 'phase': 'prepared', 'tasks': [],
        'checkpoint': str(checkpoint), 'dataset': str(dataset), 'gpus': args.gpus,
        'train_steps': args.train_steps, 'configs': configurations,
        'source_checkpoint_files': manifest(checkpoint.rglob('*'), checkpoint),
        'dataset_files': manifest([dataset / 'meta/info.json', *sorted((dataset / 'meta').rglob('*.parquet')),
                                   *sorted((dataset / 'data').rglob('*.parquet'))], dataset),
        'experiment_inputs': manifest([out / 'transition_windows.json', out / 'reference_setup.json',
                                      out / 'heldout_protocol.json', *sorted((out / 'configs').glob('*.yaml'))], out),
        'scope': 'weights-only PHW fine-tuning, independent temporal ensemble, selection, replication, conditional held-out comparison; no generated expert labels',
        'protocol': {'fps': 10, 'steps': 350, 'min_particles': 100, 'backend': 'physx',
                     'walls_during_settle': False, 'fresh_simulator': True,
                     'render_every_physics_step': True, 'development_seeds': [2, 6],
                     'selection_seeds': [12, 31], 'temporal_window': 3, 'temporal_decay': .5,
                     'global_batch_size': 16, 'learning_rate': 3e-6,
                     'training_seed': 1000, 'replication_seed': 1001}}
    write(out / 'status.json', state)
    return state


class Runner:
    def __init__(self, args):
        import fcntl

        self.args = args
        self.out = args.output.resolve()
        self.lock = (self.out / 'controller.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another controller already owns this experiment') from exc
        self.state = read(self.out / 'status.json')
        if self.state['status'] != 'prepared':
            raise ValueError('Only a prepared experiment may be started; existing runs are never overwritten')
        self.frozen = self.out / 'frozen_sources'
        self.children = []

    def save(self):
        self.state['updated_utc'] = now()
        write(self.out / 'status.json', self.state)

    def verify(self):
        from run_act_phw_diagnostics import sha
        snapshot = read(self.out / 'source_snapshot.json')
        for base, files in ((self.frozen, snapshot['files']),
            (Path(snapshot['lerobot_root']), snapshot['lerobot_files']),
            (Path(self.state['checkpoint']), self.state['source_checkpoint_files']),
            (Path(self.state['dataset']), self.state['dataset_files']),
            (self.out, self.state['experiment_inputs'])):
            for name, digest in files.items():
                if sha(base / name) != digest:
                    raise RuntimeError(f'Pinned experiment input changed: {base / name}')
        for checkpoint, files in self.state.get('generated_checkpoint_files', {}).items():
            for name, digest in files.items():
                if sha(Path(checkpoint) / name) != digest:
                    raise RuntimeError(f'Fine-tuned checkpoint changed: {Path(checkpoint) / name}')
        for path, digest in self.state.get('external_input_files', {}).items():
            if sha(Path(path)) != digest:
                raise RuntimeError(f'Pinned external input changed: {path}')

    def report(self):
        rows = ['# ACT PHW 优化实验', '', f'状态：{self.state["status"]}；阶段：{self.state["phase"]}', '',
                f'PHW 40k 起点；两组原归一化、原专家标签、batch 16、学习率 3e-6、各续训 {self.state["train_steps"]} 步。',
                '两组均重新初始化优化器。转换采样为 70% 均匀 + 30% 按 episode 均衡的关键窗口。',
                '闭环固定 PhysX 提前移墙、10 Hz、350 步、100 入车粒子、连续三步开斗；每场新模拟器、逐物理步渲染。', '',
                '| 实验 | 状态 | 成功/场数 | 平均入车粒子 |', '| --- | --- | ---: | ---: |']
        for task in self.state['tasks']:
            if task['kind'] == 'evaluation':
                count = f'{task.get("successes", "—")}/{task["episodes"]}'
                rows.append(f'| {task["name"]} | {task["status"]} | {count} | {task.get("mean_payload", "—")} |')
        rows.extend(['', '开发/选择场景用于调参，不作为独立最终成绩。',
                     '仅当候选在选择场景胜出且训练随机种子复核通过时，执行预先冻结的 100 场配对最终比较。'])
        if 'conclusion' in self.state:
            rows.extend(['', self.state['conclusion']])
        (self.out / 'REPORT.md').write_text('\n'.join(rows) + '\n')
        write(self.out / 'results.json', {k: v for k, v in self.state.items() if k not in (
            'source_checkpoint_files', 'dataset_files', 'experiment_inputs')})

    def wait_gpus(self):
        stable = 0
        while stable < 3:
            inventory = subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            samples = {int(parts[0]): [int(v) for v in parts[1:]]
                       for parts in (line.split(',') for line in inventory.splitlines())}
            free = all(samples[g][0] < 1500 and samples[g][1] < 15 for g in self.state['gpus'])
            stable = stable + 1 if free else 0
            self.state.update(status='waiting_gpu', phase='waiting_gpu', gpu_samples=samples)
            self.save()
            self.report()
            if stable < 3:
                time.sleep(10)

    def environment(self):
        env = {k: v for k, v in os.environ.items() if k != 'CUDA_VISIBLE_DEVICES'
               and not k.startswith(('EXCAVATOR_MPM_', 'EXCAVATOR_SOIL_', 'EXCAVATOR_SAND_WALLS_'))}
        env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        env.update(HF_HOME=str(ROOT / 'models/hf-cache'), TORCH_HOME=str(ROOT / 'models/torch-cache'),
                   HF_DATASETS_CACHE=str(ROOT / 'models/hf-cache/datasets'))
        return env

    def execute_group(self, tasks, phase):
        self.verify()
        self.state.update(status='running', phase=phase)
        self.children = []
        try:
            for task in tasks:
                task.update(status='running', started_utc=now())
                log = (self.out / f'{task["name"]}.log').open('x')
                env = self.environment()
                if 'cuda_mask' in task:
                    env['CUDA_VISIBLE_DEVICES'] = str(task['cuda_mask'])
                child = subprocess.Popen(task['command'], cwd=self.frozen, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                log.close()
                task['pid'] = child.pid
                self.children.append((task, child))
                self.state['tasks'].append(task)
            self.save()
            self.report()
            while any(child.poll() is None for _, child in self.children):
                for task, child in self.children:
                    if child.poll() not in (None, 0):
                        raise RuntimeError(f'{task["name"]} exited {child.returncode}; see its log')
                    output = Path(task['output'])
                    if task['kind'] == 'training' and (output / 'train.log').exists():
                        lines = (output / 'train.log').read_text().splitlines()
                        if lines:
                            task['progress'] = json.loads(lines[-1])
                    elif task['kind'] == 'evaluation':
                        task['progress'] = [read(p) for p in sorted(output.glob('workers/*/progress.json'))]
                self.save()
                time.sleep(5)
            for task, child in self.children:
                task['exit_code'] = child.returncode
                if child.returncode:
                    raise RuntimeError(f'{task["name"]} exited {child.returncode}; see its log')
                if task['kind'] == 'evaluation':
                    self.audit_evaluation(task)
                elif task['kind'] == 'training':
                    output = Path(task['output'])
                    if read(output / 'reload_check.json')['status'] != 'passed':
                        raise RuntimeError('Checkpoint reload verification did not pass')
                    task['continuation'] = read(output / 'continuation.json')
                    task['validation'] = read(output / 'checkpoints/last/validation.json')['first_action']
                    task['checkpoint'] = str((output / 'checkpoints/last/pretrained_model').resolve())
                    from run_act_phw_diagnostics import manifest
                    checkpoint = Path(task['checkpoint'])
                    self.state.setdefault('generated_checkpoint_files', {})[str(checkpoint)] = manifest(
                        checkpoint.rglob('*'), checkpoint)
                task.update(status='completed', finished_utc=now())
            self.verify()
            self.save()
            self.report()
        finally:
            for task, child in self.children:
                if child.poll() is None:
                    # psutil walks only descendants of this controller's child.
                    import psutil
                    descendants = psutil.Process(child.pid).children(recursive=True)
                    for process in reversed(descendants):
                        try:
                            process.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    _, living = psutil.wait_procs(descendants, timeout=10)
                    for process in living:
                        try:
                            process.kill()
                        except psutil.NoSuchProcess:
                            pass
                if task['status'] == 'running' and sys.exc_info()[0] is not None:
                    task.update(status='failed_or_cancelled', exit_code=child.poll())
            self.children = []

    def training(self, label, gpu, config=None):
        path = Path(config or self.state['configs'][label])
        import yaml
        settings = yaml.safe_load(path.read_text())
        return {'name': f'train_{label}', 'kind': 'training', 'output': settings['output_dir'],
                'command': [sys.executable, '-u', str(self.frozen / 'ExcavatorVLA-vlm/scripts/train_act.py'),
                            '--config', str(path), '--gpus', str(gpu)]}

    def evaluation(self, name, checkpoint, episodes, seed, gpu, temporal=False, reference=False, replay=False,
                   save_trajectory=False):
        output = self.out / name
        command = [sys.executable, '-u', str(self.frozen / 'scripts/eval_smolvla_success.py'),
            '--policy-type', 'act', '--checkpoint', str(checkpoint), '--dataset', self.state['dataset'],
            '--output', str(output), '--episodes', str(episodes), '--seed', str(seed), '--steps', '350',
            '--min-particles', '100', '--replan-interval', '1' if temporal else '3',
            '--initial-episode', '0', '--workers', '1', '--gpus', str(gpu), '--cpu-threads', '8',
            '--headless', '--fresh-simulator', '--render-every-physics-step', '--soil-backend', 'physx',
            '--no-sand-walls-during-settle', '--video' if save_trajectory or reference else '--no-video',
            '--save-trajectory' if save_trajectory else '--no-save-trajectory']
        if temporal:
            command += ['--act-temporal-window', '3', '--act-temporal-decay', '.5']
        if reference:
            command += ['--reference-episode', str(self.out / 'reference_setup.json')]
        if replay:
            command.append('--replay-reference')
        return {'name': name, 'kind': 'evaluation', 'output': str(output), 'episodes': episodes,
                'seed': seed, 'checkpoint': str(checkpoint), 'temporal': temporal,
                'reference': reference, 'replay': replay, 'command': command}

    def audit_evaluation(self, task):
        sys.path[:0] = [str(self.frozen / 'scripts'), str(self.frozen / 'ExcavatorVLA-smolvla')]
        from audit_smolvla_success import audit_episode
        output = Path(task['output'])
        statuses, audits = [], []
        for folder in sorted((output / 'episodes').glob('episode_*')):
            status = read(folder / 'episode_status.json')
            if status['status'] != 'completed' or status.get('error'):
                raise RuntimeError(f'Incomplete rollout in {task["name"]}: {status.get("error")}')
            if status['execution_source'] != ('expert_replay' if task['replay'] else 'model'):
                raise RuntimeError('Expert/model execution sources were mixed')
            if status['act_temporal_window'] != (3 if task['temporal'] else 0):
                raise RuntimeError('Temporal execution protocol differs from the requested case')
            audits.append(audit_episode(folder, status))
            statuses.append(status)
        if len(statuses) != task['episodes'] or sorted(s['scene_seed'] for s in statuses) != list(
                range(task['seed'], task['seed'] + task['episodes'])):
            raise RuntimeError('Rollout scene coverage differs from the frozen protocol')
        write(output / 'trace_audit.json', {'status': 'passed', 'episodes': audits})
        task.update(successes=sum(s['success'] for s in statuses),
                    mean_payload=sum(s['payload_particles'] for s in statuses) / len(statuses),
                    failure_modes=dict(Counter(s['failure_mode'] for s in statuses)))

    def offline(self, label, checkpoint, gpu):
        output = self.out / f'offline_{label}'
        return {'name': f'offline_{label}', 'kind': 'offline', 'output': str(output), 'cuda_mask': gpu,
            'command': [sys.executable, '-u', str(self.frozen / 'ExcavatorVLA-vlm/scripts/eval_act_offline.py'),
                '--checkpoint', str(checkpoint), '--dataset', self.state['dataset'],
                '--split', 'validation', '--batch-size', '16', '--device', 'cuda', '--output', str(output)]}

    def phase_metrics(self):
        import numpy as np
        import pyarrow.parquet as pq
        stages = {}
        for path in (Path(self.state['dataset']) / 'data').rglob('*.parquet'):
            for row in pq.read_table(path, columns=['episode_index', 'frame_index',
                                                   'observation.stage_current_id']).to_pylist():
                stages[(row['episode_index'], row['frame_index'])] = row['observation.stage_current_id']
        result = {}
        sources = {'baseline': ROOT / 'outputs/eval/act_phw_ladder/val_040000/actions.npz',
                   **{label: self.out / f'offline_{label}/actions.npz' for label in ('uniform', 'transition')}}
        for label, path in sources.items():
            arrays = np.load(path, allow_pickle=False)
            phase = np.array([stages[key] for key in zip(arrays['episode_index'], arrays['frame_index'])])
            p, t = arrays['predicted'][:, 0], arrays['target'][:, 0]
            result[label] = {}
            for i, name in enumerate(PHASES):
                mask = phase == i
                moving = np.abs(t[mask, 3]) > .02
                result[label][name] = {'frames': int(mask.sum()),
                    'bucket_mae_rad_s': float(np.abs(p[mask, 3] - t[mask, 3]).mean()),
                    'bucket_predicted_mean_abs_rad_s': float(np.abs(p[mask, 3]).mean()),
                    'bucket_target_mean_abs_rad_s': float(np.abs(t[mask, 3]).mean()),
                    'bucket_direction_accuracy_moving': float((np.sign(p[mask, 3][moving]) ==
                        np.sign(t[mask, 3][moving])).mean()) if moving.any() else None}
        write(self.out / 'offline_phase_comparison.json', result)

    def run(self):
        self.state.update(pid=os.getpid(), started_utc=now())
        self.wait_gpus()
        g0, g1 = self.state['gpus'][:2]
        training = [self.training('uniform', g0), self.training('transition', g1)]
        self.execute_group(training, 'finetuning_5000_steps')
        policies = {'baseline': self.state['checkpoint'],
                    **{label: task['checkpoint'] for label, task in zip(('uniform', 'transition'), training)},
                    'temporal': self.state['checkpoint']}
        self.execute_group([self.offline('uniform', policies['uniform'], g0),
                            self.offline('transition', policies['transition'], g1)], 'offline_validation')
        self.phase_metrics()
        gate = self.evaluation('expert_gate', policies['baseline'], 1, 2, g0, reference=True, replay=True)
        self.execute_group([gate], 'expert_environment_gate')
        if gate['successes'] != 1:
            self.state.update(status='alignment_failed', phase='stopped',
                              conclusion='专家参考回放失败；停止模型选择并保留全部证据。')
            self.save()
            self.report()
            return
        development = {}
        labels = ['baseline', 'uniform', 'transition', 'temporal']
        for start in range(0, len(labels), 2):
            tasks = [self.evaluation(f'development_{label}', policies[label], 5, 2, gpu,
                        temporal=label == 'temporal', save_trajectory=True)
                     for label, gpu in zip(labels[start:start + 2], (g0, g1))]
            self.execute_group(tasks, 'development_scenes')
            development.update({label: task for label, task in zip(labels[start:start + 2], tasks)})
        def rank(task):
            return (task['successes'], -task['failure_modes'].get('no_scoop', 0), task['mean_payload'])
        selected = sorted(('uniform', 'transition', 'temporal'),
                          key=lambda label: rank(development[label]), reverse=True)[:2]
        self.state['selected_candidates'] = selected
        self.save()
        selection = {}
        for start in range(0, 3, 2):
            labels = ['baseline', *selected][start:start + 2]
            tasks = [self.evaluation(f'selection_{label}', policies[label], 20, 12, gpu,
                                     temporal=label == 'temporal')
                     for label, gpu in zip(labels, (g0, g1))]
            self.execute_group(tasks, 'selection_scenes')
            selection.update({label: task for label, task in zip(labels, tasks)})
        winner = max(selected, key=lambda label: rank(selection[label]))
        if selection[winner]['successes'] <= selection['baseline']['successes']:
            self.state.update(status='completed', phase='no_selection_gain',
                conclusion='候选未提高选择场景的完整成功数；未消耗最终场景。下一阶段需要验证偏离状态下的专家接管并采集纠错示范，现有窗口不构成纠错标签。')
            self.save()
            self.report()
            return
        self.state['selected_winner'] = winner
        if winner != 'temporal':
            import yaml
            config = yaml.safe_load(Path(self.state['configs'][winner]).read_text())
            config.update(seed=1001, output_dir=str(ROOT / 'outputs/train' / f'{self.out.name}_{winner}_seed1001'))
            path = self.out / 'configs' / f'{winner}_seed1001.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            from run_act_phw_diagnostics import sha
            self.state['experiment_inputs'][str(path.relative_to(self.out))] = sha(path)
            self.save()
            repeated = self.training(f'{winner}_seed1001', g0, path)
            self.execute_group([repeated], 'training_seed_replication')
            repeated_eval = self.evaluation(f'replication_{winner}', repeated['checkpoint'], 20, 12, g0)
            self.execute_group([repeated_eval], 'replication_selection_scenes')
            if repeated_eval['successes'] <= selection['baseline']['successes']:
                self.state.update(status='completed', phase='replication_not_confirmed',
                    conclusion='候选在原训练随机种子上胜出，但第二个随机种子未复现；保留最终场景，暂不宣称稳定改善。')
                self.save()
                self.report()
                return
        heldout = read(self.out / 'heldout_protocol.json')
        final = [self.evaluation('heldout_baseline', policies['baseline'], 100, heldout['seed'], g0),
                 self.evaluation(f'heldout_{winner}', policies[winner], 100, heldout['seed'], g1,
                                 temporal=winner == 'temporal')]
        self.execute_group(final, 'heldout_paired_100_scenes')
        paired = []
        for i in range(1, 101):
            records = [read(Path(task['output']) / f'episodes/episode_{i:04d}/episode_status.json')
                       for task in final]
            if records[0]['scene_seed'] != records[1]['scene_seed']:
                raise RuntimeError('Final comparison has unpaired scene seeds')
            paired.append({'scene_seed': records[0]['scene_seed'],
                           'baseline_success': records[0]['success'],
                           'candidate_success': records[1]['success']})
        write(self.out / 'heldout_comparison.json', {
            'baseline': read(Path(final[0]['output']) / 'success_rate_summary.json'),
            'candidate': read(Path(final[1]['output']) / 'success_rate_summary.json'),
            'selection_winner': winner, 'seed': heldout['seed'], 'episodes_per_policy': 100,
            'paired_results': paired,
            'candidate_only_successes': sum(r['candidate_success'] and not r['baseline_success'] for r in paired),
            'baseline_only_successes': sum(r['baseline_success'] and not r['candidate_success'] for r in paired)})
        self.state.update(status='completed', phase='completed', completed_utc=now(),
            conclusion=f'独立最终场景：baseline {final[0]["successes"]}/100；{winner} {final[1]["successes"]}/100。原检查点、数据及归一化未修改。')
        self.save()
        self.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Start an existing prepared experiment only')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--train-steps', type=int, default=5000)
    args = parser.parse_args()
    args.gpus = [int(g) for g in args.gpus.split(',')]
    if len(args.gpus) != 2 or len(set(args.gpus)) != 2 or min(args.gpus) < 0 or args.train_steps < 1:
        parser.error('Exactly two distinct GPUs and a positive training-step count are required')
    if not args.resume:
        prepare(args)
    if args.prepare_only:
        return
    runner = Runner(args)
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        runner.run()
    except BaseException as exc:
        runner.state.update(status='failed', phase='failed', error=f'{type(exc).__name__}: {exc}')
        runner.save()
        runner.report()
        raise


if __name__ == '__main__':
    main()
