"""Durable dual-GPU SmolVLA -> PI05 queue with frozen sources and complete milestone checks."""
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


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Queue:
    def __init__(self, experiment, project):
        self.root, self.project = experiment, project
        self.source = experiment / 'source'
        self.child = None
        self.phase = 'starting'
        self.status = dict(status='running', controller_pid=os.getpid(), started_utc=now(),
                           order=['smolvla', 'pi05'], completed_models=[])
        self.env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
            PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0,1')
        self.lock = (experiment / 'controller.lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock.write(str(os.getpid())); self.lock.flush()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.stop)

    def stop(self, signum, frame):
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
        self.status.update(status='stopped', stopped_signal=signum)
        self.record()
        raise SystemExit(128 + signum)

    def record(self):
        self.status.update(phase=self.phase, updated_utc=now(),
            child_pid=self.child.pid if self.child is not None and self.child.poll() is None else None)
        model = 'pi05' if self.phase.startswith('pi05') else 'smolvla'
        train = self.root / model / 'train'
        progress = read(train / 'progress.json')
        if progress is None and (train / 'train.log').is_file() and model == 'pi05':
            lines = (train / 'train.log').read_text().splitlines()
            progress = json.loads(lines[-1]) if lines else None
        self.status.update(active_model=model, progress=progress,
                           pi05_queued='pi05' not in self.status['completed_models'] and model != 'pi05')
        write(self.root / 'status.json', self.status)

    def verify_sources(self):
        for relative, expected in read(self.root / 'source_manifest.json').items():
            if digest(self.source / relative) != expected:
                raise ValueError(f'Frozen source checksum changed: {relative}')

    def wait_gpus(self, phase, minimum_free_mib=22000):
        self.phase = phase + '_waiting_for_gpus'
        while True:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                                    check=True, capture_output=True, text=True)
            free = {int(a): int(b) for a, b in (row.split(',') for row in result.stdout.splitlines())}
            if all(free.get(i, 0) >= minimum_free_mib for i in (0, 1)):
                return
            self.status['gpu_free_mib'] = free
            self.record()
            time.sleep(10)

    def run(self, phase, command):
        self.phase = phase
        self.verify_sources()
        write(self.root / (phase + '_command.json'), dict(command=list(map(str, command)), started_utc=now()))
        with (self.root / (phase + '.log')).open('a') as log:
            self.child = subprocess.Popen(list(map(str, command)), cwd=self.project, env=self.env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.record()
            while self.child.poll() is None:
                time.sleep(10)
                self.record()
            code = self.child.returncode
            self.record()
            if code:
                # The owned launcher session may contain surviving distributed workers.
                try:
                    os.killpg(self.child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                raise RuntimeError(f'{phase} exited {code}; see {phase}.log')
        self.child = None

    def milestones(self, model):
        train = self.root / model / 'train'
        paths = []
        for step in (10000, 20000, 30000, 40000):
            path = train / 'checkpoints' / (f'step_{step}' if model == 'smolvla' else f'{step:06d}')
            required = ['pretrained_model/model.safetensors', 'pretrained_model/benchmark_provenance.json']
            if model == 'smolvla':
                required += ['pretrained_model/observation_history.json', 'training_state/optimizer_state.safetensors',
                    'training_state/scheduler_state.json', 'training_state/rng_state.safetensors']
            else:
                required += ['training_state.pt', 'pretrained_model/checkpoint.json']
            if any(not (path / filename).is_file() for filename in required):
                raise ValueError(f'Missing complete {model} milestone {step}')
            paths.append(str(path))
        write(self.root / model / 'milestones.json', dict(status='passed', checkpoints=paths))

    def execute(self):
        gate = read(self.root / 'smolvla/preflight.json')
        if not gate or gate.get('status') != 'passed':
            raise ValueError('SmolVLA real-model history/memory/save-reload gate has not passed')
        if gate['training_config_sha256'] != digest(self.root / 'smolvla/training_config.yaml'):
            raise ValueError('SmolVLA configuration changed after preflight')
        if shutil.disk_usage(self.root).free < 40 * 1024**3:
            raise RuntimeError('40 GiB free space is required for all eight full milestones and save overhead')
        smol_python = self.project / '.venv-smolvla-history/bin/python'
        pi_python = self.project / '.venv-pi05/bin/python'
        self.wait_gpus('smolvla')
        self.run('smolvla_training', [smol_python, self.source / 'scripts/train_smolvla.py', '--config',
                                   self.root / 'smolvla/training_config.yaml'])
        self.milestones('smolvla')
        self.run('smolvla_final_reload', [smol_python, self.source / 'scripts/verify_repaired_history_checkpoint.py',
            '--experiment', self.root, '--model', 'smolvla', '--step', '40000'])
        self.status['completed_models'].append('smolvla')
        self.wait_gpus('pi05')
        if shutil.disk_usage(self.root).free < 30 * 1024**3:
            raise RuntimeError('30 GiB free space is required for PI05 milestones and checkpoint overhead')
        command = [pi_python, self.source / 'ExcavatorVLA-vlm/scripts/train_pi05.py', '--config',
                   self.root / 'pi05/training_config.yaml']
        # These are the first two production updates under the full 40k schedule.
        # Native checks verify zero-LoRA equivalence, frozen weights, FP32 optimizer and DDP gradients.
        self.run('pi05_two_update_gate', [*command, '--stop-after', '2'])
        train = self.root / 'pi05/train'
        for filename in ('zero_lora_check.json', 'reload_check.json'):
            if read(train / filename, {}).get('status') != 'passed':
                raise ValueError(f'PI05 native preflight failed: {filename}')
        audit = read(train / 'first_update_check_000001.json', {})
        groups = audit.get('groups', {})
        if (audit.get('step') != 1 or groups.get('frozen_sample', {}).get('changed', 0)
                or not groups.get('lora_B', {}).get('changed', 0)
                or not groups.get('lora_B', {}).get('nonzero_gradient', 0)):
            raise ValueError('PI05 first-update LoRA/frozen-weight audit failed')
        self.run('pi05_initial_reload', [pi_python, self.source / 'scripts/verify_repaired_history_checkpoint.py',
            '--experiment', self.root, '--model', 'pi05', '--step', '2'])
        self.wait_gpus('pi05')
        checkpoint = train / 'checkpoints/000002/pretrained_model'
        self.run('pi05_training', [*command, '--resume', checkpoint])
        self.milestones('pi05')
        self.run('pi05_final_reload', [pi_python, self.source / 'scripts/verify_repaired_history_checkpoint.py',
            '--experiment', self.root, '--model', 'pi05', '--step', '40000'])
        self.status['completed_models'].append('pi05')
        self.phase = 'completed'
        self.status.update(status='completed', completed_utc=now())
        self.record()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    queue = Queue(args.experiment.resolve(), args.project_root.resolve())
    try:
        queue.execute()
    except Exception as exc:
        queue.status.update(status='failed', error=str(exc))
        queue.record()
        raise
