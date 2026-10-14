"""Process and GPU lifecycle for success evaluation (no ML/Isaac imports)."""
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import time


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def resolve_gpu_ids(requested=None, environ=None):
    """Resolve physical indices to UUIDs, respecting the caller's CUDA mask."""
    environ = os.environ if environ is None else environ
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        check=True, capture_output=True, text=True,
    )
    inventory = result.stdout.strip().splitlines()

    def resolve(token):
        token = token.strip()
        if token.isdecimal() and int(token) < len(inventory):
            return inventory[int(token)]
        matches = [uuid for uuid in inventory if uuid.startswith(token) and token]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f'Unknown or ambiguous GPU: {token!r}')

    mask = environ.get('CUDA_VISIBLE_DEVICES')
    visible = inventory if mask is None else (
        [] if mask.strip() in ('', '-1') else [resolve(token) for token in mask.split(',')])
    chosen = visible if requested is None else [resolve(token) for token in requested.split(',')]
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError('Select at least one GPU, without duplicates')
    if any(gpu not in visible for gpu in chosen):
        raise ValueError('--gpus must be within CUDA_VISIBLE_DEVICES')
    return chosen


def available_cpu_count():
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    quota_path = Path('/sys/fs/cgroup/cpu.max')
    if quota_path.exists():
        quota, period = quota_path.read_text().split()
        if quota != 'max':
            cores = min(cores, max(1, int(quota) // int(period)))
    return cores


def episode_indices(episodes, worker_index, worker_count):
    return range(worker_index + 1, episodes + 1, worker_count)


def prepare_shader_cache(cwd, gpu, environ):
    cache_root = Path(environ.get('EXCAVATOR_EVAL_CACHE',
                                 str(Path.home() / '.cache/excavator_success_eval'))) / gpu
    cache_root.mkdir(parents=True, exist_ok=True)
    initialized = cache_root / '.initialized'
    if not initialized.exists():
        # Seed separate writable caches from the installation's existing warm
        # cache. Otherwise introducing isolation forces expensive recompilation
        # of the same driver shaders on every GPU on the first evaluation run.
        isaac_dir = Path(environ.get('EXCAVATOR_ISAAC_DIR', str(cwd.parent / 'isaacsim')))
        for source_name, target_name in (('shadercache', 'shaders'), ('nv_shadercache', 'driver')):
            source = isaac_dir / 'kit/cache' / source_name
            if source.is_dir():
                shutil.copytree(source, cache_root / target_name, dirs_exist_ok=True)
        initialized.touch()
    (cache_root / 'driver').mkdir(exist_ok=True)
    return cache_root


def stop_simulator(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def connect_simulator(process, port, timeout=600):
    started = time.monotonic()
    next_report = started + 15
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f'Simulator exited with {process.returncode}; see simulator.log')
        try:
            sock = socket.create_connection(('127.0.0.1', port), timeout=1)
            sock.settimeout(120)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError:
            now = time.monotonic()
            if now >= next_report:
                print(f'Waiting for simulator startup: {now - started:.0f}s; see simulator.log', flush=True)
                next_report = now + 15
            time.sleep(0.25)
    raise TimeoutError('Simulator startup timed out; see simulator.log')


class SimulatorSession:
    """Own one persistent simulator and socket; always reap it on exit."""
    def __init__(self, command, cwd, log_path, video, exchange, startup_timeout=600):
        self.command = command
        self.cwd = cwd
        self.log_path = Path(log_path)
        self.video = video
        self.exchange = exchange
        self.startup_timeout = startup_timeout
        self.process = self.sock = self.log = None

    def prepare(self, seed):
        started = time.monotonic()
        reused = self.process is not None
        if reused:
            ack = self.exchange(self.sock, {'type': 'reset_episode', 'scene_seed': seed})
            if ack.get('type') != 'reset_episode_ack' or ack.get('scene_seed') != seed:
                raise ValueError(f'Invalid reset acknowledgment: {ack}')
        else:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            env = dict(os.environ, EXCAVATOR_PROJECT_ROOT=str(self.cwd),
                       EXCAVATOR_BRIDGE_PORT=str(port), EXCAVATOR_EVAL_VIDEO=str(int(self.video)))
            gpu = env.get('CUDA_VISIBLE_DEVICES', 'default')
            cache_root = prepare_shader_cache(self.cwd, gpu, env)
            driver_cache = cache_root / 'driver'
            env['__GL_SHADER_DISK_CACHE_PATH'] = str(driver_cache)
            self.log = self.log_path.open('a')
            self.process = subprocess.Popen(
                [*self.command, '--scene-seed', str(seed), '--kit-log',
                 str(self.log_path.with_name('kit.log')), '--shader-cache', str(cache_root)],
                cwd=self.cwd, env=env,
                stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            self.sock = connect_simulator(self.process, port, self.startup_timeout)
        return {'simulator_reused': reused, 'simulator_pid': self.process.pid,
                'startup_or_reset_seconds': time.monotonic() - started,
                'simulator_log': str(self.log_path)}

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self.process is not None:
            stop_simulator(self.process)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None


def run_workers(args, entrypoint, gpu_ids, summarize):
    worker_count = min(args.episodes, args.workers or len(gpu_ids), len(gpu_ids))
    cpu_threads = args.cpu_threads or max(1, min(16, available_cpu_count() // worker_count))
    config_path = args.output / 'config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config.update(resolved_workers=worker_count, resolved_gpu_ids=gpu_ids[:worker_count],
                      resolved_cpu_threads_per_simulator=cpu_threads)
        atomic_json(config_path, config)
    started = time.monotonic()
    processes = []
    logs = []
    reported = set()
    failed = False
    try:
        for index, gpu in enumerate(gpu_ids[:worker_count]):
            worker_dir = args.output / 'workers' / f'worker_{index:02d}'
            worker_dir.mkdir(parents=True)
            log = (worker_dir / 'worker.log').open('w')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS=str(min(4, cpu_threads)), MKL_NUM_THREADS=str(min(4, cpu_threads)))
            command = [os.sys.executable, str(entrypoint), *os.sys.argv[1:],
                       '--worker-index', str(index), '--worker-count', str(worker_count),
                       '--cpu-threads', str(cpu_threads), '--gpu-label', gpu]
            processes.append(subprocess.Popen(command, env=env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True))
        print(f'Evaluating {args.episodes} episodes with {worker_count} persistent GPU workers '
              f'({cpu_threads} simulator CPU threads each).', flush=True)
        while True:
            rows = summarize()
            for row in rows:
                index = row['episode_index']
                if index not in reported:
                    print(f"Episode {index}/{args.episodes}: {row['status']}, "
                          f"success={row['success']}, error={row['error']}", flush=True)
                    reported.add(index)
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                failed = True
                break
            if all(code is not None for code in codes):
                break
            time.sleep(0.5)
    finally:
        # SIGTERM is handled by each worker so its simulator is reaped too.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                stop_simulator(process)
        for log in logs:
            log.close()
        rows = summarize()
        atomic_json(args.output / 'runtime.json', {
            'wall_seconds': time.monotonic() - started, 'workers': worker_count,
            'gpu_ids': gpu_ids[:worker_count], 'cpu_threads_per_simulator': cpu_threads,
            'completed_episodes': sum(row['status'] == 'completed' for row in rows),
            'worker_exit_codes': [process.returncode for process in processes],
        })
    return int(failed or len(rows) != args.episodes or any(row['status'] != 'completed' for row in rows))
