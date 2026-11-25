"""Prepare and supervise the two explicitly requested PI05 evaluations."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one evaluator anchor: {old[:80]}')
    return text.replace(old, new, 1)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    private = out/'private_evaluator'
    private.mkdir()
    for name in ('repeat_eval.py', 'repeat_mass_physical.py', 'pi05_physx_ten.py'):
        shutil.copy2(ROOT/'scripts'/name, private/name)
    physical = (private/'repeat_mass_physical.py').read_text()
    physical = replace_once(physical,
        "            returned = in_pile and low <= height <= high\n",
        "            returned = (self.reference_return_region(reply, tip)\n"
        "                        if hasattr(self, 'reference_return_region')\n"
        "                        else in_pile and low <= height <= high)\n")
    (private/'repeat_mass_physical.py').write_text(physical)
    repeat = (private/'repeat_eval.py').read_text()
    repeat = replace_once(repeat,
        "PHYSICAL_MASS_STANDARDS = (PHYSICAL_MASS_STANDARD, MATERIAL_MASS_STANDARD)",
        "PHYSX_TEN_STANDARD = 'excavator_repeat_physx_ten_reference_v1'\n"
        "PHYSICAL_MASS_STANDARDS = (PHYSICAL_MASS_STANDARD, MATERIAL_MASS_STANDARD, PHYSX_TEN_STANDARD)")
    repeat = replace_once(repeat, "    reference = read(reference_path)\n",
        "    reference = read(reference_path)\n"
        "    if cycles == 10 and reference.get('schema_version') == 'excavator_reference_setup_v1':\n"
        "        from pi05_physx_ten import load_physx_ten_spec\n"
        "        return load_physx_ten_spec(dataset, reference_path, strict_minimum)\n")
    repeat = replace_once(repeat,
        "def make_repeated_task(spec, initial_reply, *, started_wall=None):\n",
        "def make_repeated_task(spec, initial_reply, *, started_wall=None):\n"
        "    if spec.get('version') == PHYSX_TEN_STANDARD:\n"
        "        from pi05_physx_ten import task_class\n"
        "        return task_class()(spec, initial_reply, started_wall=started_wall)\n")
    (private/'repeat_eval.py').write_text(repeat)
    entry = (ROOT/'scripts/eval_smolvla_success.py').read_text()
    entry = replace_once(entry, 'BASE = Path(__file__).resolve().parents[1]',
                         f'BASE = Path({str(ROOT)!r})\nsys.path.append(str(BASE / "scripts"))')
    entry = replace_once(entry, "    args = parser.parse_args()\n",
        "    parser.add_argument('--physx-five-to-ten', action='store_true',\n"
        "                        help='Explicitly repeat the archived five PhysX goals twice, without scene reset.')\n"
        "    args = parser.parse_args()\n")
    entry = replace_once(entry, "        if (not supported_repeat or not args.reference_episode",
        "        physx_extension = (args.physx_five_to_ten and args.policy_type == 'pi05'\n"
        "                           and args.repeat_cycles == 10 and args.repeat_cycle_steps == 0\n"
        "                           and simulation_settings['soil_backend'] == 'physx')\n"
        "        if args.physx_five_to_ten and not physx_extension:\n"
        "            parser.error('--physx-five-to-ten requires PI05, PhysX, ten cycles and no per-cycle timeout')\n"
        "        supported_repeat = supported_repeat or physx_extension\n"
        "        if (not supported_repeat or not args.reference_episode")
    entry = replace_once(entry, "        software = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda}",
        "        protocol['private_evaluator_sources'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()\n"
        "            for p in sorted(Path(__file__).parent.glob('*.py'))}\n"
        "        protocol['physx_five_to_ten'] = args.physx_five_to_ten\n"
        "        software = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda}")
    (private/'evaluate.py').write_text(entry)
    references = out/'references'
    references.mkdir()
    old_data = ROOT/'outputs/datasets/pi05_repeat_physx_bucket20_v2'
    old_source = old_data/'references/episode_0000.json'
    reference = read(old_source)
    reference['task'] = reference['task'].replace('Dig 5 consecutive scoops', 'Dig 10 consecutive scoops')
    reference['provenance']['ten_cycle_extension'] = dict(
        original_reference=str(old_source), original_sha256=digest(old_source),
        source_goal_indices=list(range(5))*2, scene_reset_between_passes=False)
    write(references/'physx_reference_0000_ten.json', reference)
    new_train = ROOT/'outputs/experiments/pi05_history_repeat20k_20261123_01/train'
    split_path = new_train/'split_manifest.json'
    split = read(split_path)
    if 24 not in split['test'] or 24 in split['train']+split['validation']:
        raise ValueError('Selected Newton source episode 24 is not held out')
    new_source = ROOT/'outputs/experiments/pi05_smolvla_test2_20261122_01/references/export_episode_0024.json'
    if read(new_source)['provenance']['source_export_episode'] != 24:
        raise ValueError('Test reference source ID differs')
    shutil.copy2(new_source, references/'newton_test_0024.json')
    old_ckpt = ROOT/'outputs/experiments/pi05_repeat_history_c_bucket20_20k_20261109_01/train/checkpoints/020000/pretrained_model'
    new_ckpt = new_train/'checkpoints/020000/pretrained_model'
    jobs = [dict(name='old_physx_ten', gpu=0, checkpoint=str(old_ckpt), dataset=str(old_data),
                 reference=str(references/'physx_reference_0000_ten.json'), seed=2, steps=10000,
                 backend='physx', full_step_budget=False, minimum_free_mib=15500,
                 note='Stop after ten physical cycles or 10000 actions; original scoop/deposit thresholds, 50kg diagnostic separately.'),
            dict(name='new_newton_test24', gpu=1, checkpoint=str(new_ckpt), dataset=str(ROOT/'data/repeat/export_repeat'),
                 reference=str(references/'newton_test_0024.json'), seed=626839174, steps=5000,
                 backend='newton_mpm', full_step_budget=True, minimum_free_mib=18000,
                 note='Fixed test parent 24, same reference/seed as previous comparison; all 5000 actions, ten-cycle scoring.')]
    protected = {str(p):digest(p) for p in private.glob('*.py')}
    for job in jobs:
        protected[job['reference']] = digest(job['reference'])
        # A model delta and all processors are immutable throughout evaluation.
        protected.update({str(p):digest(p) for p in Path(job['checkpoint']).iterdir() if p.is_file()})
        job_dir = out/'jobs'/job['name']
        job_dir.mkdir(parents=True)
        write(job_dir/'status.json', dict(status='prepared', updated_utc=now(), **job))
    manifest = dict(created_utc=now(), jobs=jobs, protected_sha256=protected,
        split_manifest=str(split_path), split_sha256=digest(split_path),
        test_selection='Source episode 24 explicitly selected from frozen test40 for continuity with prior videos; no outcome-based selection.',
        comparability='Separate native-domain diagnostics, not matched-scene/model performance ranking.',
        private_evaluator='Only private evaluator copies add PhysX ten-cycle support; shared evaluator/simulator and datasets remain unchanged.',
        resource_policy='Start each job on its assigned GPU only after free-memory check; never stop unrelated jobs.')
    write(out/'manifest.json', manifest)
    write(out/'status.json', dict(status='prepared', updated_utc=now()))
    shutil.copy2(Path(__file__), out/'controller_source.py')
    return manifest


def environment(job, job_dir):
    env = {k:v for k,v in os.environ.items() if not k.startswith('EXCAVATOR_MPM_')}
    env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        CUDA_VISIBLE_DEVICES='0,1', EXCAVATOR_EVAL_CACHE=str(job_dir/'runtime_cache'),
        EXCAVATOR_SOIL_BACKEND=job['backend'], EXCAVATOR_EVAL_MATERIAL_POSITIONS='0',
        EXCAVATOR_EVAL_MATERIAL_SNAPSHOTS='0')
    if job['backend'] == 'newton_mpm':
        env.update(EXCAVATOR_MPM_COLLECTION_RUNTIME='1', EXCAVATOR_MPM_PROFILE='coarse',
            EXCAVATOR_MPM_MAX_PARTICLES='12000', EXCAVATOR_MPM_HZ='120',
            EXCAVATOR_MPM_DISPLAY='rocks', EXCAVATOR_MPM_GRID_CAPACITY='16384',
            EXCAVATOR_MPM_EXECUTION='graph', EXCAVATOR_MPM_SLEEP_SPEED='0',
            EXCAVATOR_MPM_SOURCE_IDENTITY='newton_source_material_ids_excluding_initial_bucket_and_truck')
    return env


def command(out, job):
    cmd = [str(ROOT/'.venv-pi05/bin/python'), '-B', '-u', str(out/'private_evaluator/evaluate.py'),
        '--policy-type', 'pi05', '--checkpoint', job['checkpoint'], '--dataset', job['dataset'],
        '--output', str(out/'jobs'/job['name']/'evaluation'), '--reference-episode', job['reference'],
        '--episodes', '1', '--steps', str(job['steps']), '--repeat-cycles', '10',
        '--repeat-cycle-steps', '0', '--replan-interval', '3', '--seed', str(job['seed']),
        '--gpus', str(job['gpu']), '--workers', '1', '--soil-backend', job['backend'],
        '--cpu-threads', '8', '--video', '--preview-steps', '600', '--headless',
        '--fresh-simulator', '--render-every-physics-step', '--progress-interval', '15']
    if job['full_step_budget']:
        cmd.append('--run-full-step-budget')
    if job['backend'] == 'physx':
        cmd.append('--physx-five-to-ten')
    return cmd


def verify_protected(manifest):
    for path, sha in manifest['protected_sha256'].items():
        if digest(path) != sha:
            raise ValueError('Protected model/reference/private evaluator changed: '+path)


def free_memory(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def video_check(path):
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=codec_type,width,height,nb_frames,r_frame_rate',
        '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    data = json.loads(p.stdout)
    if float(data['format']['duration']) <= 2 or not data['streams']:
        raise ValueError('Completed video is empty')
    subprocess.run(['ffmpeg', '-v', 'error', '-sseof', '-1', '-i', str(path),
        '-frames:v', '1', '-f', 'null', '-'], check=True, capture_output=True)
    return data


def summarize(out, manifest):
    jobs = []
    for job in manifest['jobs']:
        d = out/'jobs'/job['name']
        row = read(d/'status.json')
        progress = d/'evaluation/workers/worker_00/progress.json'
        if progress.exists():
            row['progress'] = read(progress)
        jobs.append(row)
    terminal = all(j['status'] in ('completed', 'error', 'stopped_by_user') for j in jobs)
    state = ('completed' if all(j['status']=='completed' for j in jobs)
             else 'finished_with_errors' if terminal else 'running')
    result = dict(status=state, updated_utc=now(), jobs=jobs,
                  scope=manifest['comparability'])
    write(out/'summary.json', result)
    write(out/'status.json', dict(status=state, updated_utc=now(), controller_pid=os.getpid(),
                                job_states={j['name']:j['status'] for j in jobs}))
    if terminal:
        write(out/'completion.json', result)
    return result


def run_job(out, manifest, job, stop):
    folder = out/'jobs'/job['name']
    state = dict(**job, status='waiting_for_gpu_memory', updated_utc=now())
    try:
        verify_protected(manifest)
        cmd = command(out, job)
        write(folder/'launch.json', dict(command=cmd, prepared_utc=now()))
        while not stop.is_set():
            available = free_memory(job['gpu'])
            state.update(free_memory_mib=available, updated_utc=now())
            write(folder/'status.json', state)
            if available >= job['minimum_free_mib']:
                break
            stop.wait(15)
        if stop.is_set():
            state['status'] = 'stopped_by_user'
            return
        verify_protected(manifest)
        with (folder/'launch.log').open('w') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=environment(job, folder), stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', pid=proc.pid, started_utc=now())
            write(folder/'status.json', state)
            while proc.poll() is None:
                if stop.wait(10):
                    # Only this evaluator's parent is interrupted; its worker supervisor
                    # handles simulator sessions through its ordinary KeyboardInterrupt.
                    proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=60)
                    state['status'] = 'stopped_by_user'
                    return
            code = proc.returncode
        state['exit_code'] = code
        if code:
            raise RuntimeError(f'Evaluator exited {code}; see {folder / "launch.log"}')
        episode = folder/'evaluation/episodes/episode_0001'
        row = read(episode/'episode_status.json')
        if row.get('status') != 'completed' or row.get('error'):
            raise RuntimeError('Incomplete rollout: '+str(row.get('error')))
        if job['full_step_budget'] and row['policy_steps_executed'] != job['steps']:
            raise ValueError('Full-step rollout stopped early')
        if read(episode/'repeat_audit.json')['status'] != 'passed':
            raise ValueError('Independent cycle/trace audit failed')
        if not read(episode/'rollout_validation.json')['pure_model_execution_validated']:
            raise ValueError('Rollout is not pure model execution')
        info = video_check(episode/'evaluation.mp4')
        write(folder/'video_check.json', info)
        verify_protected(manifest)
        (out/'videos').mkdir(exist_ok=True)
        alias = out/'videos'/f"{job['name']}.mp4"
        alias.symlink_to(os.path.relpath(episode/'evaluation.mp4', alias.parent))
        state.update(status='completed', video=str(alias), steps=row['policy_steps_executed'],
            physical_cycles=row.get('repeat_cycles_completed'), qualified_cycles=row.get('repeat_cycles_successful'),
            cycles_ge_50kg=row.get('diagnostic_cycles_ge_50kg',row.get('repeat_cycles_successful')),
            task_success=row.get('repeat_collection_success'), failure_mode=row.get('repeat_failure_mode'),
            truck_load_kg=row.get('repeat_final_payload_mass_kg'),
            policy_latency_ms_mean=row.get('policy_latency_ms_mean'),
            rollout_seconds=row.get('rollout_time_from_policy_start_seconds'))
    except Exception as exc:
        state.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        state['updated_utc'] = now()
        write(folder/'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.prepare_only:
        manifest = prepare(out)
        for job in manifest['jobs']:
            write(out/'jobs'/job['name']/'launch.json',dict(command=command(out,job)))
        print(out, flush=True)
        return
    manifest = read(out/'manifest.json')
    if (read(out/'preflight/repeat_regressions.json')['status'] != 'passed'
            or read(out/'preflight/historical_cycle_counter.json')['status'] != 'passed'
            or not all(r['status']=='passed' for r in read(out/'preflight/configuration_checks.json'))):
        raise ValueError('Evaluation preflight checks have not passed')
    if (out/'controller.json').exists():
        raise ValueError('Refusing to start a second controller in the same experiment')
    write(out/'controller.json', dict(pid=os.getpid(), started_utc=now()))
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_:stop.set())
    signal.signal(signal.SIGTERM, lambda *_:stop.set())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_job,out,manifest,job,stop) for job in manifest['jobs']]
        while not all(f.done() for f in futures):
            summarize(out,manifest)
            time.sleep(10)
        for future in futures:
            future.result()
    summarize(out,manifest)


if __name__ == '__main__':
    main()
