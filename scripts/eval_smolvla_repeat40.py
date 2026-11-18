"""Resume the sealed test-40 SmolVLA 20k evaluation, with one video per scene.

Only immutable scene instructions are read from expert parents. No recorded
actions, expert joint paths, phase labels or expert timing enter the rollout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE), str(BASE / 'ExcavatorVLA-smolvla'), str(BASE / 'scripts')]
from success_eval_runtime import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(experiment, output, policy_type="smolvla", replan_interval=None):
    import numpy as np
    import pandas as pd
    from baseline.coarse_acceptance import validate_mass_report
    from excavator_common.vla_observation_contract import STATE_NAMES_28D, ACTION_NAMES_4D
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    split_path = experiment / 'prepared/split_manifest.json'
    split = read(split_path)
    test_ids = split['test']
    if (len(test_ids) != 40 or len(set(test_ids)) != 40
            or set(test_ids) & (set(split['train']) | set(split['validation']))):
        raise ValueError('Expected 40 distinct sealed parents with no train/validation overlap')
    dataset = (BASE / 'data/repeat/export_repeat').resolve()
    checkpoint = experiment / ('train/checkpoints/040000/pretrained_model' if policy_type == 'act' else 'train/checkpoints/step_20000/pretrained_model')
    if policy_type == 'act':
        status = read(experiment / 'status.json')
        if status.get('phase') != 'completed' or status.get('step') != 40000:
            raise ValueError('ACT training must be completed before evaluation')
    if replan_interval is None:
        replan_interval = 100 if policy_type == 'act' else 3
    if not (checkpoint / 'model.safetensors').is_file():
        raise ValueError('The completed step_20000 checkpoint is missing')
    info = read(dataset / 'meta/info.json')
    if info['fps'] != 10 or info['total_episodes'] != 200:
        raise ValueError('The repeat source publication differs from the frozen experiment')
    receipts = {}
    for path in (BASE / 'data/repeat/raw').glob('run_*/episode_*/export.json'):
        receipt = read(path)
        if not receipt.get('validated') or Path(receipt['export_dir']).resolve() != dataset:
            continue
        eid = receipt['export_episode_index']
        if eid in receipts:
            raise ValueError(f'Duplicate source receipt for export episode {eid}')
        receipts[eid] = path.resolve()
    if set(receipts) != set(range(200)):
        raise ValueError('Export receipts do not map one-to-one onto the 200 parents')
    meta = pd.concat([pd.read_parquet(p) for p in sorted(dataset.glob('meta/episodes/**/*.parquet'))])
    tasks = pd.read_parquet(dataset / 'meta/tasks.parquet')
    firsts = {}
    paths = {}
    for eid in test_ids:
        row = meta[meta.episode_index == eid].iloc[0]
        path = dataset / info['data_path'].format(chunk_index=int(row['data/chunk_index']),
                                                file_index=int(row['data/file_index']))
        paths.setdefault(path, []).append(eid)
    for path, ids in paths.items():
        frames = pd.read_parquet(path, columns=['episode_index', 'frame_index', 'task_index', 'observation.state'],
                                 filters=[('frame_index', '=', 0), ('episode_index', 'in', ids)])
        for _, row in frames.iterrows():
            eid = int(row.episode_index)
            if eid in firsts:
                raise ValueError('Duplicated first observation')
            firsts[eid] = row
    output.mkdir(parents=True, exist_ok=True)
    (output / 'references').mkdir(exist_ok=True)
    scenes = []
    for position, eid in enumerate(test_ids, 1):
        receipt = receipts[eid]
        raw = receipt.parent
        source_meta, quality, source_metrics = raw / 'meta.json', raw.parent / 'quality_gate_config.json', raw / 'repeat_metrics.json'
        m, gate = read(source_meta), read(quality)
        validate_mass_report(m, read(source_metrics))
        first = firsts[eid]
        state = np.asarray(first['observation.state'], dtype=np.float64).tolist()
        applied = m['scene_randomization']['applied']
        goals = m['scoop_results']
        source_files = [receipt, source_meta, source_metrics, quality, split_path]
        evidence = {str(p): digest(p) for p in source_files}
        reference = dict(schema_version='excavator_scene_setup_v2', fps=info['fps'], action_units='rad/s',
            state_names=list(STATE_NAMES_28D), action_names=list(ACTION_NAMES_4D),
            profile=dict(sand_xy=applied['sand_center'][:2],
                         truck_translation_xyz=applied['truck_translation_xyz'],
                         truck_yaw_deg=applied['truck_yaw_deg'],
                         unload_landing_xyz=goals[0]['unload_landing_xyz']),
            task=str(tasks.index[int(first.task_index)]),
            initial_joints_deg=np.degrees(state[:4]).tolist(),
            expected_initial_state=state, dig_target_xyz=goals[0]['target_xyz'],
            sand_amount=applied['sand_amount_multiplier'],
            provenance=dict(source_export_episode=eid, source_parent=m['episode_id'],
                            source_meta=str(source_meta), source_receipt=str(receipt),
                            source_quality_gate=str(quality), source_evidence=evidence,
                            source_scene_seed=m['repeat_baseline']['scene_seed']))
        path = output / 'references' / f'export_episode_{eid:04d}.json'
        if path.exists() and read(path) != reference:
            raise ValueError('Prepared reference changed; use a new output directory')
        atomic_json(path, reference)
        scenes.append(dict(evaluation_episode=position, source_export_episode=eid,
                           reference=str(path), reference_sha256=digest(path),
                           policy_seed=20261117 + position, source_parent=m['episode_id']))
    manifest = dict(schema_version=('act_original_40k_heldout40_v1' if policy_type == 'act' else 'smolvla_history_20k_heldout40_v3'), created_from=str(experiment),
                    checkpoint=str(checkpoint), dataset=str(dataset), split_sha256=digest(split_path),
                    requested_episodes=40, cycles_per_episode=10, cycle_step_limit=0, episode_step_limit=5000,
                    replan_interval=replan_interval, control_hz=10, scenes=scenes,
                    source_identity='newton_source_material_ids_excluding_initial_bucket_and_truck',
                    material_position_telemetry=True,
                    scope='Held-out recorded scene/task coordinates with freshly settled Newton coarse soil; '
                          'model-only closed-loop, continuous history, no scene reset between cycles.',
                    success='All ten >=50 kg new-source deposits and physical returns, >=400 kg final truck '
                            'payload, >=80% retained cumulative delivered mass; failures remain in denominator.',
                    cycle_count='Physical dig/carry/open/return attempts, independent of deposit mass. '
                                'Unsuccessful deposits still advance the next goal after returning.',
                    return_criterion='Bucket tip inside the collection Newton source search radius (recorded '
                                     'source_search_initial.radius_m) and 0.15..2.5 m above '
                                     'ground for 3 consecutive observations; score <=5 remaining source IDs separately.',
                    timeout='5000 executed 10 Hz actions for the episode; stop earlier only after all ten '
                            'physical attempts and returns. No per-cycle early termination.',
                    infrastructure='Infrastructure errors are reported separately and retried explicitly; '
                                   'completed model failures are never replaced or retried.')
    if policy_type == 'act':
        manifest.update(policy_type='act', checkpoint_step=40000)
        manifest['scope'] = manifest['scope'].replace('continuous history', 'current observation, predict 100 actions, execute ' + str(replan_interval))
    path = output / 'manifest.json'
    if path.exists() and read(path) != manifest:
        raise ValueError('Evaluation manifest differs from the existing output')
    atomic_json(path, manifest)
    return manifest


def wilson(successes, total):
    if not total:
        return None
    p, z = successes / total, 1.959963984540054
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [max(0., center-half), min(1., center+half)]


def summarize(output, manifest, active=None):
    rows, errors = [], []
    for scene in manifest['scenes']:
        parent = output / 'scenes' / f"export_episode_{scene['source_export_episode']:04d}"
        completed = []
        for result_path in sorted(parent.glob('attempt_*/launch_result.json')):
            result = read(result_path)
            status_path = result_path.parent / 'evaluation/episodes/episode_0001/episode_status.json'
            if result.get('exit_code') and not status_path.exists():
                errors.append(dict(source_export_episode=scene['source_export_episode'],
                                   error=result.get('error', 'launcher_failed_before_episode_status'),
                                   launch_result=str(result_path)))
        for path in sorted(parent.glob('attempt_*/evaluation/episodes/episode_0001/episode_status.json')):
            row = read(path)
            if row['status'] == 'completed':
                completed.append((path, row))
            else:
                errors.append(dict(source_export_episode=scene['source_export_episode'],
                                   error=row.get('error'), status_file=str(path)))
        if len(completed) > 1:
            raise ValueError('A completed policy outcome was evaluated more than once')
        if completed:
            path, row = completed[0]
            row.update(source_export_episode=scene['source_export_episode'],
                       evaluation_episode=scene['evaluation_episode'], status_file=str(path))
            rows.append(row)
    successes = sum(bool(r['repeat_collection_success']) for r in rows)
    cycles = sum(r.get('repeat_cycles_successful', r['repeat_cycles_completed']) for r in rows)
    physical_cycles = sum(r['repeat_cycles_completed'] for r in rows)
    complete = len(rows) == 40
    summary = dict(status='complete' if complete else 'running', updated_utc=now(),
        checkpoint_step=manifest.get('checkpoint_step', 20000), requested_episodes=40, completed_episodes=len(rows),
        successful_episodes=successes, task_success_rate=successes/len(rows) if rows else None,
        task_success_rate_95pct_wilson=wilson(successes, len(rows)),
        complete_test_success_rate=successes/40 if complete else None,
        successful_cycles=cycles, target_cycles_evaluated=len(rows)*10,
        physical_cycles_completed=physical_cycles,
        cycle_success_rate=cycles/(len(rows)*10) if rows else None,
        infrastructure_errors=errors, active=active or {}, episodes=rows)
    atomic_json(output / 'summary.json', summary)
    text = [f"{manifest.get('policy_type', 'smolvla')} step {manifest.get('checkpoint_step', 20000):,}: {len(rows)}/40 held-out scenes complete.",
            f"Full ten-cycle task success: {successes}/{len(rows)}" +
            (f" ({100*successes/len(rows):.1f}%)." if rows else ' (pending).'),
            f"Physical dig/dump/return cycles: {physical_cycles}/{len(rows)*10}; successful deposits/returns: {cycles}/{len(rows)*10}.",
            'Partial rates remain provisional until all 40 valid episodes finish.', '',
            manifest['scope'], manifest['success'], '',
            '| Test source episode | Task success | Physical cycles | Successful cycles | Failure | Video |',
            '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        video = Path(row['status_file']).parent / 'evaluation.mp4'
        relative = video.relative_to(output)
        text.append(f"| {row['source_export_episode']} | {bool(row['repeat_collection_success'])} | "
                    f"{row['repeat_cycles_completed']}/10 | {row.get('repeat_cycles_successful', row['repeat_cycles_completed'])}/10 | "
                    f"{row['repeat_failure_mode']} | [MP4]({relative}) |")
    (output / 'REPORT.md').write_text('\n'.join(text)+'\n')
    return summary


def evaluation_environment(output):
    return {**os.environ, 'HF_HUB_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1',
           'PYTHONDONTWRITEBYTECODE':'1', 'TOKENIZERS_PARALLELISM':'false',
           'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4',
           'EXCAVATOR_EVAL_CACHE':str(output / 'runtime_cache'),
           'EXCAVATOR_SOIL_BACKEND':'newton_mpm', 'EXCAVATOR_MPM_COLLECTION_RUNTIME':'1',
           'EXCAVATOR_MPM_PROFILE':'coarse', 'EXCAVATOR_MPM_MAX_PARTICLES':'12000',
           'EXCAVATOR_MPM_HZ':'120', 'EXCAVATOR_MPM_DISPLAY':'rocks',
           'EXCAVATOR_MPM_SOURCE_IDENTITY':'newton_source_material_ids_excluding_initial_bucket_and_truck',
           'EXCAVATOR_EVAL_MATERIAL_POSITIONS':'1',
           'EXCAVATOR_EVAL_MATERIAL_SNAPSHOTS':'1',
           'EXCAVATOR_MPM_GRID_CAPACITY':'16384', 'EXCAVATOR_MPM_EXECUTION':'graph',
           'EXCAVATOR_MPM_SLEEP_SPEED':'0'}


def evaluation_command(manifest, scene, attempt, gpu):
    return [sys.executable, str(BASE / 'scripts/eval_smolvla_success.py'),
        '--policy-type', manifest.get('policy_type', 'smolvla'), '--checkpoint', manifest['checkpoint'],
        '--dataset', manifest['dataset'], '--output', str(attempt / 'evaluation'),
        '--reference-episode', scene['reference'], '--episodes', '1',
        '--repeat-cycles', '10', '--repeat-cycle-steps', '0', '--steps', '5000',
        '--replan-interval', str(manifest['replan_interval']), '--seed', str(scene['policy_seed']),
        '--gpus', gpu, '--workers', '1', '--soil-backend', 'newton_mpm',
        '--cpu-threads', '12', '--video', '--headless', '--fresh-simulator',
        '--render-every-physics-step', '--progress-interval', '15']


def process_active(pid):
    path = Path(f'/proc/{pid}/stat')
    try:
        return path.read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def publish_completion(output, manifest, summary):
    if summary['completed_episodes'] != 40:
        return
    from repeat_eval import audit_repeat_trace
    for scene, row in zip(manifest['scenes'], summary['episodes']):
        if scene['source_export_episode'] != row['source_export_episode']:
            raise ValueError('Completed outcomes do not follow the frozen test manifest')
        folder = Path(row['status_file']).parent
        spec = read(folder / 'repeat_task_spec.json')
        for path, expected in spec['source_evidence'].items():
            if digest(path) != expected:
                raise ValueError('Original source evidence changed during evaluation')
        if audit_repeat_trace(folder, spec, row)['status'] != 'passed':
            raise ValueError('Final independent audit failed')
        video = folder / 'evaluation.mp4'
        if not video.is_file() or video.stat().st_size <= 0 or row.get('video_frames', 0) <= 20:
            raise ValueError('A completed trial is missing its evaluation video')
    completion = dict(status='complete', finished_utc=now(), checkpoint_step=manifest.get('checkpoint_step', 20000),
                      completed_episodes=40, successful_episodes=summary['successful_episodes'],
                      task_success_rate=summary['complete_test_success_rate'],
                      task_success_rate_95pct_wilson=summary['task_success_rate_95pct_wilson'],
                      successful_cycles=summary['successful_cycles'], target_cycles=400,
                      physical_cycles_completed=summary['physical_cycles_completed'],
                      source_evidence_unchanged=True, all_40_trace_audits='passed',
                      summary=str(output / 'summary.json'), report=str(output / 'REPORT.md'))
    atomic_json(output / 'completion.json', completion)
    experiment = Path(manifest['created_from'])
    atomic_json(experiment / 'closedloop_evaluation.json', completion)
    if manifest.get('policy_type') == 'act':
        return
    original_report = experiment / 'REPORT.md'
    text = original_report.read_text() if original_report.exists() else ''
    new = (f"Closed-loop evaluation of the final 20k checkpoint on all 40 sealed test episodes is complete. "
           f"Full ten-cycle task success: **{summary['successful_episodes']}/40 "
           f"({100*summary['complete_test_success_rate']:.1f}%)**. "
           f"Physical cycles: {summary['physical_cycles_completed']}/400; "
           f"successful deposits/returns: {summary['successful_cycles']}/400. "
           f"See [closed-loop report](../{output.name}/REPORT.md). Offline diagnostics below remain unchanged.")
    original_report.write_text('\n'.join(new if line.startswith('Closed-loop evaluation of the final 20k checkpoint')
                                        else line for line in text.splitlines())+'\n')


def run(args, manifest):
    output = args.output.resolve()
    lock = (output / '.controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = evaluation_environment(output)
    selected = manifest['scenes'][:args.limit] if args.limit else manifest['scenes']
    gpu_ids = args.gpus.split(',')
    active = {}
    guard = threading.Lock()
    stopped = threading.Event()

    def worker(gpu, scenes):
        for scene in scenes:
            if stopped.is_set():
                break
            eid = scene['source_export_episode']
            if digest(scene['reference']) != scene['reference_sha256']:
                raise ValueError('Reference changed after preparation')
            parent = output / 'scenes' / f'export_episode_{eid:04d}'
            parent.mkdir(parents=True, exist_ok=True)
            existing = sorted(parent.glob('attempt_*'))
            statuses = [read(p / 'evaluation/episodes/episode_0001/episode_status.json') for p in existing
                        if (p / 'evaluation/episodes/episode_0001/episode_status.json').is_file()]
            if any(row['status'] == 'completed' for row in statuses):
                continue
            attempt = parent / f'attempt_{len(existing)+1:02d}'
            attempt.mkdir()
            command = evaluation_command(manifest, scene, attempt, gpu)
            atomic_json(attempt / 'launch.json', dict(command=command, started_utc=now(),
                        source_export_episode=eid, gpu=gpu, controller_pid=os.getpid()))
            with (attempt / 'launch.log').open('w') as log:
                process = subprocess.Popen(command, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                with guard:
                    active[gpu] = dict(source_export_episode=eid, child_pid=process.pid, attempt=str(attempt))
                while process.poll() is None:
                    with guard:
                        summary = summarize(output, manifest, dict(active))
                    print(f"{now()} completed={summary['completed_episodes']}/40 active={dict(active)}", flush=True)
                    time.sleep(15)
                with guard:
                    active.pop(gpu, None)
                    summary = summarize(output, manifest, dict(active))
                atomic_json(attempt / 'launch_result.json', dict(exit_code=process.returncode, finished_utc=now()))
                path = attempt / 'evaluation/episodes/episode_0001/episode_status.json'
                if process.returncode or not path.is_file() or read(path)['status'] != 'completed':
                    stopped.set()
                    print(f'Infrastructure error in source {eid}; preserve attempt and inspect {attempt}', flush=True)
                    break
                print(f"Completed source {eid}: task_success={read(path)['repeat_collection_success']}", flush=True)
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(worker, gpu, selected[i::len(gpu_ids)]) for i, gpu in enumerate(gpu_ids)]
        for future in as_completed(futures):
            future.result()
    summary = summarize(output, manifest)
    summary['controller_status'] = 'infrastructure_error' if stopped.is_set() else ('complete' if summary['completed_episodes']==40 else 'subset_complete')
    summary['controller_pid'] = os.getpid()
    atomic_json(output / 'summary.json', summary)
    publish_completion(output, manifest, summary)
    return 1 if stopped.is_set() else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--policy-type', choices=('smolvla', 'act'), default='smolvla')
    p.add_argument('--replan-interval', type=int)
    p.add_argument('--limit', type=int, default=0, help='Run a preparation subset; manifest always retains all 40')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--wait-for-pids', type=int, nargs='*', default=[],
                   help='Wait for already launched initial trials before resuming; never rerun completed outcomes')
    args = p.parse_args()
    if args.limit < 0 or args.limit > 40:
        p.error('limit must be in 0..40')
    manifest = prepare(args.experiment, args.output, args.policy_type, args.replan_interval)
    if args.prepare_only:
        summarize(args.output.resolve(), manifest)
        print(f'Prepared 40 sealed scenes: {args.output.resolve()}', flush=True)
        return
    while any(process_active(pid) for pid in args.wait_for_pids):
        print(f'{now()} waiting for initial trial processes {args.wait_for_pids}; all 40 scenes remain queued', flush=True)
        time.sleep(15)
    raise SystemExit(run(args, manifest))


if __name__ == '__main__':
    main()
