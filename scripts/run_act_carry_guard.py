"""Bounded paired development rollouts for ACT's explicit carry-attitude guard."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(os.environ.get('ACT_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
sys.path.insert(0, str(ROOT / 'scripts'))
from run_act_phw_diagnostics import snapshot, sha, read, write
from act_carry_guard import CarryAttitudeGuard, settings_dict


def analyze(folder):
    import numpy as np
    from audit_smolvla_success import audit_episode
    status = read(folder / 'episode_status.json')
    if status['status'] != 'completed' or status.get('error'):
        raise RuntimeError(f'Incomplete rollout: {folder}: {status.get("error")}')
    audit = audit_episode(folder, status)
    scene = read(folder / 'scene_context.json')
    guard = CarryAttitudeGuard(scene)
    rows = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
    arrival, carry_start, pitch = None, None, []
    for i, row in enumerate(rows):
        state = np.asarray(row['input_state'])
        over_bed, _ = guard.release_geometry(state, row['input_bucket_landmarks_world'])
        if over_bed and arrival is None:
            arrival = i
        previous = rows[i-1]['particle_transfer_metrics']['retained_particles'] if i else 0
        if (carry_start is None and previous >= 50 and state[3] <= -1.3 and -state[20] >= .25
                and abs(row['predicted'][0]) >= .08):
            carry_start = i
        if carry_start is not None and arrival is None:
            pitch.append(float(np.degrees(np.arctan2(state[17], state[16]))))
    stop = arrival if arrival is not None else len(rows)
    carry_rows = rows[carry_start:stop] if carry_start is not None else []
    peak = max((row['particle_transfer_metrics']['retained_particles'] for row in carry_rows), default=0)
    entry_retained = (rows[arrival-1]['particle_transfer_metrics']['retained_particles']
                      if arrival is not None and arrival > 0 else None)
    transfer = rows[-1]['particle_transfer_metrics']
    return dict(task_success=bool(status['success']), execution_source=status['execution_source'],
        scene_seed=status['scene_seed'], steps=len(rows), carry_start_step=carry_start,
        first_bed_entry_step=arrival, peak_retained_during_prebed_carry=peak,
        retained_at_bed_entry=entry_retained,
        retention_at_entry_ratio=float(entry_retained / peak) if peak and entry_retained is not None else None,
        final_retained=transfer['retained_particles'], final_deposited=transfer['payload_particles'],
        final_spillage=transfer['spillage_particles'], failure_mode=status.get('failure_mode'),
        prebed_pitch_max_deg=max(pitch) if pitch else None,
        modified_steps=status.get('carry_guard_modified_steps', 0),
        guard_infeasible_steps=sum(bool((row.get('carry_guard') or {}).get('infeasible')) for row in rows),
        guard_release_step=next((row['step'] for row in rows if (row.get('carry_guard') or {}).get('released')), None),
        audit=audit, video=str(folder / 'evaluation.mp4'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alignment-source', type=Path,
                        help='Reuse an audited successful expert replay with identical physics/reference inputs')
    parser.add_argument('--baseline-source', type=Path,
                        help='Reuse completed baseline cases after checking unchanged physics and checkpoint inputs')
    parser.add_argument('--parent-projection', choices=['balanced', 'arm_first'], default='arm_first')
    parser.add_argument('--release-geometry', choices=['current', 'opening_pose', 'load_point', 'load_and_pour'], default='load_point')
    parser.add_argument('--release-interlock', action=argparse.BooleanOptionalAction, default=False,
                        help='Experimental: hold the integrated bucket target closed until release geometry is reached.')
    parser.add_argument('--cases', default='reference,seed13_loss,seed4_success')
    parser.add_argument('--expert-prefix-steps', type=int, default=0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / 'outputs/train/excavator_act_phw_20261028_150222_336724_1423224/checkpoints/040000/pretrained_model'
    expected_model = '665554b8c4b48533f0606c5506e5ea8f12558ab76912beb80191043ae1c60924'
    if sha(checkpoint / 'model.safetensors') != expected_model:
        raise ValueError('Expected original PHW 40k checkpoint')
    reference = ROOT / 'outputs/experiments/act_phw_optimization_20261103_01/reference_setup.json'
    shutil.copy2(reference, out / 'reference_setup.json')
    protocol = dict(scope='paired development diagnostics, not independent evaluation',
        policy='original ACT PHW40k plus optional deterministic carry-attitude guard; hybrid is explicit',
        checkpoint=str(checkpoint), checkpoint_model_sha256=expected_model,
        settings=settings_dict(args.parent_projection, args.release_geometry,
                               bucket_open_interlock=args.release_interlock), max_expert_replays=1,
        max_model_episodes=3 if args.baseline_source else 6,
        cases=[dict(name='reference', seed=2, reference=True),
               dict(name='seed13_loss', seed=13, reference=False),
               dict(name='seed4_success', seed=4, reference=False)],
        physics='physx_move_walls_before_settling', fps=10, steps=350, replan_interval=3,
        note='Only current observation/rigid-body geometry used by guard. Original task scoring unchanged. '
             'No reference actions used by the hybrid model. No training or label changes.')
    selected_cases = args.cases.split(',')
    if not selected_cases or not set(selected_cases) <= {case['name'] for case in protocol['cases']}:
        raise ValueError('Unknown development case selection')
    protocol['cases'] = [case for case in protocol['cases'] if case['name'] in selected_cases]
    if args.expert_prefix_steps and (args.baseline_source or any(not case['reference'] for case in protocol['cases'])):
        raise ValueError('Controlled transport diagnostic requires fresh paired reference-only cases')
    protocol['expert_prefix_steps'] = args.expert_prefix_steps
    if args.expert_prefix_steps:
        protocol['scope'] = 'expert-prefix-assisted transport diagnostic, not end-to-end ACT evaluation'
    protocol['max_model_episodes'] = len(protocol['cases']) * (1 if args.baseline_source else 2)
    if args.alignment_source:
        protocol['reused_expert_alignment'] = str(args.alignment_source.resolve())
    if args.baseline_source:
        protocol['reused_baseline_source'] = str(args.baseline_source.resolve())
    write(out / 'protocol.json', protocol)
    state = dict(status='preparing_sources', completed={}, running=[])
    write(out / 'status.json', state)
    snapshot(out)
    frozen = out / 'frozen_sources'
    protected = {str(p): sha(p) for p in [*checkpoint.glob('*'),
        *sorted((ROOT / 'data/export_phw/data').rglob('*.parquet'))] if p.is_file()}
    write(out / 'inputs_sha256.json', protected)

    def task(name, seed, gpu, *, reference=False, replay=False, guarded=False):
        destination = out / name
        command = [sys.executable, '-u', str(frozen / 'scripts/eval_smolvla_success.py'),
            '--policy-type', 'act', '--checkpoint', str(checkpoint), '--dataset', str(ROOT / 'data/export_phw'),
            '--output', str(destination), '--episodes', '1', '--seed', str(seed), '--steps', '350',
            '--min-particles', '100', '--replan-interval', '3', '--initial-episode', '0',
            '--workers', '1', '--gpus', str(gpu), '--cpu-threads', '8', '--headless',
            '--fresh-simulator', '--render-every-physics-step', '--soil-backend', 'physx',
            '--no-sand-walls-during-settle', '--video', '--no-save-trajectory']
        if reference:
            command += ['--reference-episode', str(out / 'reference_setup.json')]
        if replay:
            command.append('--replay-reference')
        if guarded:
            command += ['--act-carry-guard', '--act-carry-parent-projection', args.parent_projection,
                        '--act-carry-release-geometry', args.release_geometry]
        if args.expert_prefix_steps and not replay:
            command += ['--expert-prefix-steps', str(args.expert_prefix_steps)]
        return dict(name=name, output=str(destination), command=command, replay=replay, guarded=guarded)

    def execute(tasks):
        children, logs = [], []
        try:
            for job in tasks:
                log = (out / (job['name'] + '.log')).open('x')
                logs.append(log)
                env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                    HF_HOME=str(ROOT / 'models/hf-cache'), TORCH_HOME=str(ROOT / 'models/torch-cache'),
                    OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                child = subprocess.Popen(job['command'], cwd=frozen, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                job['pid'] = child.pid
                children.append((job, child))
            while any(child.poll() is None for _, child in children):
                state['running'] = [job for job, child in children if child.poll() is None]
                write(out / 'status.json', state)
                for job, child in children:
                    if child.poll() not in (None, 0):
                        raise RuntimeError(f"{job['name']} exited {child.returncode}; see its log")
                time.sleep(5)
            for job, child in children:
                if child.returncode:
                    raise RuntimeError(f"{job['name']} exited {child.returncode}")
                episode = Path(job['output']) / 'episodes/episode_0001'
                summary = analyze(episode)
                expected_source = ('expert_replay' if job['replay'] else
                    ('expert_prefix_then_model_with_carry_guard' if job['guarded'] else 'expert_prefix_then_model')
                    if args.expert_prefix_steps else 'model_with_carry_guard' if job['guarded'] else 'model')
                if summary['execution_source'] != expected_source:
                    raise ValueError('Hybrid/expert/model execution sources were mixed')
                state['completed'][job['name']] = summary
                write(out / 'results.json', state['completed'])
                write(out / 'status.json', state)
                print(json.dumps(dict(case=job['name'], **{k:summary[k] for k in
                    ('task_success', 'final_deposited', 'final_retained', 'first_bed_entry_step',
                     'retained_at_bed_entry', 'modified_steps')})), flush=True)
        finally:
            for _, child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            for _, child in children:
                if child.poll() is None:
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            for log in logs:
                log.close()

    try:
        state['status'] = 'expert_alignment'
        if args.alignment_source:
            source = args.alignment_source.resolve()
            old_config = read(source / 'config.json')
            if (old_config['execution_source'] != 'expert_replay' or
                    old_config['reference_setup_sha256'] != sha(out / 'reference_setup.json')):
                raise ValueError('Reused expert replay has different reference inputs')
            old_sources = read(source / 'provenance.json')['source_files']
            physics_files = [name for name in old_sources if name.startswith('ExcavatorVLA-smolvla/')
                             or name.startswith('ExcavatorVLA-isaac/assets/') or name == 'run_isaac_python.sh']
            for name in physics_files:
                if sha(frozen / name) != old_sources[name]:
                    raise ValueError(f'Reused expert replay has different physics source: {name}')
            summary = analyze(source / 'episodes/episode_0001')
            if summary['execution_source'] != 'expert_replay' or not summary['task_success']:
                raise ValueError('Reused expert replay did not complete successfully')
            (out / 'expert_reference').symlink_to(source, target_is_directory=True)
            state['completed']['expert_reference'] = summary
            state['reused_expert_alignment'] = str(source)
            write(out / 'status.json', state)
        else:
            execute([task('expert_reference', 2, 1, reference=True, replay=True)])
        if not state['completed']['expert_reference']['task_success']:
            raise RuntimeError('Reference expert alignment failed; no model comparison started')
        if args.baseline_source:
            pending = []
            for index, case in enumerate(protocol['cases']):
                source = args.baseline_source.resolve() / (case['name'] + '_baseline')
                old_config = read(source / 'config.json')
                if old_config['execution_source'] != 'model' or old_config['act_carry_guard']:
                    raise ValueError('Reused baseline must be the unmodified model')
                old_sources = read(source / 'provenance.json')['source_files']
                for name, expected in old_sources.items():
                    if (name.startswith('ExcavatorVLA-smolvla/') or
                            name.startswith('ExcavatorVLA-isaac/assets/') or name == 'run_isaac_python.sh'):
                        if sha(frozen / name) != expected:
                            raise ValueError(f'Reused baseline has different physics source: {name}')
                old_files = read(source / 'provenance.json')['checkpoint_files']
                if old_files['model.safetensors'] != expected_model:
                    raise ValueError('Reused baseline weights differ')
                (out / (case['name'] + '_baseline')).symlink_to(source, target_is_directory=True)
                state['completed'][case['name'] + '_baseline'] = analyze(source / 'episodes/episode_0001')
                pending.append(task(case['name'] + '_guard', case['seed'], index % 2,
                                    reference=case['reference'], guarded=True))
            for start in range(0, len(pending), 2):
                state['status'] = 'revised_guards'
                execute(pending[start:start+2])
        else:
            for case in protocol['cases']:
                state['status'] = 'paired_' + case['name']
                execute([task(case['name'] + '_baseline', case['seed'], 0, reference=case['reference']),
                         task(case['name'] + '_guard', case['seed'], 1, reference=case['reference'], guarded=True)])
        for path, expected in protected.items():
            if sha(Path(path)) != expected:
                raise RuntimeError(f'Original input changed: {path}')
        source_manifest = read(out / 'source_snapshot.json')
        for path, expected in source_manifest['files'].items():
            if sha(frozen / path) != expected:
                raise RuntimeError(f'Frozen source changed: {path}')
        state.update(status='completed', running=[], inputs_unchanged=True)
        write(out / 'status.json', state)
    except BaseException as error:
        state.update(status='failed', running=[], error=str(error))
        write(out / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
