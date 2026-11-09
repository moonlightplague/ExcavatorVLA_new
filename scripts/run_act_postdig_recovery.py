"""Paired, explicitly hybrid post-dig recovery experiment on reviewed ACT stalls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from act_stall_recovery import settings_dict

ROOT = Path(os.environ.get('ACT_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
REVIEWED = [1, 2, 4, 9, 10, 15, 18, 24, 29, 31]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(out, scenes, mode='secure_lift', controller_source=None):
    out.mkdir(parents=True, exist_ok=False)
    manifest = read(ROOT / 'docs/baselines/act_phw40k_carry275_20261105.json')
    archive = ROOT / manifest['archive']
    source = archive / manifest['checkpoint']
    controller_settings = settings_dict(mode)
    controller_scripts = ROOT / 'scripts'
    if controller_source is not None:
        controller_source = controller_source.resolve()
        previous = read(controller_source / 'protocol.json')
        if previous['model_sha256'] != manifest['model_sha256'] or previous['recovery_mode'] != mode:
            raise ValueError('Controller reuse requires the same baseline weights and recovery mode')
        controller_settings = previous['settings']
        controller_scripts = controller_source / 'frozen_sources/scripts'
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_id=manifest['baseline_id'], archive=str(archive), checkpoint=str(source),
        model_sha256=manifest['model_sha256'], user_reviewed_postdig_stalls=REVIEWED,
        scenes=scenes, original_success_control_scenes=[s for s in scenes if s not in REVIEWED],
        variants=['baseline', mode], recovery_mode=mode, steps=500, fps=10, settings=controller_settings,
        controller_source_experiment=str(controller_source) if controller_source is not None else None,
        intervention=('One bounded closure/lift recovery; ACT performs turning, approach and unloading'
                      if mode == 'secure_lift' else
                      'Detect a four-second low-fill stall, close/lift then turn and align over the truck; '
                      'hand back while closed so ACT performs unloading. No score information used for control'),
        curl_dip_m=controller_settings.get('curl_dip_m', 0.),
        no_training=True, weights_unchanged=True, original_physics_and_scoring=True,
        carry_constraint='Original guard/settings after closure; collision-checked progressive closure '
                         'temporarily owns the attitude constraint to avoid competing controllers',
        scope='Viewed development scenes; explicitly hybrid ACT, not a new model or independent final success rate',
        detector_selection=('Late detector: 25-second enable time selected on viewed trajectories, '
                            'all ten user stalls and none of eleven old successes' if mode == 'secure_lift' else
                            'Early low-fill detector selected on viewed trajectories. Covers ten user stalls '
                            'and also triggers three old successes (5,25,36); check these as regression controls'),
        next_gate='Expand only after actual task or sustained autonomous loaded-transport improvement; '
                  'do not count a lift or motion-only rescue as full task success')
    write(out / 'protocol.json', protocol)
    frozen = out / 'frozen_sources'; frozen.mkdir()
    for path in (archive / 'frozen_sources').iterdir():
        if path.name == 'scripts':
            shutil.copytree(path, frozen / 'scripts', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        else:
            (frozen / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    for name in ('act_stall_recovery.py', 'act_recovery_expert.py', 'excavator_reachability.py'):
        shutil.copy2(controller_scripts / name, frozen / 'scripts' / name)
    shutil.copy2(ROOT / 'scripts/run_act_postdig_recovery.py', frozen / 'scripts/run_act_postdig_recovery.py')
    robot_source = (controller_source / 'robot_model.json' if controller_source is not None else
                    ROOT / 'outputs/experiments/act_recovery_collection_progressive_20261105_02/robot_model.json')
    shutil.copy2(robot_source, out / 'robot_model.json')
    entry = frozen / 'scripts/eval_smolvla_success.py'
    code = entry.read_text()
    patches = [
        ("CAMERA_IDS = ('0', '1', '2')", "from act_stall_recovery import make_stall_guard_class, settings_dict as stall_settings\n"
         "STALL_RECOVERY_ENABLED = os.environ.get('ACT_POSTDIG_RECOVERY') == '1'\n"
         "if STALL_RECOVERY_ENABLED:\n"
         "    CarryAttitudeGuard = make_stall_guard_class(CarryAttitudeGuard, json.loads(Path(os.environ['ACT_STALL_ROBOT_MODEL']).read_text()), os.environ.get('ACT_POSTDIG_MODE', 'secure_lift'))\n\n"
         "CAMERA_IDS = ('0', '1', '2')"),
        ("    if args.expert_prefix_steps:\n        config.update", "    if STALL_RECOVERY_ENABLED:\n"
         "        config.update(execution_source='model_with_carry_guard_and_postdig_recovery',\n"
         "                      stall_recovery_settings=stall_settings(os.environ.get('ACT_POSTDIG_MODE', 'secure_lift')), weights_unchanged=True)\n"
         "    if args.expert_prefix_steps:\n        config.update"),
        ("                    status['clipped_action_dimensions'] += clipped_dimensions", "                    if carry_guard_trace and (carry_guard_trace.get('stall_recovery') or {}).get('handback'):\n"
         "                        actions = []\n"
         "                        if policy is not None:\n"
         "                            policy.reset()\n"
         "                    status['clipped_action_dimensions'] += clipped_dimensions")]
    for old, new in patches:
        if code.count(old) != 1:
            raise ValueError('Unexpected retained evaluator source; patch requires exactly one matching block')
        code = code.replace(old, new)
    entry.write_text(code)
    write(out / 'evaluator_patch.json', dict(source=str(archive / 'frozen_sources/scripts/eval_smolvla_success.py'),
        original_sha256=sha(archive / 'frozen_sources/scripts/eval_smolvla_success.py'),
        patched_sha256=sha(entry), changes=[dict(original=a, replacement=b) for a, b in patches]))
    (out / 'references').mkdir()
    original = ROOT / manifest['original_experiment'] / 'evaluation/episodes'
    template = read(ROOT / 'outputs/experiments/act_phw_optimization_20261103_01/reference_setup.json')
    for scene in scenes:
        folder = original / f'episode_{scene:04d}'
        context, initial = read(folder / 'scene_context.json'), read(folder / 'initial_observation.json')
        ref = dict(template)
        ref.update(profile=context['scene_profile'], task=initial['task'],
            initial_joints_deg=[math.degrees(v) for v in initial['state'][:4]],
            expected_initial_state=initial['state'], dig_target_xyz=context['observation_context']['dig_target_xyz'],
            sand_amount=context['scene_profile']['sand_amount_multiplier'],
            actions=[json.loads(line)['predicted'] for line in (folder / 'steps.jsonl').read_text().splitlines()],
            provenance=dict(source_episode=str(folder), purpose='Shared geometry only, no action replay'))
        write(out / 'references' / f'scene{scene}.json', ref)
    protected = [*source.glob('*'), *out.glob('*.json'), *out.glob('references/*.json'),
                 *frozen.glob('scripts/*.py')]
    write(out / 'inputs_sha256.json', {str(p): sha(p) for p in protected if p.is_file()})
    write(out / 'status.json', dict(status='prepared', expected_rollouts=2 * len(scenes)))
    return protocol


def assess(folder, variant, seed):
    from audit_smolvla_success import audit_episode
    from act_transport_diagnostics import analyze_transport
    status = read(folder / 'episode_status.json')
    if status['status'] != 'completed' or status['scene_seed'] != seed:
        raise ValueError('Episode failed infrastructure or seed check')
    expected = 'model_with_carry_guard' if variant == 'baseline' else 'model_with_carry_guard_and_postdig_recovery'
    if status['execution_source'] != expected:
        raise ValueError('Incorrect execution-source disclosure')
    audit = audit_episode(folder, status)
    transport = analyze_transport(folder)
    rows = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
    traces = [(r['step'], (r['carry_guard'].get('stall_recovery') or {})) for r in rows]
    triggers = [dict(step=i, details=t) for i, t in traces if t.get('triggered')]
    handbacks = [dict(step=i, details=t) for i, t in traces if t.get('handback')]
    overrides = sum(bool(t.get('override')) for _, t in traces)
    if variant == 'baseline' and (triggers or overrides):
        raise ValueError('Baseline received a recovery override')
    return dict(success=status['success'], policy_steps=status['policy_steps_executed'],
        triggers=triggers, handbacks=handbacks, override_steps=overrides,
        transport=transport, audit=audit, video=str(folder / 'evaluation.mp4'),
        execution_source=status['execution_source'])


def report(out, records):
    lines = ['# ACT挖后停滞：有限接管开发检查', '',
        '原PHW40k权重、原保料层、物理、评分、500步时限不变；候选为明确标注的执行层混合策略。',
        '触发条件仅使用在线28D状态、原模型动作及经过时间；不读取旧动作、旧阶段或任务评分。', '',
        '| 场景 | 变体 | 完整任务成功 | 入车 | 斗内 | 接管步数 | 交回ACT |',
        '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for item in records.values():
        result = item['result']; terminal = result['transport']['terminal']
        phases = ','.join(h['details'].get('phase', '') for h in result['handbacks'])
        lines.append(f"| {item['scene']} | {item['variant']} | {result['success']} | {terminal['payload_particles']} | "
            f"{terminal['retained_particles']} | {result['override_steps']} | {phases} |")
    lines += ['', '仅抬起或短时运动不算任务修复。粒子/接触演化不宣称逐粒子确定，报告同期对照，不替换27.5%历史基准。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def run(out, protocol):
    archive = Path(protocol['archive']); frozen = out / 'frozen_sources'
    sys.path[:0] = [str(frozen / 'scripts'), str(archive / 'runtime_dependencies')]
    template = read(archive / 'evidence/protocol.json')['command']
    jobs = [(scene, variant) for scene in protocol['scenes'] for variant in protocol['variants']]
    active, records = {}, {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    try:
        while jobs or active:
            for gpu in (0, 1):
                if gpu in active or not jobs:
                    continue
                scene, variant = jobs.pop(0); name = f'scene{scene}_{variant}'
                destination = out / 'evaluation' / name
                command = list(template); command[2] = str(frozen / 'scripts/eval_smolvla_success.py')
                for flag, value in [('--checkpoint', protocol['checkpoint']), ('--output', str(destination)),
                    ('--episodes', '1'), ('--seed', str(2000000 + scene - 1)), ('--workers', '1'), ('--gpus', str(gpu))]:
                    command[command.index(flag) + 1] = value
                command += ['--reference-episode', str(out / 'references' / f'scene{scene}.json')]
                env = dict(os.environ, ACT_POSTDIG_RECOVERY=str(int(variant != 'baseline')),
                    ACT_POSTDIG_MODE=protocol.get('recovery_mode', 'secure_lift'),
                    ACT_STALL_ROBOT_MODEL=str(out / 'robot_model.json'), PYTHONDONTWRITEBYTECODE='1',
                    HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                    HF_HOME=str(ROOT / 'models/hf-cache'), TORCH_HOME=str(ROOT / 'models/torch-cache'),
                    OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONPATH=str(archive / 'runtime_dependencies'))
                env.pop('CUDA_VISIBLE_DEVICES', None)
                log = (out / (name + '.log')).open('x')
                child = subprocess.Popen(command, cwd=frozen, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                active[gpu] = dict(name=name, scene=scene, variant=variant, destination=destination,
                                   command=command, child=child, log=log)
            write(out / 'status.json', dict(status='running', completed_rollouts=len(records),
                expected_rollouts=2 * len(protocol['scenes']), running=[dict(name=j['name'], pid=j['child'].pid,
                gpu=gpu) for gpu, j in active.items()]))
            for gpu, job in list(active.items()):
                code = job['child'].poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"{job['name']} failed; see its log")
                folder = job['destination'] / 'episodes/episode_0001'
                result = assess(folder, job['variant'], 2000000 + job['scene'] - 1)
                context, initial = read(folder / 'scene_context.json'), read(folder / 'initial_observation.json')
                geometry = dict(profile=context['scene_profile'], context=context['observation_context'],
                                initial_joints=initial['state'][:4])
                for other in records.values():
                    if other['scene'] == job['scene']:
                        if other['geometry']['profile'] != geometry['profile'] or other['geometry']['context'] != geometry['context']:
                            raise ValueError('Paired geometry differs')
                        if max(abs(a - b) for a, b in zip(other['geometry']['initial_joints'], geometry['initial_joints'])) > .05:
                            raise ValueError('Paired initial joints differ')
                records[job['name']] = dict(scene=job['scene'], variant=job['variant'], result=result,
                                            geometry=geometry, command=job['command'])
                write(out / 'results.json', records); report(out, records)
                print(json.dumps(dict(name=job['name'], success=result['success'], overrides=result['override_steps'],
                    handbacks=result['handbacks'], terminal=result['transport']['terminal'])), flush=True)
                job['log'].close(); del active[gpu]
            if jobs or active:
                time.sleep(5)
        for name, expected in read(out / 'inputs_sha256.json').items():
            if sha(name) != expected:
                raise ValueError(f'Protected input changed: {name}')
        write(out / 'status.json', dict(status='completed', completed_rollouts=len(records),
            expected_rollouts=len(records), weights_unchanged=True, baseline_replaced=False, running=[]))
    except BaseException as error:
        write(out / 'status.json', dict(status='failed', error=str(error), completed_rollouts=len(records)))
        raise
    finally:
        for job in active.values():
            if job['child'].poll() is None:
                os.killpg(job['child'].pid, signal.SIGTERM)
                try:
                    job['child'].wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(job['child'].pid, signal.SIGKILL); job['child'].wait()
            job['log'].close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', default='1,9,24,5')
    parser.add_argument('--recovery-mode', choices=['secure_lift', 'secure_align', 'secure_align_scoop'], default='secure_lift')
    parser.add_argument('--controller-source', type=Path, help='Reuse exact frozen controller code/settings from a prior experiment')
    args = parser.parse_args(); out = args.output.resolve()
    scenes = [int(s) for s in args.scenes.split(',')]
    if not scenes or len(scenes) != len(set(scenes)) or any(s < 1 or s > 40 for s in scenes):
        parser.error('Use distinct original scene indices in 1..40')
    protocol = prepare(out, scenes, args.recovery_mode, args.controller_source)
    run(out, protocol)
