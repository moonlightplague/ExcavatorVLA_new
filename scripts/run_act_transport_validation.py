"""Frozen, fixed-parameter validation of ACT plus coupled carry protection."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_act_phw_diagnostics import snapshot, sha, read, write

ROOT = Path(__file__).resolve().parents[1]


def empty_event_followup(transport, rows, folder):
    empty = transport.get('pretruck_empty_step')
    if empty is None:
        return None
    observations = [r['particle_transfer_metrics'] for r in rows[empty:empty + 31]]
    tail_file = folder / 'post_rollout_steps.jsonl'
    if len(observations) < 31 and tail_file.exists():
        tail = [json.loads(s)['particle_transfer_metrics'] for s in tail_file.read_text().splitlines()]
        observations += tail[1:1 + 31 - len(observations)]
    gap = transport['at_pretruck_empty']['bed_gap_m']
    max_deposit = max(o['payload_particles'] for o in observations)
    return dict(observations=len(observations), event_bed_gap_m=gap,
        max_followup_deposit=max_deposit,
        far_empty_no_early_deposit=bool(gap >= 2 and len(observations) == 31 and max_deposit < 100),
        deposit_during_followup=bool(max_deposit >= 100),
        note='Follow-up is at most 3 seconds. A near-truck empty bucket may be normal sand in flight; '
             'outside counts do not directly measure ground contact.')


def assess_episode(folder, expected_seed, evaluation_id):
    from audit_smolvla_success import audit_episode
    from act_transport_diagnostics import analyze_transport

    status = read(folder / 'episode_status.json')
    if status['scene_seed'] != expected_seed or status['evaluation_id'] != evaluation_id:
        raise ValueError(f'Wrong episode identity: {folder}')
    if status['status'] != 'completed':
        return dict(status='infrastructure_error', episode_status=status)
    if status['execution_source'] != 'model_with_carry_guard':
        raise ValueError('Expected ACT plus carry protection, without expert assistance')
    audit = audit_episode(folder, status)
    transport = analyze_transport(folder)
    rows = [json.loads(s) for s in (folder / 'steps.jsonl').read_text().splitlines()]
    followup = empty_event_followup(transport, rows, folder)
    prefix_stable = 0
    prefix_success = False
    for row in rows[:350]:
        prefix_stable = prefix_stable + 1 if row['deposited_particles'] >= 100 and row['bucket_position'] >= -.8 else 0
        prefix_success = prefix_success or prefix_stable >= 3
    tail = transport.get('post_rollout')
    return dict(status='completed', scene_seed=expected_seed,
        original_task_completed=status['success'], completed_within_350_steps=prefix_success,
        policy_steps=status['policy_steps_executed'], audit=audit, transport=transport,
        empty_event_followup=followup,
        guard_modified_steps=status.get('carry_guard_modified_steps', 0),
        guard_infeasible_steps=sum(bool((r.get('carry_guard') or {}).get('infeasible')) for r in rows),
        guard_release_step=next((r['step'] for r in rows if (r.get('carry_guard') or {}).get('released')), None),
        tail_diagnostic_status=(tail or {}).get('status', 'missing'),
        video=str(folder / 'evaluation.mp4'), post_rollout_video=str(folder / 'post_rollout.mp4'))


def summarize(out, protocol, records, state):
    valid = [r for r in records.values() if r['status'] == 'completed']
    starts = [r for r in valid if r['transport']['available']]
    empty = [r for r in starts if r['transport'].get('pretruck_empty_step') is not None]
    far = [r for r in empty if r['empty_event_followup']['far_empty_no_early_deposit']]
    in_flight = [r for r in empty if r['empty_event_followup']['deposit_during_followup']]
    sustained = [r for r in starts if r['transport'].get('pretruck_empty_step') is None and
        (r['transport']['original_task_success'] or r['transport'].get('actual_load_arrival_step') is not None or
         (r['transport']['loaded_bed_gap_reduction_m'] >= 3 and
          r['transport']['checkpoints']['8']['elapsed_action_steps'] >= 80))]
    summary = dict(status=state, expected_episodes=protocol['episodes'], completed_episodes=len(valid),
        infrastructure_error_episodes=sum(r['status'] != 'completed' for r in records.values()),
        task_completed=sum(r['original_task_completed'] for r in valid),
        task_completed_within_350_steps=sum(r['completed_within_350_steps'] for r in valid),
        analyzable_loaded_transport=len(starts), pretruck_empty_events=len(empty),
        far_empty_without_early_deposit=len(far), empty_then_deposit_in_followup=len(in_flight),
        sustained_transport_without_empty=len(sustained),
        no_valid_loaded_transport=len(valid)-len(starts),
        guard_intervened_episodes=sum(r['guard_modified_steps'] > 0 for r in valid),
        tail_diagnostic_errors=sum(r['tail_diagnostic_status'] != 'completed' for r in valid),
        note='Fixed candidate only, no fresh raw-ACT control. No causal improvement rate is inferred. '
             'Task noncompletion is separate from transport loss; reachability classification is pending externally.')
    write(out / 'validation_summary.json', summary)
    write(out / 'results.json', records)
    lines = ['# ACT＋联动保料：40场固定验证', '',
        f"状态：{state}；完成 {len(valid)}/{protocol['episodes']} 场。", '',
        '固定PHW40k、−130°、斗杆优先、原opening_pose解除；40场全部为约束版，不调参数，不使用专家动作。',
        f"预先固定seed {protocol['seed']}–{protocol['seed']+protocol['episodes']-1}；每场新模拟器，PhysX、10 Hz、500步上限、每3步重规划。", '',
        f"完整任务完成：{summary['task_completed']}/{len(valid)}；其中前350步内完成 {summary['task_completed_within_350_steps']} 场。",
        f"进入可分析载料运输：{len(starts)}；到车前空斗事件 {len(empty)}；其中距车斗至少2 m且3秒观察内未达到入车门槛 {len(far)}。",
        f"空斗事件后短观察内达到入车门槛：{len(in_flight)}；不能把这些直接当作漏在地上。",
        f"有持续带料/到车证据且未出现该空斗事件：{len(sustained)}；未进入可分析载料运输：{len(valid)-len(starts)}。", '',
        '距离导致的任务未完成由另一边判断，本报告不按策略结果事后删除场景或重算分母。途中漏空、没有足量挖起、带料停滞与完整任务未完成分别保存。',
        '所有计数含采样与空中转移限制。末尾30步零命令观察独立于原终点评分，保留积分目标而非冻结实测姿态。本批没有同期原ACT对照，不能据此证明改进幅度。', '',
        '| 场次/seed | 完成任务 | 运输起点 | 空斗步 | 带料接近/m | 原终点斗内/入车/外 | 观察后斗内/入车/外 |',
        '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for key, r in sorted(records.items(), key=lambda x: int(x[0])):
        if r['status'] != 'completed':
            lines.append(f'| {key} | 基础设施错误 | — | — | — | — | — |')
            continue
        t = r['transport']; tail = (t.get('post_rollout') or {}).get('final', {})
        def counts(d):
            return '/'.join(str(d.get(k, '—')) for k in ('retained_particles', 'payload_particles', 'spillage_particles'))
        gap = f"{t['loaded_bed_gap_reduction_m']:.2f}" if t['available'] else '—'
        lines.append(f"| {key}/{r['scene_seed']} | {'是' if r['original_task_completed'] else '未完成，原因待分类'} | "
            f"{t['carry_start_step']} | {t.get('pretruck_empty_step')} | {gap} | {counts(t['terminal'])} | {counts(tail)} |")
    lines += ['', '视频、动作与物理审计见 `evaluation/episodes/`；完整运输分段与空斗后续观察见 `results.json`。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary


def run_prepared(out):
    protocol = read(out / 'protocol.json')
    frozen = out / 'frozen_sources'
    command = protocol['command']
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HOME=str(Path(protocol['workspace']) / 'models/hf-cache'),
        TORCH_HOME=str(Path(protocol['workspace']) / 'models/torch-cache'),
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    records = {}
    with (out / 'evaluation.log').open('x') as log:
        child = subprocess.Popen(command, cwd=frozen, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        write(out / 'launch.json', dict(controller_pid=os.getpid(), evaluation_pid=child.pid, command=command))
        def stop(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while True:
                changed = False
                manifest = out / 'evaluation/provenance.json'
                if manifest.exists():
                    evaluation_id = read(manifest)['evaluation_id']
                    for path in sorted((out / 'evaluation/episodes').glob('*/episode_status.json')):
                        index = int(path.parent.name.split('_')[-1])
                        if str(index) in records:
                            continue
                        records[str(index)] = assess_episode(path.parent, protocol['seed'] + index - 1, evaluation_id)
                        changed = True
                code = child.poll()
                summary = summarize(out, protocol, records, 'running') if changed or not (out/'validation_summary.json').exists() else read(out/'validation_summary.json')
                progress = {p.parent.name: read(p) for p in (out/'evaluation/workers').glob('*/progress.json')}
                write(out/'status.json', dict(status='running', completed_episodes=summary['completed_episodes'],
                    expected_episodes=protocol['episodes'], updated_utc=datetime.now(timezone.utc).isoformat(),
                    task_completed=summary['task_completed'], workers=progress))
                if changed:
                    print(json.dumps(summary, ensure_ascii=False), flush=True)
                if code is not None:
                    if code or len(records) != protocol['episodes'] or summary['infrastructure_error_episodes']:
                        raise RuntimeError(f'Evaluation exited {code}, received {len(records)}/{protocol["episodes"]} episodes')
                    break
                time.sleep(5)
            for name, expected in protocol['protected_files'].items():
                if sha(Path(name)) != expected:
                    raise RuntimeError(f'Protected input changed: {name}')
            source_manifest = read(out/'source_snapshot.json')
            for name, expected in source_manifest['files'].items():
                if sha(frozen/name) != expected:
                    raise RuntimeError(f'Frozen source changed: {name}')
            final = summarize(out, protocol, records, 'completed')
            write(out/'status.json', dict(**final, inputs_unchanged=True,
                updated_utc=datetime.now(timezone.utc).isoformat(), workers={}))
            print(json.dumps(final, ensure_ascii=False), flush=True)
        except BaseException as error:
            write(out/'status.json', dict(status='error', completed_episodes=len(records), error=str(error)))
            raise
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=2000000)
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--run-prepared', type=Path)
    args = parser.parse_args()
    if args.run_prepared:
        run_prepared(args.run_prepared.resolve())
        return
    if args.output is None or args.episodes < 1 or args.seed < 0:
        parser.error('Output, positive episode count and nonnegative seed are required')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'outputs/experiments/act_spillage_four_rounds_20261104_01/transport_round4b/seed4_carry130_r1/config.json'
    selected = read(source)
    checkpoint = Path(selected['checkpoint']); dataset = Path(selected['dataset'])
    if sha(checkpoint/'model.safetensors') != '665554b8c4b48533f0606c5506e5ea8f12558ab76912beb80191043ae1c60924':
        raise ValueError('Expected original PHW40k checkpoint')
    expected = dict(max_carry_pitch_deg=-130., parent_projection='arm_first', release_geometry='opening_pose',
        near_bed_relax_distance_m=0., opening_pose_near_edge_m=.3, opening_pose_target_bucket_rad=-.8)
    if any(selected['carry_guard_settings'][k] != v for k,v in expected.items()):
        raise ValueError('Selected candidate settings differ from the validated configuration')
    protected = {str(p):sha(p) for p in checkpoint.iterdir() if p.is_file()}
    protected[str(dataset/'meta/info.json')] = sha(dataset/'meta/info.json')
    write(out/'selected_candidate_config.json', selected)
    protocol = dict(workspace=str(ROOT), episodes=args.episodes, seed=args.seed,
        seeds=list(range(args.seed,args.seed+args.episodes)), steps=500, fps=10, replan_interval=3,
        scope='fixed candidate validation on predeclared new scene seeds; no tuning or raw-ACT control',
        checkpoint=str(checkpoint), dataset=str(dataset), carry_guard_settings=selected['carry_guard_settings'],
        protected_files=protected, source_candidate_config_sha256=sha(source),
        task_success='100 initial-pile particles in truck, bucket >= -0.8 rad for three observations',
        transportation='Fixed act_transport_diagnostics definitions; report loaded motion separately from parking',
        empty_followup='Report distance >=2m and 3s without reaching deposit threshold separately from near-bed emptying',
        reachability='Distance-caused noncompletion pending external classification; no outcome-based exclusion',
        post_rollout='30 separate zero-command observations; no change to scored terminal result',
        no_training=True, gpus=args.gpus, created_utc=datetime.now(timezone.utc).isoformat())
    write(out/'protocol.json',protocol)
    snapshot(out)
    frozen=out/'frozen_sources'
    command=[sys.executable,'-u',str(frozen/'scripts/eval_smolvla_success.py'),
        '--policy-type','act','--checkpoint',str(checkpoint),'--dataset',str(dataset),
        '--output',str(out/'evaluation'),'--episodes',str(args.episodes),'--seed',str(args.seed),
        '--steps','500','--min-particles','100','--replan-interval','3','--initial-episode','0',
        '--workers','2','--gpus',args.gpus,'--cpu-threads','8','--headless','--fresh-simulator',
        '--render-every-physics-step','--soil-backend','physx','--no-sand-walls-during-settle',
        '--video','--no-save-trajectory','--act-carry-guard','--act-carry-parent-projection','arm_first',
        '--act-carry-release-geometry','opening_pose','--no-act-carry-release-interlock',
        '--act-carry-max-pitch-deg','-130',
        '--act-carry-near-bed-relax-m','0','--act-carry-opening-near-edge-m','.3',
        '--act-carry-opening-target-rad','-.8','--post-rollout-observe-steps','30']
    protocol['command']=command;write(out/'protocol.json',protocol)
    write(out/'status.json',dict(status='prepared',expected_episodes=args.episodes,completed_episodes=0))
    os.execv(sys.executable,[sys.executable,'-u',str(frozen/'scripts/run_act_transport_validation.py'),
                            '--run-prepared',str(out)])


if __name__=='__main__':
    main()
