"""Combine distinct scenes of one frozen recovery candidate without selecting reruns."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_postdig_handoff import audit as handoff_audit

REVIEWED = [1, 2, 4, 9, 10, 15, 18, 24, 29, 31]
CONTROLS = [5, 25, 36]
CORE = ['act_stall_recovery.py', 'act_recovery_expert.py', 'excavator_reachability.py', 'eval_smolvla_success.py']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(record):
    result = record['result']; folder = Path(result['video']).parent
    rows = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
    triggers = result['triggers']
    extra = dict(loaded_exit_verified=False, maximum_tip_rise_m=0., actual_override_seconds=result['override_steps'] / 10,
                 source=record.get('source_experiment'), video=result['video'])
    if triggers:
        start = triggers[0]['step']
        origin = np.asarray(rows[start]['input_bucket_landmarks_world']['tip'])
        maximum = 0.
        for index, row in enumerate(rows[start:], start):
            delta = float(row['input_bucket_landmarks_world']['tip'][2] - origin[2])
            maximum = max(maximum, delta)
            previous_particles = rows[index - 1]['particle_transfer_metrics'] if index else row['particle_transfer_metrics']
            if delta >= 1. and previous_particles['retained_particles'] >= 10:
                extra['loaded_exit_verified'] = True
        extra['maximum_tip_rise_m'] = maximum
        extra['trigger_step'] = start
    extra['handoff'] = handoff_audit(folder)
    extra['success'] = result['success']
    extra['deposited_particles'] = result['transport']['terminal']['payload_particles']
    extra['terminal_retained_particles'] = result['transport']['terminal']['retained_particles']
    extra['override_steps'] = result['override_steps']
    extra['handoff_phase'] = ','.join(h['details'].get('phase', '') for h in result['handbacks'])
    if not triggers:
        extra['assessment'] = '未接管；不将复跑差异归因于脱困'
    elif result['success']:
        extra['assessment'] = '完整任务成功；执行层混合策略'
    elif extra['handoff_phase'] == 'completed_secure_align' and extra['deposited_particles'] > 0:
        extra['assessment'] = '已对位交回并部分入车，未达100粒'
    elif 'aborted' in extra['handoff_phase']:
        extra['assessment'] = '脱困超时/中止，任务未完成'
    elif 'alignment_plan_unresolved' in extra['handoff_phase']:
        extra['assessment'] = '已离土，受约束对位规划未解；不等于机械不可达'
    else:
        extra['assessment'] = '任务未完成，需检查实际接管/交接轨迹'
    return extra


def summarize(experiments, out):
    records, protected, signatures = {}, {}, []
    for root in experiments:
        protocol = read(root / 'protocol.json')
        if protocol['recovery_mode'] != 'secure_align':
            raise ValueError('Primary report only combines the fixed secure_align candidate')
        signatures.append(dict(settings=protocol['settings'], model=protocol['model_sha256'],
            core={name: sha(root / 'frozen_sources/scripts' / name) for name in CORE}))
        for name in ('protocol.json', 'results.json'):
            path = root / name
            if path.exists():
                protected[str(path.resolve())] = sha(path)
        if not (root / 'results.json').exists():
            continue
        for item in read(root / 'results.json').values():
            key = item['scene'], item['variant']
            if key in records:
                raise ValueError(f'Duplicate scene/variant; do not choose between reruns: {key}')
            records[key] = {**item, 'source_experiment': str(root.resolve())}
    if any(s != signatures[0] for s in signatures):
        raise ValueError('Controller code, settings, or model differ between primary batches')
    cases = []
    for scene in REVIEWED + CONTROLS:
        pair = {variant: records.get((scene, variant)) for variant in ('baseline', 'secure_align')}
        value = dict(scene=scene, user_reviewed_stall=scene in REVIEWED, paired_complete=all(pair.values()))
        if pair['baseline']:
            baseline = pair['baseline']['result']
            value['baseline'] = dict(success=baseline['success'],
                deposited_particles=baseline['transport']['terminal']['payload_particles'], video=baseline['video'])
        if pair['secure_align']:
            value['candidate'] = describe(pair['secure_align'])
        cases.append(value)
    reviewed = [c for c in cases if c['user_reviewed_stall'] and c['paired_complete']]
    controls = [c for c in cases if not c['user_reviewed_stall'] and c['paired_complete']]
    summary = dict(scope='Ten user-reviewed development stalls plus three old-success controls. '
                        'Not an independent final benchmark; unchanged ACT plus explicit recovery program.',
        expected_reviewed_pairs=10, completed_reviewed_pairs=len(reviewed), expected_control_pairs=3,
        completed_control_pairs=len(controls),
        baseline_reviewed_successes=sum(c['baseline']['success'] for c in reviewed),
        candidate_reviewed_successes=sum(c['candidate']['success'] for c in reviewed),
        candidate_reviewed_success_with_intervention=sum(c['candidate']['success'] and c['candidate']['override_steps'] > 0 for c in reviewed),
        candidate_reviewed_success_without_intervention=sum(c['candidate']['success'] and c['candidate']['override_steps'] == 0 for c in reviewed),
        reviewed_recovery_triggered=sum(c['candidate']['override_steps'] > 0 for c in reviewed),
        reviewed_loaded_exit_verified=sum(c['candidate']['loaded_exit_verified'] for c in reviewed),
        baseline_control_successes=sum(c['baseline']['success'] for c in controls),
        candidate_control_successes=sum(c['candidate']['success'] for c in controls),
        intervention_control_regression_scenes=[c['scene'] for c in controls if c['baseline']['success']
            and not c['candidate']['success'] and c['candidate']['override_steps'] > 0],
        no_intervention_control_difference_scenes=[c['scene'] for c in controls if c['baseline']['success']
            != c['candidate']['success'] and c['candidate']['override_steps'] == 0],
        cases=cases, frozen_controller_signatures=signatures[0], source_artifact_sha256=protected,
        weights_unchanged=True, baseline_replaced=False)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# ACT执行层脱困：十场用户问题专项验证', '',
        f"已完成问题场景 {len(reviewed)}/10 对，正常对照 {len(controls)}/3 对。",
        f"已完成问题场景：同期基准成功{summary['baseline_reviewed_successes']}场，候选成功{summary['candidate_reviewed_successes']}场；"
        f"候选触发接管{summary['reviewed_recovery_triggered']}场，有带料离土证据{summary['reviewed_loaded_exit_verified']}场。", '',
        f"候选成功中，{summary['candidate_reviewed_success_with_intervention']}场发生接管，{summary['candidate_reviewed_success_without_intervention']}场未接管；未接管成功不归为脱困收益。", '',
        '同一冻结控制器/评估器逐字节、参数和源模型哈希在各批次一致。候选使用程序闭斗、离土、运输与对位，再交ACT卸料；不是新训练的模型。',
        '这些场景已被查看和用于开发，原27.5%基准不替换，不将本表换算成新的整体成功率。', '',
        '| 场景 | 基准成功/入车 | 候选成功/入车 | 接管步数 | 交回原因 | 判读 | 视频 |',
        '| --- | --- | --- | ---: | --- | --- | --- |']
    for case in cases:
        baseline, candidate = case.get('baseline'), case.get('candidate')
        if not case['paired_complete']:
            lines.append(f"| {case['scene']} | 待完成 | 待完成 | — | — | {'用户问题' if case['user_reviewed_stall'] else '正常对照'} | — |")
            continue
        lines.append(f"| {case['scene']} | {baseline['success']}/{baseline['deposited_particles']} | "
            f"{candidate['success']}/{candidate['deposited_particles']} | {candidate['override_steps']} | "
            f"{candidate['handoff_phase'] or '未接管'} | {candidate['assessment']} | [候选]({candidate['video']}) |")
    lines += ['', '解释限制：粒子/接触与闭环演化不宣称确定复现。没有接管的候选与基准结果不同，不能归因为脱困。',
              '有料动起来、闭斗抬起或部分入车都不等同于完整成功；完整评分仍是100粒入车且斗角≥−0.8rad连续三观测、500步时限。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('cases', 'frozen_controller_signatures', 'source_artifact_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize([p.resolve() for p in args.experiments], args.output.resolve())
