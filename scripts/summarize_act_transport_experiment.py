"""Inspectable summary of the bounded ACT transport-retention experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from act_transport_diagnostics import analyze_transport


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--complete',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    records=[]
    for status_file in sorted(root.glob('transport_round*/*/episodes/*/episode_status.json')):
        folder=status_file.parent
        if read(status_file)['status']!='completed':
            continue
        config=read(folder.parents[1]/'config.json')
        transport=analyze_transport(folder)
        record=dict(round=folder.parents[2].name,variant=folder.parents[1].name,
                    horizon=config['steps'],evaluation_id=config['evaluation_id'],transport=transport)
        if not transport['available']:
            outcome='未进入可分析的载料运输'
        elif transport['pretruck_empty_step'] is not None:
            outcome='到车前漏空/仅少量入车'
        elif transport['original_task_success']:
            outcome='带料完成原成功条件'
        elif (transport['terminal']['retained_particles']>=50 and
              transport['loaded_bed_gap_reduction_m']>=3 and
              transport['checkpoints']['8']['elapsed_action_steps']>=80):
            outcome='持续带料运动，尚未完成任务'
        elif transport['terminal']['retained_particles']>=50:
            outcome='仍留料，持续运输证据不足'
        else:
            outcome='需结合到车与卸料轨迹核对'
        record['transport_interpretation']=outcome
        records.append(record)
    source=Path(__file__).parent/'act_transport_diagnostics.py'
    payload=dict(scope='development transport loss, not independent generalization or success-rate estimation',
        records=records, completed_transport_rollouts=len(records), archived_opening_rollouts=6,
        analysis_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        interpretation='Task noncompletion and distance-caused reachability exceptions are separate from '
            'pre-truck emptying. New scooping means observed-peak ratios are not fixed-particle retention rates.')
    (root/'transport_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (root/'final_transport_analysis.py').write_bytes(source.read_bytes())
    lines=['# ACT 运输途中漏沙：开发实验汇总','',
        '目标是挖起沙后、到车前运输中漏空在地上的失败。保持原 PHW 40k；远距离任务未完成由另一边判断可达性，不能自动当作保料失败。',
        '',f'已完成运输对照 {len(records)} 条；另有此前偏离重点的开斗探索6条，已封存，不算运输优化结果。',
        '', '350步与500步为不同开发诊断上限，不重算旧成功率。粒子为观测计数；继续拾料、空中转移和未观察到经过铲斗的粒子都会影响简单比例。',
        '', '| 轮次/场景 | 变体 | 上限 | 运输起点 | 漏空步 | 原终点斗内/入车/外 | 观察后斗内/入车/外 | 带料接近/m | 判断 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in records:
        t=r['transport'];terminal=t['terminal'];progress=t.get('loaded_bed_gap_reduction_m')
        variant=r['variant'].split('_',1)[1].rsplit('_r',1)[0]
        tail=(t.get('post_rollout') or {}).get('final',{})
        def counts(values):
            return '/'.join(str(values.get(k,'—')) for k in ('retained_particles','payload_particles','spillage_particles'))
        carry=t['carry_start_step'] if t['carry_start_step'] is not None else '—'
        empty=t.get('pretruck_empty_step') if t.get('pretruck_empty_step') is not None else '—'
        lines.append(f"| {r['round'].replace('transport_','')}/{t['scene_seed']} | {variant} | {r['horizon']} | "
            f"{carry} | {empty} | {counts(terminal)} | {counts(tail)} | {progress:.2f} | {r['transport_interpretation']} |" if progress is not None else
            f"| {r['round']}/{t['scene_seed']} | {variant} | {r['horizon']} | 无 | 无 | {counts(terminal)} | {counts(tail)} | 无 | {r['transport_interpretation']} |")
    lines += ['', '每条原始视频、步骤、来源与物理审计位于对应轮次目录。末尾30步零命令观察另列于JSON，不改原终点评分；它保留积分目标，不能当成冻结机器人后的静态沉降。',
              '', '具体结论及未通过项见仓库 `docs/act_transport_spillage_20261104.md`。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    if args.complete:
        expected=['transport_round2','transport_round3','transport_round4a','transport_round4b']
        if len(records)!=14 or any(read(root/name/'status.json')['status']!='completed' for name in expected):
            raise RuntimeError('Cannot complete before all 14 authorized transport rollouts finish')
        (root/'status.json').write_text(json.dumps(dict(status='completed',
            updated_utc=datetime.now(timezone.utc).isoformat(),focus='transport loss before truck',
            completed_transport_rollouts=14,archived_opening_rollouts=6,total_model_rollouts=20,
            training_updates=0,all_four_transport_rounds_executed=True,
            stable_global_improvement_claimed=False),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
