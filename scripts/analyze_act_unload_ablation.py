"""Report matched unloading outcomes and alignment during the first opening."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def alignment(rows):
    five=rows[:50]
    empty=0
    first_opening=[]
    for row in five:
        first_opening.append(row)
        empty=empty+1 if row['state'][26]<.002 else 0
        if empty>=3:break
    def maximum(window,key):
        return max(row['geometry'][key] for row in window)
    return dict(first_five_seconds_hinge_xy_drift_m=maximum(five,'hinge_xy_drift_m'),
        first_opening_hinge_xy_drift_m=maximum(first_opening,'hinge_xy_drift_m'),
        first_opening_max_pour_bed_gap_m=maximum(first_opening,'opening_pour_bed_gap_m'),
        first_opening_parent_abs_drift_rad=[max(row['geometry']['parent_abs_drift_rad'][i] for row in first_opening) for i in range(3)],
        first_opening_steps=len(first_opening),
        first_opening_hold_active_steps=sum(row['carry_guard'].get('stall_recovery',{}).get('unload_pose_hold',{}).get('active',False) for row in first_opening))


def analyze(out):
    summary=read(out/'results.json')
    comparison=read(out/'protocol.json').get('comparison')
    latch_comparison=comparison in ('latch_handoff','handoff_wait','controlled_open','geometry_feedback')
    definition=out/'alignment_metric_definition.json'
    if not definition.exists():raise ValueError('Missing predeclared opening-window definitions')
    records=[]
    for name,result in sorted(summary['results'].items(),key=lambda item:(item[1]['scene'],item[1]['repetition'])):
        if result['status']!='completed':continue
        branches={}
        for branch,endpoint in result['branches'].items():
            path=out/name/branch/'steps.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            if len(rows)!=endpoint['branch_steps']:raise ValueError('Incomplete branch trace')
            if not latch_comparison and branch!='on' and endpoint['hold_active_steps']!=0:raise ValueError('Hold-off arm actually held')
            if branch=='on' and comparison!='geometry_feedback' and endpoint['hold_active_steps']+endpoint.get('handoff_wait_active_steps',0)<=0:raise ValueError('Eligible hold-on arm did not hold')
            if any(sum(r['unload_cohort_metrics'][k] for k in ('bucket','truck','outside'))!=result['cohort_count'] for r in rows):
                raise ValueError('Cohort identity conservation failed')
            audit=read(out/name/branch/'restore_audit.json')
            if not audit['initial_observation_equal']:raise ValueError('Initial observation mismatch')
            branches[branch]=dict(endpoint,alignment=alignment(rows),video=str(out/name/branch/'evaluation.mp4'))
        a,b=branches['off'],branches['on']
        records.append(dict(case=name,scene=result['scene'],repetition=result['repetition'],branches=branches,
            truck_gain=b['truck_fraction']-a['truck_fraction'],outside_change=b['outside_fraction']-a['outside_fraction'],
            opening_hinge_drift_change_m=b['alignment']['first_opening_hinge_xy_drift_m']-a['alignment']['first_opening_hinge_xy_drift_m']))
    values=lambda key:[r[key] for r in records]
    details=dict(matched_pairs=len(records),positive_pairs=sum(r['truck_gain']>0 for r in records),
        negative_pairs=sum(r['truck_gain']<0 for r in records),
        zero_pairs=sum(r['truck_gain']==0 for r in records),
        mean_truck_gain=statistics.mean(values('truck_gain')) if records else None,
        median_truck_gain=statistics.median(values('truck_gain')) if records else None,
        mean_opening_hinge_drift_change_m=statistics.mean(values('opening_hinge_drift_change_m')) if records else None,
        opening_window_definition_sha256=hashlib.sha256(definition.read_bytes()).hexdigest(),records=records)
    (out/'unload_analysis.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
    lines=['# 卸料姿态保持的独立开关对照','',
        f"有效配对{len(records)}；未到达可配对状态{summary['not_reached']}次。改善/退化/相同：{details['positive_pairs']}/{details['negative_pairs']}/{details['zero_pairs']}。",
        f"预设推进条件：{summary['checks']}；推进：{summary['advance']}。",'',
        '| 场景/重复 | 关闭→开启入车比例 | 关闭→开启斗外且车外比例 | 首次开斗期间斗铰点XY漂移 m，关闭→开启 | 任务完成，关闭→开启 | 视频 |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in records:
        a,b=r['branches']['off'],r['branches']['on']
        lines.append(f"| {r['scene']}/{r['repetition']} | {a['truck_fraction']:.1%}→{b['truck_fraction']:.1%} | "
            f"{a['outside_fraction']:.1%}→{b['outside_fraction']:.1%} | "
            f"{a['alignment']['first_opening_hinge_xy_drift_m']:.3f}→{b['alignment']['first_opening_hinge_xy_drift_m']:.3f} | "
            f"{a['original_task_completed']}→{b['original_task_completed']} | [关闭]({a['video']}) / [开启]({b['video']}) |")
    lines += ['',f"平均入车比例差：{details['mean_truck_gain']}；中位差：{details['median_truck_gain']}。",
        f"开斗期间平均铰点XY漂移变化（开启−关闭）：{details['mean_opening_hinge_drift_change_m']} m。",
        f"相同关闭策略重复的最大入车比例差：{summary['identical_off_max_truck_fraction_difference']}。",'',
        '入车比例仅针对快照时在斗内的固定沙粒ID；斗外且车外包含未落定的空中沙粒，3秒后续观察另存，不改变评分。',
        '开斗窗口在看结果前另行固定：从共同快照起，最多50步，到原填料阈值连续三次为空为止；同时保存固定前50步的几何指标。',
        '两组恢复相同的已观测物理状态。PhysX接触缓存与渲染历史没有序列化；若重复关闭组波动超过预设5个百分点，不认定保持逻辑的独立收益。',
        '这些是已查看场景的开发消融，场内重复不当作独立场景，不报告新的整体成功率。原模型、处理器和27.5%历史基准未替换。']
    if latch_comparison:
        lines=[line.replace('独立开关对照','交接锁定姿态对照').replace('关闭','当前版').replace('开启','候选版') for line in lines]
        if comparison=='controlled_open':
            lines=[line.replace('交接锁定姿态对照','主动开斗兜底配对验证') for line in lines]
            lines+=['','候选只在原保持未启动、已有完整开斗轨迹验证的交接点主动限速开斗；不是铲斗仍完全由ACT控制的开关实验。原成功门槛和评分预算不变。']
        if comparison=='geometry_feedback':
            lines=[line.replace('交接锁定姿态对照','斗口位置校正后卸料配对验证') for line in lines]
            lines+=['','候选在闭斗状态微调到完整斗口扫掠进入车斗后才允许开斗；实测与命令预测均须通过几何检查。普通ACT触发的旧保持保留，原成功门槛及500步总预算不变。']
    (out/'UNLOAD_REPORT.md').write_text('\n'.join(lines)+'\n')
    return details


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=analyze(args.output.resolve())
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))
