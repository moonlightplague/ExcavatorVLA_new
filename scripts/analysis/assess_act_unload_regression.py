"""Evaluate predeclared, fixed-count scene9 regression without replacing trials."""
import argparse
import json
from pathlib import Path


def read(p):return json.loads(p.read_text())


def assess(out):
    assert read(out/'status.json')['status']=='completed'
    protocol=read(out/'regression_protocol.json');summary=read(out/'results.json')
    records=summary['results'];assert len(records)==protocol['repetitions']
    assert {(v['scene'],v['repetition']) for v in records.values()}=={(9,r) for r in range(1,protocol['repetitions']+1)}
    pairs=[v for v in records.values() if v['status']=='completed']
    qualified=[v for v in pairs if min(v['branches'][b]['branch_steps'] for b in ('off','on'))>=protocol['minimum_branch_steps']]
    affected=[v for v in qualified if v['branches']['on']['geometry_feedback_active_steps']>0]
    current_success=[v for v in affected if v['branches']['off']['original_task_completed']]
    aa=[abs(v['branches']['off']['truck_fraction']-v['branches']['off_repeat']['truck_fraction']) for v in pairs]
    differences=[v['branches']['on']['truck_fraction']-v['branches']['off']['truck_fraction'] for v in pairs]
    checks=dict(coverage=len(qualified)>=protocol['minimum_matched_pairs'],
        affected_coverage=len(affected)>=protocol['minimum_affected_pairs'],
        affected_current_success_coverage=len(current_success)>=protocol['minimum_affected_current_success_pairs'],
        task_preservation=all(not v['branches']['off']['original_task_completed'] or v['branches']['on']['original_task_completed'] for v in pairs),
        material_preservation=bool(differences) and min(differences)>=-protocol['maximum_truck_fraction_loss_per_pair'],
        repeatability=bool(aa) and max(aa)<=protocol['maximum_repeated_current_difference'])
    result=dict(planned_prefixes=protocol['repetitions'],matched_pairs=len(pairs),not_reached=summary['not_reached'],
        affected_pairs=len(affected),affected_current_success_pairs=len(current_success),
        original_task_successes={b:sum(v['branches'][b]['original_task_completed'] for v in pairs) for b in ('off','on')},
        maximum_repeated_current_difference=max(aa) if aa else None,
        minimum_truck_fraction_difference=min(differences) if differences else None,
        checks=checks,regression_passed=all(checks.values()),baseline_replaced=False,
        scope='One viewed scene with repeated fresh prefixes; no overall success rate or independent new scene count')
    (out/'regression_assessment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 第9场固定四次回归','',f"配对{len(pairs)}/{protocol['repetitions']}；未到达{summary['not_reached']}。实际介入{len(affected)}对，其中原版成功{len(current_success)}对。",'',
        '| 重复 | 快照斗内粒子 | 原版→候选入车比例 | 原版→候选任务完成 | 新闭环步数 |',
        '| --- | ---: | --- | --- | ---: |']
    for v in sorted(records.values(),key=lambda r:r['repetition']):
        if v['status']!='completed':lines.append(f"| {v['repetition']} | 未到达 | — | — | — |");continue
        a,b=v['branches']['off'],v['branches']['on']
        lines.append(f"| {v['repetition']} | {v['cohort_count']} | {a['truck_fraction']:.2%}→{b['truck_fraction']:.2%} | {a['original_task_completed']}→{b['original_task_completed']} | {b['geometry_feedback_active_steps']} |")
    lines+=['',f'预设回归条件：{checks}；通过：{result["regression_passed"]}。',
        '四次前缀在主确认试验第9场启动前固定，全部保留；不按结果替换或提前停止。每个可配对状态均重复原版，顺序按重复编号交替。',
        '这些是同一开发几何的场内重复，不作为四个独立新场景；本报告不修改主试验的门槛或结论，也不发布总体成功率。']
    (out/'REGRESSION_REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    args=p.parse_args();print(json.dumps(assess(args.output.resolve()),ensure_ascii=False))
