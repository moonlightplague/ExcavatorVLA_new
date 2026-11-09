"""Read-only classification of ACT unloading handoffs and hold outcomes."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root,out):
    out.mkdir(parents=True,exist_ok=False)
    records=[];sources={}
    for folder in sorted((root/'evaluation/episodes').iterdir()):
        if not folder.is_dir():continue
        status=json.loads((folder/'episode_status.json').read_text())
        path=folder/'steps.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        starts=[];handbacks=[]
        for index,row in enumerate(rows):
            trace=row['carry_guard']['stall_recovery'];hold=trace['unload_pose_hold']
            before=rows[index-1]['particle_transfer_metrics'] if index else None
            if hold['started']:
                end=index
                while end+1<len(rows) and rows[end+1]['carry_guard']['stall_recovery']['unload_pose_hold']['active']:
                    end+=1
                release=(rows[end+1]['carry_guard']['stall_recovery']['unload_pose_hold'] if end+1<len(rows) else {})
                starts.append(dict(step=index,bucket_before=before['retained_particles'] if before else None,
                    truck_before=before['payload_particles'] if before else None,
                    hold_active_steps=end-index+1,release_reason=release.get('release_reason'),
                    end_counts=rows[end]['particle_transfer_metrics']))
            if trace.get('handback'):
                handbacks.append(dict(step=index,reason=trace.get('phase'),counts_before=before,
                    opening_intent=hold['opening_intent'],over_bed=hold['over_bed'],
                    hold_started=hold['started'],model_bucket_action=row['predicted'][3],
                    input_joint_velocity=row['input_state'][4:8],error=trace.get('error')))
        category='success'
        if not status['success']:
            reasons={h['reason'] for h in handbacks}
            if starts:
                category='hold_started_with_less_than_100_bucket_particles' if all(s['bucket_before']<100 for s in starts) else 'hold_other_failure'
            elif 'completed_secure_align' in reasons:category='aligned_but_hold_never_started'
            elif 'alignment_plan_unresolved' in reasons:category='alignment_plan_unresolved'
            elif 'aborted_budget' in reasons:category='recovery_budget_exhausted'
            elif any('empty' in str(reason) for reason in reasons):category='recovery_emptied'
            else:category='no_completed_alignment'
        records.append(dict(episode=int(folder.name.split('_')[-1]),seed=status['scene_seed'],success=status['success'],
            category=category,hold_starts=starts,handbacks=handbacks,terminal=rows[-1]['particle_transfer_metrics'],
            video=str(folder/'evaluation.mp4')))
    categories={c:sum(r['category']==c for r in records) for c in sorted({r['category'] for r in records})}
    result=dict(source=str(root),episodes=len(records),categories=categories,records=records)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'inputs_sha256.json').write_text(json.dumps(sources,indent=2)+'\n')
    lines=['# ACT随机40场：卸料保持与交接复核','',f'分类计数：{categories}。','',
        '| 场次 | 结果 | 分类 | 保持启动前斗内粒子 | 保持步数 | 最终入车 | 交接原因 |',
        '| --- | --- | --- | --- | --- | ---: | --- |']
    for r in records:
        lines.append(f"| {r['episode']} | {r['success']} | {r['category']} | "
            f"{','.join(str(s['bucket_before']) for s in r['hold_starts']) or '—'} | "
            f"{','.join(str(s['hold_active_steps']) for s in r['hold_starts']) or '—'} | "
            f"{r['terminal']['payload_particles']} | {','.join(str(h['reason']) for h in r['handbacks']) or '—'} |")
    lines+=['','斗内不足100粒并不构成物理不可能的证明；空中或此前撒出的物料仍可能进入车斗。',
        '本审计只描述实际记录。保持提前因空斗结束不支持通过延长同一保持预算解决问题。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.source.resolve(),args.output.resolve())
    print(json.dumps(result['categories'],ensure_ascii=False))
