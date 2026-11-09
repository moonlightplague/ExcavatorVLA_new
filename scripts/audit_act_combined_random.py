"""Independently check a completed random ACT batch and describe its failures."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def read(path):return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def audit(out,root):
    protocol=read(out/'protocol.json');status=read(out/'status.json');results=read(out/'results.json')
    assert status['status']=='completed' and status['result_valid']
    assert protocol['control_group'] is False
    count=protocol['episodes'];records=results['episodes']
    assert set(records)=={str(i) for i in range(1,count+1)}
    assert protocol['seeds']==list(range(protocol['seed'],protocol['seed']+count))
    assert protocol['steps']==500 and protocol['post_rollout_observe_steps']==0
    frozen=out/'frozen_sources'
    protected=read(out/'inputs_sha256.json')
    for name,digest in protected.items():assert sha(Path(name))==digest,f'Changed input: {name}'
    baseline=read(root/'docs/baselines/act_phw40k_carry275_20261105.json')
    archive=root/baseline['archive']
    assert sha(frozen/'ExcavatorVLA-smolvla/run_simulation.py')==sha(archive/'frozen_sources/ExcavatorVLA-smolvla/run_simulation.py')
    assert 'unload_snapshot_restore' not in (frozen/'ExcavatorVLA-smolvla/run_simulation.py').read_text()
    with_wait=protocol.get('unload_handoff_wait',{}).get('enabled',False)
    if with_wait:
        source=Path(protocol['handoff_wait_source']['source'])
        assert sha(frozen/'scripts/act_stall_recovery.py')==sha(source/'frozen_sources/scripts/act_stall_recovery_latched.py')
        assert sha(frozen/'scripts/act_unload_pose_latch.py')==sha(source/'frozen_sources/scripts/act_unload_pose_latch.py')
    observations=0;successes=0;profiles=[];pids=[];details=[];wait_episodes=0;timeout_events=0
    for index in range(1,count+1):
        folder=out/'evaluation/episodes'/f'episode_{index:04d}'
        endpoint=read(folder/'episode_status.json');record=records[str(index)]
        assert endpoint['status']==record['status']=='completed'
        assert endpoint['scene_seed']==record['scene_seed']==protocol['seeds'][index-1]
        assert endpoint['evaluation_profile']=='standard' and not endpoint.get('simulator_reused',False)
        pids.append(endpoint['simulator_pid'])
        scene=read(folder/'scene_context.json')
        assert scene['scene_seed']==endpoint['scene_seed']
        geometry={k:v for k,v in scene['scene_profile'].items() if k not in ('seed','sample_try')}
        profiles.append(hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest())
        rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        assert len(rows)==endpoint['policy_steps_executed']==record['policy_steps'] and 1<=len(rows)<=500
        assert (folder/'evaluation.mp4').stat().st_size>1000
        consecutive=0;first_success=None;hold_starts=[];wait_steps=0;starts=0;timeouts=0;phases=set();aligned=[]
        for offset,row in enumerate(rows):
            assert row['step']==offset
            assert len(row['state'])==28 and len(row['applied'])==4
            assert all(math.isfinite(v) for v in row['state']+row['applied'])
            assert all(abs(v)<=limit+1e-5 for v,limit in zip(row['applied'],(6,2.5,2.5,6)))
            payload=row['particle_transfer_metrics']['payload_particles']
            assert payload==row['deposited_particles']
            consecutive=consecutive+1 if payload>=100 and row['bucket_position']>=-.8 else 0
            if consecutive>=3 and first_success is None:first_success=offset
            trace=row['carry_guard']['stall_recovery'];hold=trace['unload_pose_hold'];phases.add(trace.get('phase'))
            if trace.get('handback') and trace.get('phase')=='completed_secure_align':aligned.append(offset)
            if hold['started']:
                previous=rows[offset-1]['particle_transfer_metrics'] if offset else None
                hold_starts.append(dict(step=offset,retained_before=previous['retained_particles'] if previous else None,
                    truck_before=previous['payload_particles'] if previous else None))
            wait=trace.get('unload_handoff_wait',{})
            if with_wait:assert wait.get('enabled') is True
            assert wait.get('steps',0)<=20
            assert not (wait.get('active') and hold['active'])
            if wait.get('started'):
                assert trace.get('handback') and trace.get('phase')=='completed_secure_align'
                assert hold['loaded'] and hold['over_bed'] and not hold['active']
                starts+=1
            wait_steps+=bool(wait.get('active'));timeouts+=wait.get('release_reason')=='act_opening_wait_timeout'
        assert wait_steps<=20*starts
        success=first_success is not None
        assert success==endpoint['success']==record['success']
        assert (success and first_success==len(rows)-1) or (not success and len(rows)==500)
        assert wait_steps==record.get('handoff_wait_steps',0) and timeouts==record.get('handoff_wait_timeout_events',0)
        assert (aligned[0] if aligned else None)==record.get('first_verified_alignment_step')
        successes+=success;observations+=len(rows);wait_episodes+=wait_steps>0;timeout_events+=timeouts
        tags=[]
        if not success:
            if not hold_starts:tags.append('hold_never_started')
            if hold_starts and all(s['retained_before'] is not None and s['retained_before']<100 for s in hold_starts):
                tags.append('all_hold_starts_under100_retained')
            if aligned and aligned[0]>=450:tags.append('verified_alignment_with_under50_steps_remaining')
            if timeouts:tags.append('handoff_wait_timed_out')
            if starts and not hold_starts:tags.append('waited_but_hold_never_started')
            if 'alignment_plan_unresolved' in phases:tags.append('alignment_plan_unresolved')
            if 'aborted_budget' in phases:tags.append('recovery_budget_exhausted')
            if max(r['particle_transfer_metrics']['scooped_particles'] for r in rows)==0:tags.append('no_recorded_scoop')
            if rows[-1]['particle_transfer_metrics']['retained_particles']>=100:tags.append('over100_retained_at_end')
            if max(r['deposited_particles'] for r in rows)>=100:tags.append('reached100_truck_without_three_open_observations')
        details.append(dict(episode=index,seed=endpoint['scene_seed'],success=success,steps=len(rows),
            first_verified_alignment_step=aligned[0] if aligned else None,hold_starts=hold_starts,
            handoff_wait_steps=wait_steps,handoff_wait_timeout_events=timeouts,
            terminal=rows[-1]['particle_transfer_metrics'],failure_tags=tags,video=str(folder/'evaluation.mp4')))
    assert len(set(pids))==count,'Expected one fresh simulator process per scene'
    assert successes==status['successes']==results['successes']
    assert status['success_rate']==results['success_rate']==successes/count
    assert wait_episodes==results['handoff_wait_episodes'] and timeout_events==results['handoff_wait_timeout_events']
    for name,digest in baseline['files_sha256'].items():assert sha(archive/name)==digest,f'Changed baseline: {name}'
    old=root/'outputs/experiments/act_combined_random40_20261106_01'
    prior=read(old/'evidence_sha256.json')
    for name,digest in prior.items():assert sha(old/name)==digest,f'Changed original random40 evidence: {name}'
    failure_tags=dict(Counter(tag for r in details for tag in r['failure_tags']))
    failures=[r for r in details if not r['success']]
    result=dict(passed=True,episodes=count,successes=successes,failures=count-successes,
        success_rate=successes/count,observations=observations,unique_actual_geometry_profiles=len(set(profiles)),
        unique_simulator_processes=len(set(pids)),protected_input_files=len(protected),
        preserved_baseline_files_unchanged=len(baseline['files_sha256']),previous_random40_evidence_unchanged=len(prior),
        handoff_wait_episodes=wait_episodes,handoff_wait_timeout_events=timeout_events,
        successes_among_wait_episodes=sum(r['success'] for r in details if r['handoff_wait_steps']>0),
        failure_tag_counts=failure_tags,failure_tags_are_overlapping_descriptions_not_causal_attribution=True,
        overall_improvement_not_estimated_without_control=True,baseline_replaced=False,records=details)
    (out/'postflight_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 交接等待版随机场景：失败复核','',
        f'本批成功{successes}/{count}；失败{count-successes}。等待触发{wait_episodes}场，超时{timeout_events}次。',
        f'重叠描述标签：{failure_tags}。','',
        '| 场次 | 入车/末态斗内 | 对位步 | 保持开始前斗内粒子 | 等待步数/超时次数 | 标签 | 视频 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in failures:
        lines.append(f"| {r['episode']} | {r['terminal']['payload_particles']}/{r['terminal']['retained_particles']} | "
            f"{r['first_verified_alignment_step']} | {','.join(str(s['retained_before']) for s in r['hold_starts']) or '—'} | "
            f"{r['handoff_wait_steps']}/{r['handoff_wait_timeout_events']} | {', '.join(r['failure_tags']) or '其他，见逐步记录'} | [视频]({r['video']}) |")
    lines+=['','到位步仅指已验证secure_align交接，None不等于从未自然到车。少于100粒不构成物理不可能证明，其他空中物料仍可能入车。',
        '标签可重叠，描述实际记录，不证明失败原因或新增等待的独立贡献。触发等待子集的成功数也不能当作挽救数。',
        '原成功门槛和500步预算不变，不运行对照，不计算与旧批次的因果提升。']
    (out/'FAILURES.md').write_text('\n'.join(lines)+'\n')
    evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='evidence_sha256.json' and '__pycache__' not in p.parts}
    (out/'evidence_sha256.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='records'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    print(json.dumps(audit(args.output.resolve(),args.root.resolve()),ensure_ascii=False))
