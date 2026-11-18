"""Independent trace checks for measured aperture alignment and opening."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def audit(out):
    import numpy as np
    sys.path.insert(0,str(out/'frozen_sources/scripts'))
    from excavator_reachability import AuthoredExcavatorChain
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['comparison']=='geometry_feedback'
    result=json.loads((out/'results.json').read_text())
    model=json.loads((out/'robot_model.json').read_text());chain=AuthoredExcavatorChain(model)
    corners=np.asarray(json.loads((out/'geometry_preflight.json').read_text())['calibration']['corners_local'])
    maximum_parent_delta=np.array([.12,.12,.16]);speed_limits=np.array([.12,.12,.16])
    counts=Counter();reasons=Counter();records=[]
    def clearance(q,scene):
        sequence=np.repeat(np.asarray(q)[None,:],17,0)
        sequence[:,3]=np.linspace(q[3],max(q[3],-.4),17)
        points=[]
        for transform in chain.bucket_transform(sequence):
            points.extend((np.column_stack((corners,np.ones(4)))@transform.T)[:,:3])
        points=np.asarray(points);bed=scene['truck_counting_volume'];to_local=np.asarray(bed['local_from_world'])
        local=(np.column_stack((points,np.ones(len(points))))@to_local)[:,:3]
        scale=np.linalg.norm(np.linalg.inv(to_local)[:3,:3],axis=1)
        xy=np.minimum((local[:,:2]-bed['local_min'][:2])*scale[:2],(bed['local_max'][:2]-local[:,:2])*scale[:2])
        height=(local[:,2]-bed['local_max'][2])*scale[2]
        return float(xy.min()),float(height.min())
    for name,case in sorted(result['results'].items()):
        if case['status']!='completed':continue
        folder=out/name;scene=json.loads((folder/'scene_context.json').read_text())
        rows=[json.loads(line) for line in (folder/'on/steps.jsonl').read_text().splitlines()]
        start=None;previous_phase=None;minimum=[float('inf'),float('inf')];active=0;positive=0
        for row in rows:
            t=row['carry_guard']['stall_recovery'];g=t['unload_pose_hold']['geometry_feedback']
            expected_version=protocol['geometry_feedback'].get('controller_version')
            if expected_version:assert g.get('version')==expected_version
            if g.get('release_reason'):reasons[g['release_reason']]+=1
            assert g['steps']<=50 and g['alignment_steps']<=20
            if g['started']:
                start=np.asarray(row['input_state'][:4]);counts['starts']+=1
                assert t.get('handback') and t.get('phase')=='completed_secure_align'
                assert row['step']<=450
            if not g['active']:continue
            active+=1;counts['active_observations']+=1
            assert start is not None
            goal=np.asarray(g['parent_goal'])
            assert np.all(np.abs(goal-start[:3])<=maximum_parent_delta+1e-6)
            applied=np.asarray(row['applied']);state=np.asarray(row['input_state'])
            assert np.all(np.abs(applied[:3])<=speed_limits+1e-5)
            assert abs(applied[3])<=.8+1e-5
            if g['phase']=='align_closed':
                counts['closed_alignment_observations']+=1
                assert applied[3]<=1e-5,'Opened before measured aperture alignment'
            else:
                assert g['phase']=='open_contained'
                if previous_phase!='open_contained':
                    assert g['stable_observations']>=2
                    assert np.max(np.abs(state[:3]-goal))<.020001
                    assert np.max(np.abs(state[4:7]))<=.100001
                if applied[3]>1e-5:
                    positive+=1;counts['positive_opening_observations']+=1
                    current=clearance(state[:4],scene)
                    predicted=np.asarray(row['input_joint_command_positions'])+applied*.1
                    future=clearance(predicted,scene)
                    assert min(current+future)>=.10-1e-6,(name,row['step'],current,future)
                    minimum=[min(minimum[i],current[i],future[i]) for i in range(2)]
                if g['opening_blocked']:
                    counts['blocked_opening_observations']+=1
                    assert applied[3]<=1e-5
            previous_phase=g['phase']
        assert active==case['branches']['on']['geometry_feedback_active_steps']
        records.append(dict(case=name,active_steps=active,positive_opening_steps=positive,
            minimum_verified_xy_m=minimum[0] if positive else None,
            minimum_verified_height_m=minimum[1] if positive else None))
    output=dict(passed=True,method='Independent row-vector aperture projection at the recorded measured and final applied command poses',
        counts=dict(counts),exit_reasons=dict(reasons),records=records,
        geometric_checks_do_not_replace_original_particle_success_scoring=True,
        development_gate_passed=result['advance'],baseline_replaced=False)
    (out/'geometry_execution_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    args=p.parse_args();print(json.dumps(audit(args.output.resolve()),ensure_ascii=False))
