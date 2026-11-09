"""Replay recorded controller inputs to check the narrow handoff-wait patch.

This is a controller audit, not a physical rollout or a success-rate estimate.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import types


def audit(source, candidate, output):
    import numpy as np
    frozen=source/'frozen_sources/scripts'
    sys.path.insert(0,str(frozen))
    from act_carry_guard import CarryAttitudeGuard, CarryGuardSettings
    source_code=(frozen/'act_stall_recovery.py').read_text()
    current=types.ModuleType('recorded_current_controller')
    exec(compile(source_code,str(frozen/'act_stall_recovery.py'),'exec'),current.__dict__)
    helper_code=(candidate/'frozen_sources/scripts/act_unload_pose_latch.py').read_text()
    helper=types.ModuleType('act_unload_pose_latch')
    exec(compile(helper_code,'act_unload_pose_latch.py','exec'),helper.__dict__)
    sys.modules['act_unload_pose_latch']=helper
    candidate_code=(candidate/'frozen_sources/scripts/act_stall_recovery_latched.py').read_text()
    newer=types.ModuleType('recorded_wait_controller')
    exec(compile(candidate_code,'recorded_wait_controller.py','exec'),newer.__dict__)
    model=json.loads((source/'robot_model.json').read_text())
    make_old=current.make_stall_guard_class(CarryAttitudeGuard,model,'secure_align')
    make_new=newer.make_stall_guard_class(CarryAttitudeGuard,model,'secure_align')
    limits=np.array([6.,2.5,2.5,6.],dtype=np.float32)
    records=[];checks=[];inputs={}
    for path in (frozen/'act_stall_recovery.py',candidate/'frozen_sources/scripts/act_stall_recovery_latched.py',
                 candidate/'frozen_sources/scripts/act_unload_pose_latch.py',source/'robot_model.json'):
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for folder in sorted((source/'evaluation/episodes').glob('episode_*')):
        rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        scene=json.loads((folder/'scene_context.json').read_text())
        settings=CarryGuardSettings(parent_projection='arm_first',release_geometry='opening_pose')
        old=make_old(scene,action_limits=limits,settings=settings)
        new=make_new(scene,action_limits=limits,settings=settings)
        max_old_error=0.;max_new_difference=0.;wait_steps=0;starts=[]
        for index,row in enumerate(rows):
            state=np.asarray(row['input_state'],dtype=np.float32)
            action=np.clip(np.asarray(row['predicted'],dtype=np.float32),-limits,limits)
            command=rows[index-1]['joint_command_positions'] if index else state[:4]+state[8:12]
            landmarks=row['input_bucket_landmarks_world']
            before=copy.deepcopy(old) if int(folder.name[-4:]) in (10,13,28) and row['carry_guard']['stall_recovery'].get('phase')=='completed_secure_align' else None
            applied,trace=old.update(state,action,command,landmarks)
            chosen,new_trace=new.update(state,action,command,landmarks)
            max_old_error=max(max_old_error,float(np.max(np.abs(applied-row['applied']))))
            max_new_difference=max(max_new_difference,float(np.max(np.abs(chosen-applied))))
            wait=new_trace['stall_recovery']['unload_handoff_wait']
            wait_steps+=int(wait['active'])
            if wait['started']:starts.append(index)
            if before is not None:
                at_handoff=make_new(scene,action_limits=limits,settings=settings)
                at_handoff.__dict__.update(copy.deepcopy(before.__dict__))
                first,first_trace=at_handoff.update(state,action,command,landmarks)
                event=first_trace['stall_recovery']['unload_handoff_wait']
                checks.append(dict(scene=int(folder.name[-4:]),step=index,
                    current_hold_active=trace['stall_recovery']['unload_pose_hold']['active'],
                    candidate_wait_active=event['active'],
                    bucket_command_unchanged=bool(first[3]==applied[3]),
                    model_bucket_action=float(action[3])))
        status=json.loads((folder/'episode_status.json').read_text())
        records.append(dict(scene=int(folder.name[-4:]),original_success=status['success'],
            rows=len(rows),maximum_current_reconstruction_error=max_old_error,
            maximum_candidate_action_difference=max_new_difference,
            wait_active_steps=wait_steps,wait_start_steps=starts))
        for name in ('steps.jsonl','scene_context.json','episode_status.json'):
            path=folder/name;inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    successes=[r for r in records if r['original_success']]
    assertions=dict(current_reconstruction=max(r['maximum_current_reconstruction_error'] for r in records)<=1e-5,
        all_three_known_handoffs=len(checks)==3 and all(c['candidate_wait_active'] and not c['current_hold_active'] and c['bucket_command_unchanged'] for c in checks))
    result=dict(kind='recorded-input controller audit; no counterfactual physical outcomes',
        source=str(source),candidate=str(candidate),passed=all(assertions.values()),checks=assertions,
        original_successes=len(successes),successful_recordings_with_wait=sum(r['wait_active_steps']>0 for r in successes),
        successful_recordings_with_identical_actions=sum(r['maximum_candidate_action_difference']==0 for r in successes),
        known_handoff_checks=checks,records=records)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    output.with_name('recorded_inputs_sha256.json').write_text(json.dumps(inputs,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='records'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.source.resolve(),args.candidate.resolve(),args.output.resolve()),ensure_ascii=False))
