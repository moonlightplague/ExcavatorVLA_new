"""Audit fallback admission and unchanged ordinary holds on recorded inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import types


def audit(source,candidate):
    import numpy as np
    sys.path.insert(0,str(candidate/'frozen_sources/scripts'))
    from act_carry_guard import CarryAttitudeGuard,CarryGuardSettings
    modules=[]
    for name,path in (('recorded_selected_controller',source/'frozen_sources/scripts/act_stall_recovery.py'),
                      ('act_unload_controlled_open',candidate/'frozen_sources/scripts/act_unload_controlled_open.py'),
                      ('recorded_controlled_controller',candidate/'frozen_sources/scripts/act_stall_recovery_latched.py')):
        module=types.ModuleType(name);sys.modules[name]=module
        exec(compile(path.read_text(),str(path),'exec'),module.__dict__);modules.append(module)
    model=json.loads((source/'robot_model.json').read_text())
    factories=[module.make_stall_guard_class(CarryAttitudeGuard,model,'secure_align') for module in (modules[0],modules[2])]
    limits=np.array([6.,2.5,2.5,6.],dtype=np.float32)
    checks=[];natural=[];maximum_error=0.;sources={}
    for folder in sorted((source/'evaluation/episodes').glob('episode_*')):
        rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        scene=json.loads((folder/'scene_context.json').read_text())
        settings=CarryGuardSettings(parent_projection='arm_first',release_geometry='opening_pose')
        old=factories[0](scene,action_limits=limits,settings=settings)
        for index,row in enumerate(rows):
            event=row['carry_guard']['stall_recovery']
            inspect=event.get('phase')=='completed_secure_align' or event['unload_pose_hold'].get('started')
            before=copy.deepcopy(old) if inspect else None
            state=np.asarray(row['input_state'],dtype=np.float32)
            action=np.clip(np.asarray(row['predicted'],dtype=np.float32),-limits,limits)
            command=rows[index-1]['joint_command_positions'] if index else state[:4]+state[8:12]
            args=(state,action,command,row['input_bucket_landmarks_world'])
            original,trace=old.update(*args)
            maximum_error=max(maximum_error,float(np.max(np.abs(original-row['applied']))))
            if inspect:
                newer=factories[1](scene,action_limits=limits,settings=settings)
                newer.__dict__.update(copy.deepcopy(before.__dict__))
                applied,new_trace=newer.update(*args)
                hold=new_trace['stall_recovery']['unload_pose_hold'];controlled=hold['controlled_open']
                if event['unload_pose_hold'].get('started'):
                    natural.append(dict(scene=int(folder.name[-4:]),step=index,
                        action_equal=bool(np.array_equal(applied,original)),fallback_active=controlled['active']))
                if event.get('phase')=='completed_secure_align':
                    checks.append(dict(scene=int(folder.name[-4:]),step=index,
                        old_hold_active=event['unload_pose_hold']['active'],fallback_started=controlled['started'],
                        model_bucket_action=float(action[3]),applied_bucket_action=float(applied[3]),
                        original_bucket_action=float(original[3]),arc_residual=event['contained_opening_arc_residual_m']))
        for name in ('steps.jsonl','scene_context.json'):
            p=folder/name;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    target_scenes={10,13,28,8,9,33}
    targeted=[c for c in checks if c['scene'] in target_scenes]
    passed=maximum_error<=1e-5 and len(targeted)==6 and all(c['fallback_started'] for c in targeted)
    passed=passed and bool(natural) and all(c['action_equal'] and not c['fallback_active'] for c in natural)
    result=dict(passed=passed,kind='Recorded input audit, not counterfactual physical outcomes',
        maximum_selected_controller_reconstruction_error=maximum_error,
        ordinary_hold_starts_checked=len(natural),all_ordinary_hold_start_actions_equal=all(c['action_equal'] for c in natural),
        target_handoff_checks=targeted,all_handoff_checks=checks,ordinary_hold_checks=natural)
    (candidate/'recorded_input_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (candidate/'recorded_input_sources_sha256.json').write_text(json.dumps(sources,indent=2)+'\n')
    return {k:v for k,v in result.items() if k not in ('all_handoff_checks','ordinary_hold_checks')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--candidate',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.source.resolve(),args.candidate.resolve()),ensure_ascii=False))
