"""Read-only coverage check on recorded failure states; no counterfactual physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from act_carry_guard import CarryAttitudeGuard, CarryGuardSettings
from excavator_reachability import AuthoredExcavatorChain, restored_truck_geometry


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    catalog,model=read(args.catalog),read(args.model)
    chain=AuthoredExcavatorChain(model)
    result=[]
    for case in catalog['cases']:
        folder=Path(case['folder']);scene=restored_truck_geometry(read(folder/'scene_context.json'),model)
        rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        guard=CarryAttitudeGuard(scene,settings=CarryGuardSettings(parent_projection='arm_first',release_geometry='opening_pose'))
        modified=[];errors=[];infeasible=[]
        for index,row in enumerate(rows):
            if index > case['pretruck_empty_step']:
                break
            state=np.asarray(row['input_state']);q=state[:4]
            landmarks={k:chain.point(q,k).tolist() for k in ('tip','load','pour')}
            command=rows[index-1]['joint_command_positions'] if index else q
            adjusted,detail=guard.update(state,np.clip(row['predicted'],[-6,-2.5,-2.5,-6],[6,2.5,2.5,6]),command,landmarks)
            errors.append(detail['pitch_rad'])
            if detail['modified']:modified.append(index)
            if detail['infeasible']:infeasible.append(index)
        result.append(dict(scene_seed=case['scene_seed'],source_folder=str(folder),
            recorded_empty_step=case['pretruck_empty_step'],first_intervention=modified[0] if modified else None,
            modified_recorded_states=len(modified),infeasible_recorded_states=len(infeasible),
            intervenes_before_recorded_empty=bool(modified and modified[0]<case['pretruck_empty_step'])))
    payload=dict(scope='recorded-state intervention coverage, not a closed-loop retention result',
        warning='Projected actions do not change subsequent recorded states. Late unsafe states may be infeasible '
                'even if an actual earlier intervention would have prevented reaching them.',
        files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [args.catalog,args.model,Path(__file__),Path(__file__).parent/'act_carry_guard.py']},cases=result)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    for r in result:print(r['scene_seed'],'first intervention',r['first_intervention'],'recorded empty',r['recorded_empty_step'])


if __name__=='__main__':
    main()
