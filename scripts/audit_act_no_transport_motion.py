"""Read-only check for slow/unclosed loaded motion missed by the strict start definition."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from excavator_reachability import TruckBedColumn


def audit(root):
    records=json.loads((root/'results.json').read_text())
    result=[]
    for key,r in records.items():
        if r['status']!='completed' or r['transport']['available']:
            continue
        folder=root/'evaluation/episodes'/f'episode_{int(key):04d}'
        rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        scene=json.loads((folder/'scene_context.json').read_text())
        bed=TruckBedColumn(scene,0.)
        points=[];angles=[];bucket=[];gaps=[];clearances=[];steps=[]
        for i,row in enumerate(rows):
            if not i:
                continue  # The old initial report does not save transfer counts.
            state=row['input_state']
            particles=rows[i-1]['particle_transfer_metrics']
            if particles['retained_particles']<100:
                continue
            # Use directly recorded rigid-body points, independently of the
            # main diagnostic's reconstruction from target-offset state.
            point=np.asarray(row['input_bucket_landmarks_world']['load'])
            local=bed.local(point)
            excess=np.maximum(np.maximum(bed.minimum[:2]-local[:2],local[:2]-bed.maximum[:2]),0.)
            gaps.append(float(np.linalg.norm(excess*bed.scales[:2])))
            points.append(point[:2]);angles.append(state[0]);bucket.append(state[3])
            clearances.append(-state[20]);steps.append(i)
        progress=gaps[0]-min(gaps) if gaps else None
        span=float(np.linalg.norm(np.asarray(points)-points[0],axis=1).max()) if points else None
        swing=max(angles)-min(angles) if angles else None
        flagged=bool(points and (progress>=1 or span>=2 or swing>=.2))
        result.append(dict(episode_index=int(key),scene_seed=r['scene_seed'],
            loaded_observations=len(points),
            first_loaded_step=steps[0] if steps else None,last_loaded_step=steps[-1] if steps else None,
            tip_clearance_range_m=[min(clearances),max(clearances)] if clearances else None,
            max_retained=max(row['particle_transfer_metrics']['retained_particles'] for row in rows),
            loaded_horizontal_bed_gap_reduction_m=progress,loaded_horizontal_displacement_m=span,
            loaded_swing_span_rad=swing,bucket_range_rad=[min(bucket),max(bucket)] if bucket else None,
            guard_modified_steps=r['guard_modified_steps'],needs_motion_review=flagged,
            video=r['video']))
    return dict(scope='independent recorded-landmark audit of cases excluded by the strict transport-start definition',
        note='No bucket-closure, speed or tip-clearance threshold used here. Flags request trajectory review, '
             'not automatic reclassification or a reachability judgment. Original metrics are unchanged.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=result,
        reviewed_cases=len(result),motion_review_flags=sum(r['needs_motion_review'] for r in result))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.root.resolve())
    (args.root/'no_transport_motion_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('reviewed',result['reviewed_cases'],'motion flags',result['motion_review_flags'])
    for r in result['cases']:
        print(r['episode_index'],'peak',r['max_retained'],'loaded_gap_reduction',r['loaded_horizontal_bed_gap_reduction_m'],
              'displacement',r['loaded_horizontal_displacement_m'],'flag',r['needs_motion_review'])


if __name__=='__main__':
    main()
