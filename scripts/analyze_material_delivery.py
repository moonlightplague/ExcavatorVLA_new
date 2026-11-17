"""Inspect recorded material IDs/positions without modifying rollout results."""
import argparse
import json
from pathlib import Path

import numpy as np


def position_summary(positions):
    points=np.asarray(positions,dtype=float).reshape(-1,3)
    if not len(points):return {'count':0}
    result={'count':len(points),'bounds_min':points.min(axis=0).tolist(),
            'bounds_max':points.max(axis=0).tolist(),
            'unique_positions_1mm':len(np.unique(np.round(points,3),axis=0))}
    if len(points)>1:
        distance=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=-1)
        np.fill_diagonal(distance,np.inf)
        nearest=distance.min(axis=1)
        result.update(nearest_neighbor_min_m=float(nearest.min()),nearest_neighbor_median_m=float(np.median(nearest)),
                      fraction_with_neighbor_within_1cm=float(np.mean(nearest<.01)))
    return result


def analyze(folder):
    rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    spec=json.loads((folder/'repeat_task_spec.json').read_text())
    grouped={}
    for row in rows:grouped.setdefault(row['repeat_cycle_index'],[]).append(row)
    cycles=[]
    for cycle,group in grouped.items():
        peak=max(group,key=lambda r:len(r['evaluation_particle_ids']['bucket']))
        initial=group[0]['evaluation_particle_ids']
        last=group[-1]['evaluation_particle_ids']
        eligible=peak['evaluation_particle_ids']
        positions=eligible.get('positions_world',{})
        final_positions=last.get('positions_world',{})
        initial_truck=set(initial['truck'])
        new_ids=set(last['truck'])-initial_truck
        item=dict(cycle_index=cycle,observed_steps=len(group),peak_bucket_step=peak['step'],
            peak_bucket_display_count=peak['repeat_bucket_particles'],peak_bucket_source_ids=len(eligible['bucket']),
            initial_truck_source_count=len(initial['truck']),final_truck_source_count=len(last['truck']),
            new_truck_source_ids=len(new_ids),new_truck_mass_kg=len(new_ids)*eligible['metric_point_mass_kg'],
            source_identity=eligible.get('source_identity'),eligible_source_count=eligible.get('source_count'),
            material_count=eligible['particle_count'],
            peak_bucket_geometry=position_summary(positions.get('bucket',[])),
            final_truck_geometry=position_summary(final_positions.get('truck',[])))
        cycles.append(item)
    return {'scope':'Diagnostic ID/position inspection; partial cycles are not scored outcomes.',
            'evaluation_standard':spec['version'],'observed_steps':len(rows),'cycles':cycles}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();result=analyze(args.episode)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    else:print(json.dumps(result,indent=2))


if __name__=='__main__':main()
