"""Measure loaded motion and pre-truck emptying independently of task reachability."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from excavator_reachability import TruckBedColumn, restored_truck_geometry


def read(path):
    return json.loads(path.read_text())


def analyze_transport(folder, model=None):
    folder = Path(folder)
    status, scene = read(folder/'episode_status.json'), read(folder/'scene_context.json')
    if 'local_from_world' not in scene['truck_counting_volume']:
        if model is None:
            raise ValueError('Old scene requires authored truck transform restoration')
        scene = restored_truck_geometry(scene, model)
    bed = TruckBedColumn(scene, 0.)
    rows = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    target = np.asarray(scene['observation_context']['unload_landing_xyz'])
    samples = []
    for r in rows:
        s = r['state']
        heading = s[0]
        delta = [math.cos(heading)*s[21]-math.sin(heading)*s[22],
                 math.sin(heading)*s[21]+math.cos(heading)*s[22], s[23]]
        load = target-np.asarray(delta)
        local = bed.local(load)
        excess = np.maximum(np.maximum(bed.minimum[:2]-local[:2], local[:2]-bed.maximum[:2]), 0.)
        gap = float(np.linalg.norm(excess*bed.scales[:2]))
        # Arrival is based on actual load position, not an imagined open pose.
        arrived = gap == 0 and local[2] >= bed.maximum[2]
        samples.append(dict(step=r['step'], **r['particle_transfer_metrics'], bed_gap_m=gap,
            actual_load_above_bed=bool(arrived), pitch_deg=math.degrees(math.atan2(s[17],s[16])),
            bucket_rad=s[3], motion=bool(abs(s[4]) >= .08 or math.hypot(s[5],s[6]) >= .12),
            tip_clearance_m=-s[20], load_world=load.tolist()))
    start = next((i for i,s in enumerate(samples) if s['retained_particles'] >= 100
                  and s['tip_clearance_m'] >= .25 and s['bucket_rad'] <= -1.3 and s['motion']), None)
    tail = read(folder/'post_rollout_summary.json') if (folder/'post_rollout_summary.json').exists() else None
    result = dict(folder=str(folder), scene_seed=status['scene_seed'], original_task_success=status['success'],
        task_noncompletion_interpretation='Task completion is recorded separately. Only distance-caused '
            'noncompletion may be exempted by the external reachability classification.',
        carry_start_step=start, terminal=samples[-1] if samples else None, post_rollout=tail,
        available=start is not None,
        note='Sampled particle counts, not fixed-ID retention. Outside includes airborne material. '
             'Loaded progress is measured separately so parking with sand is not successful transport.')
    if start is None:
        return result
    arrival = next((i for i in range(start,len(samples)) if samples[i]['actual_load_above_bed']), None)
    stop = arrival if arrival is not None else len(samples)-1
    empty = next((i for i in range(start,stop-1) if all(
        samples[j]['retained_particles'] <= 10
        and samples[j]['payload_particles'] < status.get('min_particles',100)
        and samples[j]['bed_gap_m'] > .5 for j in range(i,i+3))), None)
    if empty is not None:
        stop = min(stop, empty)
    segment = samples[start:stop+1]
    loaded = [s for s in segment if s['retained_particles'] >= 50]
    peak = max(s['retained_particles'] for s in segment)
    checkpoints = {}
    for seconds in (1,3,5,8,10):
        index = min(start+10*seconds, stop)
        s = samples[index]
        checkpoints[str(seconds)] = dict(**s, elapsed_action_steps=index-start,
            truncated_by_transport_end=index < start+10*seconds,
            retained_over_observed_peak=s['retained_particles']/peak if peak else None,
            scooped_increased_since_start=s['scooped_particles']-samples[start]['scooped_particles'])
    result.update(actual_load_arrival_step=arrival, pretruck_empty_step=empty,
        peak_pretruck_retained=peak,
        loaded_bed_gap_reduction_m=samples[start]['bed_gap_m']-min(s['bed_gap_m'] for s in loaded),
        maximum_pretruck_pitch_deg=max(s['pitch_deg'] for s in segment),
        minimum_pretruck_bucket_rad=min(s['bucket_rad'] for s in segment),
        at_carry_start=samples[start], at_arrival=samples[arrival] if arrival is not None else None,
        at_pretruck_empty=samples[empty] if empty is not None else None,
        checkpoints=checkpoints)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kinematics',type=Path)
    args=parser.parse_args()
    model=read(args.kinematics) if args.kinematics else None
    folders=sorted({p.parent for p in args.root.rglob('episode_status.json')
                    if 'frozen_sources' not in p.parts})
    results=[]
    for folder in folders:
        if read(folder/'episode_status.json')['status']=='completed':
            results.append(analyze_transport(folder,model))
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for r in results:
        if r['available']:
            t=r['checkpoints']['8']
            print(Path(r['folder']).parts[-3],r['scene_seed'],'empty',r['pretruck_empty_step'],
                  'loaded_progress_m',round(r['loaded_bed_gap_reduction_m'],2),
                  'retained_at_8s',t['retained_particles'],'pitch_max',round(r['maximum_pretruck_pitch_deg'],1))


if __name__=='__main__':
    main()
