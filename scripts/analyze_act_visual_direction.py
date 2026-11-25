"""Read-only phase analysis of paired ACT visual fine-tuning rollouts."""
import argparse,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/experiments/act_hdr_baseline_repeatability_20261120_02/repeats'
NEW=ROOT/'outputs/experiments/act_visual_robustness_20261120_01/hdr_repeats'


def read(path):return json.loads(Path(path).read_text())


def analyze(folder):
    folder=Path(folder)/'evaluation/episodes/episode_0001';status=read(folder/'episode_status.json')
    rows={}
    with (folder/'steps.jsonl').open() as trace:
        for line in trace:
            r=json.loads(line)
            rows[r['step']]={k:r.get(k) for k in ['action_source','carry_guard','predicted','applied','input_state','bucket_landmarks_world','input_bucket_landmarks_world']}
    cycles=[]
    for c in status['repeat_cycles']:
        if not c.get('returned') or not isinstance(c.get('end_step'),int):continue
        start=c['start_step'];dig=c.get('dig_step');depart=c.get('carry_departure_step');dump=c.get('dump_step');end=c['end_step']
        def seconds(a,b):return .1*(b-a) if isinstance(a,int) and isinstance(b,int) and b>=a else None
        r=dict(cycle=c['cycle_index'],success=c['success'],start_step=start,end_step=end,
            approach_seconds=seconds(start,dig),dig_seconds=seconds(dig,depart),carry_dump_seconds=seconds(depart,dump),return_seconds=seconds(dump,end),
            cycle_seconds=c.get('cycle_time_seconds'),max_bucket_particles=c.get('max_bucket_particles'),scooped_particles=c.get('scooped_particles'),
            delivered_mass_kg=c.get('delivered_mass_kg'),spillage_fraction=c.get('spillage_fraction'),failure_mode=c.get('failure_mode'))
        if isinstance(dump,int) and dump in rows:
            pose=rows[dump].get('bucket_landmarks_world') or {};pour=pose.get('pour');target=c['goal']['unload_landing_xyz']
            if pour:r['dump_pour_xy_distance_m']=math.hypot(pour[0]-target[0],pour[1]-target[1])
        cycles.append(r)
    incomplete=[{k:c.get(k) for k in ['cycle_index','status','start_step','dig_step','carry_departure_step','dump_step','end_step','returned','failure_mode','max_bucket_particles']}
        for c in status['repeat_cycles'] if isinstance(c.get('start_step'),int) and not c.get('returned')]
    return dict(cycles=cycles,incomplete_cycles=incomplete,model_action_audit=dict(steps=len(rows),nonmodel_steps=sum(r['action_source']!='model' for r in rows.values()),
        guarded_steps=sum(r['carry_guard'] is not None for r in rows.values()),predicted_applied_differences=sum(r['predicted']!=r['applied'] for r in rows.values())))


def means(cycles):
    keys=['approach_seconds','dig_seconds','carry_dump_seconds','return_seconds','cycle_seconds','max_bucket_particles','delivered_mass_kg','spillage_fraction','dump_pour_xy_distance_m']
    return {k:statistics.mean(c[k] for c in cycles if c.get(k) is not None) for k in keys if any(c.get(k) is not None for c in cycles)}


def main(output):
    scenes={}
    for scene in [50,112]:
        paired=[]
        for repeat in [1,2]:
            name=f'scene_{scene:04d}_repeat_{repeat:02d}'
            a=analyze(BASE/name);b=analyze(NEW/name);common=min(len(a['cycles']),len(b['cycles']))
            paired.append(dict(repeat=repeat,baseline=a,candidate=b,matched_cycle_count=common,
                matched_baseline_means=means(a['cycles'][:common]),matched_candidate_means=means(b['cycles'][:common])))
        scenes[str(scene)]=paired
    proof=dict(status='completed',scenes=scenes,scope='Read-only cycle-index-matched phase differences; observed trajectory associations, not causal attribution. Completed-cycle matching excludes unfinished cycles; two repeats only.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(proof,indent=2)+'\n')
    for scene,pairs in scenes.items():
        print(scene,json.dumps([dict(repeat=p['repeat'],matched_cycles=p['matched_cycle_count'],baseline=p['matched_baseline_means'],candidate=p['matched_candidate_means']) for p in pairs]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
