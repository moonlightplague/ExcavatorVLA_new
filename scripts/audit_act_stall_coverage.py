"""Describe training-state coverage around recorded loaded ACT stalls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def audit():
    import numpy as np
    sys.path.insert(0,str(ROOT/'ExcavatorVLA-vlm'))
    from excavator_vlm.data.contract import read_frames
    base=ROOT/'outputs/baselines/act_phw40k_carry275_20261105'
    source=base/'checkpoints/040000/pretrained_model'
    split=json.loads((source/'benchmark_provenance.json').read_text())['split_manifest']
    stats=json.loads((base/'normalization_stats.json').read_text())['observation.state']
    frames=read_frames(ROOT/'data/export_phw')
    frames=frames[frames.episode_index.isin(split['train'])]
    states=np.stack(frames['observation.state']).astype(np.float32)
    mean=np.asarray(stats['mean']);std=np.maximum(np.asarray(stats['std']),1e-8)
    norm=(states-mean)/std
    stage=frames['observation.stage_current_id'].to_numpy(int)
    training={}
    for sid in (4,5,6,7,8):
        x=states[stage==sid]
        training[str(sid)]=dict(observations=len(x),
            fill_quantiles_0_10_50_90_100=np.quantile(x[:,26],[0,.1,.5,.9,1]).tolist(),
            bucket_quantiles_10_50_90=np.quantile(x[:,3],[.1,.5,.9]).tolist(),
            fill_below_point03=int((x[:,26]<.03).sum()),fill_below_point04=int((x[:,26]<.04).sum()))
    experiment=ROOT/'outputs/experiments/act_transport_validation40_20261104_01'
    results=json.loads((experiment/'results.json').read_text())
    cases=[];hashes={}
    for key,record in sorted(results.items(),key=lambda p:int(p[0])):
        if record['transport']['available']:continue
        path=experiment/'evaluation/episodes'/f'episode_{int(key):04d}/steps.jsonl'
        contents=path.read_bytes();rows=[json.loads(s) for s in contents.splitlines()]
        loaded=[];previous=0
        for row in rows:
            # Compare the state consumed by this action with the preceding
            # returned particle count, rather than its post-action count.
            if previous>=100:loaded.append(row)
            previous=row['particle_transfer_metrics']['retained_particles']
        if len(loaded)<100:continue
        x=np.asarray([row['input_state'] for row in loaded])
        representative=loaded[len(loaded)//2]
        value=np.asarray(representative['input_state'])
        distance=np.sqrt(((norm-(value-mean)/std)**2).mean(-1))
        nearest=np.argsort(distance,kind='stable')[:5]
        cases.append(dict(scene=int(key),seed=record['scene_seed'],loaded_input_observations=len(loaded),
            fill_quantiles_10_50_90=np.quantile(x[:,26],[.1,.5,.9]).tolist(),
            bucket_quantiles_10_50_90=np.quantile(x[:,3],[.1,.5,.9]).tolist(),
            guard_modified_observations=sum(bool(row['carry_guard'].get('modified')) for row in loaded),
            representative_step=representative['step'],representative_state=representative['input_state'],
            nearest_training_states=[dict(episode_index=int(frames.iloc[i].episode_index),
                frame_index=int(frames.iloc[i].frame_index),stage_id=int(stage[i]),
                normalized_state_rms_distance=float(distance[i])) for i in nearest]))
        hashes[str(path)]=hashlib.sha256(contents).hexdigest()
    return dict(scope='Read-only coverage diagnosis on viewed development failures; no new labels, training or score changes.',
        limitations='Particle counts in a buried or open bucket do not establish readiness for loaded transport. '
            'Different state marginals or state-only neighbours do not prove missing full-input coverage or causality. '
            'Any recovery expert must recompute actions in the actual current physical scene; never copy neighbour or failure actions.',
        original_training_split=split['train'],training=training,cases=cases,
        persistent_cases=len(cases),cases_with_median_fill_below_point03=sum(c['fill_quantiles_10_50_90'][1]<.03 for c in cases),
        test_used=False,source_steps_sha256=hashes)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit();args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps({k:result[k] for k in ('persistent_cases','cases_with_median_fill_below_point03','training')},ensure_ascii=False))
