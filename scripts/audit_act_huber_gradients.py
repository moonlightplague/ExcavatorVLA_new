"""Estimate output-gradient allocation from original ACT training observations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def audit(root=ROOT):
    import numpy as np
    sys.path.insert(0,str(root/'ExcavatorVLA-vlm'))
    from excavator_vlm.data.contract import read_frames
    base=root/'outputs/baselines/act_phw40k_carry275_20261105'
    previous=root/'outputs/experiments/act_stage_auxiliary_20261105_01'
    paths=[base/'normalization_stats.json',previous/'offline_observations.json',previous/'offline_predictions.npz']
    std=np.asarray(json.loads(paths[0].read_text())['action']['std'])
    observations=json.loads(paths[1].read_text())['observations']
    arrays=np.load(paths[2])
    frames=read_frames(root/'data/export_phw')
    refs={int(ep):g.sort_values('frame_index') for ep,g in frames.groupby('episode_index')}
    records={}
    for stage in (5,6):
        ids=[i for i,r in enumerate(observations) if r['split']=='train' and r['stage_id']==stage]
        target=arrays['target'][ids];prediction=arrays['baseline'][ids]
        error=np.abs((prediction-target)/std)
        effective=[]
        for position,i in enumerate(ids):
            row=observations[i];episode=refs[row['episode_index']];frame=row['frame_index']
            indices=np.minimum(np.arange(frame,frame+10),len(episode)-1)
            expected=episode.iloc[indices]
            np.testing.assert_array_equal(target[position],np.stack(expected.action).astype(target.dtype))
            effective.append(np.asarray(expected.action_loss_weight.tolist()).reshape(-1)*
                             (np.arange(frame,frame+10)<len(episode)))
        w=np.stack(effective)[...,None]*np.ones((1,1,4))
        l1_gradient=(error>0)*w
        huber_gradient=np.minimum(error/.1,1)*w
        near=np.abs(target[:,:,3])<=.03
        bucket_l1=l1_gradient[:,:,3];bucket_huber=huber_gradient[:,:,3]
        records[str(stage)]=dict(training_observations=len(ids),
            first3_l1_error_share=float((error*w)[:,:3].sum()/(error*w).sum()),
            first3_l1_output_gradient_share=float(l1_gradient[:,:3].sum()/l1_gradient.sum()),
            first3_huber_output_gradient_share=float(huber_gradient[:,:3].sum()/huber_gradient.sum()),
            bucket_nearzero_target_l1_gradient_share=float(bucket_l1[near].sum()/bucket_l1.sum()),
            bucket_nearzero_target_huber_gradient_share=float(bucket_huber[near].sum()/bucket_huber.sum()),
            first_bucket_expert_mean_abs_rad_s=float(np.abs(target[:,0,3]).mean()),
            first_bucket_expert_median_rad_s=float(np.median(target[:,0,3])),
            first_bucket_expert_abs_above_point1_fraction=float((np.abs(target[:,0,3])>=.1).mean()))
    return dict(scope='Train-only descriptive output-gradient audit; not parameter gradients, '
        'a causal diagnosis or simulated counterfactual result.',
        beta_normalized_units=.1,test_used=False,records=records,
        estimate='Absolute residual divided by original action std; original PHW weights and episode padding. '
            'Smooth L1 derivative magnitude is min(abs(normalized residual)/beta,1).',
        sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps(result['records'],ensure_ascii=False))
