"""Inspect nearby validation poses with different command directions; optionally ablate history."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch.utils.data import default_collate

from smolvla_history import CAMERAS, MASK, configure_dataset, history_spec, inference_batch, load_history
from excavator_vlm.data.contract import read_frames


def write(path,payload):path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def find_pairs(frames,episodes):
    candidates=[]
    for episode in episodes:
        group=frames[frames.episode_index==episode]
        unload=group[(group['observation.stage_current_id']==9)&(group['observation.return_to_pile']==0)]
        returning=group[group['observation.return_to_pile']==1]
        if unload.empty or returning.empty:continue
        states_a=np.stack(unload['observation.state'])[:,:4]
        states_b=np.stack(returning['observation.state'])[:,:4]
        distances,neighbors=cKDTree(states_b).query(states_a,k=min(32,len(states_b)))
        if distances.ndim==1:distances=distances[:,None];neighbors=neighbors[:,None]
        actions_a=np.stack(unload.action);actions_b=np.stack(returning.action)
        for i in range(len(unload)):
            for distance,j in zip(distances[i],neighbors[i]):
                a,b=actions_a[i],actions_b[j]
                if distance>.5 or np.linalg.norm(a)<.1 or np.linalg.norm(b)<.1:continue
                cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                if cosine>-.25 or abs(float(unload.iloc[i].timestamp-returning.iloc[j].timestamp))<.6:continue
                candidates.append({'episode_index':int(episode),'unload_index':int(unload.iloc[i]['index']),
                    'return_index':int(returning.iloc[j]['index']),'joint_pose_l2_rad':float(distance),
                    'target_direction_cosine':cosine,'unload_action':a.tolist(),'return_action':b.tolist()})
        if candidates:
            # Only the best pair from an episode is needed for a qualitative contact sheet.
            local=[row for row in candidates if row['episode_index']==episode]
            candidates=[row for row in candidates if row['episode_index']!=episode]
            if local:candidates.append(min(local,key=lambda row:row['joint_pose_l2_rad']))
    return sorted(candidates,key=lambda row:row['joint_pose_l2_rad'])[:3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--wait-for-model',action='store_true')
    parser.add_argument('--gpu',default='1')
    args=parser.parse_args()
    root=args.experiment.resolve();prepared=root/'prepared';output=root/'phase_pairs'
    output.mkdir(exist_ok=False)
    state={'status':'running','phase':'selecting_validation_pairs'}
    write(output/'status.json',state)
    split=json.loads((prepared/'split_manifest.json').read_text())
    audit=json.loads((prepared/'dataset_audit.json').read_text())
    frames=read_frames(Path(audit['dataset']))
    pairs=find_pairs(frames,split['validation'])
    write(output/'selected_pairs.json',{'selection':'exploratory validation only; same episode, nearby joint pose, negative target-direction cosine',
        'criteria':{'joint_l2_max_rad':.5,'target_direction_cosine_max':-.25,'separation_seconds_min':.6},
        'prior_strict_search':'No pairs met joint L2 <=0.25 rad and target cosine <=-0.5; preserved separately.',
        'pairs':pairs})
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset=LeRobotDataset('local/excavator',root=Path(audit['dataset']),episodes=split['validation'],
        delta_timestamps={'action':[i/10 for i in range(10)]},video_backend='pyav')
    configure_dataset(dataset,history_spec())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    samples=[]
    for number,pair in enumerate(pairs,1):
        fig,axes=plt.subplots(2,9,figsize=(16,4.5),constrained_layout=True)
        for row,name in enumerate(('unload','return')):
            absolute=pair[f'{name}_index']
            relative=dataset._absolute_to_relative_idx[absolute] if dataset._absolute_to_relative_idx is not None else absolute
            sample=dataset[relative];samples.append((number,name,absolute,sample))
            for time_index,offset in enumerate((-.6,-.3,0)):
                for camera_index,key in enumerate(CAMERAS):
                    axis=axes[row,time_index*3+camera_index]
                    axis.imshow(sample[key][time_index].permute(1,2,0).numpy());axis.axis('off')
                    axis.set_title(f'{name}: {offset:g}s / cam {camera_index}',fontsize=8)
        fig.suptitle(f'Validation episode {pair["episode_index"]}: joint distance {pair["joint_pose_l2_rad"]:.3f} rad; '
            f'target cosine {pair["target_direction_cosine"]:.3f}\n'
            'Columns are chronological history / camera identity. Labels are evaluation annotations only.',fontsize=10)
        fig.savefig(output/f'pair_{number}.png',dpi=150);plt.close(fig)
    print(json.dumps({'pairs':len(pairs),'output':str(output)}),flush=True)
    if not args.wait_for_model:
        write(output/'status.json',{'status':'completed','phase':'data_only_visual_inspection','pairs':len(pairs)});return
    state.update(phase='waiting_for_final_model',pairs=len(pairs));write(output/'status.json',state)
    progress=root/'train/progress.json'
    while not progress.exists() or json.loads(progress.read_text()).get('status')!='completed':
        controller=root/'status.json'
        if controller.exists() and json.loads(controller.read_text())['status']=='failed':
            write(output/'status.json',{'status':'failed','error':'Training controller failed; no final-model analysis performed'});return
        time.sleep(10)
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    checkpoint=root/'train/checkpoints/step_20000/pretrained_model'
    policy,pre,post=load_history(checkpoint,'cuda')
    results=[]
    with torch.inference_mode():
        for pair,name,absolute,sample in samples:
            for mode in ('full_history','current_images_only_masked_history'):
                raw=inference_batch(default_collate([sample]))
                if mode!='full_history':
                    mask=torch.tensor([[True,True,False]])
                    raw[MASK]=mask
                    for key in ('observation.state',*CAMERAS):raw[f'{key}_is_pad']=mask.clone()
                noise=torch.randn(1,10,32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(1000+absolute))
                action=post(policy.predict_action_chunk(pre(raw),noise=noise)[:,0]).cpu()[0]
                target=sample['action'][0]
                results.append({'pair':pair,'phase':name,'absolute_index':absolute,'mode':mode,
                    'predicted_action':action.tolist(),'target_action':target.tolist(),
                    'mae_rad_s':(action-target).abs().mean().item()})
    write(output/'model_ablation.json',{'status':'completed','checkpoint':str(checkpoint),'observations':len(samples),
        'scope':'Selected validation-pose diagnostic. Masking history at inference is not a separately trained baseline or a task-success test.',
        'full_28d_state_retained_in_both_conditions':True,'results':results})
    write(output/'status.json',{'status':'completed','phase':'final_model_context_ablation','pairs':len(pairs)})


if __name__=='__main__':main()
