"""Phase-stratified action diagnostics; annotations select evaluation rows only."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import default_collate

from prepare_smolvla_history import phase_subsets, write
from smolvla_history import CAMERAS, MANIFEST, configure_dataset, history_spec, load_history, inference_batch


def load_model(checkpoint):
    if (checkpoint / MANIFEST).is_file():
        return load_history(checkpoint,'cuda')
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    cfg=PreTrainedConfig.from_pretrained(checkpoint,local_files_only=True)
    if cfg.n_obs_steps!=1:
        raise ValueError('Baseline must be a single-timestamp checkpoint')
    cfg.device='cuda';cfg.load_vlm_weights=False
    policy=SmolVLAPolicy.from_pretrained(checkpoint,config=cfg,strict=True,local_files_only=True).eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor':{'device':'cuda'}})
    return policy,pre,post


def evaluate(checkpoint, prepared, split_name, output):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from excavator_vlm.data.contract import read_frames
    report=json.loads((prepared/'dataset_audit.json').read_text())
    split=json.loads((prepared/'split_manifest.json').read_text())
    root=Path(report['dataset'])
    frames=read_frames(root)
    phases=phase_subsets(frames,split[split_name])
    dataset=LeRobotDataset('local/excavator',root=root,episodes=split[split_name],video_backend='pyav',
        delta_timestamps={'action':[i/10 for i in range(10)]})
    configure_dataset(dataset,history_spec())
    policy,pre,post=load_model(checkpoint)
    history=(checkpoint/MANIFEST).is_file()
    results={}
    all_rows=[]
    for phase,indices in phases.items():
        errors=[];chunk_numerator=chunk_denominator=correct=nonzero=0
        for absolute in indices:
            relative=(dataset._absolute_to_relative_idx[absolute]
                if dataset._absolute_to_relative_idx is not None else absolute)
            sample=dataset[relative]
            raw=default_collate([sample])
            if not history:
                for key in ('observation.state',*CAMERAS):raw[key]=raw[key][:,-1]
            raw=inference_batch(raw)
            if not history:
                raw={key:value for key,value in raw.items() if key in ('task','observation.state',*CAMERAS)}
            noise=torch.randn(1,policy.config.chunk_size,policy.config.max_action_dim,device='cuda',
                generator=torch.Generator(device='cuda').manual_seed(1000+absolute))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=policy.predict_action_chunk(pre(raw),noise=noise)
                physical=torch.stack([post(prediction[:,i]) for i in range(prediction.shape[1])],dim=1)[0].cpu()
            target=sample['action']
            if physical.shape!=target.shape or not torch.isfinite(physical).all():
                raise ValueError('Invalid inference chunk')
            error=(physical-target).abs()
            weights=(~sample['action_is_pad']).float()
            # Evaluation reports ordinary physical action MAE, masking episode padding.
            chunk_numerator+=(error*weights[:,None]).sum().item()
            chunk_denominator+=weights.sum().item()*4
            active=target[0].abs()>=.05
            correct+=((physical[0].sign()==target[0].sign()) & active).sum().item()
            nonzero+=active.sum().item()
            errors.append(error[0].mean().item())
            all_rows.append({'absolute_index':absolute,'episode_index':int(sample['episode_index']),
                'phase':phase,'target_first_action':target[0].tolist(),'predicted_first_action':physical[0].tolist(),
                'first_action_mae_rad_s':errors[-1]})
        results[phase]={'decision_observations':len(indices),'first_action_mae_rad_s':float(np.mean(errors)) if errors else None,
            'chunk_mae_rad_s':chunk_numerator/chunk_denominator if chunk_denominator else None,
            'direction_accuracy_abs_target_at_least_0_05':correct/nonzero if nonzero else None,
            'active_joint_decisions':nonzero}
        print(json.dumps({'phase':phase,**results[phase]}),flush=True)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'predictions.json',all_rows)
    final={'status':'passed','checkpoint':str(checkpoint),'split':split_name,'episodes':len(split[split_name]),
        'dataset_fingerprint':report['fingerprint'],'history':history,'phase_metrics':results,
        'scope':'Offline recorded-observation action diagnostics; not closed-loop task success.',
        'phase_labels_used_as_inputs':False,'sampling':'8 uniformly spaced decision observations per phase per episode'}
    write(output/'report.json',final)
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--split',choices=('validation','test'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',default='0')
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    evaluate(args.checkpoint.resolve(),args.prepared.resolve(),args.split,args.output.resolve())
