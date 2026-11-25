"""Stratified teacher-forced diagnostics using cached causal features."""
import argparse,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from excavator_vlm.policies.act.inference import load_act


def main(args):
    torch.set_num_threads(4)
    policy,_,_=load_act(args.checkpoint,'cuda:0',original_act=True)
    metadata=json.loads((args.checkpoint/'benchmark_provenance.json').read_text())
    split=metadata['split_manifest']; statistics=json.loads((args.checkpoint.parents[2]/'normalization_stats.json').read_text())
    std=np.asarray(statistics['action']['std'])
    groups={}; coverage={}
    for path in sorted((args.dataset/'data').rglob('*.parquet')):
        table=pq.read_table(path,columns=['episode_index','frame_index','observation.return_to_pile','observation.stage_current_id'])
        ep=int(table['episode_index'][0].as_py())
        if ep not in split['train']+split['validation']:continue
        with np.load(args.cache/f'episode_{ep:04d}.npz') as z:
            frames=z['frame_index']; features=torch.as_tensor(z['features'],device='cuda')[None]
            target=z['target'][:,0]*std; baseline=z['base'][:,0]*std
            ret=np.asarray(table['observation.return_to_pile'].to_pylist()).reshape(-1)[frames]>0
            stage=np.asarray(table['observation.stage_current_id'].to_pylist()).reshape(-1)[frames]
            masks=dict(all=np.ones(len(ret),bool),returning=ret,unload=(stage==9)&~ret,other=~ret)
            if ep in split['train']:
                for name,mask in masks.items():coverage[name]=coverage.get(name,0)+int(mask.sum())
                continue
            with torch.no_grad():delta=policy.model.memory_adapter(features)[0][0].cpu().numpy().reshape(-1,10,4)
            predicted=baseline+delta[:,0]*std
            for name,mask in masks.items():
                active=mask&(abs(target[:,0])>.2)
                row=groups.setdefault(name,dict(count=0,error=0.,source_error=0.,active_swing=0,wrong_direction=0,source_wrong_direction=0,predicted_swing_sum=0.,target_swing_sum=0.))
                row['count']+=int(mask.sum());row['error']+=float(abs(predicted[mask]-target[mask]).sum());row['source_error']+=float(abs(baseline[mask]-target[mask]).sum())
                row['active_swing']+=int(active.sum());row['wrong_direction']+=int((predicted[active,0]*target[active,0]<=0).sum());row['source_wrong_direction']+=int((baseline[active,0]*target[active,0]<=0).sum())
                row['predicted_swing_sum']+=float(abs(predicted[active,0]).sum());row['target_swing_sum']+=float(abs(target[active,0]).sum())
    for row in groups.values():
        row['mae']=row.pop('error')/max(1,4*row['count']);row['source_mae']=row.pop('source_error')/max(1,4*row['count'])
        row['wrong_swing_fraction']=row['wrong_direction']/max(1,row['active_swing'])
        row['swing_magnitude_ratio']=row['predicted_swing_sum']/max(1e-10,row['target_swing_sum'])
    traces={}
    for name in ('0016','0035'):
        path=args.rollout/f'scene_{name}/evaluation/episodes/episode_0001/steps.jsonl'
        if path.exists():
            rows=[json.loads(line) for line in path.open()]; tail=rows[-100:]
            s=np.asarray([r['input_state'] for r in tail]);a=np.asarray([r['predicted'] for r in tail])
            traces[name]=dict(steps=len(rows),last100_joint_range=np.ptp(s[:,:4],axis=0).tolist(),
                 last100_mean_action=np.mean(a,axis=0).tolist(),last100_tracking_error_mean=np.mean(s[:,8:12],axis=0).tolist(),
                 current_state=rows[-1]['input_state'])
    result=dict(validation=groups,training_coverage=coverage,rollout=traces,policy_inputs_exclude_selection_labels=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--rollout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
