"""Quantify frozen-physics pixel variation and its effect on fixed-state ACT."""
import argparse,json,os
from pathlib import Path
import numpy as np


def analyze(root,actions=False):
    report={'scope':'Diagnostic capture study, no training or task-success measurement','runs':{},'cross_launch':{}}
    for folder in sorted(root.glob('scene_*_repeat_*')):
        base=folder/'evaluation/episodes/episode_0001';path=base/'frozen_capture/report.json'
        if not path.exists():continue
        proof=json.loads(path.read_text());entry=dict(physical_fixed=all(s['physical_fixed'] for s in proof['stages']),
            transport_exact=all(s['transport_roundtrip_exact'] for s in proof['stages']),settings_before=proof['settings_before'],stages={})
        for stage in proof['stages']:
            entry['stages'][stage['mode']]={k:{key:value for key,value in v.items() if key!='shape'} for k,v in stage['cameras'].items() if k!='overview'}
        report['runs'][folder.name]=entry
    for scene in [50,112]:
        paths=[root/f'scene_{scene:04d}_repeat_{i:02d}/evaluation/episodes/episode_0001' for i in [1,2]]
        if not all((p/'frozen_capture/report.json').exists() for p in paths):continue
        a,b=[np.load(p/'frozen_capture/initial_particles.npy') for p in paths]
        first=json.loads((paths[0]/'initial_observation.json').read_text());second=json.loads((paths[1]/'initial_observation.json').read_text())
        cross={'initial_state_max_difference':float(np.max(abs(np.asarray(first['state'])-second['state']))),
               'particle_shapes_equal':a.shape==b.shape}
        if a.shape==b.shape:cross.update(particle_max_abs_difference=float(np.max(abs(a-b))),particle_mae=float(np.mean(abs(a-b))),particles_exact_equal=bool(np.array_equal(a,b)))
        matrices=[np.load(p/'frozen_capture/initial_camera_transforms.npz') for p in paths]
        cross['camera_transform_max_abs_difference']=max(float(np.max(abs(matrices[0][k]-matrices[1][k]))) for k in matrices[0].files)
        cross['pixels']={}
        modes=[stage['mode'] for stage in json.loads((paths[0]/'frozen_capture/report.json').read_text())['stages']]
        for mode in modes:
            cross['pixels'][mode]={}
            for camera in ['0','1','2']:
                a,b=[np.load(p/f'frozen_capture/{mode}/camera_{camera}.npy',mmap_mode='r')[-1] for p in paths]
                difference=abs(a.astype(np.int16)-b.astype(np.int16))
                cross['pixels'][mode][camera]=dict(last_frame_mae=float(difference.mean()),changed_pixel_fraction=float(np.any(difference>0,-1).mean()),max_abs_difference=int(difference.max()))
        report['cross_launch'][str(scene)]=cross
    if actions:
        import torch
        from excavator_vlm.policies.act.inference import load_act
        torch.set_num_threads(4)
        base=Path(__file__).resolve().parents[1]
        checkpoint=base/'outputs/experiments/act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model'
        policy,pre,post=load_act(checkpoint,'cuda:0',original_act=True)
        report['action_analysis_visible_gpu']=os.environ.get('CUDA_VISIBLE_DEVICES')
        for name,entry in report['runs'].items():
            folder=root/name/'evaluation/episodes/episode_0001';scene=int(name.split('_')[1])
            historical=base/f'outputs/experiments/act_velocity_holdout8_20261119_01/scene_{scene:04d}/evaluation/episodes/episode_0001/initial_observation.json'
            state=torch.tensor(json.loads(historical.read_text())['state'],dtype=torch.float32)[None]
            entry['fixed_state_action_sensitivity']={}
            for mode in entry['stages']:
                frames={k:np.load(folder/f'frozen_capture/{mode}/camera_{k}.npy',mmap_mode='r') for k in ['0','1','2']}
                predictions=[]
                with torch.inference_mode():
                    for index in range(24):
                        policy.reset()
                        batch={'observation.state':state,**{f'observation.images.{k}':torch.from_numpy(np.array(value[index])).permute(2,0,1).float()[None]/255 for k,value in frames.items()}}
                        prediction=post(policy.predict_action_chunk(pre(batch))[:,0]).cpu().numpy()[0]
                        predictions.append(prediction)
                predictions=np.stack(predictions)
                entry['fixed_state_action_sensitivity'][mode]=dict(state_reference='identical historical initial state for this scene',
                    mean_action=predictions.mean(0).tolist(),std_action=predictions.std(0).tolist(),range_action=np.ptp(predictions,axis=0).tolist(),
                    last8_range_action=np.ptp(predictions[-8:],axis=0).tolist())
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'analyzed_runs':list(report['runs']),'cross_launch':report['cross_launch']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--actions',action='store_true');args=parser.parse_args();analyze(args.output,args.actions)
