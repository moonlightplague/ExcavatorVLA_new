"""Train two model-only alternatives using immutable causal ACT feature caches."""
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
PREVIOUS=ROOT/'outputs/experiments/act_persistent_temporal5k_20261119_03'
SOURCE=PREVIOUS/'train/checkpoints/005000/pretrained_model'
DATA=ROOT/'outputs/experiments/act_repeat_control_repair_data_20261118_01/dataset'
CACHE=PREVIOUS/'features'
RUNTIME=ROOT/'outputs/experiments/act_training_alternatives_runtime_20261119_01/python'


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata():return json.loads((SOURCE/'benchmark_provenance.json').read_text())


def assert_external():
    expected=json.loads((PREVIOUS/'external_files.json').read_text())
    for name,sha in expected.items():
        if digest(ROOT/name)!=sha:raise ValueError('External evaluator/controller changed: '+name)
    return expected


def cpu_data(episode):
    import numpy as np
    with np.load(CACHE/f'episode_{episode:04d}.npz') as data:return {k:data[k].copy() for k in data.files}


def fit_pose_statistics(episodes,stats):
    import numpy as np
    state_mean=np.asarray(stats['observation.state']['mean']);state_std=np.asarray(stats['observation.state']['std'])+1e-8
    action_mean=np.asarray(stats['action']['mean']);action_std=np.asarray(stats['action']['std'])
    minimum=np.full(4,np.inf);maximum=np.full(4,-np.inf)
    for episode in episodes:
        data=cpu_data(episode);state=data['features'][:,:28]*state_std+state_mean
        q=state[:,:4]+state[:,8:12]
        poses=q[:,None]+.1*np.cumsum(data['target']*action_std+action_mean,axis=1)
        selected=poses[~data['padding']];minimum=np.minimum(minimum,selected.min(0));maximum=np.maximum(maximum,selected.max(0))
    center=(minimum+maximum)/2;radius=(maximum-minimum)/2+.02
    xx=np.eye(29)*.001;xy=np.zeros((29,40))
    for episode in episodes:
        data=cpu_data(episode);valid=~data['padding'].any(1);x=data['features'][valid,:28].astype(np.float64)
        state=x*state_std+state_mean;q=state[:,:4]+state[:,8:12]
        pose=q[:,None]+.1*np.cumsum(data['target'][valid]*action_std+action_mean,axis=1)
        y=np.arctanh(np.clip((pose-center)/radius,-.9999,.9999)).reshape(-1,40)
        x=np.concatenate((x,np.ones((len(x),1))),1);xx+=x.T@x;xy+=x.T@y
    coefficients=np.linalg.solve(xx,xy)
    return dict(center=center.tolist(),radius=radius.tolist(),minimum=minimum.tolist(),maximum=maximum.tolist(),
                fit_episodes=episodes,state_normalization_mean=state_mean.tolist(),state_normalization_std=state_std.tolist(),
                action_normalization_mean=action_mean.tolist(),action_normalization_std=action_std.tolist()),coefficients


def construct(kind,pose_stats=None,coefficients=None):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from safetensors.torch import load_file
    config=PreTrainedConfig.from_pretrained(str(SOURCE));config.device='cuda:0'
    config.action_parameterization='command_pose' if kind=='pose' else 'velocity'
    if kind=='pose':
        config.command_pose_center=pose_stats['center'];config.command_pose_radius=pose_stats['radius']
        for key in ('state_normalization_mean','state_normalization_std','action_normalization_mean','action_normalization_std'):
            setattr(config,key,pose_stats[key])
    policy=ACTPolicy(config).cuda().eval();values=load_file(str(SOURCE/'model.safetensors'))
    if kind=='pose':
        values={k:v for k,v in values.items() if not k.startswith('model.memory_adapter.output.')}
        result=policy.load_state_dict(values,strict=False)
        allowed={'model.memory_adapter.'+s for s in ('output.weight','output.bias','pose_state.weight','pose_state.bias',
            'pose_center','pose_radius','state_mean','state_std','action_mean','action_std')}
        if set(result.missing_keys)!=allowed or result.unexpected_keys:raise ValueError(str(result))
        with torch.no_grad():
            policy.model.memory_adapter.pose_state.weight.copy_(torch.as_tensor(coefficients[:-1].T,device='cuda',dtype=torch.float32))
            policy.model.memory_adapter.pose_state.bias.copy_(torch.as_tensor(coefficients[-1],device='cuda',dtype=torch.float32))
    else:policy.load_state_dict(values,strict=True)
    return policy


def annotations(episodes):
    import numpy as np
    import pyarrow.parquet as pq
    needed=set(episodes);result={}
    for path in sorted((DATA/'data').rglob('*.parquet')):
        table=pq.read_table(path,columns=['episode_index','observation.return_to_pile','observation.stage_current_id'])
        episode=int(table['episode_index'][0].as_py())
        if episode not in needed:continue
        frames=cpu_data(episode)['frame_index'];ret=np.asarray(table['observation.return_to_pile'].to_pylist()).reshape(-1)[frames]>0
        stage=np.asarray(table['observation.stage_current_id'].to_pylist()).reshape(-1)[frames]
        changes=np.flatnonzero(np.r_[False,(ret[1:]!=ret[:-1])|(stage[1:]!=stage[:-1])])
        focused=ret.copy()
        for change in changes:focused[max(0,change-6):min(len(ret),change+7)]=True
        result[episode]=(ret, np.flatnonzero(focused))
    if set(result)!=needed:raise ValueError('Missing episode annotations')
    return result


def evaluate(memory,episodes,std):
    import torch,numpy as np
    memory.eval();groups={name:np.zeros(5) for name in ('all','return','other')}
    with torch.no_grad():
        for feature,base,target,pad,ret,focus in episodes:
            delta,_=memory(feature[None]);pred=base+delta[0].reshape_as(base)
            error=(pred-target).abs()*std
            for name,select in [('all',torch.ones_like(ret)),('return',ret),('other',~ret)]:
                mask=(~pad)&select[:,None];active=select&(target[:,0,0].abs()*std[0]>.2)
                groups[name]+=np.array([error[mask].sum().item(),mask.sum().item()*4,
                    error[select,0].sum().item(),select.sum().item()*4,
                    ((pred[active,0,0]*target[active,0,0])<=0).sum().item()])
    memory.train()
    return {name:dict(mae=float(v[0]/max(1,v[1])),first_action_mae=float(v[2]/max(1,v[3])),
                      wrong_active_swing_directions=int(v[4])) for name,v in groups.items()}


def train(args):
    import numpy as np,torch
    from safetensors.torch import load_file
    torch.set_num_threads(4);torch.manual_seed(20261119);rng=np.random.default_rng(20261119)
    out=args.output/args.kind;out.mkdir(parents=True,exist_ok=False)
    original=metadata();split=original['split_manifest'];stats=json.loads((SOURCE.parents[2]/'normalization_stats.json').read_text())
    assert not set(split['train'])&set(split['validation']+split['test'])
    pose_stats,coefficients=(fit_pose_statistics(split['train'],stats) if args.kind=='pose' else (None,None))
    if pose_stats:write(out/'pose_statistics.json',pose_stats)
    policy=construct(args.kind,pose_stats,coefficients);memory=policy.model.memory_adapter
    note=annotations(split['train']+split['validation'])
    def load(episode):
        data=cpu_data(episode);ret,focus=note[episode]
        return tuple(torch.as_tensor(data[k],device='cuda') for k in ('features','base','target','padding'))+(torch.as_tensor(ret,device='cuda'),focus)
    training=[load(ep) for ep in split['train']];validation=[load(ep) for ep in split['validation']]
    probs=np.array([len(ep[0]) for ep in training],dtype=np.float64);probs/=probs.sum()
    std=torch.tensor(stats['action']['std'],device='cuda',dtype=torch.float32)
    mean=torch.tensor(stats['action']['mean'],device='cuda',dtype=torch.float32)
    initial=evaluate(memory,validation,std);write(out/'validation_initial.json',initial)
    unroll=64;batch_size=8;optimizer=torch.optim.AdamW(memory.parameters(),lr=1e-4 if args.kind=='velocity' else 3e-4,weight_decay=1e-4)
    steps=args.velocity_steps if args.kind=='velocity' else args.pose_steps
    started=time.time();best=None
    for step in range(1,steps+1):
        chosen=rng.choice(len(training),batch_size,p=probs);starts=[]
        for e in chosen:
            if rng.random()<.5 and len(training[e][5]):
                anchor=int(rng.choice(training[e][5]));start=int(np.clip(anchor-8,0,len(training[e][0])-unroll))
            else:start=int(rng.integers(0,len(training[e][0])-unroll+1))
            starts.append(start)
        prefix=torch.zeros(batch_size,max(1,max(starts)),1604,device='cuda')
        fields=[[] for _ in range(5)]
        for i,(e,start) in enumerate(zip(chosen,starts,strict=True)):
            prefix[i,:start]=training[e][0][:start]
            for slot in range(5):fields[slot].append(training[e][slot][start:start+unroll])
        feature,base,target,pad,ret=[torch.stack(x) for x in fields]
        hidden=memory.prefix_hidden(prefix,torch.tensor(starts))
        delta,_=memory(feature,hidden);prediction=base+delta.reshape_as(base)
        horizon=torch.tensor([2.,2.,2.,1.,1.,1.,1.,1.,1.,1.],device='cuda')
        joint=torch.tensor([2.,1.,1.,1.],device='cuda')
        weight=(~pad).float()[...,None]*(1.+2.*ret.float()[...,None,None])*horizon[None,None,:,None]*joint
        l1=((prediction-target).abs()*weight).sum()/weight.sum()
        true_physical=target*std+mean;pred_physical=prediction*std+mean
        active=true_physical.abs()>.2
        margin=torch.relu(.35*true_physical.abs()-true_physical.sign()*pred_physical)/std
        direction=(margin*weight*active).sum()/(weight*active).sum().clamp_min(1)
        loss=l1+.1*direction
        if args.kind=='pose':
            predicted_pose=memory.target_poses(feature,prediction)
            actual_pose=memory.target_poses(feature,target)
            pose_error=(predicted_pose-actual_pose).abs()/memory.pose_radius
            pose_loss=(pose_error*weight).sum()/weight.sum()
            loss=loss+.25*pose_loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(memory.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        # Decay only after stable initial fitting of the new position head.
        for group in optimizer.param_groups:
            group['lr']=(1e-4 if args.kind=='velocity' else 3e-4)*(.1+.9*.5*(1+np.cos(np.pi*step/steps)))
        if step==1 or step%100==0:
            status=dict(status='training',variant=args.kind,step=step,steps=steps,loss=loss.item(),l1=l1.item(),direction_loss=direction.item(),gradient_norm=norm.item(),elapsed_seconds=time.time()-started)
            write(out/'status.json',status);print(json.dumps(status),flush=True)
        if step%2000==0 or step==steps:
            metrics=evaluate(memory,validation,std);write(out/f'validation_{step:06d}.json',metrics);print(json.dumps(dict(step=step,validation=metrics)),flush=True)
    # Source core, all processors and deployment contract stay identical.
    original_weights=load_file(str(SOURCE/'model.safetensors'))
    for name,value in original_weights.items():
        if not name.startswith('model.memory_adapter.') and not torch.equal(value,policy.state_dict()[name].cpu()):raise ValueError('Frozen core changed: '+name)
    destination=out/f'train/checkpoints/{steps:06d}/pretrained_model';destination.mkdir(parents=True)
    policy.eval();policy.save_pretrained(destination)
    for file in SOURCE.glob('policy_*'):shutil.copy2(file,destination/file.name)
    shutil.copy2(SOURCE.parents[2]/'normalization_stats.json',out/'train/normalization_stats.json')
    original['model_training_alternative']=dict(variant=args.kind,source=str(SOURCE),source_sha256=digest(SOURCE/'model.safetensors'),steps=steps,
        sampling='50% training-only return/transition focused; 50% uniform episode-time; full causal prefix reconstructed',
        unroll=unroll,loss='weighted L1 plus 0.1 active command-direction margin; pose adds 0.25 normalized pose loss',
        weights=dict(return_factor=3,executed_first_three_factor=2,swing_factor=2),pose_statistics=pose_stats,
        no_task_labels_in_model_input=True,external_files_unchanged=True)
    write(destination/'benchmark_provenance.json',original)
    torch.save(dict(optimizer=optimizer.state_dict(),step=steps,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),destination.parent/'training_state.pt')
    from excavator_vlm.policies.act.inference import load_act
    restored,_,_=load_act(destination,'cuda:0',original_act=True)
    with torch.no_grad():
        sample=validation[0][0][:64][None];torch.testing.assert_close(memory(sample)[0],restored.model.memory_adapter(sample)[0],rtol=0,atol=0)
    assert_external()
    write(out/'status.json',dict(status='completed',variant=args.kind,step=steps,checkpoint=str(destination),validation=metrics,
                                frozen_core_unchanged=True,strict_reload='passed',external_files_unchanged=True))


def run(args):
    preflight=RUNTIME.parent/'preflight.json'
    if not preflight.is_file() or json.loads(preflight.read_text()).get('status')!='passed':
        raise RuntimeError('Both alternative trainers must pass real-data preflight')
    previous=json.loads((PREVIOUS/'experiment.json').read_text())
    if previous['fingerprint']!=metadata()['dataset_fingerprint']:
        raise ValueError('Cached dataset fingerprint differs from source checkpoint')
    required=metadata()['split_manifest']['train']+metadata()['split_manifest']['validation']
    if len(required)!=160 or any(not (CACHE/f'episode_{ep:04d}.npz').is_file() for ep in required):
        raise ValueError('Complete train/validation causal feature cache required')
    args.output.mkdir(parents=True,exist_ok=False);assert_external()
    shutil.copy2(preflight,args.output/'preflight.json')
    shutil.copy2(RUNTIME.parent/'runtime_manifest.json',args.output/'runtime_manifest.json')
    write(args.output/'cache_hashes.json',{str(ep):digest(CACHE/f'episode_{ep:04d}.npz') for ep in required})
    shutil.copy2(Path(__file__),args.output/'run.py')
    write(args.output/'external_files.json',assert_external())
    write(args.output/'manifest.json',dict(source=str(SOURCE),cache=str(CACHE),dataset=str(DATA),source_sha256=digest(SOURCE/'model.safetensors'),
        cache_experiment_sha256=digest(PREVIOUS/'experiment.json'),split=metadata()['split_manifest'],
        variants=dict(velocity_steps=args.velocity_steps,pose_steps=args.pose_steps),existing_evaluation_preserved=True))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',ACT_PROJECT_ROOT=str(ROOT),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    for kind in ('velocity','pose'):
        with (args.output/f'{kind}.log').open('w') as log:
            subprocess.run([str(RUNTIME),'-u','-B',str(args.output/'run.py'),'--mode','train','--kind',kind,'--output',str(args.output),
                            '--velocity-steps',str(args.velocity_steps),'--pose-steps',str(args.pose_steps)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(args.output/'status.json',dict(status='waiting_for_existing_evaluation',reference=str(PREVIOUS/'evaluation_pair/completion.json')))
    while not (PREVIOUS/'evaluation_pair/completion.json').exists():time.sleep(15)
    for kind,steps in [('velocity',args.velocity_steps),('pose',args.pose_steps)]:
        trained=json.loads((args.output/kind/'status.json').read_text())
        score=trained['validation']['all']
        if (score['mae'] > .5 or score['first_action_mae'] > .5):
            write(args.output/kind/'evaluation_gate.json',dict(status='rejected',reason='Validation action MAE exceeds 0.5 rad/s',validation=score))
            continue
        assert_external();folder=args.output/kind/'evaluation_pair';folder.mkdir()
        template=(PREVIOUS/'evaluation_pair/run.py').read_text()
        old=str(SOURCE.relative_to(ROOT/'outputs/experiments'))
        new=str((args.output/kind/f'train/checkpoints/{steps:06d}/pretrained_model').relative_to(ROOT/'outputs/experiments'))
        if template.count(old)!=1:raise ValueError('Evaluation template checkpoint anchor changed')
        template=template.replace(old,new).replace('Model-internal persistent GRU memory checkpoint',f'Model-only {kind} action representation checkpoint')
        (folder/'run.py').write_text(template)
        write(args.output/'status.json',dict(status='evaluating',variant=kind,output=str(folder)))
        with (folder/'orchestrator.log').open('w') as log:
            subprocess.run([str(RUNTIME),'-B',str(folder/'run.py')],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1'},stdout=log,stderr=subprocess.STDOUT,check=True)
    write(args.output/'status.json',dict(status='completed',variants=['velocity','pose']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=lambda s:Path(s).resolve(),required=True)
    parser.add_argument('--mode',choices=['run','train'],default='run');parser.add_argument('--kind',choices=['velocity','pose'],default='velocity')
    parser.add_argument('--velocity-steps',type=int,default=8000);parser.add_argument('--pose-steps',type=int,default=20000)
    args=parser.parse_args()
    try: {'run':run,'train':train}[args.mode](args)
    except Exception as error:
        if args.output.exists():write(args.output/f'error_{args.mode}_{args.kind}.json',dict(error=repr(error)))
        raise
