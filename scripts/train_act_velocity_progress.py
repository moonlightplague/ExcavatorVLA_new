"""Continue velocity ACT with dig/carry/recovery balance and optional learned progress.

Task/recovery labels select training examples and loss weights only. Public
model observations and velocity actions, controllers and evaluators stay fixed.
"""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'scripts'))
import run_act_training_alternatives as common
SOURCE=ROOT/'outputs/experiments/act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model'
common.SOURCE=SOURCE
RUNTIME=ROOT/'outputs/experiments/act_velocity_progress_runtime_20261119_01/python'
DIMS=[0,1,2,3,18,19,20,21,22,23,26]


def future_delta(features,horizon=4,segment_ids=None):
    import numpy as np
    index=np.arange(len(features))+horizon;valid=index<len(features)
    if segment_ids is not None:
        valid &= segment_ids[np.minimum(index,len(features)-1)]==segment_ids
    return features[np.minimum(index,len(features)-1)][:,DIMS]-features[:,DIMS],valid


def construct(progress):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from safetensors.torch import load_file
    config=PreTrainedConfig.from_pretrained(str(SOURCE));config.device='cuda:0';config.memory_progress=progress
    stats=json.loads((SOURCE.parents[2]/'normalization_stats.json').read_text())
    config.state_normalization_mean=stats['observation.state']['mean']
    config.state_normalization_std=[float(s)+1e-8 for s in stats['observation.state']['std']]
    policy=ACTPolicy(config).cuda().eval();result=policy.load_state_dict(load_file(str(SOURCE/'model.safetensors')),strict=False)
    new={k for k in policy.state_dict() if k.startswith(('model.memory_adapter.angle_','model.memory_adapter.future_state_head.'))}
    if set(result.missing_keys)!=new or result.unexpected_keys:raise ValueError(str(result))
    return policy


def load_episodes(ids,stats):
    import numpy as np,torch,pyarrow.parquet as pq
    tables={}
    for path in sorted((common.DATA/'data').rglob('*.parquet')):
        table=pq.read_table(path,columns=['episode_index','observation.stage_current_id','observation.return_to_pile','observation.recovery_active','observation.scoop_index'])
        ep=int(table['episode_index'][0].as_py())
        if ep in ids:tables[ep]=table
    episodes=[]
    for ep in ids:
        data=common.cpu_data(ep);frame=data['frame_index'];table=tables[ep]
        stage=np.asarray(table['observation.stage_current_id'].to_pylist()).reshape(-1)[frame]
        ret=np.asarray(table['observation.return_to_pile'].to_pylist()).reshape(-1)[frame]>0
        recovery=np.asarray(table['observation.recovery_active'].to_pylist()).reshape(-1)[frame]>0
        dig=np.isin(stage,[2,3,4,5])&~ret;carry=(stage>=6)&~ret
        moving=np.max(abs(data['target'][:,0]*np.asarray(stats['action']['std'])+np.asarray(stats['action']['mean'])),axis=1)>.05
        group=np.stack((dig,carry,ret,recovery),1)
        scoop=np.asarray(table['observation.scoop_index'].to_pylist()).reshape(-1)[frame]
        future,valid=future_delta(data['features'],segment_ids=scoop)
        item={k:torch.as_tensor(data[k],device='cuda') for k in ('features','base','target','padding')}
        item.update(group=torch.as_tensor(group,device='cuda'),moving=torch.as_tensor(moving,device='cuda'),
                    future=torch.as_tensor(future,device='cuda'),future_valid=torch.as_tensor(valid,device='cuda'),episode=ep,
                    anchors=[np.flatnonzero(dig&moving),np.flatnonzero(carry&moving),np.flatnonzero(ret),np.flatnonzero(recovery),np.arange(len(frame))])
        episodes.append(item)
    return episodes


def evaluate(memory,episodes,std):
    import torch,numpy as np
    sums={key:np.zeros(4) for key in ['all','dig','carry','return','recovery']};memory.eval()
    with torch.no_grad():
        for ep in episodes:
            delta,_=memory(ep['features'][None]);pred=ep['base']+delta[0].reshape_as(ep['base'])
            error=abs(pred-ep['target'])*std
            for i,key in enumerate(sums):
                selected=torch.ones(len(pred),device='cuda',dtype=torch.bool) if key=='all' else ep['group'][:,i-1]
                valid=(~ep['padding'])&selected[:,None]
                sums[key]+=np.array([error[valid].sum().item(),valid.sum().item()*4,error[selected,0].sum().item(),selected.sum().item()*4])
    memory.train()
    return {key:dict(mae=float(s[0]/max(1,s[1])),first_action_mae=float(s[2]/max(1,s[3])),observations=int(s[3]/4)) for key,s in sums.items()}


def score(metrics):return .4*metrics['dig']['first_action_mae']+.4*metrics['carry']['first_action_mae']+.2*metrics['return']['first_action_mae']


def train(args):
    import torch,numpy as np
    from safetensors.torch import load_file
    torch.set_num_threads(4);torch.manual_seed(20261119);rng=np.random.default_rng(20261119)
    output=args.output/args.variant;output.mkdir(parents=True,exist_ok=False)
    progress=args.variant=='progress';policy=construct(progress);memory=policy.model.memory_adapter
    split=common.metadata()['split_manifest'];stats=json.loads((SOURCE.parents[2]/'normalization_stats.json').read_text())
    training=load_episodes(split['train'],stats);validation=load_episodes(split['validation'],stats)
    std=torch.tensor(stats['action']['std'],device='cuda',dtype=torch.float32);mean=torch.tensor(stats['action']['mean'],device='cuda',dtype=torch.float32)
    baseline=evaluate(memory,validation,std);common.write(output/'validation_initial.json',baseline)
    probability=np.array([len(ep['features']) for ep in training],dtype=np.float64);probability/=probability.sum()
    coverage={name:sum(int(ep['group'][:,i].sum()) for ep in training) for i,name in enumerate(['dig','carry','return','recovery'])}
    common.write(output/'training_coverage.json',coverage)
    optimizer=torch.optim.AdamW(memory.parameters(),lr=7e-5,weight_decay=1e-4)
    steps=args.balanced_steps if not progress else args.progress_steps;unroll=48;batch=8;started=time.time()
    for step in range(1,steps+1):
        chosen=rng.choice(len(training),batch,p=probability);starts=[]
        for e in chosen:
            category=int(rng.choice(5,p=[.30,.25,.15,.20,.10]));anchors=training[e]['anchors'][category]
            if len(anchors):start=int(np.clip(int(rng.choice(anchors))-8,0,len(training[e]['features'])-unroll))
            else:start=int(rng.integers(0,len(training[e]['features'])-unroll+1))
            starts.append(start)
        prefix=torch.zeros(batch,max(1,max(starts)),1604,device='cuda');fields={k:[] for k in ['features','base','target','padding','group','moving','future','future_valid']}
        for i,(e,start) in enumerate(zip(chosen,starts,strict=True)):
            prefix[i,:start]=training[e]['features'][:start]
            for k in fields:fields[k].append(training[e][k][start:start+unroll])
        part={k:torch.stack(v) for k,v in fields.items()};feature=part['features']
        hidden=memory.prefix_hidden(prefix,torch.tensor(starts));encoded,_=memory.encode(feature,hidden)
        delta=memory.output(encoded);prediction=part['base']+delta.reshape_as(part['base']);target=part['target'];group=part['group']
        state_weight=1.+2*group[:,:,0]+2*(group[:,:,1]&part['moving'])+group[:,:,2]+group[:,:,3]
        horizon=torch.tensor([2.,2.,2.,1.,1.,1.,1.,1.,1.,1.],device='cuda')
        joint=torch.tensor([1.5,1.5,1.5,2.],device='cuda')
        weight=(~part['padding']).float()[...,None]*state_weight[...,None,None]*horizon[None,None,:,None]*joint
        l1=(abs(prediction-target)*weight).sum()/weight.sum()
        truth=target*std+mean;physical=prediction*std+mean;active=abs(truth)>.15
        direction=torch.relu(.3*abs(truth)-truth.sign()*physical)/std
        direction=(direction*weight*active).sum()/(weight*active).sum().clamp_min(1)
        displacement=(prediction-target).cumsum(-2)/torch.arange(1,11,device='cuda').sqrt()[None,None,:,None]
        displacement=(abs(displacement)*weight).sum()/weight.sum()
        loss=l1+.15*direction+.15*displacement
        if progress:
            future=memory.future_state_head(encoded)
            element=torch.nn.functional.smooth_l1_loss(future,part['future'],reduction='none')
            future_loss=element[part['future_valid']].mean();loss=loss+.05*future_loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(memory.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        for setting in optimizer.param_groups:setting['lr']=7e-5*(.1+.9*.5*(1+np.cos(np.pi*step/steps)))
        if step==1 or step%100==0:
            status=dict(status='training',variant=args.variant,step=step,steps=steps,loss=loss.item(),l1=l1.item(),direction=direction.item(),displacement=displacement.item(),gradient_norm=norm.item(),elapsed_seconds=time.time()-started)
            common.write(output/'status.json',status);print(json.dumps(status),flush=True)
        if step%2000==0 or step==steps:
            metrics=evaluate(memory,validation,std);common.write(output/f'validation_{step:06d}.json',metrics);print(json.dumps(dict(step=step,validation=metrics)),flush=True)
    original=load_file(str(SOURCE/'model.safetensors'))
    for name,tensor in original.items():
        if not name.startswith('model.memory_adapter.') and not torch.equal(policy.state_dict()[name].cpu(),tensor):raise RuntimeError('Frozen core changed: '+name)
    dest=output/f'train/checkpoints/{steps:06d}/pretrained_model';dest.mkdir(parents=True);policy.eval();policy.save_pretrained(dest)
    for p in SOURCE.glob('policy_*'):shutil.copy2(p,dest/p.name)
    shutil.copy2(SOURCE.parents[2]/'normalization_stats.json',output/'train/normalization_stats.json')
    provenance=common.metadata();provenance['dig_carry_progress']=dict(source=str(SOURCE),source_sha256=common.digest(SOURCE/'model.safetensors'),variant=args.variant,steps=steps,trainable='model.memory_adapter.*',
        sampling_probabilities=dict(dig=.30,carry_active=.25,return_motion=.15,recovery=.20,uniform=.10),unroll=48,
        loss='weighted normalized action L1 + 0.15 active direction margin + 0.15 integrated action displacement; progress adds 0.05 future state smooth L1',
        future_target='delta over four decision observations: joint positions, dig/unload relative geometry, bucket fill; no cross-parent or scoop/goal-boundary targets',
        task_or_recovery_labels_as_inputs=False)
    common.write(dest/'benchmark_provenance.json',provenance)
    torch.save(dict(optimizer=optimizer.state_dict(),step=steps,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),dest.parent/'training_state.pt')
    from excavator_vlm.policies.act.inference import load_act
    restored,_,_=load_act(dest,'cuda:0',original_act=True)
    with torch.no_grad():
        feature=validation[0]['features'][:64][None];torch.testing.assert_close(memory(feature)[0],restored.model.memory_adapter(feature)[0],rtol=0,atol=0)
    common.assert_external()
    common.write(output/'status.json',dict(status='completed',variant=args.variant,step=steps,checkpoint=str(dest),validation=metrics,validation_score=score(metrics),frozen_core_unchanged=True,strict_reload='passed',external_files_unchanged=True))


def run(args):
    preflight=RUNTIME.parent/'preflight.json'
    if not preflight.exists() or common.read(preflight).get('status')!='passed':raise RuntimeError('Real-data preflight required')
    args.output.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),args.output/'run.py');shutil.copy2(preflight,args.output/'preflight.json')
    common.write(args.output/'manifest.json',dict(source=str(SOURCE),source_sha256=common.digest(SOURCE/'model.safetensors'),cache=str(common.CACHE),dataset=str(common.DATA),
        split=common.metadata()['split_manifest'],balanced_steps=args.balanced_steps,progress_steps=args.progress_steps,
        regression_scenes=[11,50,112],regression_scope='Development regression after diagnosis, not independent new held-out evidence'))
    common.write(args.output/'external_files.json',common.assert_external())
    env=dict(os.environ,ACT_PROJECT_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    for variant in ['balanced','progress']:
        common.write(args.output/'status.json',dict(status='training',active_variant=variant))
        with (args.output/f'{variant}.log').open('w') as log:
            subprocess.run([str(RUNTIME),'-u','-B',str(args.output/'run.py'),'--mode','train','--variant',variant,'--output',str(args.output),
                '--balanced-steps',str(args.balanced_steps),'--progress-steps',str(args.progress_steps)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    candidates=[]
    for variant in ['balanced','progress']:
        state=common.read(args.output/variant/'status.json');initial=common.read(args.output/variant/'validation_initial.json');value=state['validation']
        gate=value['all']['first_action_mae']<=1.2*initial['all']['first_action_mae'] and value['return']['first_action_mae']<=1.25*initial['return']['first_action_mae'] and score(value)<score(initial)
        common.write(args.output/variant/'validation_gate.json',dict(passed=gate,score=score(value),source_score=score(initial)))
        if gate:candidates.append((score(value),variant,state['checkpoint']))
    if not candidates:
        common.write(args.output/'status.json',dict(status='validation_gate_rejected',evaluations_started=False));return
    _,winner,checkpoint=min(candidates);common.write(args.output/'selection.json',dict(variant=winner,checkpoint=checkpoint,selection='Fixed dig/carry/return validation score, with overall and return regression gates'))
    template=ROOT/'outputs/experiments/act_velocity_holdout8_20261119_01/run.py';text=template.read_text()
    text=text.replace(str(Path('act_training_alternatives_20261119_01')),str(args.output.relative_to(ROOT/'outputs/experiments'))+'/'+winner)
    text=text.replace("CHECKPOINT=SOURCE/'velocity/train/checkpoints/008000/pretrained_model'",'CHECKPOINT=Path('+repr(checkpoint)+')')
    # Fixed development regression scenes include two failure modes and an existing success.
    text=text.replace('ids=sorted(random.Random(20261119).sample(eligible,8))','ids=[11,50,112]')
    text=text.replace('episodes=8,','episodes=3,').replace('requested_episodes=8','requested_episodes=3').replace("requested_episodes=8)","requested_episodes=3)")
    text=text.replace("'/8 completed'","'/3 completed'").replace("result['completed_episodes']==8","result['completed_episodes']==3")
    text=text.replace("'uniform random sample without replacement from 38 eligible held-out parents; sorted execution order'","'fixed development regression scenes 11,50,112; not independent evaluation'")
    text=text.replace('Further fixed-weight validation on eight additional held-out parents, not a forty-scene population estimate; no checkpoint selection or training during this run','Development regression of diagnosed empty-load/carry failures and a prior full success; these scenes informed this optimization and do not estimate unseen success rate')
    # The evaluator's protection list reads SOURCE/external_files.json.
    common.write(args.output/winner/'external_files.json',common.assert_external())
    evaluator=args.output/'evaluate_regression.py';evaluator.write_text(text);compile(text,str(evaluator),'exec')
    common.write(args.output/'status.json',dict(status='evaluating_regression',selected_variant=winner,scenes=[11,50,112]))
    with (args.output/'regression.log').open('w') as log:
        subprocess.run([str(RUNTIME),'-u','-B',str(evaluator),'--output',str(args.output/'regression')],env={**os.environ,'ACT_PROJECT_ROOT':str(ROOT),'CUDA_VISIBLE_DEVICES':'0,1'},cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    common.write(args.output/'status.json',dict(status='completed',selected_variant=winner,evaluation=str(args.output/'regression')))


if __name__=='__main__':
    # Keep file operations local; no evaluator/controller source mutations.
    common.read=lambda path:json.loads(Path(path).read_text())
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=lambda s:Path(s).resolve(),required=True)
    parser.add_argument('--mode',choices=['run','train'],default='run');parser.add_argument('--variant',choices=['balanced','progress'],default='balanced')
    parser.add_argument('--balanced-steps',type=int,default=8000);parser.add_argument('--progress-steps',type=int,default=12000)
    args=parser.parse_args()
    try: {'run':run,'train':train}[args.mode](args)
    except Exception as error:
        if args.output.exists():common.write(args.output/f'error_{args.mode}_{args.variant}.json',dict(error=repr(error)))
        raise
