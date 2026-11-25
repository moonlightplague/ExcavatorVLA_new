"""Conservative persistent ACT fine-tuning across physical-return/cycle junctions."""
import argparse,copy,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'scripts'))
import run_act_training_alternatives as common
SOURCE=ROOT/'outputs/experiments/act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model'
common.SOURCE=SOURCE
RUNTIME=ROOT/'outputs/experiments/act_training_alternatives_runtime_20261119_01/python'
DEPENDENCY=ROOT/'outputs/experiments/act_visual_direction_validation_20261120_02'
CHECKS=ROOT/'outputs/experiments/act_return_bridge_checks_20261121_01/preflight.json'


def read(path):return json.loads(Path(path).read_text())


def annotations(ids):
    import numpy as np,pyarrow.parquet as pq
    result={};needed=set(ids)
    for path in sorted((common.DATA/'data').rglob('*.parquet')):
        table=pq.read_table(path,columns=['episode_index','observation.return_to_pile','observation.stage_current_id'])
        episode=int(table['episode_index'][0].as_py())
        if episode not in needed:continue
        if set(table['episode_index'].to_pylist())!={episode}:raise ValueError('Unexpected shared episode parquet')
        cached=common.cpu_data(episode);frames=cached['frame_index'];assert np.array_equal(frames,np.arange(0,len(table),3))
        ret=np.asarray(table['observation.return_to_pile'].to_pylist()).reshape(-1)[frames]>0
        stage=np.asarray(table['observation.stage_current_id'].to_pylist()).reshape(-1)[frames]
        ends=np.flatnonzero(ret[:-1]&~ret[1:])+1
        bridge=np.zeros(len(ret),dtype=bool)
        for end in ends:bridge[max(0,end-12):min(len(ret),end+40)]=True
        focus=ret.copy();changes=np.flatnonzero(np.r_[False,(ret[1:]!=ret[:-1])|(stage[1:]!=stage[:-1])])
        for change in changes:focus[max(0,change-6):min(len(ret),change+7)]=True
        result[episode]=dict(ret=ret,stage=stage,ends=ends,bridge=bridge,focus=np.flatnonzero(focus))
    if set(result)!=needed:raise ValueError('Missing parent annotations')
    return result


def choose_start(note,length,unroll,rng,forced=None):
    import numpy as np
    mode=forced or rng.choice(['bridge','return_transition','uniform'],p=[.5,.25,.25])
    if mode=='bridge':
        if not len(note['ends']):raise ValueError('No true return-to-next-cycle junction')
        boundary=int(rng.choice(note['ends']));start=int(np.clip(boundary-32,0,length-unroll))
        if not start<boundary<start+unroll:raise ValueError('Junction not spanned')
    elif mode=='return_transition':start=int(np.clip(int(rng.choice(note['focus']))-16,0,length-unroll))
    else:start=int(rng.integers(0,length-unroll+1))
    return start,mode


def evaluate(memory,episodes,std,mean):
    import torch,numpy as np
    memory.eval();sums={k:np.zeros(6,dtype=np.float64) for k in ['all','return','bridge','dig','carry']}
    with torch.no_grad():
        for ep in episodes:
            prediction=ep['base']+memory(ep['features'][None])[0][0].reshape_as(ep['base'])
            err=abs(prediction-ep['target'])*std;true=ep['target']*std+mean;physical=prediction*std+mean
            groups=dict(all=torch.ones_like(ep['ret']),return_=ep['ret'],bridge=ep['bridge'],dig=ep['dig'],carry=ep['carry'])
            groups['return']=groups.pop('return_')
            for name,select in groups.items():
                mask=(~ep['padding'])&select[:,None];active=select&(~ep['padding'][:,0])&(true[:,0,0].abs()>.2)
                sums[name]+=np.array([err[mask].sum().item(),mask.sum().item()*4,err[select,0].sum().item(),select.sum().item()*4,
                    (physical[active,0,0]*true[active,0,0]<=0).sum().item(),active.sum().item()])
    memory.train()
    return {name:dict(mae=float(s[0]/max(1,s[1])),first_action_mae=float(s[2]/max(1,s[3])),wrong_active_swing_directions=int(s[4]),active_swing_observations=int(s[5])) for name,s in sums.items()}


def loss_terms(prediction,target,reference,pad,ret,std,mean):
    import torch
    horizon=prediction.new_tensor([2.,2.,2.,1.,1.,1.,1.,1.,1.,1.]);joint=prediction.new_tensor([2.,1.,1.,1.])
    weights=(~pad).float()[...,None]*(1+2*ret.float()[...,None,None])*horizon[None,None,:,None]*joint
    l1=((prediction-target).abs()*weights).sum()/weights.sum().clamp_min(1)
    truth=target*std+mean;physical=prediction*std+mean;active=truth.abs()>.2
    margin=torch.relu(.35*truth.abs()-truth.sign()*physical)/std
    direction=(margin*weights*active).sum()/(weights*active).sum().clamp_min(1)
    anchor=((prediction-reference).square()*weights).sum()/weights.sum().clamp_min(1)
    return l1+.1*direction+.1*anchor,(l1,direction,anchor)


def train(args):
    import numpy as np,torch
    from safetensors.torch import load_file
    from excavator_vlm.policies.act.inference import load_act
    torch.set_num_threads(4);torch.manual_seed(20261121);rng=np.random.default_rng(20261121)
    device=args.device;policy,_,_=load_act(SOURCE,device,original_act=True);teacher,_,_=load_act(SOURCE,device,original_act=True)
    for n,p in policy.named_parameters():p.requires_grad_(n.startswith('model.memory_adapter.'))
    for p in teacher.parameters():p.requires_grad_(False)
    memory=policy.model.memory_adapter;teacher_memory=teacher.model.memory_adapter
    metadata=common.metadata();split=metadata['split_manifest'];train_ids=split['train'];val_ids=split['validation']
    if args.smoke:train_ids=train_ids[:2];val_ids=val_ids[:2]
    assert not set(train_ids)&set(split['validation']+split['test'])
    note=annotations(train_ids+val_ids)
    def load(ep):
        data=common.cpu_data(ep);n=note[ep];result={k:torch.as_tensor(data[k],device=device) for k in ['features','base','target','padding']}
        result.update({k:torch.as_tensor(value,device=device) for k,value in dict(ret=n['ret'],bridge=n['bridge'],dig=np.isin(n['stage'],[2,3,4,5])&~n['ret'],carry=(n['stage']>=6)&~n['ret']).items()})
        with torch.no_grad():result['reference']=result['base']+teacher_memory(result['features'][None])[0][0].reshape_as(result['base'])
        result['episode']=ep;return result
    args.output.mkdir(parents=True,exist_ok=False);out=args.output
    common.write(out/'status.json',dict(status='loading_readonly_features',device=device,training_episodes=train_ids,validation_episodes=val_ids))
    training=[load(ep) for ep in train_ids];validation=[load(ep) for ep in val_ids]
    stats=read(SOURCE.parents[2]/'normalization_stats.json');std=torch.tensor(stats['action']['std'],device=device);mean=torch.tensor(stats['action']['mean'],device=device)
    initial=evaluate(memory,validation,std,mean);common.write(out/'validation_initial.json',initial)
    probs=np.array([len(ep['features']) for ep in training],dtype=np.float64);probs/=probs.sum()
    unroll=128;batch_size=2 if args.smoke else 8;steps=2 if args.smoke else args.steps
    optimizer=torch.optim.AdamW(memory.parameters(),lr=1e-7,weight_decay=1e-4);counts=dict(bridge=0,return_transition=0,uniform=0);started=time.time()
    for step in range(1,steps+1):
        multiplier=min(1.,step/100.)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/max(1,steps))))
        for group in optimizer.param_groups:group['lr']=1e-5*multiplier
        chosen=rng.choice(len(training),batch_size,p=probs);starts=[]
        for e in chosen:
            ep=training[e];start,mode=choose_start(note[ep['episode']],len(ep['features']),unroll,rng,forced='bridge' if args.smoke else None)
            starts.append(start);counts[mode]+=1
        prefix=torch.zeros(batch_size,max(1,max(starts)),1604,device=device)
        fields={k:[] for k in ['features','base','target','padding','ret','reference']}
        for lane,(e,start) in enumerate(zip(chosen,starts,strict=True)):
            ep=training[e];prefix[lane,:start]=ep['features'][:start]
            for key in fields:fields[key].append(ep[key][start:start+unroll])
        fields={k:torch.stack(v) for k,v in fields.items()}
        hidden=memory.prefix_hidden(prefix,torch.tensor(starts))
        prediction=fields['base']+memory(fields['features'],hidden)[0].reshape_as(fields['base'])
        loss,parts=loss_terms(prediction,fields['target'],fields['reference'],fields['padding'],fields['ret'],std,mean)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite bridge training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(memory.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%50==0 or step==steps:
            state=dict(status='training',step=step,steps=steps,unroll=unroll,loss=loss.item(),expert_l1=parts[0].item(),direction=parts[1].item(),source_anchor=parts[2].item(),gradient_norm=norm.item(),sampling_counts=counts,elapsed_seconds=time.time()-started)
            common.write(out/'status.json',state);print(json.dumps(state),flush=True)
        if step%500==0 or step==steps:
            metrics=evaluate(memory,validation,std,mean);common.write(out/f'validation_{step:06d}.json',metrics);print(json.dumps(dict(step=step,validation=metrics)),flush=True)
    original=load_file(str(SOURCE/'model.safetensors'));state=policy.state_dict();changed=[]
    for name,value in original.items():
        if not torch.equal(value,state[name].cpu()):
            if not name.startswith('model.memory_adapter.'):raise ValueError('Frozen core changed: '+name)
            changed.append(name)
    if not changed:raise ValueError('Memory weights unchanged')
    destination=out/f'train/checkpoints/{steps:06d}/pretrained_model';destination.mkdir(parents=True);policy.eval();policy.save_pretrained(destination)
    for path in SOURCE.glob('policy_*'):shutil.copy2(path,destination/path.name)
    shutil.copy2(SOURCE.parents[2]/'normalization_stats.json',out/'train/normalization_stats.json')
    metadata['return_bridge_finetuning']=dict(source=str(SOURCE),source_sha256=common.digest(SOURCE/'model.safetensors'),steps=steps,unroll=128,
        sampling=dict(true_return_end_centered=.5,original_return_transition=.25,uniform=.25),
        bridge='128-decision sequence starting 32 decisions before true return-end label; full current-weight episode prefix, then truncated BPTT',
        weights=dict(return_factor=3,executed_first_three_factor=2,swing_factor=2,direction=.1,source_anchor=.1),
        learning_rate=dict(maximum=1e-5,minimum=1e-6,warmup_steps=100),trainable='model.memory_adapter.*',frozen_core_unchanged=True,
        task_metadata_inputs=False,test_rollouts_used_for_fitting=False)
    common.write(destination/'benchmark_provenance.json',metadata)
    torch.save(dict(optimizer=optimizer.state_dict(),step=steps,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()),destination.parent/'training_state.pt')
    restored,_,_=load_act(destination,device,original_act=True)
    with torch.no_grad():
        sequence=validation[0]['features'][None]
        torch.testing.assert_close(memory(sequence)[0],restored.model.memory_adapter(sequence)[0],rtol=0,atol=0)
    # Public native RGB inference must remain loadable and preserve reset/memory behavior.
    import importlib.util
    path=ROOT/'ExcavatorVLA-vlm/excavator_vlm/policies/act/visual_training.py';spec=importlib.util.spec_from_file_location('bridge_raw_reader',path);visual=importlib.util.module_from_spec(spec);spec.loader.exec_module(visual)
    real=visual.RGBEpisodes(common.DATA).load(val_ids[0],3)
    from excavator_vlm.policies.act.inference import load_act as reload_act
    public,pre,_=reload_act(destination,device,original_act=True);policy.reset();public.reset()
    with torch.no_grad():
        for frame in range(3):
            raw,_,_,_=visual.batch_at([real],frame);batch=pre(raw)
            torch.testing.assert_close(policy.predict_action_chunk(dict(batch)),public.predict_action_chunk(dict(batch)),rtol=0,atol=0)
    common.assert_external()
    gate=(metrics['all']['first_action_mae']<=1.02*initial['all']['first_action_mae'] and metrics['return']['first_action_mae']<=1.02*initial['return']['first_action_mae']
        and metrics['return']['wrong_active_swing_directions']<=initial['return']['wrong_active_swing_directions']+1 and metrics['bridge']['first_action_mae']<=initial['bridge']['first_action_mae'])
    common.write(out/'validation_gate.json',dict(passed=gate,source=initial,candidate=metrics,scope='Expert-trajectory filter only, not proof of closed-loop improvement'))
    common.write(out/'status.json',dict(status='completed',checkpoint=str(destination),steps=steps,validation=metrics,validation_gate_passed=gate,
        strict_reload='passed',native_rgb_reload='passed',frozen_core_unchanged=True,external_files_unchanged=True,changed_parameters=changed))


def prepare_checks(output):
    import numpy as np
    from safetensors.torch import load_file
    original=read(common.PREVIOUS/'experiment.json');metadata=common.metadata();split=metadata['split_manifest']
    if original['fingerprint']!=metadata['dataset_fingerprint'] or original['split_manifest']!=split:raise ValueError('Feature cache provenance mismatch')
    if common.digest(SOURCE.parents[2]/'normalization_stats.json')!=metadata['normalization_stats_sha256']:raise ValueError('Normalization statistics changed')
    cache_source=load_file(str(Path(original['source'])/'model.safetensors'));source=load_file(str(SOURCE/'model.safetensors'))
    for name,value in cache_source.items():
        if not name.startswith('model.memory_adapter.'):
            import torch
            if not torch.equal(value,source[name]):raise ValueError('Cached core differs from source: '+name)
    del source,cache_source
    expected=read(ROOT/'outputs/experiments/act_training_alternatives_20261119_01/cache_hashes.json')
    for ep in split['train']+split['validation']:
        if common.digest(common.CACHE/f'episode_{ep:04d}.npz')!=expected[str(ep)]:raise ValueError('Frozen feature cache changed')
    note=annotations(split['train']+split['validation']);rng=np.random.default_rng(20261121)
    for ep in split['train']:
        length=len(note[ep]['ret'])
        for _ in range(20):
            start,mode=choose_start(note[ep],length,128,rng,forced='bridge')
            assert any(start<b<start+128 for b in note[ep]['ends'])
    audit={part:dict(episodes=len(split[part]),return_to_next_cycle_junctions=sum(len(note[ep]['ends']) for ep in split[part])) for part in ['train','validation']}
    common.write(output/'dataset_audit.json',dict(split_audit=audit,test_parents_loaded=False,source_core_matches_feature_cache=True,feature_hashes_verified=160,
        task_labels='Training sampling and validation grouping only; never model input'))
    return audit


def run(args):
    proof=read(CHECKS)
    if proof.get('status')!='passed' or proof['trainer_sha256']!=common.digest(Path(__file__)):raise RuntimeError('Current trainer CPU preflight required')
    args.output.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),args.output/'run.py');shutil.copy2(CHECKS,args.output/'preflight.json')
    common.write(args.output/'external_files.json',common.assert_external())
    common.write(args.output/'manifest.json',dict(source=str(SOURCE),source_sha256=common.digest(SOURCE/'model.safetensors'),wait_for=str(DEPENDENCY),steps=args.steps,
        source_preserved=True,visual_core_frozen=True,trainable='memory_adapter',unroll=128,default_next_training_budget=1500,checkpoint_selection='Final only; original validation gates before development rollout'))
    common.write(args.output/'status.json',dict(status='queued_after_three_arm_evaluation',dependency=str(DEPENDENCY),new_training_started=False))
    while not (DEPENDENCY/'completion.json').exists():time.sleep(15)
    if read(DEPENDENCY/'completion.json')['status']!='completed':
        common.write(args.output/'status.json',dict(status='dependency_infrastructure_error',new_training_started=False));return
    prepare_checks(args.output)
    common.write(args.output/'status.json',dict(status='training',new_training_started=True))
    env={**os.environ,'ACT_PROJECT_ROOT':str(ROOT),'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    with (args.output/'training.log').open('w') as log:
        subprocess.run([str(RUNTIME),'-u','-B',str(args.output/'run.py'),'--mode','train','--output',str(args.output/'pilot'),'--steps',str(args.steps),'--device','cuda:0'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    state=read(args.output/'pilot/status.json')
    if not state['validation_gate_passed']:
        common.write(args.output/'status.json',dict(status='validation_gate_rejected',baseline_replaced=False));return
    # Reuse the proven, versioned HDR evaluator; standard native model interface.
    import train_act_visual_robustness as visual_run
    visual_run.evaluate_candidate(args.output,state['checkpoint'],'return_bridge')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=lambda p:Path(p).resolve(),required=True);p.add_argument('--mode',choices=['run','train'],default='run');p.add_argument('--steps',type=int,default=1500);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    try:{'run':run,'train':train}[args.mode](args)
    except Exception as e:
        if args.output.exists():common.write(args.output/f'error_{args.mode}.json',dict(error=repr(e)))
        raise
