"""Conservative head-only refinement from the established return-trained ACT.

Freeze visual, temporal and recurrent encoders and swing outputs. Train only
boom/arm/bucket outputs in the first three executed actions; no action rules.
"""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'scripts'))
import train_act_velocity_progress as groups
common=groups.common
SOURCE=groups.SOURCE
RUNTIME=ROOT/'outputs/experiments/act_training_alternatives_runtime_20261119_01/python'
SOURCE_EVAL=ROOT/'outputs/experiments/act_velocity_holdout8_20261119_01'


def read(path):return json.loads(Path(path).read_text())


def encode(policy,episodes):
    import torch
    memory=policy.model.memory_adapter;memory.eval()
    with torch.no_grad():
        for episode in episodes:
            hidden,_=memory.gru(memory.project(episode['features'][None]))
            episode['encoded']=hidden[0].detach()
            episode['teacher_delta']=memory.output(hidden)[0].detach()
    return episodes


def metric(head,episodes,std):
    import torch,numpy as np
    values={name:np.zeros(4) for name in ['all','dig','carry','return','recovery']};differences=[];swing=0.
    with torch.no_grad():
        for episode in episodes:
            delta=head(episode['encoded']).reshape_as(episode['base']);pred=episode['base']+delta
            teacher=episode['base']+episode['teacher_delta'].reshape_as(episode['base'])
            errors=abs(pred-episode['target'])*std
            differences.append(((pred[:,:3]-teacher[:,:3])*std).abs().flatten().cpu().numpy())
            swing=max(swing,(pred[:,:,0]-teacher[:,:,0]).abs().max().item())
            for i,name in enumerate(values):
                choose=torch.ones(len(pred),device='cuda',dtype=torch.bool) if name=='all' else episode['group'][:,i-1]
                valid=(~episode['padding'])&choose[:,None]
                values[name]+=np.array([errors[valid].sum().item(),valid.sum().item()*4,errors[choose,0].sum().item(),choose.sum().item()*4])
    metrics={name:dict(mae=float(v[0]/max(1,v[1])),first_action_mae=float(v[2]/max(1,v[3]))) for name,v in values.items()}
    differences=np.concatenate(differences)
    metrics['source_deviation']=dict(executed_action_mae=float(differences.mean()),p99=float(np.quantile(differences,.99)),max=float(differences.max()),swing_max_normalized_difference=swing)
    return metrics


def train(args):
    import torch,numpy as np
    torch.set_num_threads(4);torch.manual_seed(20261120);rng=np.random.default_rng(20261120)
    policy=common.construct('velocity');policy.eval()
    for parameter in policy.parameters():parameter.requires_grad_(False)
    head=policy.model.memory_adapter.output
    for parameter in head.parameters():parameter.requires_grad_(True)
    split=common.metadata()['split_manifest'];stats=read(SOURCE.parents[2]/'normalization_stats.json')
    training=encode(policy,groups.load_episodes(split['train'],stats));validation=encode(policy,groups.load_episodes(split['validation'],stats))
    original={name:value.detach().clone() for name,value in head.state_dict().items()}
    std=torch.tensor(stats['action']['std'],device='cuda',dtype=torch.float32)
    initial=metric(head,validation,std);common.write(args.output/'validation_initial.json',initial)
    flat={key:torch.cat([episode[key] for episode in training]) for key in ['encoded','base','target','padding','teacher_delta','group']}
    masks=[(flat['group'][:,0]).nonzero().flatten(),(flat['group'][:,1]).nonzero().flatten(),(flat['group'][:,3]).nonzero().flatten(),torch.arange(len(flat['encoded']),device='cuda')]
    row_mask=torch.zeros(40,device='cuda');row_mask[torch.tensor([1,2,3,5,6,7,9,10,11],device='cuda')]=1
    optimizer=torch.optim.Adam(head.parameters(),lr=1e-5)
    started=time.time()
    for step in range(1,args.steps+1):
        pieces=[]
        for pool,size in zip(masks,[204,128,77,103]):
            if len(pool)==0:pool=masks[-1]
            pieces.append(pool[torch.randint(len(pool),(size,),device='cuda')])
        index=torch.cat(pieces);encoded=flat['encoded'][index]
        delta=head(encoded).reshape(-1,10,4);base=flat['base'][index];target=flat['target'][index]
        teacher=flat['teacher_delta'][index].reshape_as(delta)
        valid=(~flat['padding'][index,:3]).float()[...,None]
        joint=torch.tensor([0.,1.,1.,2.],device='cuda')
        weight=valid*joint
        fit=torch.nn.functional.smooth_l1_loss(base[:,:3]+delta[:,:3],target[:,:3],beta=.1,reduction='none')
        bc=(fit*weight).sum()/weight.sum()
        return_weight=1.+3.*flat['group'][index,2].float()
        anchor=(((delta[:,:3]-teacher[:,:3]).square()*joint).mean((1,2))*return_weight).mean()
        loss=bc+10.*anchor
        optimizer.zero_grad(set_to_none=True);loss.backward()
        head.weight.grad.mul_(row_mask[:,None]);head.bias.grad.mul_(row_mask)
        torch.nn.utils.clip_grad_norm_(head.parameters(),.25,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():
            head.weight[~row_mask.bool()]=original['weight'][~row_mask.bool()]
            head.bias[~row_mask.bool()]=original['bias'][~row_mask.bool()]
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite head optimization')
        if step==1 or step%100==0:
            status=dict(status='training',step=step,steps=args.steps,loss=loss.item(),bc=bc.item(),anchor=anchor.item(),elapsed_seconds=time.time()-started)
            common.write(args.output/'status.json',status);print(json.dumps(status),flush=True)
    # Limit the final change in parameter space, entirely inside the saved model.
    with torch.no_grad():
        for name,parameter in head.named_parameters():parameter.copy_(original[name]+.25*(parameter-original[name]))
    assert torch.equal(head.weight[~row_mask.bool()],original['weight'][~row_mask.bool()])
    assert torch.equal(head.bias[~row_mask.bool()],original['bias'][~row_mask.bool()])
    final=metric(head,validation,std);common.write(args.output/'validation_final.json',final)
    assert final['source_deviation']['swing_max_normalized_difference']==0
    from safetensors.torch import load_file
    source_values=load_file(str(SOURCE/'model.safetensors'))
    for name,value in source_values.items():
        if not name.startswith('model.memory_adapter.output.') and not torch.equal(value,policy.state_dict()[name].cpu()):raise ValueError('Frozen model changed: '+name)
    destination=args.output/f'train/checkpoints/{args.steps:06d}/pretrained_model';destination.mkdir(parents=True);policy.save_pretrained(destination)
    for file in SOURCE.glob('policy_*'):shutil.copy2(file,destination/file.name)
    shutil.copy2(SOURCE.parents[2]/'normalization_stats.json',args.output/'train/normalization_stats.json')
    provenance=common.metadata();provenance['conservative_head']=dict(source=str(SOURCE),source_sha256=common.digest(SOURCE/'model.safetensors'),optimizer_steps=args.steps,learning_rate=1e-5,
        trainable_rows=[1,2,3,5,6,7,9,10,11],effective_trainable_parameters=2313,source_anchor_weight=10.,parameter_interpolation=.25,
        frozen_encoder_and_swing_weights=True,stage_labels_as_model_inputs=False)
    common.write(destination/'benchmark_provenance.json',provenance)
    from excavator_vlm.policies.act.inference import load_act
    restored,_,_=load_act(destination,'cuda:0',original_act=True)
    with torch.no_grad():
        sample=validation[0]['features'][:32][None]
        torch.testing.assert_close(policy.model.memory_adapter(sample)[0],restored.model.memory_adapter(sample)[0],rtol=0,atol=0)
    common.assert_external()
    gate=final['all']['first_action_mae']<=1.05*initial['all']['first_action_mae'] and final['return']['first_action_mae']<=1.05*initial['return']['first_action_mae'] and final['source_deviation']['p99']<=.05
    common.write(args.output/'training_result.json',dict(checkpoint=str(destination),validation=final,validation_gate=gate,strict_reload='passed',frozen_model_unchanged=True))


def evaluation_script(output,checkpoint,scenes,steps):
    text=(SOURCE_EVAL/'run.py').read_text()
    text=text.replace("SOURCE=ROOT/'outputs/experiments/act_training_alternatives_20261119_01'",'SOURCE=Path('+repr(str(output))+')')
    text=text.replace("CHECKPOINT=SOURCE/'velocity/train/checkpoints/008000/pretrained_model'",'CHECKPOINT=Path('+repr(checkpoint)+')')
    text=text.replace('ids=sorted(random.Random(20261119).sample(eligible,8))','ids='+repr(scenes))
    text=text.replace('episodes=8,','episodes='+str(len(scenes))+',').replace('requested_episodes=8','requested_episodes='+str(len(scenes)))
    text=text.replace("result['completed_episodes']==8","result['completed_episodes']=="+str(len(scenes)))
    text=text.replace("'/8 completed'",repr('/'+str(len(scenes))+' completed'))
    text=text.replace("'uniform random sample without replacement from 38 eligible held-out parents; sorted execution order'","'fixed development regression scenes; not independent testing'")
    text=text.replace('steps=5000,','steps='+str(steps)+',').replace("'--steps','5000'","'--steps',"+repr(str(steps)))
    text=text.replace('Further fixed-weight validation on eight additional held-out parents, not a forty-scene population estimate; no checkpoint selection or training during this run',
        'Development diagnostic; step budget '+str(steps)+'. A short prefix run is not a full ten-cycle success evaluation.')
    return text


def execute(out,checkpoint,scenes,budget,label):
    script=out/(label+'.py');script.write_text(evaluation_script(out,checkpoint,scenes,budget));compile(script.read_text(),str(script),'exec')
    with (out/(label+'.log')).open('w') as log:
        subprocess.run([str(RUNTIME),'-u','-B',str(script),'--output',str(out/label)],cwd=ROOT,env={**os.environ,'ACT_PROJECT_ROOT':str(ROOT),'CUDA_VISIBLE_DEVICES':'0,1'},stdout=log,stderr=subprocess.STDOUT,check=True)


def run(args):
    preflight=RUNTIME.parent/'conservative_head_preflight.json'
    if not preflight.exists() or read(preflight).get('status')!='passed':raise RuntimeError('Head-only preflight must pass')
    args.output.mkdir(parents=True,exist_ok=False);common.write(args.output/'external_files.json',common.assert_external())
    shutil.copy2(preflight,args.output/'preflight.json')
    shutil.copy2(Path(__file__),args.output/'run.py')
    common.write(args.output/'manifest.json',dict(source=str(SOURCE),source_sha256=common.digest(SOURCE/'model.safetensors'),steps=args.steps,
        strategy='head only, exact frozen swing weights, anchor to source outputs, quarter-weight interpolation',prefix_scenes=[50,112],prefix_budget=2000,
        full_scenes=[11,50,112],full_budget=5000,prefix_is_task_success_evaluation=False))
    train(args)
    import gc,torch
    gc.collect();torch.cuda.empty_cache()
    result=read(args.output/'training_result.json')
    if not result['validation_gate']:
        common.write(args.output/'status.json',dict(status='validation_gate_rejected',baseline_replaced=False));return
    checkpoint=result['checkpoint'];common.write(args.output/'status.json',dict(status='evaluating_short_prefix',scenes=[50,112]))
    execute(args.output,checkpoint,[50,112],2000,'prefix_checks')
    comparison=[];gate=True
    for ep in [50,112]:
        baseline=read(SOURCE_EVAL/f'scene_{ep:04d}/evaluation/episodes/episode_0001/episode_status.json')
        cycles=[c for c in baseline['repeat_cycles'] if isinstance(c.get('end_step'),int) and c['end_step']<2000 and c.get('returned')]
        required=len(cycles);success=sum(bool(c['success']) for c in cycles)
        current=read(args.output/f'prefix_checks/scene_{ep:04d}/evaluation/episodes/episode_0001/episode_status.json')
        passed=not current.get('error') and current.get('repeat_cycles_completed',0)>=max(0,required-1) and current.get('repeat_cycles_successful',0)>=max(0,success-1)
        comparison.append(dict(scene=ep,baseline_cycles_at_2000=required,baseline_successful_cycles_at_2000=success,candidate_cycles=current.get('repeat_cycles_completed'),candidate_successful_cycles=current.get('repeat_cycles_successful'),passed=passed))
        gate &= passed
    common.write(args.output/'prefix_gate.json',dict(passed=bool(gate),comparison=comparison,scope='Early regression screen only; task completion criterion unchanged'))
    if not gate:
        common.write(args.output/'status.json',dict(status='prefix_regression_rejected',baseline_replaced=False));return
    common.write(args.output/'status.json',dict(status='evaluating_full_regression',scenes=[11,50,112]))
    execute(args.output,checkpoint,[11,50,112],5000,'full_regression')
    common.write(args.output/'status.json',dict(status='completed',baseline_replaced=False,results=str(args.output/'full_regression')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=lambda p:Path(p).resolve(),required=True);parser.add_argument('--steps',type=int,default=1500)
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if args.output.exists():common.write(args.output/'error.json',dict(error=repr(error)))
        raise
