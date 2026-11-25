"""Paired raw-image visual/decoder fine-tuning with frozen ACT episode memory."""
import argparse,hashlib,importlib.util,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'scripts'))
import run_act_training_alternatives as common
SOURCE=ROOT/'outputs/experiments/act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model'
common.SOURCE=SOURCE
RUNTIME=ROOT/'outputs/experiments/act_training_alternatives_runtime_20261119_01/python'
HDR=ROOT/'outputs/experiments/act_hdr_baseline_repeatability_20261120_02'


def read(path):return json.loads(Path(path).read_text())


def helper(path=None):
    path=Path(path or os.environ.get('ACT_VISUAL_HELPER',ROOT/'ExcavatorVLA-vlm/excavator_vlm/policies/act/visual_training.py'))
    spec=importlib.util.spec_from_file_location('act_raw_visual_training',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def load_policy():
    from excavator_vlm.policies.act.inference import load_act
    return load_act(SOURCE,'cuda:0',original_act=True)


def evaluate(policy,pre,post,reader,ids,visual,lanes=2,max_decisions=0):
    import torch,numpy as np
    policy.eval();std=torch.tensor(read(SOURCE.parents[2]/'normalization_stats.json')['action']['std'],device='cuda',dtype=torch.float32)
    sums=torch.zeros(4,4,device='cuda');noise=torch.zeros(2,device='cuda')
    with torch.no_grad():
        for offset in range(0,len(ids),lanes):
            print(json.dumps(dict(status='validation',parent_episodes=ids[offset:offset+lanes],parent_groups_done=offset//lanes,parent_groups=math.ceil(len(ids)/lanes))),flush=True)
            episodes=reader.group(ids[offset:offset+lanes],max_decisions);clean=visual.VisualSequence(policy.model);altered=visual.VisualSequence(policy.model)
            rng=np.random.default_rng(20261120+int(episodes[0]['episode']));generator=torch.Generator().manual_seed(20261120+int(episodes[0]['episode']))
            brightness=torch.tensor(rng.uniform(.97,1.03,len(episodes)),dtype=torch.float32)[:,None,None,None]
            contrast=torch.tensor(rng.uniform(.97,1.03,len(episodes)),dtype=torch.float32)[:,None,None,None]
            for frame in range(max(len(ep['frames']) for ep in episodes)):
                raw,pad,groups,valid=visual.batch_at(episodes,frame)
                noisy={**raw,**visual.photometric({k:raw[k] for k in policy.config.image_features},brightness,contrast,.5,generator)}
                batch=pre(dict(raw));target=batch['action'];prediction=clean.step(batch)
                perturbed=altered.step(pre(noisy));mask=torch.as_tensor(~pad,device='cuda');group=torch.as_tensor(groups,device='cuda')
                error=abs(prediction-target)*std
                for i in range(4):
                    select=torch.as_tensor(valid,device='cuda') if i==0 else group[:,i-1]
                    selected=mask&select[:,None]
                    sums[i,0]+=error[selected].sum();sums[i,1]+=selected.sum()*4
                    sums[i,2]+=error[select,0].sum();sums[i,3]+=select.sum()*4
                difference=abs(prediction[:,:3]-perturbed[:,:3])*std
                noise[0]+=difference[mask[:,:3]].sum();noise[1]+=mask[:,:3].sum()*4
            del episodes
    values=sums.cpu().numpy();n=noise.cpu().numpy()
    return {**{name:dict(mae=float(s[0]/max(1,s[1])),first_action_mae=float(s[2]/max(1,s[3])),observations=int(s[3]/4)) for name,s in zip(['all','dig','carry','return'],values,strict=True)},
        'appearance_action_mae':float(n[0]/max(1,n[1])),
        'scope':'Sequential raw-image validation, separate clean/augmented episode memories; augmentation labels never enter model inputs'}


def train(args):
    import torch,numpy as np
    from concurrent.futures import ThreadPoolExecutor
    torch.set_num_threads(4);torch.manual_seed(20261120)
    visual=helper();policy,pre,post=load_policy();source,source_pre,_=load_policy()
    names=visual.enable_visual_gradients(policy)
    if not names:raise ValueError('No selected visual/decoder parameters')
    reader=visual.RGBEpisodes(common.DATA);split=common.metadata()['split_manifest']
    train_ids=split['train'];val_ids=split['validation']
    if args.smoke:train_ids=train_ids[:2];val_ids=val_ids[:2]
    max_decisions=6 if args.smoke else 0
    epochs=1 if args.smoke else args.epochs
    # No held-out test episodes are loaded for training or selection.
    if set(train_ids)&set(split['validation']+split['test']):raise ValueError('Training split overlap')
    output=args.output/args.variant;output.mkdir(parents=True,exist_ok=False)
    common.write(output/'status.json',dict(status='initial_validation',variant=args.variant,episodes=len(val_ids)))
    common.write(output/'trainable_parameters.json',dict(names=names,count=sum(p.numel() for p in policy.parameters() if p.requires_grad),
        frozen_memory=True,optimizer_step='Only after every complete simultaneous parent group; model weights fixed throughout each episode'))
    initial=evaluate(policy,pre,post,reader,val_ids,visual,args.lanes,max_decisions);common.write(output/'validation_initial.json',initial)
    parameters=list(policy.named_parameters())
    optimizer=torch.optim.AdamW([
        dict(params=[p for n,p in parameters if p.requires_grad and n.startswith('model.backbone.layer4.')],lr=2e-7),
        dict(params=[p for n,p in parameters if p.requires_grad and not n.startswith('model.backbone.layer4.')],lr=6e-7)],weight_decay=1e-4)
    updated=0;started=time.time();orders=[]
    for epoch in range(epochs):
        rng=np.random.default_rng(20261120+epoch);order=rng.permutation(train_ids).tolist();orders.append(order)
        batches=[order[i:i+args.lanes] for i in range(0,len(order),args.lanes)]
        with ThreadPoolExecutor(max_workers=1) as fetcher:
            pending=fetcher.submit(reader.group,batches[0],max_decisions)
            for group_index,ids in enumerate(batches):
                episodes=pending.result()
                if group_index+1<len(batches):pending=fetcher.submit(reader.group,batches[group_index+1],max_decisions)
                clean=visual.VisualSequence(policy.model,gradients=True)
                augmented=visual.VisualSequence(policy.model,gradients=True) if args.variant=='robust' else None
                teacher=visual.VisualSequence(source.model)
                brightness=torch.tensor(rng.uniform(.97,1.03,len(episodes)),dtype=torch.float32)[:,None,None,None]
                contrast=torch.tensor(rng.uniform(.97,1.03,len(episodes)),dtype=torch.float32)[:,None,None,None]
                generator=torch.Generator().manual_seed(20261120+epoch*1000+group_index)
                lengths=torch.tensor([len(ep['frames']) for ep in episodes],device='cuda',dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True);running=torch.zeros(3,device='cuda');count=0
                for frame in range(max(int(v) for v in lengths.tolist())):
                    raw,pad,groups,valid=visual.batch_at(episodes,frame)
                    noisy=None
                    if augmented is not None:noisy={**raw,**visual.photometric({k:raw[k] for k in policy.config.image_features},brightness,contrast,.5,generator)}
                    processed=pre(dict(raw));target=processed['action']
                    with torch.no_grad():reference=teacher.step(processed)
                    prediction=clean.step(processed)
                    mask=torch.as_tensor(~pad,device='cuda',dtype=torch.float32)[...,None]
                    denominator=(mask.sum((1,2))*4).clamp_min(1)
                    bc=(abs(prediction-target)*mask).sum((1,2))/denominator
                    anchor=((prediction-reference).square()*mask).sum((1,2))/denominator
                    consistency=torch.zeros_like(bc)
                    if augmented is not None:
                        altered=augmented.step(pre(noisy))
                        bc=.5*(bc+(abs(altered-target)*mask).sum((1,2))/denominator)
                        consistency=((prediction-altered).square()*mask).sum((1,2))/denominator
                    returning=torch.as_tensor(groups[:,2],device='cuda',dtype=torch.float32)
                    per_lane=bc+.1*(1.+2.*returning)*anchor+consistency
                    loss=(per_lane/lengths/len(episodes)).sum()
                    loss.backward()  # Free this step's graph; recurrent states are detached.
                    running+=torch.stack((bc.mean(),anchor.mean(),consistency.mean())).detach();count+=1
                    if frame==0 or (frame+1)%100==0:
                        if not torch.isfinite(running).all():raise RuntimeError('Nonfinite training loss')
                        status=dict(status='training',variant=args.variant,epoch=epoch+1,epochs=epochs,parent_group=group_index+1,parent_groups=len(batches),
                            parent_episodes=ids,decision=frame+1,max_decisions=int(lengths.max().item()),optimizer_updates=updated,
                            mean_bc=float(running[0].item()/count),mean_source_anchor=float(running[1].item()/count),
                            mean_consistency=float(running[2].item()/count),elapsed_seconds=time.time()-started,
                            peak_gpu_bytes=torch.cuda.max_memory_allocated())
                        common.write(output/'status.json',status);print(json.dumps(status),flush=True)
                gradient=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step();updated+=1
                ratio=updated/(epochs*len(batches));warm=min(1.,(updated+1)/5.)
                multiplier=warm*(.1+.9*.5*(1.+math.cos(math.pi*ratio)))
                for g,rate in zip(optimizer.param_groups,[1e-6,3e-6],strict=True):g['lr']=rate*multiplier
                del episodes
        metrics=evaluate(policy,pre,post,reader,val_ids,visual,args.lanes,max_decisions)
        common.write(output/f'validation_epoch_{epoch+1:02d}.json',metrics)
        print(json.dumps(dict(epoch=epoch+1,validation=metrics)),flush=True)
    # Only named visual/decoder parameters may differ; memory and all other buffers stay exact.
    from safetensors.torch import load_file
    original=load_file(str(SOURCE/'model.safetensors'));state=policy.state_dict()
    for name,tensor in original.items():
        if name not in names and not torch.equal(tensor,state[name].cpu()):raise ValueError('Frozen parameter changed: '+name)
    if not any(not torch.equal(original[name],state[name].cpu()) for name in names):raise ValueError('No visual/decoder weights updated')
    destination=output/f'train/checkpoints/{updated:06d}/pretrained_model';destination.mkdir(parents=True);policy.eval();policy.save_pretrained(destination)
    for p in SOURCE.glob('policy_*'):shutil.copy2(p,destination/p.name)
    shutil.copy2(SOURCE.parents[2]/'normalization_stats.json',output/'train/normalization_stats.json')
    metadata=common.metadata();metadata['raw_visual_finetuning']=dict(variant=args.variant,source=str(SOURCE),source_sha256=common.digest(SOURCE/'model.safetensors'),
        epochs=epochs,optimizer_updates=updated,trainable=names,learning_rates=dict(last_visual_block=1e-6,decoder_and_action_head=3e-6),memory_frozen=True,
        recurrent_gradient='One decision at a time; hidden and prior feature graphs detached; weights update only at complete episode-group boundaries',
        latent='Zero prior and eval-mode core in training, same as inference; no posterior action input',
        augmentation=dict(brightness=[.97,1.03],contrast=[.97,1.03],noise_std_rgb255=.5,lighting_shared_between_cameras=True) if args.variant=='robust' else None,
        loss='Control: expert L1 plus 0.1 source-action squared anchor, tripled on return. Robust: mean clean/augmented BC plus same source anchor and action consistency MSE weight 1.',
        test_episodes_used_for_training=False)
    common.write(destination/'benchmark_provenance.json',metadata)
    common.write(output/'episode_orders.json',orders)
    torch.save(dict(optimizer=optimizer.state_dict(),updates=updated,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),destination.parent/'training_state.pt')
    from excavator_vlm.policies.act.inference import load_act
    restored,_,_=load_act(destination,'cuda:0',original_act=True)
    real=reader.load(val_ids[0],3);policy.reset();restored.reset()
    with torch.no_grad():
        for frame in range(3):
            raw,_,_,_=visual.batch_at([real],frame);batch=pre(raw)
            torch.testing.assert_close(policy.predict_action_chunk(dict(batch)),restored.predict_action_chunk(dict(batch)),rtol=0,atol=0)
    common.assert_external()
    common.write(output/'status.json',dict(status='completed',variant=args.variant,epochs=epochs,optimizer_updates=updated,checkpoint=str(destination),validation=metrics,
        memory_and_other_frozen_weights_unchanged=True,strict_online_reload='passed',external_files_unchanged=True))


def run(args):
    preflight=ROOT/'outputs/experiments/act_visual_runtime_checks_20261120_01/preflight.json'
    if not preflight.exists() or read(preflight).get('status')!='passed':raise RuntimeError('Raw visual trainer preflight required')
    proof=read(preflight)
    if proof['trainer_sha256']!=common.digest(Path(__file__)) or proof['helper_sha256']!=common.digest(ROOT/'ExcavatorVLA-vlm/excavator_vlm/policies/act/visual_training.py'):
        raise RuntimeError('Training source changed since preflight')
    args.output.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),args.output/'run.py')
    source_helper=ROOT/'ExcavatorVLA-vlm/excavator_vlm/policies/act/visual_training.py'
    shutil.copy2(source_helper,args.output/'visual_training.py');shutil.copy2(preflight,args.output/'preflight.json')
    common.write(args.output/'manifest.json',dict(source=str(SOURCE),source_sha256=common.digest(SOURCE/'model.safetensors'),dataset=str(common.DATA),
        split=common.metadata()['split_manifest'],epochs=args.epochs,lanes=args.lanes,source_helper_sha256=common.digest(source_helper),
        variants=['control','robust'],raw_rgb=True,feature_cache_used=False,uniform_hdr_baseline=str(HDR),model_memory_frozen=True))
    common.write(args.output/'external_files.json',common.assert_external())
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','ACT_PROJECT_ROOT':str(ROOT),'ACT_VISUAL_HELPER':str(args.output/'visual_training.py'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    for variant in ['control','robust']:
        common.write(args.output/'status.json',dict(status='training',active_variant=variant))
        with (args.output/f'{variant}.log').open('w') as log:
            subprocess.run([str(RUNTIME),'-u','-B',str(args.output/'run.py'),'--mode','train','--variant',variant,'--output',str(args.output),
                '--epochs',str(args.epochs),'--lanes',str(args.lanes)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    # Select solely on original validation episodes and measured appearance sensitivity.
    choices=[]
    for variant in ['control','robust']:
        state=read(args.output/variant/'status.json');initial=read(args.output/variant/'validation_initial.json');v=state['validation']
        passed=v['all']['first_action_mae']<=1.05*initial['all']['first_action_mae'] and v['return']['first_action_mae']<=1.1*initial['return']['first_action_mae'] and v['appearance_action_mae']<=initial['appearance_action_mae']
        common.write(args.output/variant/'validation_gate.json',dict(passed=passed,metrics=v,source_metrics=initial))
        if passed:choices.append((v['all']['first_action_mae']+v['appearance_action_mae'],variant,state['checkpoint']))
    if not choices:
        common.write(args.output/'status.json',dict(status='validation_gate_rejected',baseline_replaced=False));return
    _,winner,checkpoint=min(choices);common.write(args.output/'selection.json',dict(variant=winner,checkpoint=checkpoint,selection='Validation clean first-action error plus appearance-induced action change, with source and return regression gates'))
    common.write(args.output/'status.json',dict(status='waiting_for_hdr_baseline',selected_variant=winner))
    while not (HDR/'repeats/completion.json').exists():time.sleep(15)
    if read(HDR/'repeats/status.json')['status']!='completed':raise RuntimeError('HDR baseline has infrastructure errors')
    evaluate_candidate(args.output,checkpoint,winner)


def evaluate_candidate(output,checkpoint,winner):
    # Two repeats per known scene under the SAME versioned HDR evaluator and GPU.
    template=(ROOT/'scripts/evaluate_act_baseline_repeatability.py').read_text()
    old="CHECKPOINT=ROOT/'outputs/experiments/act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model'"
    if template.count(old)!=1:raise ValueError('Repeat evaluation checkpoint anchor changed')
    template=template.replace(old,'CHECKPOINT=Path('+repr(checkpoint)+')')
    marker="            changes={'--output':str(folder/'evaluation'),'--steps':'2000','--gpus':job['gpu_uuid']}"
    if template.count(marker)!=1:raise ValueError('Repeat command checkpoint anchor changed')
    template=template.replace(marker,marker.replace("{'--output':","{'--checkpoint':str(CHECKPOINT),'--output':"))
    inference=(ROOT/'scripts/check_act_inference_repeatability.py').read_text()
    inference=inference.replace("root=Path(__file__).resolve().parents[1]",'root=Path('+repr(str(ROOT))+')')
    inference=inference.replace("checkpoint=root/'outputs/experiments/act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model'",'checkpoint=Path('+repr(checkpoint)+')')
    inference_path=output/'check_selected_inference.py';inference_path.write_text(inference)
    template=template.replace("str(ROOT/'scripts/check_act_inference_repeatability.py')",'str(Path('+repr(str(inference_path))+'))')
    script=output/'evaluate_selected.py';script.write_text(template);compile(template,str(script),'exec')
    common.write(output/'status.json',dict(status='evaluating_hdr_development_repeats',selected_variant=winner,scenes=[50,112],repeats_each=2,steps=2000))
    with (output/'evaluation.log').open('w') as log:
        subprocess.run([str(RUNTIME),'-u','-B',str(script),'--output',str(output/'hdr_repeats'),'--evaluator',str(HDR/'evaluator_single_hdr.py'),
            '--lighting-profile','single_authored_hdr_v1'],cwd=ROOT,env={**os.environ,'ACT_PROJECT_ROOT':str(ROOT),'CUDA_VISIBLE_DEVICES':'0,1'},stdout=log,stderr=subprocess.STDOUT,check=True)
    results=read(output/'hdr_repeats/summary.json');baseline=read(HDR/'repeats/summary.json')
    comparison={}
    for scene in ['50','112']:
        def samples(report):return [s for s in report['scenes'][scene]['samples'] if s['lighting_profile']=='single_authored_hdr_v1']
        a=samples(baseline);b=samples(results)
        comparison[scene]=dict(baseline=a,candidate=b,mean_successful_cycle_change=sum(s['successful_cycles'] for s in b)/len(b)-sum(s['successful_cycles'] for s in a)/len(a)) if a and b else dict(error='Incomplete comparable samples')
    common.write(output/'paired_hdr_comparison.json',dict(scenes=comparison,scope='Two repeats per known scene, 2000 steps, development only; not independent generalization or full ten-cycle success rates'))
    common.write(output/'status.json',dict(status='completed' if results['completed_runs']==4 and not results['infrastructure_errors'] else 'evaluation_incomplete',selected_variant=winner,baseline_replaced=False,results=str(output/'hdr_repeats')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=lambda p:Path(p).resolve(),required=True)
    parser.add_argument('--mode',choices=['run','train'],default='run');parser.add_argument('--variant',choices=['control','robust'],default='control')
    parser.add_argument('--epochs',type=int,default=2);parser.add_argument('--lanes',type=int,default=2);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    try:{'run':run,'train':train}[args.mode](args)
    except Exception as error:
        if args.output.exists():common.write(args.output/f'error_{args.mode}_{args.variant}.json',dict(error=repr(error)))
        raise
