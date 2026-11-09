"""Paired bounded ACT fine-tuning: original L1 versus normalized Smooth L1."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',str(Path(__file__).resolve().parents[1]))).resolve()
import run_act_stage_auxiliary as common
from run_act_stage_auxiliary import read,write,sha


def worker(out,arm):
    protocol=read(out/'protocol.json')
    os.environ['CUDA_VISIBLE_DEVICES']=str(protocol['arms'][arm]['gpu'])
    common.setup(out)
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from excavator_vlm.data.contract import read_frames
    from excavator_vlm.policies.act.adapter import training_batch
    from excavator_vlm.policies.act.config import build_train_config
    from excavator_vlm.policies.act.continuation import validate_continuation
    from excavator_vlm.policies.act.dataset import make_act_dataset,training_stats
    from excavator_vlm.policies.act.inference import load_act
    from excavator_vlm.policies.act.sampling import make_sampler
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.utils.train_utils import save_checkpoint,update_last_checkpoint
    from act_huber_objective import policy_objective

    torch.set_num_threads(4);torch.manual_seed(1000);np.random.seed(1000)
    common.check_inputs(out)
    settings=read(out/'settings.json');settings['output_dir']=str(out/'train'/arm)
    settings['job_name']='act_huber_'+arm
    cfg=build_train_config(settings)
    source=Path(protocol['checkpoint']);split=read(out/'split.json')
    frames=read_frames(settings['dataset']['root']);stats=training_stats(frames,split['train'])
    continuation=validate_continuation(source,cfg.policy,read(out/'dataset_contract.json'),split,split['train'],stats)
    run=out/'train'/arm;run.mkdir(parents=True,exist_ok=False)
    write(run/'normalization_stats.json',stats);write(run/'resolved_settings.json',settings)
    write(run/'continuation.json',continuation)
    cfg.dataset.episodes=split['train'];cfg.policy.device='cuda'
    policy,pre,post=load_act(source,'cuda')
    expected=load_file(str(source/'model.safetensors'))
    for name,value in policy.state_dict().items():torch.testing.assert_close(value.cpu(),expected[name],rtol=0,atol=0)
    del expected
    optimizer,scheduler=make_optimizer_and_scheduler(cfg,policy)
    if scheduler is not None:raise ValueError('Expected the original no-scheduler preset')
    training=frames[frames.episode_index.isin(split['train'])]
    references={int(ep):group.sort_values('frame_index') for ep,group in training.groupby('episode_index')}
    dataset=make_act_dataset('local/excavator',Path(settings['dataset']['root']),split['train'],10)
    sampler,sampling=make_sampler(dataset,split['train'],settings['sampling'],1000)
    loader=DataLoader(dataset,batch_size=16,sampler=sampler,num_workers=2,multiprocessing_context='spawn',
        persistent_workers=True,timeout=60,generator=torch.Generator().manual_seed(1000))
    iterator=iter(loader);checked=set();sample_hash=hashlib.sha256()
    torch.manual_seed(1000);started=time.monotonic()
    initial={name:value.detach().clone() for name,value in policy.named_parameters()
             if name in ('model.action_head.weight','model.encoder_robot_state_input_proj.weight','model.backbone.conv1.weight')}
    kind=protocol['arms'][arm]['loss']
    with (run/'train.jsonl').open('x') as log:
        for step in range(1,401):
            raw=next(iterator)
            identities=list(zip(raw['episode_index'].reshape(-1).tolist(),raw['frame_index'].reshape(-1).tolist(),strict=True))
            sample_hash.update(json.dumps(identities,separators=(',',':')).encode())
            for i,(ep,frame) in enumerate(identities):
                key=int(ep),int(frame)
                if key in checked:continue
                if key[0] not in references:raise ValueError('Held-out training observation')
                episode=references[key[0]];indices=np.minimum(np.arange(key[1],key[1]+10),len(episode)-1)
                expected=episode.iloc[indices]
                for name,value in [('action',np.stack(expected.action)),('action_loss_weight',np.asarray(expected.action_loss_weight.tolist()))]:
                    actual=raw[name][i].numpy()
                    np.testing.assert_array_equal(actual.reshape(-1),value.astype(actual.dtype).reshape(-1))
                actual=raw['observation.state'][i].numpy()
                np.testing.assert_array_equal(actual,episode.iloc[key[1]]['observation.state'].astype(actual.dtype))
                np.testing.assert_array_equal(raw['action_is_pad'][i].numpy(),np.arange(key[1],key[1]+10)>=len(episode))
                checked.add(key)
            policy.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,parts=policy_objective(policy,training_batch(pre,raw),kind,.1)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),10.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            if step==1 or step%50==0:
                row=dict(step=step,objective=kind,**parts,grad_norm=float(norm),elapsed_seconds=time.monotonic()-started)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                write(run/'status.json',dict(status='training',step=step,steps=400))
    checkpoint=run/'checkpoints/000400'
    save_checkpoint(checkpoint,400,cfg,policy,optimizer,None);update_last_checkpoint(checkpoint)
    pretrained=checkpoint/'pretrained_model'
    for path in source.glob('*processor*'):
        if path.is_file():shutil.copy2(path,pretrained/path.name)
    provenance=read(source/'benchmark_provenance.json')
    provenance.update(continuation=continuation,finetune_steps=400,reconstruction_loss_kind=kind,
        smooth_l1_beta=.1 if kind!='l1' else None,
        executed_prefix_l1_steps=policy.config.n_action_steps if kind=='executed_prefix_l1' else None,
        normalization_stats_sha256=sha(run/'normalization_stats.json'))
    write(pretrained/'benchmark_provenance.json',provenance)
    policy.eval()
    with torch.inference_mode():
        observed=pre({k:raw[k][:2] for k in raw if k=='observation.state' or k.startswith('observation.images.')})
        predicted=policy.predict_action_chunk(observed).cpu();restored,_,_=load_act(pretrained,'cuda')
        torch.testing.assert_close(restored.predict_action_chunk(observed).cpu(),predicted,rtol=1e-5,atol=1e-6)
    if set(policy.state_dict())!=set(restored.state_dict()):raise ValueError('Inference schema changed')
    for path in source.glob('*processor*'):
        if path.is_file() and sha(path)!=sha(pretrained/path.name):raise ValueError('Processor bytes changed')
    common.check_inputs(out)
    result=dict(status='completed',steps=400,objective=kind,checkpoint=str(pretrained),model_sha256=sha(pretrained/'model.safetensors'),
        sample_order_sha256=sample_hash.hexdigest(),parquet_aligned_distinct_observations=len(checked),sampling=sampling,
        reload_check='passed',processors_byte_identical=True,inputs_unchanged=True,
        parameter_changes={name:float((dict(policy.named_parameters())[name]-value).abs().max().detach()) for name,value in initial.items()},
        elapsed_seconds=time.monotonic()-started)
    write(run/'result.json',result);write(run/'status.json',dict(status='completed',step=400))


def prepare(out,candidate_loss):
    # Reuse the audited split, sampler, evaluation identities and carry-guard
    # scene preparation. Replace the unexecuted auxiliary-task template before
    # any optimizer updates, and freeze the actual loss experiment below.
    template=common.prepare(out)
    protocol={k:v for k,v in template.items() if k not in ('arms','auxiliary_head_lr','intervention','auxiliary_features','stage_class_weights',
                                                          'expected_observation_stage_probability','export','inference','offline_gate')}
    protocol.update(experiment='act_l1_vs_smooth_l1',candidate_arm='huber',
        arms=dict(control=dict(gpu=0,loss='l1'),huber=dict(gpu=1,loss=candidate_loss)),
        beta=.1,executed_prefix=3,
        intervention=('Only replace normalized weighted L1 reconstruction by Smooth L1; PHW/padding/denominator/KL unchanged'
            if candidate_loss=='smooth_l1' else 'Keep original L1 for the three actually executed actions; '
            'use Smooth L1 for the remaining seven steps. PHW/padding/denominator/KL/architecture unchanged'),
        export='Original ACT architecture and original processors; no additional inference modules or phase inputs',
        offline_gate='Original seven gates plus strong stage6 first-bucket (expert abs>=0.1 rad/s) '
            'MAE improves 10% vs baseline and 5% vs control; strong-action sign accuracy does not decline vs either',
        scope='Paired 400-step development experiment; no claim that output-gradient imbalance is the causal failure')
    write(out/'protocol.json',protocol)
    for name in ('run_act_huber.py','act_huber_objective.py'):
        shutil.copy2(ROOT/'scripts'/name,out/'frozen_scripts'/name)
    protected=read(out/'inputs_sha256.json')
    protected[str(out/'protocol.json')]=sha(out/'protocol.json')
    for name in ('run_act_huber.py','act_huber_objective.py'):
        path=out/'frozen_scripts'/name;protected[str(path)]=sha(path)
    write(out/'inputs_sha256.json',protected)
    return protocol


def offline(out):
    common.offline(out)
    import numpy as np
    arrays=np.load(out/'offline_predictions.npz')
    obs=read(out/'offline_observations.json')['observations']
    ids=np.array([r['split']=='validation' and r['stage_id']==6 for r in obs])
    target=arrays['target'][ids,0,3];strong=np.abs(target)>=.1
    if not strong.any():raise ValueError('No predeclared strong-action validation observations')
    details={name:dict(observations=int(strong.sum()),
        mae=float(np.abs(arrays[name][ids,0,3][strong]-target[strong]).mean()),
        sign_accuracy=float((np.sign(arrays[name][ids,0,3][strong])==np.sign(target[strong])).mean()))
        for name in ('baseline','control','huber')}
    d=read(out/'offline_metrics.json');v=details['huber'];b=details['baseline'];c=details['control']
    d['strong_stage6_first_bucket']=details
    d['gate'].update(strong_bucket_mae=v['mae']<=.9*b['mae'] and v['mae']<=.95*c['mae'],
        strong_bucket_sign=v['sign_accuracy']>=max(b['sign_accuracy'],c['sign_accuracy']))
    d['advance']=all(d['gate'].values());write(out/'offline_metrics.json',d)
    return d['advance']


def report(out):
    data=read(out/'offline_metrics.json')
    lines=['# ACT L1与Smooth L1限额对照','',
        '两组各400次更新，同一PHW40k、数据、采样、原归一化/PHW/KL。唯一差别为重建损失形式；实际候选见protocol.json，beta=0.1。','',
        '| 模型 | 持料首步铲斗MAE | 持料预测幅度 | 转换前三步MAE | 抽样卸料前三步MAE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,metrics in data['policies'].items():
        v=metrics['validation'];lines.append(f"| {name} | {v['6']['first_bucket_mae_rad_s']:.6f} | "
            f"{v['6']['first_bucket_predicted_mean_abs_rad_s']:.6f} | {v['transitions']['first_three_mae_rad_s']:.6f} | "
            f"{v['9']['first_three_mae_rad_s']:.6f} |")
    lines += ['',f"预设推进条件：{'通过' if data['advance'] else '未通过'}。",json.dumps(data['gate'],ensure_ascii=False),'',
        '单位rad/s。固定专家记录状态的开发诊断，持料验证仅22帧，不是完整任务成功率。27.5%保留基准不修改。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def controller(out,candidate_loss):
    protocol=prepare(out,candidate_loss);children=[];logs=[];state=dict(status='training',training={})
    def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for arm in protocol['arms']:
            log=(out/(arm+'.log')).open('x');logs.append(log)
            command=[sys.executable,'-u',str(out/'frozen_scripts/run_act_huber.py'),'--output',str(out),'--worker',arm]
            child=subprocess.Popen(command,cwd=out,env=dict(os.environ,ACT_PROJECT_ROOT=str(ROOT)),stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
            children.append((arm,child));state['training'][arm]=dict(pid=child.pid,command=command,gpu=protocol['arms'][arm]['gpu'])
        while any(child.poll() is None for _,child in children):
            for arm,child in children:
                state['training'][arm]['returncode']=child.poll()
                path=out/'train'/arm/'status.json'
                if path.exists():state['training'][arm]['progress']=read(path)
                if child.poll() not in (None,0):raise RuntimeError(f'{arm} failed; inspect its log')
            write(out/'status.json',state);time.sleep(5)
        training={arm:read(out/'train'/arm/'result.json') for arm,_ in children}
        if len({v['sample_order_sha256'] for v in training.values()})!=1:raise ValueError('Different sample order')
        write(out/'training_results.json',training)
        state['status']='offline_evaluation';write(out/'status.json',state)
        advance=offline(out)
        state.update(status='offline_passed' if advance else 'completed_no_advance',advance=advance,
                     training_complete=True,source_inputs_unchanged=True)
        write(out/'status.json',state)
        if advance:common.closed_loop(out,state)
        report(out);print(json.dumps(dict(status=state['status'],advance=advance)),flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error));write(out/'status.json',state);raise
    finally:
        for _,child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for _,child in children:
            if child.poll() is None:
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for log in logs:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',choices=['control','huber'],help=argparse.SUPPRESS)
    parser.add_argument('--candidate-loss',choices=['smooth_l1','executed_prefix_l1'],default='smooth_l1')
    args=parser.parse_args();out=args.output.resolve()
    if args.worker:worker(out,args.worker)
    else:controller(out,args.candidate_loss)
