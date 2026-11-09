"""Paired 400-step ACT fine-tuning with an observation-only stage auxiliary task."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(os.environ.get('ACT_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def setup(out):
    protocol = read(out / 'protocol.json')
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HOME=str(ROOT / 'models/hf-cache'), TORCH_HOME=str(ROOT / 'models/torch-cache'),
        HF_DATASETS_CACHE=str(ROOT / 'models/hf-cache/datasets'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    sys.path[:0] = [str(out / 'frozen_sources/ExcavatorVLA-vlm'),
                    str(Path(protocol['archive']) / 'runtime_dependencies')]
    return protocol


def check_inputs(out):
    for name, expected in read(out / 'inputs_sha256.json').items():
        if sha(name) != expected:
            raise ValueError(f'Frozen input changed: {name}')


def worker(out, arm):
    protocol = read(out / 'protocol.json')
    gpu = protocol['arms'][arm]['gpu']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    setup(out)
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader
    from excavator_vlm.data.contract import read_frames
    from excavator_vlm.policies.act.adapter import training_batch
    from excavator_vlm.policies.act.config import build_train_config
    from excavator_vlm.policies.act.continuation import validate_continuation
    from excavator_vlm.policies.act.dataset import make_act_dataset, training_stats
    from excavator_vlm.policies.act.inference import load_act
    from excavator_vlm.policies.act.sampling import make_sampler
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.utils.train_utils import save_checkpoint, update_last_checkpoint
    from act_stage_auxiliary import CurrentObservationFeatures, StageAuxiliaryHead, auxiliary_loss, stage_labels

    torch.set_num_threads(4)
    torch.manual_seed(1000)
    np.random.seed(1000)
    check_inputs(out)
    settings = read(out / 'settings.json')
    settings['output_dir'] = str(out / 'train' / arm)
    settings['job_name'] = 'act_stage_auxiliary_' + arm
    cfg = build_train_config(settings)
    source = Path(protocol['checkpoint'])
    split = read(out / 'split.json')
    frames = read_frames(settings['dataset']['root'])
    stats = training_stats(frames, split['train'])
    continuation = validate_continuation(source, cfg.policy, read(out / 'dataset_contract.json'),
                                         split, split['train'], stats)
    run = out / 'train' / arm
    run.mkdir(parents=True, exist_ok=False)
    write(run / 'normalization_stats.json', stats)
    write(run / 'resolved_settings.json', settings)
    write(run / 'continuation.json', continuation)
    cfg.dataset.episodes = split['train']
    cfg.policy.device = 'cuda'
    policy, pre, post = load_act(source, 'cuda')
    # Keep the exact source architecture, original fresh AdamW preset and source processors.
    expected = load_file(str(source / 'model.safetensors'))
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected[name], rtol=0, atol=0)
    del expected
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(2000)
        head = StageAuxiliaryHead(policy.config.dim_model, len(policy.config.image_features)).to('cuda')
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    if scheduler is not None:
        raise ValueError('This experiment preserves the original no-scheduler preset')
    optimizer.add_param_group(dict(params=list(head.parameters()), lr=protocol['auxiliary_head_lr']))
    coefficient = protocol['arms'][arm]['coefficient']
    weights = torch.tensor(protocol['stage_class_weights'], device='cuda')
    allowed = set(split['train'])
    training_frames = frames[frames.episode_index.isin(allowed)]
    stage_map = {(int(row.episode_index), int(row.frame_index)): int(stage)
        for row, stage in zip(training_frames.itertuples(), training_frames['observation.stage_current_id'], strict=True)}
    references = {int(ep): group.sort_values('frame_index') for ep, group in training_frames.groupby('episode_index')}
    checked = set()
    dataset = make_act_dataset('local/excavator', Path(settings['dataset']['root']), split['train'], 10)
    sampler, sampling = make_sampler(dataset, split['train'], settings['sampling'], 1000)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=2,
        multiprocessing_context='spawn', persistent_workers=True, timeout=60,
        generator=torch.Generator().manual_seed(1000))
    iterator = iter(loader)
    sample_hash = hashlib.sha256()
    counts = np.zeros(10, dtype=np.int64)
    torch.manual_seed(1000)
    started = time.monotonic()
    initial_head = policy.model.action_head.weight.detach().clone()
    initial_state = policy.model.encoder_robot_state_input_proj.weight.detach().clone()
    initial_backbone = next(policy.model.backbone.parameters()).detach().clone()
    with CurrentObservationFeatures(policy) as captured, (run / 'train.jsonl').open('x') as log:
        for step in range(1, protocol['steps'] + 1):
            raw = next(iterator)
            labels = stage_labels(raw, stage_map, allowed).to('cuda')
            identities = list(zip(raw['episode_index'].reshape(-1).tolist(), raw['frame_index'].reshape(-1).tolist(), strict=True))
            sample_hash.update(json.dumps(identities, separators=(',', ':')).encode())
            counts += np.bincount(labels.cpu().numpy(), minlength=10)
            for i, (ep, frame) in enumerate(identities):
                key = int(ep), int(frame)
                if key in checked:
                    continue
                episode = references[key[0]]
                indices = np.minimum(np.arange(key[1],key[1]+10),len(episode)-1)
                expected = episode.iloc[indices]
                for name, value in [('action',np.stack(expected.action)),
                                    ('action_loss_weight',np.asarray(expected.action_loss_weight.tolist()))]:
                    actual = raw[name][i].numpy()
                    np.testing.assert_array_equal(actual.reshape(-1),value.astype(actual.dtype).reshape(-1))
                actual_state = raw['observation.state'][i].numpy()
                np.testing.assert_array_equal(actual_state,episode.iloc[key[1]]['observation.state'].astype(actual_state.dtype))
                np.testing.assert_array_equal(raw['action_is_pad'][i].numpy(),np.arange(key[1],key[1]+10)>=len(episode))
                checked.add(key)
            policy.train()
            head.train()
            captured.reset()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ordinary, parts = policy(training_batch(pre, raw))
                auxiliary, logits = auxiliary_loss(head, captured, labels, weights)
                loss = ordinary + coefficient * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([*policy.parameters(), *head.parameters()],10.)
            if not torch.isfinite(norm):
                raise FloatingPointError(f'Nonfinite gradient at {step}')
            optimizer.step()
            if step == 1 or step % 50 == 0:
                row = dict(step=step,**parts,stage_loss=float(auxiliary.detach()),
                    stage_accuracy=float((logits.argmax(-1)==labels).float().mean()),
                    coefficient=coefficient,combined_loss=float(loss.detach()),grad_norm=float(norm),
                    elapsed_seconds=time.monotonic()-started)
                log.write(json.dumps(row)+'\n'); log.flush()
                print(json.dumps(row),flush=True)
                write(run / 'status.json',dict(status='training',step=step,steps=protocol['steps']))
    checkpoint = run / 'checkpoints' / f'{protocol["steps"]:06d}'
    # Only the original ACT policy is exported. No stage module or label becomes
    # part of inference; the auxiliary parameters are kept in training_state.
    save_checkpoint(checkpoint, protocol['steps'], cfg, policy, optimizer, None)
    update_last_checkpoint(checkpoint)
    pretrained = checkpoint / 'pretrained_model'
    for path in source.glob('*processor*'):
        if path.is_file():
            shutil.copy2(path, pretrained / path.name)
    provenance = read(source / 'benchmark_provenance.json')
    provenance.update(continuation=continuation, finetune_steps=protocol['steps'],
        stage_auxiliary_coefficient=coefficient, inference_auxiliary_head=False,
        normalization_stats_sha256=sha(run / 'normalization_stats.json'))
    write(pretrained / 'benchmark_provenance.json',provenance)
    save_file(head.state_dict(),str(checkpoint / 'training_state/stage_auxiliary_head.safetensors'))
    write(checkpoint / 'training_state/stage_auxiliary.json',dict(coefficient=coefficient,
        class_weights=protocol['stage_class_weights'],head_lr=protocol['auxiliary_head_lr'],
        inference_used=False,features='current state/image projections, before latent-conditioned attention'))
    policy.eval()
    with torch.inference_mode():
        observed = pre({k:raw[k][:2] for k in raw if k=='observation.state' or k.startswith('observation.images.')})
        prediction = policy.predict_action_chunk(observed).cpu()
        restored, _, _ = load_act(pretrained,'cuda')
        torch.testing.assert_close(restored.predict_action_chunk(observed).cpu(),prediction,rtol=1e-5,atol=1e-6)
    if set(policy.state_dict()) != set(restored.state_dict()):
        raise ValueError('Export added auxiliary inference parameters')
    for path in source.glob('*processor*'):
        if path.is_file() and sha(path)!=sha(pretrained/path.name):
            raise ValueError('Processor bytes changed')
    check_inputs(out)
    result = dict(status='completed', steps=protocol['steps'], coefficient=coefficient,
        checkpoint=str(pretrained), model_sha256=sha(pretrained/'model.safetensors'),
        sample_order_sha256=sample_hash.hexdigest(), sampled_stage_counts=counts.tolist(),
        parquet_aligned_distinct_observations=len(checked),sampling=sampling,
        reload_check='passed', processors_byte_identical=True, inputs_unchanged=True,
        original_inference_schema=True, elapsed_seconds=time.monotonic()-started,
        parameter_changes=dict(action_head_max=float((policy.model.action_head.weight-initial_head).abs().max()),
            state_projection_max=float((policy.model.encoder_robot_state_input_proj.weight-initial_state).abs().max()),
            backbone_max=float((next(policy.model.backbone.parameters())-initial_backbone).abs().max())))
    write(run/'result.json',result)
    write(run/'status.json',dict(status='completed',step=protocol['steps']))


def action_metrics(prediction, target, padding, observations):
    import numpy as np
    result = {}
    error = np.abs(prediction-target)
    for group in ('train','validation'):
        group_mask = np.array([r['split']==group for r in observations])
        result[group] = {}
        for stage in [*range(10),'transitions']:
            mask = group_mask & np.array([r['stage_id'] in (5,6) if stage=='transitions'
                                         else r['stage_id']==stage for r in observations])
            valid = ~padding[mask,:3]
            result[group][str(stage)] = dict(observations=int(mask.sum()),
                first_three_mae_rad_s=float((error[mask,:3]*valid[:,:,None]).sum()/(valid.sum()*4)),
                first_bucket_mae_rad_s=float(error[mask,0,3].mean()),
                first_bucket_predicted_mean_abs_rad_s=float(np.abs(prediction[mask,0,3]).mean()),
                first_bucket_expert_mean_abs_rad_s=float(np.abs(target[mask,0,3]).mean()))
    return result


def offline(out):
    protocol = setup(out)
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from excavator_vlm.policies.act.dataset import make_act_dataset
    from excavator_vlm.policies.act.inference import load_act
    torch.set_num_threads(4)
    observations = read(out/'offline_observations.json')['observations']
    split = read(out/'split.json')
    checkpoints = dict(baseline=Path(protocol['checkpoint']),**{arm:Path(read(out/'train'/arm/'result.json')['checkpoint'])
                                                            for arm in protocol['arms']})
    policies = {name:load_act(path,'cuda') for name,path in checkpoints.items()}
    predictions = {name:[] for name in checkpoints}
    targets,padding=[],[]
    for group in ('train','validation'):
        chosen=[r for r in observations if r['split']==group]
        dataset=make_act_dataset('local/excavator',Path(protocol['dataset']),split[group],10)
        identity=dataset.hf_dataset.select_columns(['episode_index','frame_index']).to_pandas()
        lookup={(int(ep),int(frame)):i for i,(ep,frame) in enumerate(zip(identity.episode_index,identity.frame_index,strict=True))}
        for start in range(0,len(chosen),16):
            raw=default_collate([dataset[lookup[(r['episode_index'],r['frame_index'])]] for r in chosen[start:start+16]])
            observation={k:v for k,v in raw.items() if k=='observation.state' or k.startswith('observation.images.')}
            with torch.inference_mode():
                for name,(policy,pre,post) in policies.items():
                    chunk=policy.predict_action_chunk(pre(observation))
                    predictions[name].append(post(chunk.reshape(-1,4)).reshape_as(chunk).cpu().numpy())
            targets.append(raw['action'].numpy()); padding.append(raw['action_is_pad'].numpy())
        print(f'Offline {group}: {len(chosen)} observations, three policies',flush=True)
    target=np.concatenate(targets); pad=np.concatenate(padding)
    predictions={name:np.concatenate(v) for name,v in predictions.items()}
    metrics={name:action_metrics(value,target,pad,observations) for name,value in predictions.items()}
    val=metrics[protocol.get('candidate_arm','stage_aux')]['validation']; base=metrics['baseline']['validation']; control=metrics['control']['validation']
    gate=dict(
        stage6_bucket_vs_baseline=val['6']['first_bucket_mae_rad_s']<=.9*base['6']['first_bucket_mae_rad_s'],
        transitions_vs_baseline=val['transitions']['first_three_mae_rad_s']<=.9*base['transitions']['first_three_mae_rad_s'],
        stage6_bucket_vs_control=val['6']['first_bucket_mae_rad_s']<=.95*control['6']['first_bucket_mae_rad_s'],
        transitions_vs_control=val['transitions']['first_three_mae_rad_s']<=.95*control['transitions']['first_three_mae_rad_s'],
        target_stages=all(val[str(s)]['first_three_mae_rad_s']<=1.05*base[str(s)]['first_three_mae_rad_s'] for s in (5,6)),
        controls=all(val[str(s)]['first_three_mae_rad_s']<=max(1.05*base[str(s)]['first_three_mae_rad_s'],
            base[str(s)]['first_three_mae_rad_s']+.002) for s in range(10) if s not in (5,6)),
        stage6_amplitude=val['6']['first_bucket_predicted_mean_abs_rad_s']>=.2*val['6']['first_bucket_expert_mean_abs_rad_s'])
    np.savez_compressed(out/'offline_predictions.npz',target=target,padding=pad,**predictions)
    write(out/'offline_metrics.json',dict(policies=metrics,gate=gate,advance=all(gate.values()),test_used=False,
        scope='Expert recorded observations; not evidence of actual loaded transport or task success'))
    check_inputs(out)
    del policies, policy, pre, post, chunk
    torch.cuda.empty_cache()
    return all(gate.values())


def closed_loop(out, state):
    protocol=read(out/'protocol.json')
    archive=Path(protocol['archive'])
    sys.path.insert(0,str(archive/'frozen_sources/scripts'))
    from run_act_transport_validation import assess_episode
    candidate_arm=protocol.get('candidate_arm','stage_aux')
    candidate=read(out/'train'/candidate_arm/'result.json')['checkpoint']
    original_command=read(archive/'evidence/protocol.json')['command']
    records={}
    for index,scene in enumerate(protocol['development_scenes']):
        jobs=[];handles=[]
        try:
            for variant,checkpoint in [('baseline',protocol['checkpoint']),(candidate_arm,candidate)]:
                gpu=(index+(variant==candidate_arm))%2
                name=f'scene{scene}_{variant}'
                destination=out/'closed_loop'/name
                command=original_command.copy()
                # Replace the absolute entrypoint explicitly; all remaining
                # physics/guard flags are exactly those in the retained run.
                command[1]='-u'
                command[2]=str(archive/'frozen_sources/scripts/eval_smolvla_success.py')
                for flag,value in [('--checkpoint',checkpoint),('--output',str(destination)),
                    ('--episodes','1'),('--seed',str(2000000+scene-1)),('--workers','1'),('--gpus',str(gpu))]:
                    command[command.index(flag)+1]=value
                command += ['--reference-episode',str(out/'references'/f'scene{scene}.json')]
                env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                    HF_HOME=str(ROOT/'models/hf-cache'),TORCH_HOME=str(ROOT/'models/torch-cache'),
                    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                    PYTHONPATH=str(archive/'runtime_dependencies'))
                # Offline evaluation masks the controller to one card. Each
                # rollout must resolve its explicit physical GPU independently.
                env.pop('CUDA_VISIBLE_DEVICES',None)
                log=(out/(name+'.log')).open('x');handles.append(log)
                child=subprocess.Popen(command,cwd=archive/'frozen_sources',env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                jobs.append((name,variant,destination,command,child))
            state.update(status='closed_loop',scene=scene,running=[dict(name=j[0],pid=j[4].pid,command=j[3]) for j in jobs])
            write(out/'status.json',state)
            while any(j[4].poll() is None for j in jobs):
                for name,_,_,_,child in jobs:
                    if child.poll() not in (None,0):raise RuntimeError(f'{name} failed; inspect its log')
                time.sleep(5)
            for name,variant,destination,command,child in jobs:
                if child.returncode:raise RuntimeError(f'{name} failed')
                evaluation_id=read(destination/'provenance.json')['evaluation_id']
                result=assess_episode(destination/'episodes/episode_0001',2000000+scene-1,evaluation_id)
                records[name]=dict(scene=scene,variant=variant,command=command,result=result)
                write(out/'closed_loop_results.json',records)
        finally:
            for *_,child in jobs:
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            for *_,child in jobs:
                if child.poll() is None:
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
            for handle in handles:handle.close()
    check_inputs(out)
    state.update(status='completed_with_closed_loop',running=[],
        development_only=True,completed_rollouts=len(records))
    write(out/'status.json',state)


def report(out):
    metrics=read(out/'offline_metrics.json')
    lines=['# ACT 阶段识别辅助微调小试','',
        '两组均从保留的 PHW40k 权重启动，400步、batch16、相同采样和原动作/PHW/KL损失。',
        '候选仅增加0.05倍阶段分类损失；分类特征不经过未来动作条件化的注意力。导出为原ACT结构，阶段头不参与推理。','',
        '| 策略 | 持料首步铲斗MAE | 持料首步预测幅度 | 转换前三步MAE | 抽样卸料前三步MAE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,values in metrics['policies'].items():
        v=values['validation']
        lines.append(f"| {name} | {v['6']['first_bucket_mae_rad_s']:.6f} | "
            f"{v['6']['first_bucket_predicted_mean_abs_rad_s']:.6f} | "
            f"{v['transitions']['first_three_mae_rad_s']:.6f} | {v['9']['first_three_mae_rad_s']:.6f} |")
    lines += ['',f"预设离线推进门槛：{'通过' if metrics['advance'] else '未通过'}。",
              json.dumps(metrics['gate'],ensure_ascii=False),'',
              '全部数值为原专家记录状态的离线误差，单位rad/s；持料验证仅22帧，不是完整任务成功率。',
              '原27.5%基准保持。各组只保留第400步检查点，处理器字节不变；两组抽样顺序一致。']
    closed=out/'closed_loop_results.json'
    if closed.exists():
        lines += ['','已运行预先固定的开发案例闭环；见closed_loop_results.json，不能当作独立最终成功率。']
    else:
        lines += ['','没有启动额外闭环或修改步数、损失系数。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    manifest=read(ROOT/'docs/baselines/act_phw40k_carry275_20261105.json')
    archive=ROOT/manifest['archive']
    # All ACT inference/training code is copied from the retained baseline;
    # only the two explicitly named experiment scripts are new.
    frozen=out/'frozen_sources/ExcavatorVLA-vlm'
    shutil.copytree(archive/'frozen_sources/ExcavatorVLA-vlm/excavator_vlm',frozen/'excavator_vlm',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (out/'frozen_scripts').mkdir()
    for name in ('run_act_stage_auxiliary.py','act_stage_auxiliary.py'):
        shutil.copy2(ROOT/'scripts'/name,out/'frozen_scripts'/name)
    source=archive/manifest['checkpoint']
    split=read(source/'benchmark_provenance.json')['split_manifest']
    write(out/'split.json',split)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),baseline_id=manifest['baseline_id'],
        archive=str(archive),checkpoint=str(source),model_sha256=manifest['model_sha256'],
        dataset=manifest['external_dependencies']['dataset'],steps=400,seed=1000,batch_size=16,
        learning_rate=1e-5,auxiliary_head_lr=1e-4,precision='bf16 training, float32 evaluation',
        arms=dict(control=dict(gpu=0,coefficient=0.),stage_aux=dict(gpu=1,coefficient=.05)),
        sampling='Same 70% uniform + 30% episode-balanced stages 5/6 replacement sampling in both arms',
        intervention='Only add 0.05 * training-only current-stage cross entropy; original ACT L1/PHW/KL unchanged',
        auxiliary_features='Current state projection plus three ordered spatially averaged image projections; '
            'before latent-conditioned attention, no future action or posterior latent',
        stage_class_weights=[],export='Original ACT schema only; stage head retained separately in training_state',
        inference='Same three images and 28D state; no stage labels or auxiliary head',
        selection='Only final 400-step weights; no budget/hyperparameter changes after seeing results',
        offline_gate='At least 10% stage6 bucket/transition first3 MAE improvement vs baseline, 5% vs ordinary '
            'fine-tuning; each transition stage at most 5% worse; every other stage at most '
            'max(5%, 0.002 rad/s) worse vs baseline; recover at least 20% of expert stage6 first bucket amplitude',
        if_gate_fails='Stop and retain the 27.5% baseline; no added steps or new loss settings',
        if_gate_passes='Paired baseline/candidate rollouts on predeclared development scenes 1,9,20,6; '
            '500 steps, original carry guard/physics/scoring, same initial geometry; no new final success rate',
        development_scenes=[1,9,20,6],test_used=False)
    write(out/'protocol.json',protocol)
    setup(out)
    import numpy as np
    from excavator_vlm.policies.act.config import resolve_settings
    from excavator_vlm.policies.act.train import prepare as audit_prepare
    from excavator_vlm.policies.act.sampling import observation_probabilities
    settings=resolve_settings(archive/'frozen_sources/ExcavatorVLA-vlm/configs/act/full_phw.yaml',dict(
        dataset=dict(root=protocol['dataset']),benchmark=dict(split_file=str(out/'split.json'),validation_freq=0),
        launcher=dict(gpu_ids=[0],cpu_threads=4,hf_home=str(ROOT/'models/hf-cache'),torch_home=str(ROOT/'models/torch-cache')),
        continuation=dict(checkpoint=str(source)),steps=400,batch_size=16,num_workers=2,save_freq=400,log_freq=50,
        output_dir=str(out/'unused')))
    report,actual_split,frames,episodes=audit_prepare(settings)
    if actual_split!=split:
        raise ValueError('Frozen split differs from source checkpoint')
    write(out/'dataset_contract.json',report)
    chosen=frames[frames.episode_index.isin(episodes) & frames['observation.stage_current_id'].isin([5,6])]
    windows=dict(labels_changed=False,training_episodes=episodes,windows=[dict(episode_index=int(r.episode_index),
        observation_frame=int(r.frame_index)) for r in chosen.itertuples()])
    write(out/'focused_observations.json',windows)
    settings['sampling']=dict(manifest=str(out/'focused_observations.json'),transition_probability=.3)
    write(out/'settings.json',settings)
    training=frames[frames.episode_index.isin(episodes)]
    keys=list(zip(training.episode_index,training.frame_index,strict=True))
    probabilities,_=observation_probabilities(keys,episodes,windows,.3)
    stages=training['observation.stage_current_id'].to_numpy(dtype=int)
    mass=np.bincount(stages,weights=probabilities,minlength=10)
    if (mass<=0).any():
        raise ValueError('Missing training stage')
    weights=1/np.sqrt(mass);weights/=float(np.dot(mass,weights))
    protocol['stage_class_weights']=weights.tolist()
    protocol['expected_observation_stage_probability']=mass.tolist()
    write(out/'protocol.json',protocol)
    # Reuse the fixed, already declared diagnostic identities, including all
    # conversion validation frames. They never enter either training sampler.
    observations=read(ROOT/'outputs/experiments/act_head_probe_20261105_02/protocol.json')['observations']
    for row in observations:
        if row['split'] not in ('train','validation') or row['episode_index'] not in split[row['split']]:
            raise ValueError('Invalid offline observation split')
    write(out/'offline_observations.json',dict(observations=observations,test_used=False))
    (out/'references').mkdir()
    origin=ROOT/manifest['original_experiment']/'evaluation/episodes'
    template=read(ROOT/'outputs/experiments/act_phw_optimization_20261103_01/reference_setup.json')
    for scene in protocol['development_scenes']:
        folder=origin/f'episode_{scene:04d}'
        context=read(folder/'scene_context.json');initial=read(folder/'initial_observation.json')
        reference=dict(template)
        reference.update(profile=context['scene_profile'],task=initial['task'],
            initial_joints_deg=[math.degrees(v) for v in initial['state'][:4]],
            expected_initial_state=initial['state'],dig_target_xyz=context['observation_context']['dig_target_xyz'],
            sand_amount=context['scene_profile']['sand_amount_multiplier'],
            actions=[json.loads(line)['predicted'] for line in (folder/'steps.jsonl').read_text().splitlines()],
            provenance=dict(source_episode=str(folder),scene_sha256=sha(folder/'scene_context.json'),
                purpose='Recorded model development geometry. Stored model actions are never replayed or used as expert labels.'))
        write(out/'references'/f'scene{scene}.json',reference)
    protected=[*source.glob('*'),*Path(protocol['dataset']).glob('data/**/*.parquet'),
        *Path(protocol['dataset']).glob('meta/**/*.parquet'),Path(protocol['dataset'])/'meta/info.json',
        *out.glob('*.json'),*out.rglob('*.py'),*out.glob('references/*.json')]
    protected={str(p):sha(p) for p in protected if p.is_file() and p.name not in ('status.json','inputs_sha256.json')}
    write(out/'inputs_sha256.json',protected)
    write(out/'status.json',dict(status='prepared',steps_per_arm=400))
    return protocol


def controller(out):
    protocol=prepare(out)
    children=[]; logs=[]
    state=dict(status='training',training={})
    def stop(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for arm in protocol['arms']:
            log=(out/(arm+'.log')).open('x');logs.append(log)
            command=[sys.executable,'-u',str(out/'frozen_scripts/run_act_stage_auxiliary.py'),
                     '--output',str(out),'--worker',arm]
            child=subprocess.Popen(command,cwd=out,env=dict(os.environ,ACT_PROJECT_ROOT=str(ROOT)),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((arm,child))
            state['training'][arm]=dict(pid=child.pid,gpu=protocol['arms'][arm]['gpu'],command=command)
        while any(child.poll() is None for _,child in children):
            for arm,child in children:
                state['training'][arm]['returncode']=child.poll()
                progress=out/'train'/arm/'status.json'
                if progress.exists():state['training'][arm]['progress']=read(progress)
                if child.poll() not in (None,0):raise RuntimeError(f'{arm} failed; see {out/(arm+".log")}')
            write(out/'status.json',state)
            time.sleep(5)
        training={arm:read(out/'train'/arm/'result.json') for arm,_ in children}
        if len({v['sample_order_sha256'] for v in training.values()})!=1:
            raise ValueError('Training arms saw different sample order')
        write(out/'training_results.json',training)
        state['status']='offline_evaluation';write(out/'status.json',state)
        advance=offline(out)
        state.update(status='offline_passed' if advance else 'completed_no_advance',advance=advance,
                     training_complete=True,source_inputs_unchanged=True)
        write(out/'status.json',state)
        if advance:closed_loop(out,state)
        report(out)
        print(json.dumps(dict(status=state['status'],advance=advance)),flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error));write(out/'status.json',state)
        raise
    finally:
        for _,child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        for _,child in children:
            if child.poll() is None:
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
        for log in logs:log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',choices=['control','stage_aux'],help=argparse.SUPPRESS)
    args=parser.parse_args()
    out=args.output.resolve()
    if args.worker:worker(out,args.worker)
    else:controller(out)


if __name__=='__main__':
    main()
