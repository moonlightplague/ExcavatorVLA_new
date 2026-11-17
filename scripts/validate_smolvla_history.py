"""Real-data, real-model GPU gate before full SmolVLA history training."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import torch
from torch.utils.data import default_collate

from train_smolvla import build_train_config, install_trainer_adapters, load_settings
from smolvla_history import bind_history, inference_batch, load_history, window_mask


def physical_actions(policy, pre, post, raw, noise):
    policy.reset()
    chunk = policy.predict_action_chunk(pre(inference_batch(raw)), noise=noise)
    return torch.stack([post(chunk[:, i]) for i in range(chunk.shape[1])], dim=1).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'status':'running'})+'\n')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    settings, launcher = load_settings(args.config)
    os.environ['HF_HOME'] = launcher['hf_home']
    cfg = build_train_config(settings)
    cfg.validate()
    bind_history(cfg.policy, launcher['observation_history'])
    prepared = Path(launcher['prepared_data'])
    stats = json.loads((prepared / 'normalization_stats.json').read_text())
    from lerobot.scripts import lerobot_train as trainer
    install_trainer_adapters(trainer, cfg, launcher['observation_history'], stats)
    cfg.dataset.episodes = cfg.dataset.episodes[:2]
    dataset = trainer.make_dataset(cfg)
    item, first = dataset[20], dataset[0]
    raw = default_collate([item])
    start = default_collate([first])
    if raw['action'].shape != (1,10,4) or raw['observation.state'].shape != (1,3,28):
        raise AssertionError('Incorrect real history/action tensor dimensions')
    if not torch.equal(window_mask(start, launcher['observation_history']), torch.tensor([[True,True,False]])):
        raise AssertionError('Episode-start padding differs from PI05')
    for time_index, frame_index in enumerate((14,17,20)):
        comparison=dataset[frame_index]
        for key in ('observation.state', *launcher['observation_history']['cameras']):
            torch.testing.assert_close(item[key][time_index],comparison[key][-1],rtol=0,atol=0)
    torch.testing.assert_close(item['action'],torch.stack(dataset.hf_dataset[20:30]['action']),rtol=0,atol=0)
    boundary=int(dataset.meta.episodes[cfg.dataset.episodes[0]]['length'])
    second=default_collate([dataset[boundary]])
    if not torch.equal(window_mask(second,launcher['observation_history']),torch.tensor([[True,True,False]])):
        raise AssertionError('History crossed an episode boundary')
    policy = trainer.make_policy(cfg=cfg.policy, ds_meta=dataset.meta).to('cuda')
    policy.train()
    tensors = {key:{name:torch.tensor(value,dtype=torch.float32) for name,value in row.items()} for key,row in stats.items()}
    pre, post = trainer.make_pre_post_processors(policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={'device_processor':{'device':'cuda'},'normalizer_processor':{
            'features':{**cfg.policy.input_features,**cfg.policy.output_features},
            'norm_map':cfg.policy.normalization_mapping,'stats':tensors},
            'tokenizer_processor':{'max_length':cfg.policy.tokenizer_max_length}},
        postprocessor_overrides={'unnormalizer_processor':{'features':cfg.policy.output_features,
            'norm_map':cfg.policy.normalization_mapping,'stats':tensors}})
    results = []
    for size in (1,2,4,8,16):
        policy.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        processed = loss = None
        try:
            processed = pre(default_collate([item]*size))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,_ = policy(processed)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in policy.parameters()):
                raise FloatingPointError('Nonfinite loss/gradients in memory probe')
            torch.cuda.synchronize()
            row={'batch_size':size,'status':'passed','loss':loss.detach().item(),
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'seconds':time.monotonic()-started}
        except torch.cuda.OutOfMemoryError:
            row={'batch_size':size,'status':'cuda_out_of_memory','seconds':time.monotonic()-started}
        results.append(row)
        print(json.dumps(row),flush=True)
        del processed,loss
        policy.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        if row['status'] != 'passed':
            break
    passing=[row['batch_size'] for row in results if row['status']=='passed']
    if not passing:
        raise RuntimeError('Even microbatch 1 does not fit')
    processed=pre(raw)
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        images,masks=policy.prepare_images(processed)
        prefix=policy.model.embed_prefix(images,masks,processed['observation.language.tokens'],
            processed['observation.language.attention_mask'],policy.prepare_state(processed))[0]
        prefix_tokens=prefix.shape[1]
        max_positions=policy.model.vlm_with_expert.config.text_config.max_position_embeddings
        if prefix_tokens+cfg.policy.chunk_size > max_positions:
            raise RuntimeError('Nine-image prefix exceeds pretrained position budget')
    del prefix,images,masks,processed
    optimizer=cfg.optimizer.build(policy.get_optim_params())
    scheduler=cfg.scheduler.build(optimizer,cfg.steps)
    watched={name:p.detach().flatten()[:32].cpu().clone() for name,p in policy.named_parameters()
             if p.requires_grad and ('state_proj' in name or 'action_out_proj' in name)}
    frozen={name:p.detach().flatten()[:32].cpu().clone() for name,p in policy.named_parameters() if not p.requires_grad}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss,_=policy(pre(raw))
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),cfg.policy.optimizer_grad_clip_norm)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    changed=[name for name,p in policy.named_parameters() if name in watched and
             not torch.equal(watched[name],p.detach().flatten()[:32].cpu())]
    if not changed or any(not torch.equal(frozen[name],p.detach().flatten()[:32].cpu())
            for name,p in policy.named_parameters() if name in frozen):
        raise RuntimeError('First update did not preserve frozen/change trainable sampled weights')
    policy.eval()
    noise=torch.randn(1,10,32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(20261116))
    expected=physical_actions(policy,pre,post,raw,noise)
    if expected.shape != (1,10,4) or not torch.isfinite(expected).all():
        raise AssertionError('Invalid physical action chunk')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    from lerobot.utils.train_utils import save_checkpoint, load_training_state
    with tempfile.TemporaryDirectory(prefix='smolvla_history_gate_',dir='/tmp') as temp:
        checkpoint=Path(temp)/'step_1'
        save_checkpoint(checkpoint,1,cfg,policy,optimizer,scheduler,pre,post)
        del policy,optimizer,scheduler,pre,post
        gc.collect();torch.cuda.empty_cache()
        policy,pre,post=load_history(checkpoint/'pretrained_model','cuda')
        actual=physical_actions(policy,pre,post,raw,noise)
        difference=(expected-actual).abs().max().item()
        if difference > 1e-5:
            raise AssertionError(f'Checkpoint inference roundtrip differs: {difference}')
        changed_raw={key:value.clone() if torch.is_tensor(value) else value for key,value in raw.items()}
        for key in launcher['observation_history']['cameras']:changed_raw[key][:,:2]=0
        influence=(actual-physical_actions(policy,pre,post,changed_raw,noise)).abs().max().item()
        if influence<=1e-6:raise AssertionError('Historical visual inputs have no observed influence')
        padded_expected=physical_actions(policy,pre,post,start,noise)
        for key in launcher['observation_history']['cameras']:start[key][:,:2]=1
        padding_difference=(padded_expected-physical_actions(policy,pre,post,start,noise)).abs().max().item()
        if padding_difference>1e-5:raise AssertionError('Padded historical pixels changed inference')
        optimizer=cfg.optimizer.build(policy.get_optim_params())
        scheduler=cfg.scheduler.build(optimizer,cfg.steps)
        step,optimizer,scheduler=load_training_state(checkpoint,optimizer,scheduler)
        if step!=1 or not optimizer.state or scheduler.last_epoch!=1:
            raise AssertionError('Optimizer/scheduler/step resume failed')
    from prepare_smolvla_history import file_hash
    report={'status':'passed','training_config_sha256':file_hash(args.config),
        'normalization_stats_sha256':file_hash(prepared/'normalization_stats.json'),
        'memory_probes':results,'recommended_micro_batch_size':max(passing),
        'logical_batch_per_gpu':cfg.batch_size,'history_shape':[1,3,3,256,256],
        'state_shape':[1,3,28],'effective_state_shape':[1,32],'action_shape':[1,10,4],
        'image_order':'time then cameras 0,1,2','visual_images':9,'prefix_tokens':prefix_tokens,
        'max_position_embeddings':max_positions,'finite_gradient_norm':float(norm),
        'changed_sampled_trainable_parameters':changed,'sampled_frozen_parameters_unchanged':True,
        'reload_action_max_abs_error_rad_s':difference,'resumed_optimizer_step':step,
        'valid_historical_image_influence_rad_s':influence,
        'masked_padding_image_influence_rad_s':padding_difference,
        'temporary_checkpoints_removed':True,'expert_only_initial_validation_not_task_success':True}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
