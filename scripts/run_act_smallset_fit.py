"""Bounded, train-only ACT fit diagnostic; never saves deployable weights."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(os.environ.get('ACT_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
sys.path.insert(0, str(ROOT / 'ExcavatorVLA-vlm'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/train/'
        'excavator_act_phw_20261028_150222_336724_1423224/checkpoints/040000/pretrained_model')
    parser.add_argument('--probe', type=Path, default=ROOT / 'outputs/analysis/'
        'act_incremental_20261103_01/transition_inference_probe.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/export_phw')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.steps <= 200:
        parser.error('This diagnostic is capped at 200 optimizer steps')
    args.output.mkdir(parents=True, exist_ok=False)
    if Path(__file__).resolve() != (args.output / Path(__file__).name).resolve():
        shutil.copyfile(__file__, args.output / Path(__file__).name)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', HF_HOME=str(ROOT / 'models/hf-cache'),
        HF_DATASETS_CACHE=str(ROOT / 'models/hf-cache/datasets'),
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')

    import numpy as np
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import default_collate
    from excavator_vlm.data.contract import read_frames, STATE_NAMES
    from excavator_vlm.policies.act.adapter import training_batch, weighted_l1
    from excavator_vlm.policies.act.dataset import make_act_dataset
    from excavator_vlm.policies.act.inference import load_act
    from lerobot.optim.optimizers import AdamWConfig

    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.monotonic()
    checkpoint = args.checkpoint.resolve()
    probe = json.loads(args.probe.read_text())
    provenance = json.loads((checkpoint / 'benchmark_provenance.json').read_text())
    split = provenance['split_manifest']
    records = [row for row in probe['records'] if row['split'] == 'train']
    identities = [(row['episode_index'], row['frame_index']) for row in records]
    if len(identities) != 16 or len(set(identities)) != 16:
        raise ValueError('Expected the same 16 distinct train observations from the previous probe')
    if any(ep not in split['train'] or ep in split['validation'] + split['test'] for ep, _ in identities):
        raise ValueError('Held-out observation in fit subset')
    if digest(checkpoint / 'model.safetensors') != probe['checkpoint_sha256']:
        raise ValueError('Source checkpoint differs from the previous read-only probe')

    # Record the exact loaded code, data, and processor/checkpoint files. Inputs
    # are checked again at exit; no output checkpoint is ever written.
    import excavator_vlm.policies.act.adapter as adapter
    import excavator_vlm.policies.act.dataset as dataset_module
    import excavator_vlm.policies.act.inference as inference
    import lerobot.policies.act.modeling_act as modeling
    import lerobot.optim.optimizers as optimizers
    source_paths = [Path(inspect.getfile(module)).resolve()
                    for module in (adapter, dataset_module, inference, modeling, optimizers)]
    frozen_files = [*sorted(checkpoint.glob('*')), args.probe.resolve(),
                    *sorted(args.data.glob('data/**/*.parquet')),
                    *sorted(args.data.glob('meta/**/*.parquet')),
                    args.data / 'meta/info.json', *source_paths]
    before = {str(path): digest(path) for path in frozen_files if path.is_file()}
    policy, pre, post = load_act(checkpoint, 'cuda')
    expected = load_file(str(checkpoint / 'model.safetensors'))
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
    del expected
    saved_training = json.loads((checkpoint / 'train_config.json').read_text())
    optimizer_settings = saved_training['optimizer'].copy()
    if optimizer_settings.pop('type') != 'adamw' or saved_training['scheduler'] is not None:
        raise ValueError('Expected original AdamW preset and no scheduler')
    optimizer = AdamWConfig(**optimizer_settings).build(policy.get_optim_params())
    optimized = {id(param) for group in optimizer.param_groups for param in group['params']}
    trainable = {id(param) for param in policy.parameters() if param.requires_grad}
    if optimized != trainable:
        raise ValueError('Trainable parameters missing from optimizer')

    frames = read_frames(args.data)
    indexed = frames.set_index(['episode_index', 'frame_index'])
    dataset = make_act_dataset('local/excavator', args.data.resolve(), split['train'], 10)
    identity = dataset.hf_dataset.select_columns(['episode_index', 'frame_index']).to_pandas()
    lookup = {(int(ep), int(frame)): i for i, (ep, frame) in
              enumerate(zip(identity.episode_index, identity.frame_index))}
    samples = [dataset[lookup[key]] for key in identities]
    raw = default_collate(samples)
    processed = training_batch(pre, raw)
    padding = processed['action_is_pad']
    supervision = processed['action_loss_weight']
    if supervision.ndim == 3:
        supervision = supervision.squeeze(-1)
    weights = (~padding).float() * supervision
    target = raw['action'].numpy()
    roundtrip = post(processed['action'].reshape(-1, 4)).reshape_as(raw['action']).cpu()
    torch.testing.assert_close(roundtrip, raw['action'], rtol=1e-5, atol=1e-6)
    audit_records, future_stages = [], []
    alignment_errors = dict(action=0., weight=0., state=0.)
    for i, ((ep, frame), record) in enumerate(zip(identities, records)):
        episode = frames[frames.episode_index == ep].sort_values('frame_index')
        indices = np.minimum(np.arange(frame, frame + 10), len(episode) - 1)
        expected_action = np.stack(episode.iloc[indices]['action'])
        expected_weight = np.asarray(episode.iloc[indices]['action_loss_weight'].tolist()).reshape(-1)
        expected_padding = np.arange(frame, frame + 10) >= len(episode)
        # Pandas and HF's Arrow conversion differ at float64 roundoff (~1e-15).
        # A 1e-12 absolute tolerance still rejects any frame/dtype-scale mismatch.
        np.testing.assert_allclose(target[i], expected_action, rtol=0, atol=1e-12)
        alignment_errors['action'] = max(alignment_errors['action'],
            float(np.abs(target[i] - expected_action).max()))
        np.testing.assert_allclose(raw['action_loss_weight'][i].numpy().reshape(-1),
                                   expected_weight, rtol=0, atol=1e-12)
        alignment_errors['weight'] = max(alignment_errors['weight'],
            float(np.abs(raw['action_loss_weight'][i].numpy().reshape(-1) - expected_weight).max()))
        np.testing.assert_array_equal(padding[i].cpu().numpy(), expected_padding)
        np.testing.assert_allclose(target[i, 0], record['target'], rtol=0, atol=0)
        expected_state = indexed.loc[(ep, frame), 'observation.state']
        # State features are declared float32 by the dataset contract, while
        # parquet's list values are stored as doubles. Compare after that cast.
        np.testing.assert_allclose(raw['observation.state'][i].numpy(),
            expected_state.astype(raw['observation.state'][i].numpy().dtype), rtol=0, atol=1e-12)
        alignment_errors['state'] = max(alignment_errors['state'],
            float(np.abs(raw['observation.state'][i].numpy() - expected_state).max()))
        stages = episode.iloc[indices]['observation.stage_current_id'].to_numpy(dtype=int)
        if stages[0] != record['stage_id']:
            raise ValueError('Selected observation stage changed')
        future_stages.append(stages)
        audit_records.append(dict(episode_index=ep, frame_index=frame, stage_id=int(stages[0]),
            future_stage_ids=stages.tolist(), target_chunk=target[i].tolist(),
            raw_weights=expected_weight.tolist(), padding=expected_padding.tolist(),
            effective_weight=float(weights[i].sum()),
            normalized_state=processed['observation.state'][i].cpu().tolist()))
    future_stages = np.stack(future_stages)
    weight_np = weights.cpu().numpy()
    stage_ids = np.asarray([row['stage_id'] for row in records])
    stage_mass = {str(stage): float(weight_np[future_stages == stage].sum() / weight_np.sum())
                  for stage in np.unique(future_stages)}

    # Nearby states can still be distinguished by images. Report distances and
    # label disagreement; do not infer irreducible ambiguity from state alone.
    state = processed['observation.state'].cpu().numpy()
    distances = np.sqrt(((state[:, None] - state[None]) ** 2).mean(-1))
    np.fill_diagonal(distances, np.inf)
    neighbors = []
    for i in range(len(records)):
        j = int(distances[i].argmin())
        image_mse = {key: float(((raw[key][i] - raw[key][j]) ** 2).mean())
                     for key in policy.config.image_features}
        neighbors.append(dict(first=identities[i], second=identities[j],
            normalized_state_rms_distance=float(distances[i, j]), image_mse=image_mse,
            first_action_abs_difference=(np.abs(target[i, 0] - target[j, 0])).tolist(),
            exact_same_model_inputs=bool(np.array_equal(state[i], state[j]) and
                                         all(value == 0 for value in image_mse.values()))))

    initial_tensors = {key: value.detach().clone() for key, value in policy.named_parameters()
                       if key in ('model.action_head.weight', 'model.action_head.bias',
                                  'model.encoder_robot_state_input_proj.weight',
                                  'model.backbone.0.body.conv1.weight')}
    payload = dict(status='prepared', scope='train-only fit diagnostic, not a deployable candidate '
        'or evidence of closed-loop/generalization improvement', checkpoint=str(checkpoint),
        checkpoint_sha256=probe['checkpoint_sha256'], source_probe=str(args.probe),
        settings=dict(steps=args.steps, observations=len(records), seed=args.seed, gpu=args.gpu,
            batch_size=16, precision='bf16 autocast training; float32 deterministic diagnostics',
            optimizer='original AdamW preset; fresh optimizer state', optimizer_settings=optimizer_settings,
            kl_weight=policy.config.kl_weight, dropout=policy.config.dropout,
            chunk_size=10, n_action_steps=3, normalization='saved original processors, unchanged',
            selection='all 16 train observations of the previous 32-observation probe',
            checkpoint_saved=False, pass_reduction_threshold=0.5),
        audit=dict(exact_initial_weights=True, action_chunks_and_weights_match_parquet=True,
            max_parquet_loader_abs_differences=alignment_errors,
            padding_matches_episode_boundaries=True, normalization_roundtrip=True,
            heldout_observations_used=False, optimizer_covers_all_trainable_parameters=True,
            supervised_action_fraction=float((weights > 0).float().mean()),
            total_effective_weight=float(weights.sum()), future_stage_weight_fraction=stage_mass,
            normalized_state_feature_names=STATE_NAMES,
            min_nonself_state_distance=float(distances.min())),
        inputs_sha256=before, subset=audit_records, nearest_state_pairs=neighbors,
        evaluations=[], gradients=[], updates=[])
    write(args.output / 'result.json', payload)

    def metrics(prediction):
        absolute = post(prediction.reshape(-1, 4)).reshape_as(prediction).float().cpu().numpy()
        error = np.abs(absolute - target)
        normalized, _ = weighted_l1(prediction, processed['action'], padding,
                                     processed['action_loss_weight'])
        result = dict(weighted_normalized_l1=float(normalized),
            weighted_chunk_mae_rad_s=float((error * weight_np[..., None]).sum() / (4 * weight_np.sum())),
            first_action_mae_rad_s=error[:, 0].mean(0).tolist(),
            first_three_action_mae_rad_s=error[:, :3].mean((0, 1)).tolist(),
            first_action_predictions=absolute[:, 0].tolist())
        for stage in (5, 6):
            mask = stage_ids == stage
            result[f'observation_stage{stage}'] = dict(
                first_action_mae_rad_s=error[mask, 0].mean(0).tolist(),
                bucket_first_mae_rad_s=float(error[mask, 0, 3].mean()),
                bucket_first_prediction_mean_abs_rad_s=float(np.abs(absolute[mask, 0, 3]).mean()),
                bucket_first_target_mean_abs_rad_s=float(np.abs(target[mask, 0, 3]).mean()),
                first_three_mae_rad_s=float(error[mask, :3].mean()),
                chunk_mae_rad_s=float(error[mask].mean()))
        return result

    def evaluate(step):
        policy.eval()
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            torch.manual_seed(12345)
            prior = policy.predict_action_chunk(processed)
            model_batch = {**processed, 'observation.images':
                           [processed[key] for key in policy.config.image_features]}
            policy.model.training = True
            try:
                with patch('torch.randn_like', side_effect=lambda value: torch.zeros_like(value)):
                    reconstructed, (mu, logvar) = policy.model(model_batch)
                posterior = metrics(reconstructed)
                kl = float((-.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean())
            finally:
                policy.model.training = False
            row = dict(step=step, elapsed_seconds=time.monotonic() - started,
                       prior=metrics(prior), posterior_mean=posterior, posterior_kl=kl)
        payload['evaluations'].append(row)
        payload['completed_steps'] = step
        payload['status'] = 'running'
        write(args.output / 'result.json', payload)
        print(json.dumps(dict(step=step, elapsed_seconds=row['elapsed_seconds'],
            prior_l1=row['prior']['weighted_normalized_l1'],
            posterior_l1=posterior['weighted_normalized_l1'],
            stage6_bucket_mae=row['prior']['observation_stage6']['bucket_first_mae_rad_s'])), flush=True)

    def parameter_group(name):
        if 'vae_encoder' in name:
            return 'vae_encoder'
        for prefix in ('backbone', 'action_head', 'encoder_robot_state_input_proj', 'encoder', 'decoder'):
            if name.startswith('model.' + prefix):
                return prefix
        return 'other'

    evaluate(0)
    # Verify baseline consistency across CPU/GPU and the previous tiny probe.
    np.testing.assert_allclose(payload['evaluations'][0]['prior']['first_action_predictions'],
        np.asarray([row['prior'] for row in records]), rtol=1e-3, atol=1e-4)
    with (args.output / 'train.jsonl').open('x') as log:
        for step in range(1, args.steps + 1):
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            inspect_gradient = step in (1, 100, args.steps)
            captured = []

            def capture(module, inputs, output):
                output[0].retain_grad()
                captured.append(output[0])

            handle = policy.model.register_forward_hook(capture) if inspect_gradient else None
            try:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, parts = policy(processed)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step}')
                loss.backward()
            finally:
                if handle is not None:
                    handle.remove()
            if inspect_gradient:
                prediction = captured[0]
                expected_grad = (prediction.detach().float() - processed['action']).sign()
                expected_grad *= weights.unsqueeze(-1) / (weights.sum() * 4)
                torch.testing.assert_close(prediction.grad.float(), expected_grad, rtol=.01, atol=1e-6)
                groups = {}
                for name, param in policy.named_parameters():
                    if not param.requires_grad:
                        continue
                    group = groups.setdefault(parameter_group(name), dict(parameters=0,
                        elements=0, elements_with_gradient=0, nonzero_gradient_elements=0,
                        gradient_squared_norm=0., missing_gradients=[]))
                    group['parameters'] += 1
                    group['elements'] += param.numel()
                    if param.grad is None:
                        group['missing_gradients'].append(name)
                        continue
                    grad = param.grad.detach().float()
                    if not torch.isfinite(grad).all():
                        raise FloatingPointError(f'Nonfinite gradient: {name}')
                    group['elements_with_gradient'] += grad.numel()
                    group['nonzero_gradient_elements'] += int(torch.count_nonzero(grad))
                    group['gradient_squared_norm'] += float(grad.square().sum())
                for group in groups.values():
                    group['gradient_norm'] = group.pop('gradient_squared_norm') ** .5
                payload['gradients'].append(dict(step=step, groups=groups,
                    output_gradient_matches_effective_weights=True,
                    output_gradient_abs_sum_per_joint=prediction.grad.abs().sum((0, 1)).float().cpu().tolist()))
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), optimizer_settings['grad_clip_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError(f'Nonfinite gradient norm at step {step}')
            optimizer.step()
            log.write(json.dumps(dict(step=step, **parts, grad_norm=float(norm),
                lr=optimizer.param_groups[0]['lr'], elapsed_seconds=time.monotonic() - started)) + '\n')
            log.flush()
            if step in (25, 50, 100, args.steps):
                evaluate(step)

    for name, initial in initial_tensors.items():
        change = dict(policy.named_parameters())[name].detach() - initial
        payload['updates'].append(dict(parameter=name, max_abs_update=float(change.abs().max()),
            l2_update=float(change.norm()), changed_elements=int(torch.count_nonzero(change))))
    after = {path: digest(path) for path in before}
    if after != before:
        raise RuntimeError('Source input files changed while the diagnostic was running')
    baseline, final = payload['evaluations'][0], payload['evaluations'][-1]
    reductions = {kind: 1 - final[kind]['weighted_normalized_l1'] /
                   baseline[kind]['weighted_normalized_l1'] for kind in ('prior', 'posterior_mean')}
    payload.update(status='completed', elapsed_seconds=time.monotonic() - started,
        inputs_unchanged=True, weighted_normalized_l1_reduction=reductions,
        pass_50_percent_train_fit=reductions['posterior_mean'] >= .5,
        pass_50_percent_prior_fit=reductions['prior'] >= .5,
        weights_discarded=True, checkpoint_saved=False)
    write(args.output / 'result.json', payload)
    print(json.dumps(dict(status='completed', reductions=reductions,
        elapsed_seconds=payload['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
