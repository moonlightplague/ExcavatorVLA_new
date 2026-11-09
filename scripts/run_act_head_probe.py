"""Bounded action-head diagnostic on frozen ACT features; never saves a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', HF_HOME=str(ROOT / 'models/hf-cache'),
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    manifest_path = ROOT / 'docs/baselines/act_phw40k_carry275_20261105.json'
    manifest = json.loads(manifest_path.read_text())
    archive = ROOT / manifest['archive']
    sys.path[:0] = [str(archive / 'frozen_sources/ExcavatorVLA-vlm'),
                    str(archive / 'runtime_dependencies')]
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from excavator_vlm.data.contract import read_frames
    from excavator_vlm.policies.act.adapter import training_batch, weighted_l1
    from excavator_vlm.policies.act.dataset import make_act_dataset
    from excavator_vlm.policies.act.inference import load_act

    torch.set_num_threads(4)
    torch.manual_seed(1000)
    np.random.seed(1000)
    started = time.monotonic()
    source = archive / manifest['checkpoint']
    for name, expected in manifest['files_sha256'].items():
        if name.startswith(('checkpoints/', 'frozen_sources/ExcavatorVLA-vlm/excavator_vlm/',
                            'runtime_dependencies/lerobot/')) and digest(archive / name) != expected:
            raise ValueError(f'Preserved model/source changed: {name}')
    split = json.loads((source / 'benchmark_provenance.json').read_text())['split_manifest']
    if any(set(split[a]) & set(split[b]) for a, b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Episode split overlaps')
    data = Path(manifest['external_dependencies']['dataset'])
    protected = {str(p): digest(p) for p in sorted(data.glob('data/**/*.parquet'))}
    protected[str(data / 'meta/info.json')] = digest(data / 'meta/info.json')
    frames = read_frames(data)
    observations = []
    for group in ('train', 'validation'):
        group_frames = frames[frames.episode_index.isin(split[group])]
        for stage in range(10):
            chosen = group_frames[group_frames['observation.stage_current_id'] == stage]
            if stage not in (5, 6) and len(chosen) > 32:
                chosen = chosen.iloc[np.linspace(0, len(chosen) - 1, 32, dtype=int)]
            observations.extend(dict(split=group, stage_id=stage,
                episode_index=int(row.episode_index), frame_index=int(row.frame_index))
                for row in chosen.itertuples())
    identities = [(r['split'], r['episode_index'], r['frame_index']) for r in observations]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate probe observation')
    for row in observations:
        if row['episode_index'] not in split[row['split']] or row['episode_index'] in split['test']:
            raise ValueError('Probe observation leaks held-out episodes')
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_id=manifest['baseline_id'], source_checkpoint=str(source),
        model_sha256=manifest['model_sha256'], scope='Local action-head fit/validation diagnostic; '
        'not full-dataset fine-tuning or closed-loop evidence. No deployable weights saved.',
        fixed_settings=dict(steps=200, seed=1000, gpu=args.gpu, learning_rate=1e-4,
            optimizer='AdamW', weight_decay=1e-4, betas=[.9,.999], eps=1e-8,
            precision='float32', features='frozen original model, eval mode, zero latent, no dropout',
            trainable='Only the 2052 parameters of the final linear action head',
            target_observations='All train stages 5/6; original expert chunks, padding and PHW weights',
            control_observations='Up to 32 evenly spaced train observations per other stage',
            observation_mass='50% transitions, 50% controls before future-action PHW weights',
            arms=dict(expert_controls='Expert labels on both transitions and controls',
                      teacher_controls='Expert labels on transitions; original model output on controls'),
            final_step_only_selection=True, checkpoints_saved=False),
        advance_gate=dict(stage6_first_bucket_mae_ratio_max=.9,
            stages5_and6_first_three_mae_ratio_max=.9,
            each_target_stage_first_three_mae_ratio_max=1.05,
            each_control_stage_first_three_mae_allowed='max(baseline * 1.05, baseline + 0.002 rad/s)',
            stage6_first_bucket_amplitude_min_fraction_of_expert=.2,
            if_pass='Separately predeclare full-dataset control/candidate and a small paired closed-loop check',
            if_fail='Stop this action-head-only approach; do not add steps or change thresholds'),
        observations=observations, test_used=False, inputs_sha256=protected)
    write(out / 'protocol.json', protocol)
    write(out / 'status.json', dict(status='extracting_features', observations=len(observations)))
    policy, pre, post = load_act(source, 'cuda')
    head = policy.model.action_head
    if sum(p.numel() for p in head.parameters()) != 2052:
        raise ValueError('Unexpected action head architecture')
    for param in policy.parameters():
        param.requires_grad_(False)
    features, targets, paddings, weights, teachers = [], [], [], [], []
    indexed = frames.set_index(['episode_index', 'frame_index'])
    checked = 0
    for group in ('train', 'validation'):
        chosen = [r for r in observations if r['split'] == group]
        dataset = make_act_dataset('local/excavator', data, split[group], 10)
        ids = dataset.hf_dataset.select_columns(['episode_index','frame_index']).to_pandas()
        lookup = {(int(ep), int(frame)): i for i, (ep, frame) in
                  enumerate(zip(ids.episode_index, ids.frame_index))}
        for offset in range(0, len(chosen), 16):
            current = chosen[offset:offset + 16]
            raw = default_collate([dataset[lookup[(r['episode_index'],r['frame_index'])]] for r in current])
            # Independently align actions, original PHW supervision and padding
            # with parquet before they enter any head fit.
            for i, row in enumerate(current):
                ep = row['episode_index']; frame = row['frame_index']
                episode = frames[frames.episode_index == ep].sort_values('frame_index')
                positions = np.minimum(np.arange(frame, frame + 10), len(episode) - 1)
                expected = episode.iloc[positions]
                # The dataset contract converts numeric features to their
                # declared tensor dtype. Match that conversion before doing
                # an exact check against parquet's stored list values.
                raw_action = raw['action'][i].numpy()
                np.testing.assert_array_equal(raw_action, np.stack(expected.action).astype(raw_action.dtype))
                raw_weight = raw['action_loss_weight'][i].numpy().reshape(-1)
                np.testing.assert_array_equal(raw_weight,
                    np.asarray(expected.action_loss_weight.tolist()).reshape(-1).astype(raw_weight.dtype))
                raw_state = raw['observation.state'][i].numpy()
                np.testing.assert_array_equal(raw_state,
                    indexed.loc[(ep,frame),'observation.state'].astype(raw_state.dtype))
                np.testing.assert_array_equal(raw['action_is_pad'][i].numpy(), np.arange(frame, frame + 10) >= len(episode))
                if int(indexed.loc[(ep,frame),'observation.stage_current_id']) != row['stage_id']:
                    raise ValueError('Observation stage differs from parquet')
                checked += 1
            batch = training_batch(pre, raw)
            captured = []
            hook = head.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().cpu()))
            try:
                with torch.inference_mode():
                    prediction = policy.predict_action_chunk({k:v for k,v in batch.items()
                        if k == 'observation.state' or k.startswith('observation.images.')})
                    features.append(captured[0])
                    teachers.append(prediction.cpu())
            finally:
                hook.remove()
            targets.append(batch['action'].cpu())
            paddings.append(batch['action_is_pad'].cpu())
            weights.append(batch['action_loss_weight'].cpu())
            if offset % 128 == 0:
                print(json.dumps(dict(split=group, decoded=min(offset + 16,len(chosen)),
                                      total=len(chosen),elapsed_seconds=time.monotonic()-started)),flush=True)
                write(out / 'status.json', dict(status='extracting_features', decoded=checked,
                                                observations=len(observations)))
    x, y, pad, supervision, teacher = [torch.cat(v).to('cuda') for v in
                                     (features, targets, paddings, weights, teachers)]
    if supervision.ndim == 3:
        supervision = supervision.squeeze(-1)
    del features, targets, paddings, weights, teachers
    with torch.no_grad():
        torch.testing.assert_close(head(x), teacher, rtol=1e-5, atol=1e-6)
    train = np.array([r['split'] == 'train' for r in observations])
    stages = np.array([r['stage_id'] for r in observations])
    transition = np.isin(stages, [5,6])
    probability = np.zeros(len(observations), dtype=np.float32)
    for mask in (train & transition, train & ~transition):
        probability[mask] = .5 / mask.sum()
    train_ids = torch.as_tensor(np.flatnonzero(train), device='cuda')
    effective = supervision[train_ids] * torch.as_tensor(probability[train], device='cuda')[:,None]
    target_raw = post(y.reshape(-1,4)).reshape_as(y).float().cpu().numpy()
    train_features = x[train_ids].detach().clone()
    train_pad = pad[train_ids].detach().clone()
    original = {name:value.detach().clone() for name,value in head.state_dict().items()}
    initial_policy = {name:value.detach().cpu().clone() for name,value in policy.state_dict().items()}

    def metrics(prediction):
        with torch.no_grad():
            predicted = post(prediction.reshape(-1,4)).reshape_as(prediction).float().cpu().numpy()
        errors = np.abs(predicted - target_raw)
        result = {}
        for group in ('train','validation'):
            group_mask = np.array([r['split'] == group for r in observations])
            result[group] = {}
            for stage in range(10):
                ids = group_mask & (stages == stage)
                valid = ~pad[ids].cpu().numpy()
                err = errors[ids]
                first3 = valid[:,:3]
                result[group][str(stage)] = dict(observations=int(ids.sum()),
                    first_bucket_mae_rad_s=float(err[:,0,3].mean()),
                    first_bucket_predicted_mean_abs_rad_s=float(np.abs(predicted[ids,0,3]).mean()),
                    first_bucket_expert_mean_abs_rad_s=float(np.abs(target_raw[ids,0,3]).mean()),
                    first_three_mae_rad_s=float((err[:,:3] * first3[:,:,None]).sum() / (first3.sum()*4)))
            ids = group_mask & transition
            valid = ~pad[ids,:3].cpu().numpy()
            result[group]['transitions'] = dict(observations=int(ids.sum()),
                first_three_mae_rad_s=float((errors[ids,:3] * valid[:,:,None]).sum() / (valid.sum()*4)))
        return result, predicted

    baseline, baseline_prediction = metrics(teacher)
    results = dict(scope=protocol['scope'], baseline=baseline, arms={},
                   audit=dict(parquet_aligned_observations=checked, cached_head_matches_original=True,
                              test_used=False, source_weights_unchanged=True))
    saved_predictions = dict(baseline=baseline_prediction, target=target_raw)
    for arm in ('expert_controls','teacher_controls'):
        local_head = torch.nn.Linear(512,4,device='cuda')
        local_head.load_state_dict(original)
        optimizer = torch.optim.AdamW(local_head.parameters(),lr=1e-4,weight_decay=1e-4)
        train_target = y[train_ids].detach().clone()
        if arm == 'teacher_controls':
            control = torch.as_tensor(~transition[train],device='cuda')
            train_target[control] = teacher[train_ids][control]
        write(out / 'status.json', dict(status='fitting_head',arm=arm,steps=200))
        with (out / f'{arm}.jsonl').open('x') as log:
            for step in range(1,201):
                optimizer.zero_grad(set_to_none=True)
                predicted = local_head(train_features)
                loss, _ = weighted_l1(predicted,train_target,train_pad,effective)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite head loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(local_head.parameters(),10.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite head gradient')
                optimizer.step()
                if step in (1,50,100,200):
                    log.write(json.dumps(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm)))+'\n')
        with torch.no_grad():
            actual, predicted_raw = metrics(local_head(x))
        val, base = actual['validation'],baseline['validation']
        gate = dict(
            stage6_bucket=val['6']['first_bucket_mae_rad_s'] <= .9*base['6']['first_bucket_mae_rad_s'],
            transitions=val['transitions']['first_three_mae_rad_s'] <= .9*base['transitions']['first_three_mae_rad_s'],
            target_stages=all(val[str(s)]['first_three_mae_rad_s'] <= 1.05*base[str(s)]['first_three_mae_rad_s'] for s in (5,6)),
            controls=all(val[str(s)]['first_three_mae_rad_s'] <= max(1.05*base[str(s)]['first_three_mae_rad_s'],
                          base[str(s)]['first_three_mae_rad_s']+.002) for s in range(10) if s not in (5,6)),
            stage6_amplitude=val['6']['first_bucket_predicted_mean_abs_rad_s'] >= .2*val['6']['first_bucket_expert_mean_abs_rad_s'])
        results['arms'][arm] = dict(metrics=actual,gate=gate,advance=all(gate.values()),
            head_update_l2=float(sum((v-original[k]).square().sum() for k,v in local_head.state_dict().items()).sqrt()))
        saved_predictions[arm] = predicted_raw
        print(json.dumps(dict(arm=arm,gate=gate,advance=all(gate.values()),
                              stage6_bucket_mae=val['6']['first_bucket_mae_rad_s'])),flush=True)
    for name,value in policy.state_dict().items():
        torch.testing.assert_close(value.cpu(), initial_policy[name],rtol=0,atol=0)
    if any(digest(Path(name)) != expected for name,expected in protected.items()):
        raise ValueError('Original training inputs changed during diagnostic')
    if digest(source / 'model.safetensors') != manifest['model_sha256']:
        raise ValueError('Preserved source checkpoint changed')
    np.savez_compressed(out / 'offline_predictions.npz', **saved_predictions)
    results.update(status='completed', elapsed_seconds=time.monotonic()-started,
                   checkpoints_saved=False, probe_weights_discarded=True,inputs_unchanged=True)
    write(out / 'result.json',results)
    write(out / 'status.json',dict(status='completed',elapsed_seconds=results['elapsed_seconds'],
        advance_arms=[name for name,r in results['arms'].items() if r['advance']],checkpoints_saved=False))
    print(json.dumps(dict(status='completed',elapsed_seconds=results['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()
