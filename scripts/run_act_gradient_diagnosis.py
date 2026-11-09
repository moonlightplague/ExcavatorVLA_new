"""Execute the predeclared, train-only ACT gradient diagnostic without updates."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(os.environ.get('ACT_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def parameter_group(name):
    if 'vae_encoder' in name:
        return 'vae_encoder'
    for prefix in ('backbone', 'action_head', 'encoder_robot_state_input_proj', 'encoder', 'decoder'):
        if name.startswith('model.' + prefix):
            return prefix
    return 'other'


def split_losses(policy, processed, target_stages):
    """Use original PHW normalization for both disjoint future-target partitions."""
    import torch
    from excavator_vlm.policies.act.adapter import weighted_l1

    model_batch = {**processed, 'observation.images':
                   [processed[key] for key in policy.config.image_features]}
    prediction, (mu, logvar) = policy.model(model_batch)
    original_l1, weights = weighted_l1(prediction, processed['action'],
                                       processed['action_is_pad'], processed['action_loss_weight'])
    error = (prediction.float() - processed['action'].float()).abs()
    denominator = (weights.sum() * prediction.shape[-1]).clamp_min(1)
    hold_weights = weights * (target_stages == 6)
    other_weights = weights * (target_stages != 6)
    hold = (error * hold_weights.unsqueeze(-1)).sum() / denominator
    other = (error * other_weights.unsqueeze(-1)).sum() / denominator
    active = (weights.sum(-1) > 0).float()
    kl = ((-.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1) * active).sum()
    kl = kl / active.sum().clamp_min(1)
    torch.testing.assert_close(hold + other, original_l1, rtol=1e-6, atol=1e-7)
    return hold, other, policy.config.kl_weight * kl, dict(
        original_l1=float(original_l1.detach()), hold_l1=float(hold.detach()),
        other_l1=float(other.detach()), kl=float(kl.detach()),
        effective_hold_weight=float(hold_weights.sum()),
        effective_other_weight=float(other_weights.sum()),
        denominator=float(denominator),
        total_loss=float((original_l1 + policy.config.kl_weight * kl).detach()))


def component_gradients(losses, named_parameters):
    import torch
    parameters = [parameter for _, parameter in named_parameters]
    return tuple(torch.autograd.grad(loss, parameters, retain_graph=i < len(losses) - 1,
                                     allow_unused=True) for i, loss in enumerate(losses))


def gradient_statistics(named_parameters, gradients):
    """Reduce three gradient vectors once on GPU; KL includes its original weight."""
    import torch
    groups = sorted({parameter_group(name) for name, _ in named_parameters})
    indices = {name: i for i, name in enumerate(groups)}
    gram = torch.zeros(len(groups), 3, 3, device=named_parameters[0][1].device)
    counts = {name: dict(trainable_elements=0, missing_gradient_parameters={
        key: [] for key in ('hold_l1', 'other_l1', 'weighted_kl')}) for name in groups}
    for index, (name, parameter) in enumerate(named_parameters):
        group = parameter_group(name)
        counts[group]['trainable_elements'] += parameter.numel()
        values = [gradient[index] for gradient in gradients]
        for key, value in zip(('hold_l1', 'other_l1', 'weighted_kl'), values):
            if value is None:
                counts[group]['missing_gradient_parameters'][key].append(name)
        for i, first in enumerate(values):
            if first is None:
                continue
            for j in range(i, 3):
                second = values[j]
                if second is not None:
                    value = torch.dot(first.float().reshape(-1), second.float().reshape(-1))
                    gram[indices[group], i, j] += value
                    if i != j:
                        gram[indices[group], j, i] += value
    if not torch.isfinite(gram).all():
        raise FloatingPointError('Nonfinite gradient Gram matrix')
    matrices = gram.cpu().double().numpy()

    def summarize(matrix):
        import math
        hold2, other2, kl2 = [max(0., float(matrix[i, i])) for i in range(3)]
        keep2 = max(0., other2 + kl2 + 2 * float(matrix[1, 2]))
        hold_other = float(matrix[0, 1])
        hold_keep = hold_other + float(matrix[0, 2])

        def cosine(dot, first2, second2):
            denominator = math.sqrt(first2 * second2)
            return float(dot / denominator) if denominator > 0 else None

        return dict(hold_norm=math.sqrt(hold2), other_norm=math.sqrt(other2),
            weighted_kl_norm=math.sqrt(kl2), keep_norm=math.sqrt(keep2),
            hold_other_dot=hold_other, hold_keep_dot=hold_keep,
            hold_other_cosine=cosine(hold_other, hold2, other2),
            hold_weighted_kl_cosine=cosine(float(matrix[0, 2]), hold2, kl2),
            other_weighted_kl_cosine=cosine(float(matrix[1, 2]), other2, kl2),
            hold_keep_cosine=cosine(hold_keep, hold2, keep2),
            projection_coefficient=hold_keep / keep2 if keep2 > 1e-20 and hold_keep < 0 else 0.)

    return dict(global_parameters=summarize(matrices.sum(0)),
                parameter_groups={name: {**summarize(matrices[i]), **counts[name]}
                                  for i, name in enumerate(groups)})


def neighbor_audit(frames, episodes, pre, policy, get_sample, out, settings):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from excavator_vlm.data.contract import STATE_NAMES

    pool = frames[frames.episode_index.isin(episodes) &
                  frames['observation.stage_current_id'].isin(settings['neighbor_candidate_stages'])]
    pool = pool.sort_values(['episode_index', 'frame_index']).reset_index(drop=True)
    physical_states = np.stack(pool['observation.state']).astype(np.float32)
    with torch.no_grad():
        normalized = pre({'observation.state': torch.from_numpy(physical_states)})['observation.state']
        normalized = normalized.cpu().numpy()
    actions = np.stack(pool.action).astype(np.float32)
    keys = [(int(ep), int(frame)) for ep, frame in zip(pool.episode_index, pool.frame_index)]
    stages = pool['observation.stage_current_id'].to_numpy(dtype=int)
    episode_ids = pool.episode_index.to_numpy(dtype=int)
    holds = np.flatnonzero(stages == 6)
    if len(holds) != settings['train_hold_observations']:
        raise ValueError('Training hold observation count changed')
    records = []
    for index in holds:
        distances = np.sqrt(((normalized - normalized[index]) ** 2).mean(-1))
        distances[index] = np.inf
        for partition in settings['neighbor_partitions']:
            allowed = (episode_ids == episode_ids[index]) if partition == 'same_episode' else (
                episode_ids != episode_ids[index])
            candidates = np.flatnonzero(allowed & np.isfinite(distances))
            order = candidates[np.argsort(distances[candidates], kind='stable')]
            for neighbor in order[:settings['neighbors_per_partition']]:
                records.append(dict(partition=partition, first=list(keys[index]), second=list(keys[neighbor]),
                    first_stage=6, second_stage=int(stages[neighbor]),
                    state_rms_distance=float(distances[neighbor]),
                    first_action=actions[index].tolist(), second_action=actions[neighbor].tolist(),
                    bucket_action_abs_difference=float(abs(actions[index, 3] - actions[neighbor, 3])),
                    opposite_bucket_sign=bool(actions[index, 3] * actions[neighbor, 3] < 0),
                    first_state=physical_states[index].tolist(), second_state=physical_states[neighbor].tolist(),
                    normalized_state_delta=(normalized[neighbor] - normalized[index]).tolist()))
    pairs, used = [], set()
    # Predeclared here before gradient measurements: four nearest pairs per
    # partition with >=0.3 rad/s bucket label disagreement, no duplicate pair.
    per_partition = settings['max_image_pairs'] // len(settings['neighbor_partitions'])
    for partition in settings['neighbor_partitions']:
        candidates = sorted((row for row in records if row['partition'] == partition and
                             row['bucket_action_abs_difference'] >= .3),
                            key=lambda row: (row['state_rms_distance'], -row['bucket_action_abs_difference'],
                                             row['first'], row['second']))
        selected = 0
        for row in candidates:
            identity = tuple(sorted((tuple(row['first']), tuple(row['second']))))
            if identity in used:
                continue
            used.add(identity)
            first, second = get_sample(tuple(row['first'])), get_sample(tuple(row['second']))
            measured = {key: float(((first[key] - second[key]) ** 2).mean())
                        for key in policy.config.image_features}
            pairs.append({**row, 'image_mse': measured,
                          'identical_full_inputs': bool(torch.equal(first['observation.state'],
                              second['observation.state']) and all(value == 0 for value in measured.values()))})
            selected += 1
            if selected == per_partition:
                break
    if pairs:
        width, height = 192, 192
        sheet = Image.new('RGB', (width * 6, (height + 45) * len(pairs)), 'white')
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
        for index, pair in enumerate(pairs):
            y = index * (height + 45)
            label = (f"{index+1} {pair['partition']}: A={pair['first']} stage6 "
                     f"bucket={pair['first_action'][3]:+.3f}; B={pair['second']} "
                     f"stage{pair['second_stage']} bucket={pair['second_action'][3]:+.3f}; "
                     f"state RMS={pair['state_rms_distance']:.4f}")
            draw.text((4, y + 3), label, font=font, fill='black')
            first, second = get_sample(tuple(pair['first'])), get_sample(tuple(pair['second']))
            for camera, key in enumerate(policy.config.image_features):
                for side, sample in enumerate((first, second)):
                    array = np.clip(sample[key].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
                    sheet.paste(Image.fromarray(array).resize((width, height)),
                                ((camera * 2 + side) * width, y + 45))
                    draw.text(((camera * 2 + side) * width + 4, y + 24),
                              f"camera {camera} / {'A' if side == 0 else 'B'}", font=font, fill='black')
        sheet.save(out / 'neighbor_pairs.png')
    payload = dict(scope='train-only state neighborhood diagnosis; image differences do not prove '
        'the learned policy uses those differences', hold_observations=len(holds),
        candidate_observations=len(pool), state_names=STATE_NAMES, records=records,
        image_pair_rule='four nearest distinct pairs per partition with bucket label delta >= 0.3 rad/s',
        image_pairs=pairs, decoded_image_pairs=len(pairs),
        exact_full_input_conflicts_in_decoded_pairs=sum(pair['identical_full_inputs'] for pair in pairs))
    write(out / 'neighbor_audit.json', payload)
    return dict(hold_observations=len(holds), candidate_observations=len(pool), neighbor_records=len(records),
                decoded_image_pairs=len(pairs), exact_full_input_conflicts_in_decoded_pairs=
                payload['exact_full_input_conflicts_in_decoded_pairs'])


def run_diagnosis(protocol, out, gpu, state):
    started = time.monotonic()
    checkpoint = ROOT / protocol['source_checkpoint']
    protected = dict(protocol['pinned_inputs_sha256'])
    protected[str((checkpoint / 'model.safetensors').relative_to(ROOT))] = protocol['source_model_sha256']
    previous = read(ROOT / 'outputs/experiments/act_incremental_sampling_20261103_01/status.json')
    for path, expected in previous['protected_inputs_sha256'].items():
        protected[path] = expected
    for path, expected in protected.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'Pinned input changed: {path}')
    frozen = out / 'frozen_sources/ExcavatorVLA-vlm'
    shutil.copytree(ROOT / 'ExcavatorVLA-vlm/excavator_vlm', frozen / 'excavator_vlm',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    sys.path.insert(0, str(frozen))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HOME=str(ROOT / 'models/hf-cache'), HF_DATASETS_CACHE=str(ROOT / 'models/hf-cache/datasets'),
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import default_collate
    from excavator_vlm.data.contract import read_frames
    from excavator_vlm.policies.act.dataset import make_act_dataset
    from excavator_vlm.policies.act.inference import load_act
    from excavator_vlm.policies.act.adapter import training_batch
    from excavator_vlm.policies.act.sampling import make_sampler
    import lerobot.policies.act.modeling_act as modeling

    torch.set_num_threads(4)
    source_paths = [*sorted(frozen.rglob('*.py')), Path(inspect.getfile(modeling)).resolve(),
                    out / 'protocol.json', out / Path(__file__).name]
    for path in source_paths:
        protected[str(path)] = digest(path)
    write(out / 'inputs_sha256.json', protected)
    provenance = read(checkpoint / 'benchmark_provenance.json')
    split = provenance['split_manifest']
    if any(set(split[a]) & set(split[b]) for a, b in
           (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('Frozen episode partitions overlap')
    frames = read_frames(ROOT / 'data/export_phw')
    tables = {int(ep): group.sort_values('frame_index') for ep, group in frames.groupby('episode_index')}
    policy, pre, _ = load_act(checkpoint, 'cuda')
    original = load_file(str(checkpoint / 'model.safetensors'))
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value.cpu(), original[key], rtol=0, atol=0)
    dataset = make_act_dataset('local/excavator', (ROOT / 'data/export_phw').resolve(), split['train'], 10)
    identity = dataset.hf_dataset.select_columns(['episode_index', 'frame_index']).to_pandas()
    keys = [(int(ep), int(frame)) for ep, frame in zip(identity.episode_index, identity.frame_index)]
    lookup = {key: i for i, key in enumerate(keys)}
    cache, labels = {}, {}
    alignment_errors = dict(action_max_abs_error=0., state_max_abs_error_after_float32_cast=0.)

    def target_metadata(key):
        ep, frame = key
        if ep not in split['train'] or ep in split['validation'] + split['test']:
            raise ValueError('Diagnostic selected a held-out observation')
        table = tables[ep]
        rows = table.iloc[np.minimum(np.arange(frame, frame + 10), len(table) - 1)]
        return dict(stages=rows['observation.stage_current_id'].to_numpy(dtype=np.int64),
            action=np.stack(rows.action), weights=np.asarray(rows.action_loss_weight.tolist()).reshape(-1),
            padding=np.arange(frame, frame + 10) >= len(table))

    def get_sample(key):
        if key not in cache:
            sample = dataset[lookup[key]]
            if (int(sample['episode_index']), int(sample['frame_index'])) != key:
                raise ValueError('HF observation identity differs from sampling manifest')
            metadata = target_metadata(key)
            actual = sample['action'].numpy()
            expected = metadata['action'].astype(actual.dtype)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
            alignment_errors['action_max_abs_error'] = max(alignment_errors['action_max_abs_error'],
                                                         float(np.abs(actual - expected).max()))
            np.testing.assert_array_equal(sample['action_is_pad'].numpy(), metadata['padding'])
            np.testing.assert_array_equal(sample['action_loss_weight'].numpy().reshape(-1),
                                          metadata['weights'])
            expected_state = np.asarray(tables[key[0]].iloc[key[1]]['observation.state']).astype(np.float32)
            actual_state = sample['observation.state'].numpy().astype(np.float32)
            np.testing.assert_allclose(actual_state, expected_state, rtol=0, atol=1e-6)
            alignment_errors['state_max_abs_error_after_float32_cast'] = max(
                alignment_errors['state_max_abs_error_after_float32_cast'],
                float(np.abs(actual_state - expected_state).max()))
            cache[key], labels[key] = sample, metadata
        return cache[key]

    settings = protocol['diagnosis']
    spec = protocol['paired_training']['sampling']
    sampler, sampling_report = make_sampler(dataset, split['train'],
        dict(manifest=str(ROOT / spec['candidate_manifest']),
             transition_probability=spec['episode_balanced_stage5_6_probability']), settings['sampling_seed'])
    draws = list(iter(sampler))[:settings['max_batches_inspected_for_selection'] * settings['batch_size']]
    selected = []
    for batch_index in range(settings['max_batches_inspected_for_selection']):
        batch_keys = [keys[index] for index in draws[batch_index * settings['batch_size']:
                                                    (batch_index + 1) * settings['batch_size']]]
        metadata = [target_metadata(key) for key in batch_keys]
        stages = np.stack([item['stages'] for item in metadata])
        weights = np.stack([item['weights'] * ~item['padding'] for item in metadata])
        if (weights * (stages == 6)).sum() > 0 and (weights * (stages != 6)).sum() > 0:
            selected.append(dict(sampler_batch_index=batch_index, identities=batch_keys,
                future_stages=stages.tolist(), effective_weights=weights.tolist()))
            if len(selected) == settings['selected_batches']:
                break
    write(out / 'diagnostic_batches.json', dict(sampling=sampling_report, seed=settings['sampling_seed'],
        heldout_used=False, selected_batches=selected,
        unique_observations=len({tuple(key) for batch in selected for key in batch['identities']})))
    state.update(status='neighbor_audit', selected_batches=len(selected))
    write(out / 'status.json', state)
    state['neighbor_summary'] = neighbor_audit(frames, split['train'], pre, policy, get_sample, out, settings)
    raw_batches = [default_collate([get_sample(tuple(key)) for key in batch['identities']]) for batch in selected]
    write(out / 'alignment_audit.json', dict(**alignment_errors, actions_weights_padding_and_states_checked=True,
        selected_observations=len(cache), heldout_used=False))
    named_parameters = [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad]
    rows, seed_results = [], []
    gate = settings['activation_gate']
    state.update(status='gradient_diagnosis', precision='original BF16 training autocast',
                 optimizer_updates=0, seed_results=[])
    write(out / 'status.json', state)
    with (out / 'gradient_batches.jsonl').open('x') as log:
        for seed in settings['forward_seeds']:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            conflicts, triggers = 0, 0
            for batch_index, raw in enumerate(raw_batches):
                policy.train()
                processed = training_batch(pre, raw)
                stages = torch.tensor(selected[batch_index]['future_stages'], device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    hold, other, weighted_kl, parts = split_losses(policy, processed, stages)
                gradients = component_gradients((hold, other, weighted_kl), named_parameters)
                measured = gradient_statistics(named_parameters, gradients)
                global_stats = measured['global_parameters']
                conflict = (global_stats['hold_norm'] > gate['minimum_gradient_norm'] and
                    global_stats['other_norm'] > gate['minimum_gradient_norm'] and
                    global_stats['hold_other_cosine'] is not None and
                    global_stats['hold_other_cosine'] < gate['cosine_below'])
                trigger = (global_stats['hold_keep_dot'] < 0 and global_stats['keep_norm'] ** 2 >
                           protocol['paired_training']['minimum_protected_gradient_squared_norm_for_projection'])
                conflicts += int(conflict)
                triggers += int(trigger)
                row = dict(forward_seed=seed, selected_batch_index=batch_index,
                    sampler_batch_index=selected[batch_index]['sampler_batch_index'], parts=parts,
                    **measured, conflict=bool(conflict), projection_would_trigger=bool(trigger))
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                state['gradient_forwards'] += 1
                state.update(active_seed=seed, active_seed_completed_batches=batch_index + 1,
                             active_seed_conflicts=conflicts, active_seed_projection_triggers=triggers)
                write(out / 'status.json', state)
                del gradients, hold, other, weighted_kl, processed
            passed = (len(selected) == settings['selected_batches'] and
                conflicts >= gate['minimum_conflicting_batches_per_forward_seed'] and
                triggers >= gate['minimum_projection_trigger_batches_per_forward_seed_including_kl'])
            seed_results.append(dict(forward_seed=seed, batches=len(selected),
                conflicting_batches=conflicts, projection_trigger_batches=triggers, passed=bool(passed)))
            state['seed_results'] = seed_results
            write(out / 'status.json', state)
            print(json.dumps(seed_results[-1]), flush=True)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value.cpu(), original[key], rtol=0, atol=0)
    if any(parameter.grad is not None for _, parameter in named_parameters):
        raise ValueError('Read-only autograd diagnostic unexpectedly populated parameter.grad')
    for path, expected in protected.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'Pinned input changed during diagnosis: {path}')
    passed = len(seed_results) == len(settings['forward_seeds']) and all(row['passed'] for row in seed_results)
    state.update(status='diagnosis_passed_ready_for_training' if passed else 'diagnosis_gate_failed',
        activation_gate_passed=bool(passed), seed_results=seed_results, inputs_unchanged=True,
        model_state_unchanged=True, optimizer_updates=0, checkpoint_saved=False,
        gradient_forwards=len(rows), elapsed_seconds=time.monotonic() - started)
    write(out / 'status.json', state)
    write(out / 'diagnosis_result.json', dict(**state,
        scope='train-only diagnostic; not root-cause proof or closed-loop improvement',
        activation_gate=gate, alignment_audit=read(out / 'alignment_audit.json')))
    print(json.dumps(dict(status=state['status'], elapsed_seconds=state['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'docs/act_gradient_protocol_20261103.json')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    protocol = read(args.protocol)
    out = ROOT / protocol['planned_output']
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, out / 'protocol.json')
    shutil.copy2(__file__, out / Path(__file__).name)
    state = dict(status='checking_inputs', optimizer_updates=0, gradient_forwards=0,
                 gpu=args.gpu, source_model_sha256=protocol['source_model_sha256'])
    write(out / 'status.json', state)
    try:
        run_diagnosis(protocol, out, args.gpu, state)
    except BaseException as error:
        state.update(status='infrastructure_or_implementation_failed', error=str(error))
        write(out / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
