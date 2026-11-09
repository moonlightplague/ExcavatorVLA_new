"""Bounded ACT continuation with train-only successful recovery observations."""
import argparse
from collections import Counter
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
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sample_plans(original_keys, expert_keys, total=6400, seed=1000):
    """Same original draws; replace exactly 10%, balancing expert episodes."""
    import numpy as np
    original_keys = [tuple(map(int, key)) for key in original_keys]
    expert_keys = [tuple(map(int, key)) for key in expert_keys]
    if total % 10 or not original_keys or not expert_keys:
        raise ValueError('Need nonempty source pools and a multiple-of-ten budget')
    if len(set(original_keys)) != len(original_keys) or len(set(expert_keys)) != len(expert_keys):
        raise ValueError('Duplicate observation identities within a source')
    rng = np.random.default_rng(seed)
    control = [dict(source='original', episode_index=original_keys[i][0], frame_index=original_keys[i][1])
               for i in rng.integers(len(original_keys), size=total)]
    positions = rng.choice(total, size=total // 10, replace=False)
    groups = {ep: [key for key in expert_keys if key[0] == ep] for ep in sorted({k[0] for k in expert_keys})}
    episodes = np.resize(list(groups), len(positions))
    rng.shuffle(episodes)
    recovery = [dict(row) for row in control]
    for position, ep in zip(positions, episodes, strict=True):
        key = groups[int(ep)][int(rng.integers(len(groups[int(ep)])))]
        recovery[int(position)] = dict(source='recovery', episode_index=key[0], frame_index=key[1])
    return dict(control=control, recovery=recovery)


class SelectedObservations:
    """Strip non-input metadata before collation and namespace the two sources."""
    def __init__(self, datasets, plans):
        self.datasets = datasets
        self.plan = plans
        self.lookups = {}
        for source, dataset in datasets.items():
            ids = dataset.hf_dataset.select_columns(['episode_index', 'frame_index']).to_pandas()
            self.lookups[source] = {(int(ep), int(frame)): i for i, (ep, frame) in
                                    enumerate(zip(ids.episode_index, ids.frame_index, strict=True))}

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        row = self.plan[index]
        key = row['episode_index'], row['frame_index']
        raw = self.datasets[row['source']][self.lookups[row['source']][key]]
        result = {k: v for k, v in raw.items() if k in ('observation.state', 'action', 'action_is_pad',
                  'action_loss_weight', 'episode_index', 'frame_index') or k.startswith('observation.images.')}
        result['sample_source'] = int(row['source'] == 'recovery')
        return result


def setup(out):
    protocol = read(out / 'protocol.json')
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HOME=str(ROOT / 'models/hf-cache'), TORCH_HOME=str(ROOT / 'models/torch-cache'),
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    sys.path[:0] = [str(out / 'frozen_sources/ExcavatorVLA-vlm'),
                    str(Path(protocol['archive']) / 'runtime_dependencies')]
    return protocol


def check_inputs(out):
    for name, expected in read(out / 'inputs_sha256.json').items():
        if sha(name) != expected:
            raise ValueError(f'Protected input changed: {name}')


def prepare(out):
    import numpy as np
    out.mkdir(parents=True, exist_ok=False)
    manifest = read(ROOT / 'docs/baselines/act_phw40k_carry275_20261105.json')
    archive = ROOT / manifest['archive']
    source = archive / manifest['checkpoint']
    recovery = (ROOT / 'data/act_recovery_20261105_v1').resolve()
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_id=manifest['baseline_id'], archive=str(archive), checkpoint=str(source),
        model_sha256=manifest['model_sha256'], dataset=manifest['external_dependencies']['dataset'],
        recovery_dataset=str(recovery), steps=400, batch_size=16, seed=1000, learning_rate=1e-5,
        arms=dict(control=dict(gpu=0), recovery=dict(gpu=1)), candidate_arm='recovery',
        intervention='Replace exactly 640 of 6400 uniform original-training observations with episode-balanced '
                     'eligible expert suffix observations. Original PHW/L1/KL/padding and all model settings unchanged.',
        normalization='Original PHW40k processors and statistics; never recomputed on the recovery data',
        selection='Only final 400-step checkpoints; no extra steps or hyperparameter changes after results',
        development_scenes=[1, 9, 20], development_scope='Training scene repair checks, not held-out performance',
        unseen_seeds=[3100000, 3100001, 3100002, 3100003],
        offline_guardrail='Every original validation stage first-three MAE <= max(1.10*baseline, baseline+0.005 rad/s)',
        unseen_gate='Run fresh paired seeds only if offline guardrail passes and recovery has strictly more '
                    'full successes on the three development scenes than both baseline and control',
        rollout='500 steps, 10 Hz, original -130 degree arm_first opening_pose guard, no expert takeover, '
                'original physics/assets/scoring; fresh simulator for each run',
        test_used=False, scope='Bounded development experiment; four fresh seeds are a small paired pilot, not a final benchmark')
    write(out / 'protocol.json', protocol)
    frozen = out / 'frozen_sources/ExcavatorVLA-vlm'
    shutil.copytree(archive / 'frozen_sources/ExcavatorVLA-vlm/excavator_vlm', frozen / 'excavator_vlm',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (out / 'frozen_scripts').mkdir()
    shutil.copy2(Path(__file__), out / 'frozen_scripts/run_act_recovery_finetune.py')
    split = read(source / 'benchmark_provenance.json')['split_manifest']
    write(out / 'split.json', split)
    setup(out)
    from excavator_vlm.policies.act.config import resolve_settings
    from excavator_vlm.policies.act.train import prepare as audit_prepare
    from excavator_vlm.data.contract import read_frames
    settings = resolve_settings(archive / 'frozen_sources/ExcavatorVLA-vlm/configs/act/full_phw.yaml', dict(
        dataset=dict(root=protocol['dataset']), benchmark=dict(split_file=str(out / 'split.json'), validation_freq=0),
        launcher=dict(gpu_ids=[0], cpu_threads=4, hf_home=str(ROOT / 'models/hf-cache'),
                      torch_home=str(ROOT / 'models/torch-cache')),
        continuation=dict(checkpoint=str(source)), steps=400, batch_size=16, num_workers=2,
        save_freq=400, log_freq=50, output_dir=str(out / 'unused')))
    settings['sampling'] = dict(manifest=None, transition_probability=0.0)
    write(out / 'settings.json', settings)
    contract, actual_split, frames, episodes = audit_prepare(settings)
    if split != actual_split or sha(source / 'model.safetensors') != protocol['model_sha256']:
        raise ValueError('Source split/model changed')
    write(out / 'dataset_contract.json', contract)
    expert = read(recovery / 'eligible_training_observations.json')['observations']
    expert_keys = [(r['episode_index'], r['frame_index']) for r in expert]
    rf = read_frames(recovery)
    eligible = {(int(r.episode_index), int(r.frame_index)) for r in rf.itertuples()
                if bool(r.action_is_expert) and float(r.action_loss_weight) > 0}
    if set(expert_keys) != eligible or len(expert_keys) != 634:
        raise ValueError('Recovery eligibility must equal the 634 successful expert observations')
    training = frames[frames.episode_index.isin(episodes)]
    keys = [(int(ep), int(frame)) for ep, frame in zip(training.episode_index, training.frame_index, strict=True)]
    plans = sample_plans(keys, expert_keys)
    for arm, plan in plans.items():
        write(out / f'{arm}_samples.json', plan)
    counts = Counter(r['episode_index'] for r in plans['recovery'] if r['source'] == 'recovery')
    write(out / 'sampling_audit.json', dict(total=6400, replacement_observations=640, fraction=.1,
        original_pool=len(keys), expert_pool=len(expert_keys), expert_episode_draws=dict(counts),
        unchanged_original_positions=sum(a == b for a, b in zip(plans['control'], plans['recovery'], strict=True)),
        original_validation_test_used_for_training=False, model_prefix_sampled=False,
        note='10% is the observation fraction; original PHW and expert weight=1 are preserved'))
    observations = read(ROOT / 'outputs/experiments/act_huber_20261105_01/offline_observations.json')
    write(out / 'offline_observations.json', observations)
    (out / 'references').mkdir()
    origin = ROOT / manifest['original_experiment'] / 'evaluation/episodes'
    template = read(ROOT / 'outputs/experiments/act_phw_optimization_20261103_01/reference_setup.json')
    for scene in protocol['development_scenes']:
        folder = origin / f'episode_{scene:04d}'
        context, initial = read(folder / 'scene_context.json'), read(folder / 'initial_observation.json')
        reference = dict(template)
        reference.update(profile=context['scene_profile'], task=initial['task'],
            initial_joints_deg=[math.degrees(v) for v in initial['state'][:4]],
            expected_initial_state=initial['state'], dig_target_xyz=context['observation_context']['dig_target_xyz'],
            sand_amount=context['scene_profile']['sand_amount_multiplier'],
            actions=[json.loads(line)['predicted'] for line in (folder / 'steps.jsonl').read_text().splitlines()],
            provenance=dict(source_episode=str(folder), purpose='Geometry only; actions are never replayed'))
        write(out / 'references' / f'scene{scene}.json', reference)
    paths = [*source.glob('*'), archive / 'normalization_stats.json',
             *Path(protocol['dataset']).glob('data/**/*.parquet'),
             *Path(protocol['dataset']).glob('meta/**/*.parquet'), Path(protocol['dataset']) / 'meta/info.json',
             *recovery.rglob('*.json'), *recovery.rglob('*.parquet'), *recovery.rglob('*.mp4'),
             *out.glob('*.json'), *out.rglob('*.py'), *out.glob('references/*.json')]
    protected = {str(p): sha(p) for p in paths if p.is_file() and p.name not in ('inputs_sha256.json', 'status.json')}
    write(out / 'inputs_sha256.json', protected)
    write(out / 'status.json', dict(status='prepared', steps_per_arm=400))
    return protocol


def worker(out, arm):
    protocol = read(out / 'protocol.json')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(protocol['arms'][arm]['gpu'])
    setup(out)
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from excavator_vlm.data.contract import read_frames
    from excavator_vlm.policies.act.adapter import training_batch
    from excavator_vlm.policies.act.config import build_train_config
    from excavator_vlm.policies.act.continuation import validate_continuation
    from excavator_vlm.policies.act.dataset import make_act_dataset, training_stats
    from excavator_vlm.policies.act.inference import load_act
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.utils.train_utils import save_checkpoint, update_last_checkpoint
    torch.set_num_threads(4)
    torch.manual_seed(1000)
    np.random.seed(1000)
    check_inputs(out)
    run = out / 'train' / arm
    settings = read(out / 'settings.json')
    settings.update(output_dir=str(run), job_name='act_recovery_' + arm)
    cfg = build_train_config(settings)
    split = read(out / 'split.json')
    source = Path(protocol['checkpoint'])
    frames = read_frames(protocol['dataset'])
    stats = training_stats(frames, split['train'])
    continuation = validate_continuation(source, cfg.policy, read(out / 'dataset_contract.json'), split, split['train'], stats)
    run.mkdir(parents=True, exist_ok=False)
    write(run / 'normalization_stats.json', stats)
    write(run / 'resolved_settings.json', settings)
    write(run / 'continuation.json', continuation)
    cfg.dataset.episodes = split['train']
    cfg.policy.device = 'cuda'
    policy, pre, post = load_act(source, 'cuda')
    expected = load_file(str(source / 'model.safetensors'))
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected[name], rtol=0, atol=0)
    del expected
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    if scheduler is not None:
        raise ValueError('Expected original no-scheduler optimizer')
    datasets = dict(original=make_act_dataset('local/excavator', Path(protocol['dataset']), split['train'], 10))
    rf = read_frames(protocol['recovery_dataset'])
    if arm == 'recovery':
        datasets['recovery'] = make_act_dataset('local/act-recovery', Path(protocol['recovery_dataset']), [0, 1, 2], 10)
    plan = read(out / f'{arm}_samples.json')
    selected = SelectedObservations(datasets, plan)
    loader = DataLoader(selected, batch_size=16, shuffle=False, num_workers=2,
        multiprocessing_context='spawn', persistent_workers=True, timeout=90,
        generator=torch.Generator().manual_seed(1000))
    references = {'original': {int(ep): g.sort_values('frame_index') for ep, g in
                  frames[frames.episode_index.isin(split['train'])].groupby('episode_index')},
                  'recovery': {int(ep): g.sort_values('frame_index') for ep, g in rf.groupby('episode_index')}}
    allowed = {(r['episode_index'], r['frame_index']) for r in
               read(Path(protocol['recovery_dataset']) / 'eligible_training_observations.json')['observations']}
    checked = set()
    source_counts = Counter()
    source_weight = Counter()
    sample_hash = hashlib.sha256()
    initial = {name: value.detach().clone() for name, value in policy.named_parameters() if name in
               ('model.action_head.weight', 'model.encoder_robot_state_input_proj.weight', 'model.backbone.conv1.weight')}
    torch.manual_seed(1000)
    started = time.monotonic()
    with (run / 'train.jsonl').open('x') as log:
        for step, raw in enumerate(loader, 1):
            identities = []
            for i in range(len(raw['sample_source'])):
                src = 'recovery' if int(raw['sample_source'][i]) else 'original'
                ep, frame = int(raw['episode_index'][i]), int(raw['frame_index'][i])
                identities.append(dict(source=src, episode_index=ep, frame_index=frame))
                source_counts[src] += 1
                source_weight[src] += float((raw['action_loss_weight'][i].reshape(-1) * (~raw['action_is_pad'][i])).sum())
                if src == 'recovery' and (ep, frame) not in allowed:
                    raise ValueError('A model-prefix observation reached training')
                key = src, ep, frame
                if key in checked:
                    continue
                episode = references[src][ep]
                rows = episode.iloc[np.minimum(np.arange(frame, frame + 10), len(episode) - 1)]
                for name, value in [('action', np.stack(rows.action)),
                                    ('action_loss_weight', np.asarray(rows.action_loss_weight.tolist()))]:
                    actual = raw[name][i].numpy()
                    np.testing.assert_array_equal(actual.reshape(-1), value.astype(actual.dtype).reshape(-1))
                actual = raw['observation.state'][i].numpy()
                np.testing.assert_array_equal(actual, episode.iloc[frame]['observation.state'].astype(actual.dtype))
                np.testing.assert_array_equal(raw['action_is_pad'][i].numpy(), np.arange(frame, frame + 10) >= len(episode))
                checked.add(key)
            if identities != plan[(step - 1) * 16:step * 16]:
                raise ValueError('Actual sample order differs from frozen plan')
            sample_hash.update(json.dumps(identities, separators=(',', ':')).encode())
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, parts = policy(training_batch(pre, raw))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            if step == 1 or step % 50 == 0:
                row = dict(step=step, **parts, grad_norm=float(norm), elapsed_seconds=time.monotonic() - started,
                           sampled_sources=dict(source_counts))
                log.write(json.dumps(row) + '\n'); log.flush()
                print(json.dumps(row), flush=True)
                write(run / 'status.json', dict(status='training', step=step, steps=400))
    if step != 400 or source_counts['recovery'] != (640 if arm == 'recovery' else 0):
        raise ValueError('Training budget/source counts differ')
    checkpoint = run / 'checkpoints/000400'
    save_checkpoint(checkpoint, 400, cfg, policy, optimizer, None)
    update_last_checkpoint(checkpoint)
    pretrained = checkpoint / 'pretrained_model'
    for path in source.glob('*processor*'):
        if path.is_file():
            shutil.copy2(path, pretrained / path.name)
    provenance = read(source / 'benchmark_provenance.json')
    provenance.update(continuation=continuation, finetune_steps=400, finetune_arm=arm,
        additional_training_datasets=([dict(root=protocol['recovery_dataset'], train_only=True,
            eligible_observations=634, sampled_observations=640,
            provenance_sha256=sha(Path(protocol['recovery_dataset']) / 'recovery_provenance.json'))]
            if arm == 'recovery' else []),
        normalization_stats_sha256=sha(run / 'normalization_stats.json'),
        sample_plan_sha256=sha(out / f'{arm}_samples.json'))
    write(pretrained / 'benchmark_provenance.json', provenance)
    policy.eval()
    with torch.inference_mode():
        observed = pre({k: raw[k][:2] for k in raw if k == 'observation.state' or k.startswith('observation.images.')})
        predicted = policy.predict_action_chunk(observed).cpu()
        restored, _, _ = load_act(pretrained, 'cuda')
        torch.testing.assert_close(restored.predict_action_chunk(observed).cpu(), predicted, rtol=1e-5, atol=1e-6)
    if set(policy.state_dict()) != set(restored.state_dict()):
        raise ValueError('Inference schema changed')
    for path in source.glob('*processor*'):
        if path.is_file() and sha(path) != sha(pretrained / path.name):
            raise ValueError('Processor bytes changed')
    check_inputs(out)
    result = dict(status='completed', steps=400, checkpoint=str(pretrained),
        model_sha256=sha(pretrained / 'model.safetensors'), sample_order_sha256=sample_hash.hexdigest(),
        observed_source_counts=dict(source_counts), valid_supervision_weight=dict(source_weight),
        parquet_aligned_distinct_observations=len(checked), reload_check='passed',
        processors_byte_identical=True, inputs_unchanged=True,
        parameter_changes={name: float((dict(policy.named_parameters())[name] - value).abs().max().detach())
                           for name, value in initial.items()}, elapsed_seconds=time.monotonic() - started)
    write(run / 'result.json', result)
    write(run / 'status.json', dict(status='completed', step=400))


def offline(out):
    protocol = setup(out)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from excavator_vlm.policies.act.dataset import make_act_dataset
    from excavator_vlm.policies.act.inference import load_act
    torch.set_num_threads(4)
    checkpoints = dict(baseline=protocol['checkpoint'], **{arm: read(out / 'train' / arm / 'result.json')['checkpoint']
                                                         for arm in protocol['arms']})
    policies = {name: load_act(path, 'cuda') for name, path in checkpoints.items()}
    split = read(out / 'split.json')
    observations = read(out / 'offline_observations.json')['observations']
    groups = [(group, Path(protocol['dataset']), split[group], [r for r in observations if r['split'] == group])
              for group in ('train', 'validation')]
    expert = read(Path(protocol['recovery_dataset']) / 'eligible_training_observations.json')['observations']
    groups.append(('recovery_training', Path(protocol['recovery_dataset']), [0, 1, 2], expert))
    all_metrics, guardrail = {}, {}
    for group, root, episodes, chosen in groups:
        dataset = make_act_dataset('local/excavator', root, episodes, 10)
        ids = dataset.hf_dataset.select_columns(['episode_index', 'frame_index']).to_pandas()
        lookup = {(int(ep), int(frame)): i for i, (ep, frame) in enumerate(zip(ids.episode_index, ids.frame_index, strict=True))}
        targets, pads = [], []
        predictions = {name: [] for name in policies}
        for start in range(0, len(chosen), 16):
            raw = default_collate([dataset[lookup[(r['episode_index'], r['frame_index'])]] for r in chosen[start:start + 16]])
            observation = {k: v for k, v in raw.items() if k == 'observation.state' or k.startswith('observation.images.')}
            with torch.inference_mode():
                for name, (policy, pre, post) in policies.items():
                    chunk = policy.predict_action_chunk(pre(observation))
                    predictions[name].append(post(chunk.reshape(-1, 4)).reshape_as(chunk).cpu().numpy())
            targets.append(raw['action'].numpy()); pads.append(raw['action_is_pad'].numpy())
        target, padding = np.concatenate(targets), np.concatenate(pads)
        predictions = {name: np.concatenate(values) for name, values in predictions.items()}
        def metrics(predicted, mask):
            error = np.abs(predicted[mask] - target[mask])
            valid = ~padding[mask, :3]
            return dict(observations=int(mask.sum()), first_three_mae_rad_s=float(
                (error[:, :3] * valid[:, :, None]).sum() / (valid.sum() * 4)),
                first_bucket_mae_rad_s=float(error[:, 0, 3].mean()))
        group_metrics = {}
        for name, predicted in predictions.items():
            if not np.isfinite(predicted).all():
                raise FloatingPointError('Nonfinite offline predictions')
            result = dict(all=metrics(predicted, np.ones(len(chosen), dtype=bool)))
            if group != 'recovery_training':
                for stage in range(10):
                    mask = np.array([r['stage_id'] == stage for r in chosen])
                    if mask.any():
                        result[str(stage)] = metrics(predicted, mask)
            else:
                for episode in episodes:
                    result[f'episode{episode}'] = metrics(predicted, np.array([r['episode_index'] == episode for r in chosen]))
            group_metrics[name] = result
        all_metrics[group] = group_metrics
        np.savez_compressed(out / f'offline_{group}.npz', target=target, padding=padding, **predictions)
        print(f'Offline {group}: {len(chosen)} observations', flush=True)
        if group == 'validation':
            for stage in range(10):
                key = str(stage)
                if key in group_metrics['baseline']:
                    base = group_metrics['baseline'][key]['first_three_mae_rad_s']
                    candidate = group_metrics['recovery'][key]['first_three_mae_rad_s']
                    guardrail[key] = candidate <= max(1.10 * base, base + .005)
    write(out / 'offline_metrics.json', dict(groups=all_metrics, validation_guardrail=guardrail,
        guardrail_passed=all(guardrail.values()), recovery_scope='Training fit only; not held-out validation', test_used=False))
    check_inputs(out)
    return all(guardrail.values())


def rollout_jobs(out, unseen=False):
    protocol = read(out / 'protocol.json')
    checkpoints = dict(baseline=protocol['checkpoint'], **{arm: read(out / 'train' / arm / 'result.json')['checkpoint']
                                                         for arm in protocol['arms']})
    cases = ([dict(label=f'new{seed}', seed=seed, reference=None) for seed in protocol['unseen_seeds']]
             if unseen else [dict(label=f'scene{scene}', seed=2000000 + scene - 1,
                                 reference=str(out / 'references' / f'scene{scene}.json'))
                             for scene in protocol['development_scenes']])
    return [dict(case=case, variant=variant, checkpoint=checkpoint) for case in cases
            for variant, checkpoint in checkpoints.items()]


def run_rollouts(out, jobs, state):
    protocol = read(out / 'protocol.json')
    archive = Path(protocol['archive'])
    sys.path.insert(0, str(archive / 'frozen_sources/scripts'))
    from run_act_transport_validation import assess_episode
    results_path = out / 'closed_loop_results.json'
    results = read(results_path) if results_path.exists() else {}
    template = read(archive / 'evidence/protocol.json')['command']
    pending = list(jobs)
    active = {}
    def launch(job, gpu):
        case = job['case']
        name = case['label'] + '_' + job['variant']
        destination = out / 'closed_loop' / name
        command = list(template)
        command[2] = str(archive / 'frozen_sources/scripts/eval_smolvla_success.py')
        for flag, value in [('--checkpoint', job['checkpoint']), ('--output', str(destination)),
                            ('--episodes', '1'), ('--seed', str(case['seed'])), ('--workers', '1'), ('--gpus', str(gpu))]:
            command[command.index(flag) + 1] = value
        if case['reference']:
            command += ['--reference-episode', case['reference']]
        env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
            HF_HOME=str(ROOT / 'models/hf-cache'), TORCH_HOME=str(ROOT / 'models/torch-cache'),
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONPATH=str(archive / 'runtime_dependencies'))
        env.pop('CUDA_VISIBLE_DEVICES', None)
        log = (out / (name + '.log')).open('x')
        child = subprocess.Popen(command, cwd=archive / 'frozen_sources', env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return dict(job=job, name=name, destination=destination, command=command, child=child, log=log)
    try:
        while pending or active:
            for gpu in (0, 1):
                if gpu not in active and pending:
                    active[gpu] = launch(pending.pop(0), gpu)
            state.update(status='closed_loop', running=[dict(name=j['name'], pid=j['child'].pid,
                gpu=gpu, command=j['command']) for gpu, j in active.items()], completed_rollouts=len(results))
            write(out / 'status.json', state)
            for gpu, job in list(active.items()):
                code = job['child'].poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"Rollout {job['name']} exited {code}; inspect its log")
                destination, case = job['destination'], job['job']['case']
                evaluation_id = read(destination / 'provenance.json')['evaluation_id']
                folder = destination / 'episodes/episode_0001'
                assessment = assess_episode(folder, case['seed'], evaluation_id)
                # audit_episode raises on a failed audit; its return value is a
                # diagnostic record, rather than a second boolean verdict.
                if assessment['status'] != 'completed':
                    raise ValueError(f"Invalid episode audit: {job['name']}")
                context = read(folder / 'scene_context.json')
                initial = read(folder / 'initial_observation.json')
                scene_geometry = dict(profile=context['scene_profile'], context=context['observation_context'],
                                      initial_joints=initial['state'][:4])
                for other in results.values():
                    if other['case']['label'] == case['label']:
                        if other['geometry']['profile'] != scene_geometry['profile'] or other['geometry']['context'] != scene_geometry['context']:
                            raise ValueError('Paired rollout geometry differs')
                        joint_error = max(abs(a - b) for a, b in zip(other['geometry']['initial_joints'],
                                          scene_geometry['initial_joints'], strict=True))
                        if joint_error > .05:
                            raise ValueError('Paired initial joint states differ by more than 0.05 rad')
                results[job['name']] = dict(case=case, variant=job['job']['variant'],
                    result=assessment, geometry=scene_geometry, command=job['command'])
                write(results_path, results)
                job['log'].close()
                print(json.dumps(dict(rollout=job['name'], success=assessment['original_task_completed'],
                    terminal=assessment['transport']['terminal'])), flush=True)
                del active[gpu]
            if pending or active:
                time.sleep(5)
    finally:
        for job in active.values():
            child = job['child']
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
            job['log'].close()
    check_inputs(out)
    return results


def report(out):
    offline_data = read(out / 'offline_metrics.json')
    closed = read(out / 'closed_loop_results.json') if (out / 'closed_loop_results.json').exists() else {}
    gate = read(out / 'advance_decision.json') if (out / 'advance_decision.json').exists() else {}
    lines = ['# ACT成功纠错数据：400步对照', '',
        '两组均从27.5%保留基准的PHW40k权重开始，原归一化、网络、L1/PHW/KL及推理设置不变。',
        'control均匀抽取6400个原训练观测；recovery替换其中640个为按episode均衡的成功专家后缀。',
        '每组仅保留第400步检查点。三条采集场景属于训练开发检查，不能作为独立泛化成绩。', '',
        '| 模型 | 纠错训练状态前三步MAE / rad/s | 原验证状态前三步MAE / rad/s |',
        '| --- | ---: | ---: |']
    for name in ('baseline', 'control', 'recovery'):
        r = offline_data['groups']['recovery_training'][name]['all']['first_three_mae_rad_s']
        v = offline_data['groups']['validation'][name]['all']['first_three_mae_rad_s']
        lines.append(f'| {name} | {r:.6f} | {v:.6f} |')
    lines += ['', '原验证阶段保护条件：' + str(offline_data['validation_guardrail']), '',
              '| 场景 | 模型 | 完整任务成功 | 入车粒子 | 末态斗内粒子 |', '| --- | --- | --- | ---: | ---: |']
    for item in closed.values():
        result = item['result']; terminal = result['transport']['terminal']
        lines.append(f"| {item['case']['label']} | {item['variant']} | {result['original_task_completed']} | "
                     f"{terminal['payload_particles']} | {terminal['retained_particles']} |")
    lines += ['', '新场景推进决定：' + json.dumps(gate, ensure_ascii=False), '',
              '27.5%历史基准不替换；本轮不追加步数、改变混合比例或按结果删除场景。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def controller(out):
    protocol = prepare(out)
    state = dict(status='training', training={})
    children, logs = [], []
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for arm, settings in protocol['arms'].items():
            log = (out / (arm + '.log')).open('x'); logs.append(log)
            command = [sys.executable, '-u', str(out / 'frozen_scripts/run_act_recovery_finetune.py'),
                       '--output', str(out), '--worker', arm]
            child = subprocess.Popen(command, cwd=out, env=dict(os.environ, ACT_PROJECT_ROOT=str(ROOT)),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((arm, child))
            state['training'][arm] = dict(pid=child.pid, gpu=settings['gpu'], command=command)
        while any(child.poll() is None for _, child in children):
            for arm, child in children:
                state['training'][arm]['returncode'] = child.poll()
                progress = out / 'train' / arm / 'status.json'
                if progress.exists():
                    state['training'][arm]['progress'] = read(progress)
                if child.poll() not in (None, 0):
                    raise RuntimeError(f'{arm} failed; inspect its log')
            write(out / 'status.json', state)
            time.sleep(5)
        for arm, child in children:
            state['training'][arm].update(returncode=child.returncode,
                progress=read(out / 'train' / arm / 'status.json'))
        write(out / 'training_results.json', {arm: read(out / 'train' / arm / 'result.json') for arm, _ in children})
        state['status'] = 'offline_evaluation'; write(out / 'status.json', state)
        guardrail = offline(out)
        results = run_rollouts(out, rollout_jobs(out), state)
        counts = {variant: sum(r['variant'] == variant and r['result']['original_task_completed']
                              for r in results.values()) for variant in ('baseline', 'control', 'recovery')}
        advance = guardrail and counts['recovery'] > max(counts['baseline'], counts['control'])
        write(out / 'advance_decision.json', dict(advance=advance, offline_guardrail=guardrail,
            development_successes=counts, reason='Predeclared original-validation guardrail and strictly more full '
            'development successes than both controls; development scenes already used in recovery training'))
        if advance:
            run_rollouts(out, rollout_jobs(out, unseen=True), state)
        report(out)
        check_inputs(out)
        state.update(status='completed', running=[], unseen_pilot_executed=advance,
                     source_inputs_unchanged=True, baseline_replaced=False)
        write(out / 'status.json', state)
    except BaseException as error:
        state.update(status='failed', error=str(error)); write(out / 'status.json', state)
        raise
    finally:
        for _, child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for _, child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=['control', 'recovery'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.output.resolve(), args.worker)
    else:
        controller(args.output.resolve())
