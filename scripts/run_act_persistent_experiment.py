"""Frozen ACT feature caching and episode-prefix recurrent imitation training.

Only model weights change. The original episode split, processors, targets and
rollout entrypoints are retained. Run using the private persistent ACT runtime.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ.get('ACT_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
SOURCE = ROOT / 'outputs/experiments/act_temporal_repaired10k_20261118_01/train/checkpoints/010000/pretrained_model'
DATA = ROOT / 'outputs/experiments/act_repeat_control_repair_data_20261118_01/dataset'
EXTERNAL = ['scripts/eval_smolvla_success.py', 'scripts/eval_smolvla_repeat40.py',
            'scripts/repeat_eval.py', 'ExcavatorVLA-smolvla/run_simulation.py']


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    return json.loads((SOURCE / 'benchmark_provenance.json').read_text())


def model():
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from safetensors.torch import load_file
    config = PreTrainedConfig.from_pretrained(str(SOURCE))
    config.device = 'cuda:0'
    config.persistent_memory = True
    config.memory_hidden_dim = 256
    policy = ACTPolicy(config).cuda().eval()
    result = policy.load_state_dict(load_file(str(SOURCE / 'model.safetensors')), strict=False)
    expected = {k for k in policy.state_dict() if k.startswith('model.memory_adapter.')}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(str(result))
    pre, post = make_pre_post_processors(config, pretrained_path=str(SOURCE),
        preprocessor_overrides={'device_processor': {'device': 'cuda:0'}})
    return policy, pre, post


def video_frames(path, frames, offset, fps):
    import av
    import numpy as np
    frames = iter(frames)
    wanted = next(frames, None)
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 1
        container.streams.video[0].thread_type = 'SLICE'
        for frame in container.decode(video=0):
            if wanted is None:
                break
            timestamp = float(frame.pts * frame.time_base)
            target = offset + wanted / fps
            if timestamp < target - 0.5 / fps:
                continue
            if abs(timestamp - target) > 0.02:
                raise ValueError(f'Video timestamp mismatch {path}: {timestamp} vs {target}')
            yield np.ascontiguousarray(frame.to_ndarray(format='rgb24').transpose(2, 0, 1))
            wanted = next(frames, None)
    if wanted is not None:
        raise ValueError(f'Missing video frame {wanted} in {path}')


def cache(args):
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    torch.set_num_threads(4)
    torch.manual_seed(20261119)
    policy, pre, _ = model()
    split = provenance()['split_manifest']
    selected = sorted(split['train'] + split['val']) if 'val' in split else sorted(split['train'] + split['validation'])
    assert not set(selected) & set(split['test'])
    selected = selected[args.worker::2]
    rows = [row for path in sorted((DATA/'meta/episodes').rglob('*.parquet'))
            for row in pq.read_table(path).to_pylist()]
    rows = {r['episode_index']: r for r in rows}
    cameras = list(policy.config.image_features)
    folder = args.output / 'features'
    folder.mkdir(exist_ok=True)
    for episode in selected:
        output = folder / f'episode_{episode:04d}.npz'
        if output.exists():
            raise FileExistsError(output)
        start = time.time()
        row = rows[episode]
        table = pq.read_table(DATA / f'data/chunk-{row["data/chunk_index"]:03d}/file-{row["data/file_index"]:03d}.parquet')
        table = table.filter(__import__('pyarrow').compute.equal(table['episode_index'], episode))
        states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        if len(states) != row['length']:
            raise ValueError('Episode boundary mismatch')
        frames = np.arange(0, len(states), policy.config.n_action_steps)
        future = frames[:, None] + np.arange(policy.config.chunk_size)
        padding = future >= len(states)
        targets = actions[np.minimum(future, len(states)-1)]
        normal = pre({'observation.state': torch.from_numpy(states[frames]), 'action': torch.from_numpy(targets)})
        normalized_state = normal['observation.state']
        normalized_target = normal['action']
        streams = [video_frames(DATA / f'videos/{cam}/chunk-{row[f"videos/{cam}/chunk_index"]:03d}/file-{row[f"videos/{cam}/file_index"]:03d}.mp4',
                    frames, row[f'videos/{cam}/from_timestamp'], 10) for cam in cameras]
        pooled_history = []
        features, base_chunks = [], []
        policy.reset()
        for offset in range(0, len(frames), args.cache_batch):
            count = min(args.cache_batch, len(frames)-offset)
            images = {cam: torch.from_numpy(np.stack([next(stream) for _ in range(count)])).float().div_(255)
                      for cam, stream in zip(cameras, streams, strict=True)}
            current = pre({'observation.state': torch.from_numpy(states[frames[offset:offset+count]]), **images})
            with torch.no_grad():
                spatial = [policy.model.backbone(current[cam])['feature_map'] for cam in cameras]
                pooled = torch.stack([x.mean((-2, -1)) for x in spatial], dim=1)
                pooled_history.append(pooled)
                all_pooled = torch.cat(pooled_history)
                idx = torch.arange(offset, offset+count, device='cuda')[:, None] + torch.arange(-2, 1, device='cuda')
                hist_pad = idx < 0
                idx = idx.clamp_min(0)
                feature, base = policy.model.features_from_spatial(normalized_state[idx], all_pooled[idx], hist_pad, spatial)
                if offset == 0:
                    # Real pixel parity: cached frozen maps and the ordinary public API.
                    raw = {k: current[k][:1] for k in ('observation.state', *cameras)}
                    actual = policy.predict_action_chunk(raw)
                    torch.testing.assert_close(actual, base[:1], rtol=1e-4, atol=2e-5)
                    policy.reset()
                features.append(feature.cpu().numpy())
                base_chunks.append(base.cpu().numpy())
        for stream in streams:
            # Exhaust the generators to execute their missing-frame checks/close files.
            if next(stream, None) is not None:
                raise ValueError('Unexpected extra selected video frame')
        tmp = output.with_suffix('.tmp.npz')
        np.savez(tmp, features=np.concatenate(features), base=np.concatenate(base_chunks),
                 target=normalized_target.cpu().numpy(), padding=padding, frame_index=frames)
        tmp.replace(output)
        write(args.output / f'cache_worker_{args.worker}.json',
              dict(episode=episode, decisions=len(frames), seconds=time.time()-start,
                   completed=len(list(folder.glob('*.npz'))), selected_total=160))
        print(f'cached episode={episode} decisions={len(frames)} seconds={time.time()-start:.1f}', flush=True)
    write(args.output / f'cache_worker_{args.worker}_done.json', dict(episodes=selected))


def validation(memory, episodes, action_std):
    import torch
    import numpy as np
    totals = np.zeros(4, dtype=np.float64)
    memory.eval()
    with torch.no_grad():
        for episode in episodes:
            feat, base, target, pad = episode
            delta, _ = memory(feat[None])
            pred = base + delta[0].reshape_as(base)
            valid = (~pad).unsqueeze(-1).expand_as(base)
            err = (pred-target).abs() * action_std
            baseline = (base-target).abs() * action_std
            totals += [err[valid].sum().item(), baseline[valid].sum().item(), valid.sum().item(),
                       err[:, 0].sum().item()]
    memory.train()
    return dict(physical_mae=totals[0]/totals[2], source_physical_mae=totals[1]/totals[2],
                first_action_mae=totals[3]/sum(len(ep[0])*4 for ep in episodes))


def train(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    torch.manual_seed(20261119)
    rng = np.random.default_rng(20261119)
    policy, pre, post = model()
    memory = policy.model.memory_adapter
    split = provenance()['split_manifest']
    val_ids = split.get('val', split.get('validation'))
    def load(episode):
        with np.load(args.output / 'features' / f'episode_{episode:04d}.npz') as z:
            return tuple(torch.as_tensor(z[k], device='cuda') for k in ('features', 'base', 'target', 'padding'))
    training = [load(e) for e in split['train']]
    val = [load(e) for e in val_ids]
    probabilities = np.array([len(e[0]) for e in training], dtype=np.float64)
    probabilities /= probabilities.sum()
    stats = json.loads((SOURCE.parents[2] / 'normalization_stats.json').read_text())
    action_std = torch.tensor(stats['action']['std'], device='cuda', dtype=torch.float32)
    optimizer = torch.optim.AdamW(memory.parameters(), lr=1e-4, weight_decay=1e-4)
    baseline = validation(memory, val, action_std)
    write(args.output / 'validation_initial.json', baseline)
    unroll = 32
    batch_size = 8
    memory.train()
    started = time.time()
    for step in range(1, args.steps+1):
        chosen = rng.choice(len(training), batch_size, p=probabilities)
        starts = [int(rng.integers(0, max(1, len(training[e][0])-unroll+1))) for e in chosen]
        lengths = torch.tensor(starts, dtype=torch.long)
        prefix = torch.zeros(batch_size, max(max(starts), 1), training[0][0].shape[-1], device='cuda')
        feature, bases, targets, masks = [], [], [], []
        for i, (e, start) in enumerate(zip(chosen, starts, strict=True)):
            feat, base, target, pad = training[e]
            prefix[i, :start] = feat[:start]
            feature.append(feat[start:start+unroll]); bases.append(base[start:start+unroll])
            targets.append(target[start:start+unroll]); masks.append(pad[start:start+unroll])
        hidden = memory.prefix_hidden(prefix, lengths)
        delta, _ = memory(torch.stack(feature), hidden)
        base, target, pad = torch.stack(bases), torch.stack(targets), torch.stack(masks)
        prediction = base + delta.reshape_as(base)
        valid = (~pad).unsqueeze(-1).expand_as(prediction)
        l1 = (prediction-target).abs()[valid].mean()
        loss = l1 + 1e-4 * delta.square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite recurrent training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(memory.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0 or step == 1:
            status = dict(status='training', step=step, steps=args.steps, loss=loss.item(),
                          grad_norm=norm.item(), elapsed_seconds=time.time()-started)
            write(args.output / 'status.json', status)
            print(json.dumps(status), flush=True)
        if step % 1000 == 0 or step == args.steps:
            metrics = validation(memory, val, action_std)
            write(args.output / f'validation_{step:06d}.json', metrics)
            print(json.dumps(dict(step=step, validation=metrics)), flush=True)
    source_weights = load_file(str(SOURCE / 'model.safetensors'))
    for name, tensor in source_weights.items():
        if not torch.equal(policy.state_dict()[name].cpu(), tensor):
            raise ValueError(f'Frozen source model changed: {name}')
    destination = args.output / f'train/checkpoints/{args.steps:06d}/pretrained_model'
    destination.mkdir(parents=True)
    policy.save_pretrained(destination)
    for path in SOURCE.glob('policy_*'):
        shutil.copy2(path, destination / path.name)
    meta = provenance()
    meta['persistent_memory'] = dict(version='act_persistent_action_memory_v1', source=str(SOURCE),
        source_sha256=digest(SOURCE/'model.safetensors'), trainable='model.memory_adapter.*',
        hidden_dim=256, unroll=unroll, optimizer_steps=args.steps, learning_rate=1e-4,
        prefix='full same-episode causal prefix recomputed with current weights; detached before unroll',
        loss='uniform valid-action normalized L1 plus 1e-4 residual squared penalty',
        frozen_core_bitwise_unchanged=True, task_metadata_inputs=False)
    write(destination / 'benchmark_provenance.json', meta)
    shutil.copy2(SOURCE.parents[2] / 'normalization_stats.json', args.output / 'train/normalization_stats.json')
    torch.save(dict(optimizer=optimizer.state_dict(), step=args.steps, rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()),
               destination.parent / 'training_state.pt')
    from excavator_vlm.policies.act.inference import load_act
    reloaded, _, _ = load_act(destination, 'cuda:0', original_act=True)
    with torch.no_grad():
        test = val[0][0][:32][None]
        expected = memory(test)[0]
        actual = reloaded.model.memory_adapter(test)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, sha in json.loads((args.output/'external_files.json').read_text()).items():
        if digest(ROOT/name) != sha:
            raise ValueError('External evaluator/controller file changed: '+name)
    write(args.output/'status.json', dict(status='completed', step=args.steps, checkpoint=str(destination),
        validation=metrics, frozen_core_bitwise_unchanged=True, strict_reload='passed', external_files_unchanged=True))


def orchestrate(args):
    runtime = Path(args.runtime).resolve()
    preflight = runtime.parent / 'preflight.json'
    if (not preflight.is_file() or json.loads(preflight.read_text()).get('status') != 'passed'
            or 'batch16_max_difference' not in json.loads(preflight.read_text())):
        raise RuntimeError('Real checkpoint, pixel-cache and reload checks must pass before training')
    from excavator_vlm.data.contract import audit_dataset
    audit, _ = audit_dataset(DATA)
    if audit['fingerprint'] != provenance()['dataset_fingerprint']:
        raise ValueError('Source dataset changed since the source model was trained')
    previous_external = ROOT/'outputs/experiments/act_temporal_runtime_20261118_02/unchanged_external_files.json'
    for name, sha in json.loads(previous_external.read_text()).items():
        if digest(ROOT/name) != sha:
            raise ValueError('External file changed since previous experiment: '+name)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(preflight, args.output/'preflight.json')
    shutil.copy2(runtime.parent/'runtime_manifest.json', args.output/'runtime_manifest.json')
    write(args.output/'dataset_audit.json', audit)
    snapshot = args.output/'run.py'
    shutil.copy2(Path(__file__), snapshot)
    write(args.output/'external_files.json', {p:digest(ROOT/p) for p in EXTERNAL})
    write(args.output/'experiment.json', dict(source=str(SOURCE), source_sha256=digest(SOURCE/'model.safetensors'),
        dataset=str(DATA), fingerprint=provenance()['dataset_fingerprint'], split_manifest=provenance()['split_manifest'],
        steps=args.steps, runtime=sys.path[0], inputs='current 28D state + three RGB views only',
        training='frozen ACT core; learned episode-long GRU residual; full-prefix reconstruction'))
    write(args.output/'status.json', dict(status='caching', workers=2))
    workers = []
    runtime = Path(args.runtime).resolve()
    for worker in range(2):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker), ACT_PROJECT_ROOT=str(ROOT))
        log = (args.output/f'cache_worker_{worker}.log').open('w')
        workers.append(subprocess.Popen([str(runtime), str(snapshot), '--mode','cache','--worker',str(worker),
            '--output',str(args.output),'--cache-batch',str(args.cache_batch)], env=env, stdout=log, stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in workers):
        if any(p.returncode not in (None, 0) for p in workers):
            for p in workers:
                if p.poll() is None:
                    p.terminate()
            raise RuntimeError(f'Cache worker failed: {[p.wait() for p in workers]}')
        time.sleep(2)
    codes = [p.wait() for p in workers]
    if any(codes):
        raise RuntimeError(f'Cache workers failed: {codes}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', ACT_PROJECT_ROOT=str(ROOT))
    with (args.output/'train.log').open('w') as log:
        subprocess.run([str(runtime),str(snapshot),'--mode','train','--output',str(args.output),
                        '--steps',str(args.steps)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    evaluation = args.output/'evaluation_pair'
    evaluation.mkdir()
    template = ROOT/'outputs/experiments/act_temporal_10k_repeat10_pair_20261119_01/run.py'
    text = template.read_text()
    text = text.replace('act_temporal_repaired10k_20261118_01/train/checkpoints/010000/pretrained_model',
                        str((args.output/f'train/checkpoints/{args.steps:06d}/pretrained_model').relative_to(ROOT/'outputs/experiments')))
    text = text.replace('Model-internal temporal checkpoint', 'Model-internal persistent GRU memory checkpoint')
    (evaluation/'run.py').write_text(text)
    write(args.output/'evaluation_queue.json',dict(status='running',output=str(evaluation),
          scenes=[16,35],steps=5000,repeat_cycles=10,evaluator_source_unchanged=True))
    with (evaluation/'orchestrator.log').open('w') as log:
        subprocess.run([str(runtime),'-B',str(evaluation/'run.py')], cwd=ROOT,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    write(args.output/'evaluation_queue.json',dict(status='completed',output=str(evaluation)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['run','cache','train'], default='run')
    parser.add_argument('--output', type=lambda s:Path(s).resolve(), required=True)
    parser.add_argument('--runtime', default=str(ROOT/'outputs/experiments/act_persistent_runtime_20261119_03/python'))
    parser.add_argument('--worker',type=int, default=0)
    parser.add_argument('--cache-batch',type=int, default=16)
    parser.add_argument('--steps',type=int, default=5000)
    args = parser.parse_args()
    try:
        {'run':orchestrate, 'cache':cache, 'train':train}[args.mode](args)
    except Exception as error:
        if args.output.exists():
            write(args.output/f'error_{args.mode}_{args.worker}.json', dict(error=repr(error)))
        raise
