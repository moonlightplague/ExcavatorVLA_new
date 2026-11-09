"""A 200-step paired ACT sampling pilot with a frozen offline diagnostic set."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ.get('ACT_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
sys.path.insert(0, str(ROOT / 'ExcavatorVLA-vlm'))
sys.path.insert(0, str(ROOT / 'scripts'))


def evaluate_completed(out, state):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from excavator_vlm.data.contract import file_hash
    from run_act_optimization import write, read

    base = read(out / 'train/uniform/resolved_settings.json')
    split = read(out / 'split.json')
    evaluation = read(out / 'offline_observations.json')['observations']
    source = Path(state['source_checkpoint'])
    continuation = state['continuation']
    protected = state['protected_inputs_sha256']
    frozen_hashes = state['source_hashes']
    for path, expected in {**protected, **frozen_hashes}.items():
        if file_hash(path) != expected:
            raise ValueError(f'Frozen input changed before evaluation: {path}')
    processor_audit = {}
    for name in ('uniform', 'focused'):
        step_file = out / 'train' / name / 'checkpoints/last/training_state/training_step.json'
        if read(step_file)['step'] != state['steps_per_arm']:
            raise ValueError(f'{name} has not completed the predeclared training budget')
    # Verify each arm's final checkpoint and saved processors before using it.
    for name in ('uniform', 'focused'):
        run = out / 'train' / name
        if read(run / 'reload_check.json')['status'] != 'passed':
            raise ValueError('Training checkpoint reload check failed')
        last = run / 'checkpoints/last/pretrained_model'
        processor_audit[name] = {}
        for filename, expected in continuation['source_processor_files'].items():
            a, b = source / filename, last / filename
            identical_file = file_hash(b) == expected
            changes = []
            if filename.endswith('.safetensors'):
                original, saved = load_file(str(a)), load_file(str(b))
                if original.keys() != saved.keys():
                    raise ValueError(f'Processor tensor keys changed: {name}/{filename}')
                for key in original:
                    x, y = original[key], saved[key]
                    if key.endswith('.count') and x.numel() == y.numel() == 1:
                        torch.testing.assert_close(x.reshape(-1), y.reshape(-1), rtol=0, atol=0)
                        if x.shape != y.shape:
                            changes.append(dict(key=key, source_shape=list(x.shape),
                                                saved_shape=list(y.shape), value=float(x.item())))
                    else:
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
            elif read(a) != read(b):
                raise ValueError(f'Processor configuration changed: {name}/{filename}')
            processor_audit[name][filename] = dict(file_sha256_identical=identical_file,
                values_identical=True, scalar_count_metadata_shape_changes=changes)
    state['status'] = 'offline_evaluation'
    state.pop('error', None)
    write(out / 'processor_audit.json', processor_audit)
    write(out / 'status.json', state)
    # One fixed observation batch is decoded once, then evaluated by all arms.
    # No data or policy fitting takes place during these forward passes.
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['HF_HOME'] = base['launcher']['hf_home']
    from torch.utils.data import default_collate
    from excavator_vlm.policies.act.dataset import make_act_dataset
    from excavator_vlm.policies.act.inference import load_act
    from excavator_vlm.evaluation.offline_metrics import error_metrics
    torch.set_num_threads(4)
    checkpoints = dict(baseline=source, **{name: out / 'train' / name /
        'checkpoints/last/pretrained_model' for name in ('uniform', 'focused')})
    policies = {name: load_act(path, 'cuda') for name, path in checkpoints.items()}
    predictions = {name: [] for name in checkpoints}
    targets, padding = [], []
    for split_name in ('train', 'validation'):
        chosen = [row for row in evaluation if row['split'] == split_name]
        dataset = make_act_dataset('local/excavator', Path(base['dataset']['root']),
                                   split[split_name], 10)
        identity = dataset.hf_dataset.select_columns(['episode_index', 'frame_index']).to_pandas()
        lookup = {(int(ep), int(frame)): i for i, (ep, frame) in
                  enumerate(zip(identity.episode_index, identity.frame_index))}
        for start in range(0, len(chosen), 16):
            batch = default_collate([dataset[lookup[(row['episode_index'], row['frame_index'])]]
                                     for row in chosen[start:start+16]])
            observation = {key: batch[key] for key in batch if key.startswith('observation.images.')
                           or key == 'observation.state'}
            with torch.inference_mode():
                for name, (policy, pre, post) in policies.items():
                    chunk = policy.predict_action_chunk(pre(observation))
                    predictions[name].append(post(chunk.reshape(-1, 4)).reshape_as(chunk).cpu().numpy())
            targets.append(batch['action'].numpy())
            padding.append(batch['action_is_pad'].numpy())
        print(f'Offline {split_name}: {len(chosen)} observations, three policies', flush=True)
    target, pad = np.concatenate(targets), np.concatenate(padding)
    predictions = {name: np.concatenate(value) for name, value in predictions.items()}
    summary = {}
    for name, predicted in predictions.items():
        result = {}
        for split_name in ('train', 'validation'):
            ids = np.array([row['split'] == split_name for row in evaluation])
            result[split_name] = {}
            stages = sorted({row['stage_id'] for row in evaluation if row['split'] == split_name})
            for stage in stages:
                mask = ids & np.array([row['stage_id'] == stage for row in evaluation])
                valid = ~pad[mask]
                result[split_name][str(stage)] = dict(observations=int(mask.sum()),
                    first_action=error_metrics(predicted[mask, 0], target[mask, 0],
                                               valid[:, 0], [6., 2.5, 2.5, 6.]),
                    first_three=error_metrics(predicted[mask, :3], target[mask, :3],
                                              valid[:, :3], [6., 2.5, 2.5, 6.]),
                    chunk=error_metrics(predicted[mask], target[mask], valid, [6., 2.5, 2.5, 6.]))
        summary[name] = result
    np.savez_compressed(out / 'offline_predictions.npz', target=target, padding=pad, **predictions)
    write(out / 'offline_metrics.json', dict(scope=state['scope'], policies=summary,
        observation_manifest=str(out / 'offline_observations.json'), test_used=False))
    for path, expected in {**protected, **frozen_hashes}.items():
        if file_hash(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    state.update(status='completed', inputs_unchanged=True, processor_values_unchanged=True,
                 processor_audit=processor_audit, checkpoints_reloaded=True)
    write(out / 'status.json', state)
    print(json.dumps(dict(status='completed', output=str(out))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-evaluation', action='store_true',
                        help='Evaluate existing completed arms without any optimizer updates')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.resume_evaluation:
        from run_act_optimization import read, write
        state = read(out / 'status.json')
        write(out / 'before_resume_evaluation.json', state)
        shutil.copy2(__file__, out / 'resume_act_transition_pilot.py')
        try:
            evaluate_completed(out, state)
        except BaseException as error:
            state.update(status='failed', error=str(error))
            write(out / 'status.json', state)
            raise
        return
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import yaml
    from excavator_vlm.policies.act.config import resolve_settings, build_train_config
    from excavator_vlm.policies.act.train import prepare
    from excavator_vlm.policies.act.dataset import training_stats
    from excavator_vlm.policies.act.continuation import validate_continuation
    from excavator_vlm.data.contract import file_hash
    from run_act_optimization import sampling_audit, write, read

    source = ROOT / 'outputs/train/excavator_act_phw_20261028_150222_336724_1423224/'
    source = source / 'checkpoints/040000/pretrained_model'
    fit = read(ROOT / 'outputs/analysis/act_incremental_20261103_02/result.json')
    if (fit['status'] != 'completed' or not fit['pass_50_percent_train_fit'] or
            file_hash(source / 'model.safetensors') != fit['checkpoint_sha256']):
        raise ValueError('The local fit gate must pass on this exact PHW 40k checkpoint first')
    config_path = ROOT / 'ExcavatorVLA-vlm/configs/act/full_phw.yaml'
    base = resolve_settings(config_path, dict(steps=200, batch_size=16, num_workers=2,
        save_freq=200, log_freq=25, output_dir=str(out / 'unused'),
        benchmark=dict(validation_freq=0), launcher=dict(gpu_ids=[0], cpu_threads=4)))
    report, split, frames, episodes = prepare(base)
    continuation = validate_continuation(source, build_train_config(base).policy,
                                         report, split, episodes, training_stats(frames, episodes))
    selected = frames[frames.episode_index.isin(episodes) &
                      frames['observation.stage_current_id'].isin([5, 6])]
    manifest = dict(schema_version='act_transition_windows_v2', labels_changed=False,
        training_episodes=episodes, context_frames=0, action_chunk_size=10,
        selection='only actual train observation stages 5 and 6; no unload windows',
        windows=[dict(episode_index=int(row.episode_index), observation_frame=int(row.frame_index))
                 for row in selected.itertuples()])
    write(out / 'focused_observations.json', manifest)
    sampling = sampling_audit(frames, episodes, manifest)
    sampling['focused'] = sampling.pop('transition')
    write(out / 'sampling_audit.json', sampling)
    evaluation = []
    for split_name in ('train', 'validation'):
        rows = frames[frames.episode_index.isin(split[split_name])]
        for stage, group in rows.groupby('observation.stage_current_id', sort=True):
            if split_name == 'train' and stage not in (5, 6):
                continue
            group = group.sort_values(['episode_index', 'frame_index'])
            if stage not in (5, 6):
                group = group.iloc[np.linspace(0, len(group)-1, min(32, len(group)), dtype=int)]
            evaluation.extend(dict(split=split_name, stage_id=int(stage),
                episode_index=int(row.episode_index), frame_index=int(row.frame_index))
                for row in group.itertuples())
    write(out / 'offline_observations.json', dict(selection='all train stages 5/6; all validation '
        'stages 5/6 and at most 32 evenly spaced validation frames per other stage',
        test_used=False, observations=evaluation))
    frozen = out / 'frozen_sources/ExcavatorVLA-vlm'
    shutil.copytree(ROOT / 'ExcavatorVLA-vlm/excavator_vlm', frozen / 'excavator_vlm',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (frozen / 'scripts').mkdir()
    shutil.copy2(ROOT / 'ExcavatorVLA-vlm/scripts/train_act.py', frozen / 'scripts/train_act.py')
    shutil.copy2(__file__, out / Path(__file__).name)
    write(out / 'split.json', split)
    frozen_hashes = {str(p): file_hash(p) for p in frozen.rglob('*.py')}
    protected_paths = [*source.glob('*'), *Path(base['dataset']['root']).glob('data/**/*.parquet'),
                       Path(base['dataset']['root']) / 'meta/info.json']
    protected = {str(p): file_hash(p) for p in protected_paths if p.is_file()}
    state = dict(status='prepared', source_checkpoint=str(source), steps_per_arm=200,
        global_batch_size=16, seed=1000, learning_rate=1e-5,
        intervention='only observation sampling distribution', continuation=continuation,
        scope='small offline diagnostic pilot, not closed-loop task evaluation',
        offline_observations=len(evaluation), source_hashes=frozen_hashes,
        protected_inputs_sha256=protected, training={})
    write(out / 'status.json', state)
    (out / 'configs').mkdir()
    jobs, handles = [], []
    try:
        for gpu, (name, probability) in enumerate((('uniform', 0.), ('focused', .3))):
            cfg = json.loads(json.dumps(base))
            cfg['benchmark']['split_file'] = str(out / 'split.json')
            cfg['continuation'] = dict(checkpoint=str(source))
            cfg['sampling'] = dict(manifest=str(out / 'focused_observations.json'),
                                   transition_probability=probability)
            cfg['launcher']['gpu_ids'] = [gpu]
            cfg['output_dir'] = str(out / 'train' / name)
            cfg['job_name'] = 'act_transition_pilot_' + name
            path = out / 'configs' / (name + '.yaml')
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            handle = (out / (name + '.log')).open('x')
            handles.append(handle)
            env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            command = [sys.executable, str(frozen / 'scripts/train_act.py'), '--config', str(path)]
            job = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
            jobs.append((name, job))
            state['training'][name] = dict(pid=job.pid, command=command, gpu=gpu)
        while any(job.poll() is None for _, job in jobs):
            state['status'] = 'training'
            for name, job in jobs:
                state['training'][name]['returncode'] = job.poll()
                if job.poll() not in (None, 0):
                    raise RuntimeError(f'{name} training failed; see {out / (name + ".log")}')
            write(out / 'status.json', state)
            time.sleep(5)
        for name, job in jobs:
            state['training'][name]['returncode'] = job.returncode
            if job.returncode:
                raise RuntimeError(f'{name} training failed')
        evaluate_completed(out, state)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        write(out / 'status.json', state)
        raise
    finally:
        for _, job in jobs:
            if job.poll() is None:
                job.terminate()
        for _, job in jobs:
            if job.poll() is None:
                try:
                    job.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    job.kill()
                    job.wait()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
