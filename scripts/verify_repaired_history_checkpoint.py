"""Independently load a new repaired-data model and infer from one held-out observation."""
import argparse
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'ExcavatorVLA-vlm'), str(BASE / 'scripts')]


def verify(experiment, model, step):
    import torch
    from torch.utils.data import default_collate
    from excavator_vlm.policies.pi05.dataset import make_pi05_dataset
    from excavator_vlm.policies.pi05.history import make_history, observation_batch
    from excavator_vlm.data.contract import file_hash
    from success_eval_runtime import atomic_json

    prepared = experiment / 'prepared'
    split = json.loads((prepared / 'split_manifest.json').read_text())
    audit = json.loads((prepared / 'dataset_audit.json').read_text())
    train = experiment / model / 'train'
    directory = train / 'checkpoints' / (f'step_{step}' if model == 'smolvla' else f'{step:06d}')
    checkpoint = directory / 'pretrained_model'
    provenance = json.loads((checkpoint / 'benchmark_provenance.json').read_text())
    if (provenance['dataset_fingerprint'] != audit['fingerprint']
            or provenance['action_contract'] != audit['action_contract']
            or provenance['training_episodes'] != split['train']):
        raise ValueError('Saved checkpoint differs from the repaired data/action/split contract')
    dataset = make_pi05_dataset('local/excavator', audit['dataset'], [split['validation'][0]],
                                10, 'pyav', make_history([-6, -3, 0]))
    raw = observation_batch(default_collate([dataset[20]]))
    if model == 'smolvla':
        from smolvla_history import load_history, STATE_HISTORY_VERSION
        policy, pre, post = load_history(checkpoint, 'cuda')
        if policy.history['version'] != STATE_HISTORY_VERSION or policy.model.state_proj.in_features != 96:
            raise ValueError('SmolVLA checkpoint lost numeric-state history')
    else:
        from excavator_vlm.policies.pi05.inference import load_pi05
        policy, pre, post = load_pi05(checkpoint, 'cuda')
        if policy.config.n_obs_steps != 3:
            raise ValueError('PI05 checkpoint lost causal history')
        stats = json.loads((checkpoint / 'normalization_stats.json').read_text())
        if stats != json.loads((prepared / 'normalization_stats.json').read_text()):
            raise ValueError('PI05 statistics differ from shared train-only statistics')
    torch.manual_seed(1000)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = policy.predict_action_chunk(pre(raw))
        physical = torch.stack([post(prediction[:, i]) for i in range(prediction.shape[1])], 1)
    if physical.shape != (1, 10, 4) or not torch.isfinite(physical).all():
        raise ValueError('Reloaded model did not produce a finite physical action chunk')
    report = dict(status='passed', model=model, step=step, checkpoint=str(checkpoint),
        observation_split='validation', observation_episode=split['validation'][0],
        action_shape=list(physical.shape), action_contract=provenance['action_contract'],
        dataset_fingerprint=audit['fingerprint'], weight_sha256=file_hash(checkpoint / 'model.safetensors'),
        scope='strict independent reload and inference; no closed-loop success estimate')
    atomic_json(experiment / model / f'checkpoint_check_{step:06d}.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--model', choices=('smolvla', 'pi05'), required=True)
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false')
    verify(args.experiment.resolve(), args.model, args.step)
