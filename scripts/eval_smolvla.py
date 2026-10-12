"""Offline first-action error on recorded observations, not a simulator rollout."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--num-episodes', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    meta = LeRobotDatasetMetadata('local/excavator', root=args.dataset)
    if not 1 <= args.num_episodes <= meta.total_episodes:
        parser.error(f'num-episodes must be between 1 and {meta.total_episodes}')
    dataset = LeRobotDataset('local/excavator', root=args.dataset,
                            episodes=list(range(args.num_episodes)), video_backend='pyav')
    policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint)).to('cuda').eval()
    if policy.config.input_features['observation.state'].shape != meta.features['observation.state']['shape']:
        raise ValueError('Checkpoint and dataset state dimensions differ')
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint),
        preprocessor_overrides={'device_processor': {'device': 'cuda'}},
    )
    predictions, targets, episode_ids = [], [], []
    with torch.inference_mode():
        for batch in tqdm(DataLoader(dataset, batch_size=8, num_workers=0), desc='Evaluation'):
            targets.append(batch['action'].numpy().copy())
            episode_ids.append(batch['episode_index'].numpy())
            observation = {key: batch[key] for key in policy.config.input_features if key in batch}
            observation['task'] = batch['task']
            policy.reset()
            actions = policy.predict_action_chunk(preprocess(observation))[:, 0]
            predictions.append(postprocess(actions).cpu().numpy())
    predicted, target = np.concatenate(predictions), np.concatenate(targets)
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError('Non-finite actions found')
    error = predicted - target
    names = meta.features['action']['names']
    metrics = {
        'checkpoint': str(args.checkpoint.resolve()), 'dataset': str(args.dataset.resolve()),
        'evaluation': 'offline_first_action_per_recorded_observation',
        'note': 'First N episodes; may overlap training data. Not a closed-loop success metric.',
        'seed': 0, 'episodes': args.num_episodes, 'frames': len(target), 'units': 'rad/s',
        'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt(np.mean(error ** 2))),
        'per_joint': {name: {'mae': float(np.abs(error[:, i]).mean()),
                             'rmse': float(np.sqrt(np.mean(error[:, i] ** 2)))}
                      for i, name in enumerate(names)},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    np.savez_compressed(args.output / 'actions.npz', predicted=predicted, target=target,
                        episode_index=np.concatenate(episode_ids))
    print(json.dumps(metrics, indent=2))
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
