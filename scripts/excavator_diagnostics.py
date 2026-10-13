"""Recorded-episode setup shared by rollout diagnostics and dataset audits."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def exporter_module():
    path = BASE / 'ExcavatorVLA-isaac/excavator_dataset_tools.py'
    spec = importlib.util.spec_from_file_location('excavator_export_diagnostics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_initial_pose(root, episode_index=0):
    root = Path(root)
    metadata = pd.concat([pd.read_parquet(p) for p in sorted(root.glob('meta/episodes/**/*.parquet'))])
    selected = metadata[metadata.episode_index == episode_index]
    if len(selected) != 1:
        raise ValueError(f'Dataset episode {episode_index} is missing or duplicated')
    row = selected.iloc[0]
    info = json.loads((root / 'meta/info.json').read_text())
    path = root / info['data_path'].format(chunk_index=int(row['data/chunk_index']),
                                          file_index=int(row['data/file_index']))
    frames = pd.read_parquet(path)
    first = frames[(frames.episode_index == episode_index) & (frames.frame_index == 0)]
    if len(first) != 1:
        raise ValueError('Expected exactly one initial observation')
    q = np.asarray(first.iloc[0]['observation.state'][:4], dtype=np.float64)
    return np.degrees(q).tolist()


def reference_episode(path, fps):
    path = Path(path)
    meta = json.loads((path / 'meta.json').read_text())
    if not meta.get('success'):
        raise ValueError('Reference must be a successful expert episode')
    samples = [json.loads(line) for line in (path / 'trajectory.jsonl').read_text().splitlines() if line]
    exporter = exporter_module()
    applied = meta['scene_randomization']['applied']
    profile = {
        'sand_xy': applied['sand_center'][:2],
        'truck_translation_xyz': applied['truck_translation_xyz'],
        'truck_yaw_deg': applied['truck_yaw_deg'],
        'unload_landing_xyz': meta['unload_landing_xyz'],
    }
    task = exporter.build_episode_task_text(samples[0], meta, state_names=meta['state_names'])
    transformed = exporter.apply_export_time_policy_to_trajectory(samples, base_fps=fps, state_names=meta['state_names'])
    return {
        'profile': profile, 'task': task,
        'initial_joints_deg': np.degrees(samples[0]['obs.q']).tolist(),
        'dig_target_xyz': meta['target_xyz'],
        'sand_amount': applied.get('sand_amount_multiplier', 1.0),
        'expected_initial_state': samples[0]['observation.state'],
        'actions': np.asarray([row['action'] for row in transformed['samples']], dtype=np.float32),
    }
