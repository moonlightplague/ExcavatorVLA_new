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
    if path.is_file():
        return reference_setup(path, fps)
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


def reference_setup(path, fps):
    """Load an explicit, source-recorded setup without requiring raw images.

    This is a distinct format, not a fabricated raw episode. Exported actions
    are already at the requested control cadence and must not be shifted again.
    """
    from excavator_common.vla_observation_contract import STATE_NAMES_28D, ACTION_NAMES_4D

    data = json.loads(Path(path).read_text())
    if (data.get('schema_version') != 'excavator_reference_setup_v1'
            or data.get('fps') != fps or data.get('action_units') != 'rad/s'):
        raise ValueError('Reference setup schema, action units or control FPS differ')
    if data.get('state_names') != list(STATE_NAMES_28D) or data.get('action_names') != list(ACTION_NAMES_4D):
        raise ValueError('Reference setup state/action order differs from deployment')
    if not data.get('provenance') or not isinstance(data.get('task'), str) or not data['task'].strip():
        raise ValueError('Reference setup requires task text and recorded source provenance')
    result = {key: data[key] for key in ('profile', 'task', 'initial_joints_deg',
                                        'dig_target_xyz', 'sand_amount', 'expected_initial_state')}
    for name, shape in [('initial_joints_deg', (4,)), ('dig_target_xyz', (3,)),
                        ('expected_initial_state', (28,))]:
        value = np.asarray(result[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid finite reference {name}: expected {shape}')
    for name, shape in [('sand_xy', (2,)), ('truck_translation_xyz', (3,)),
                        ('truck_yaw_deg', ()), ('unload_landing_xyz', (3,))]:
        value = np.asarray(result['profile'][name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid finite reference profile {name}')
    if not np.isfinite(result['sand_amount']) or result['sand_amount'] <= 0:
        raise ValueError('Reference sand amount must be finite and positive')
    if not np.allclose(np.degrees(result['expected_initial_state'][:4]),
                       result['initial_joints_deg'], rtol=0, atol=1e-5):
        raise ValueError('Reference initial pose and expected state disagree')
    actions = np.asarray(data['actions'], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 4 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError('Reference setup requires nonempty finite [T,4] exported actions')
    result['actions'] = actions
    return result
