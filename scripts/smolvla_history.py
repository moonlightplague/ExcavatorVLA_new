"""Local SmolVLA causal visual history; pretrained modules and action space are unchanged."""
import copy
import json
from pathlib import Path
import sys

import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'ExcavatorVLA-vlm'))
from excavator_vlm.data.contract import CAMERAS
from excavator_vlm.policies.pi05.history import (
    MASK, ObservationHistory as PI05ObservationHistory, make_history,
    temporal_images, window_mask,
)

VERSION = 'excavator_smolvla_visual_history_v1'
MANIFEST = 'observation_history.json'


def history_spec(indices=(-6, -3, 0)):
    sampling = make_history(indices)
    if len(indices) != 3:
        raise ValueError('SmolVLA visual history requires exactly three causal timestamps')
    return {**sampling, 'version': VERSION, 'state_encoding': 'current_normalized_vector',
            'action_alignment': 'current_observation_then_actions_0_to_H_minus_1'}


def validate_history(spec):
    if not isinstance(spec, dict) or spec != history_spec(spec.get('delta_indices', [])):
        raise ValueError('Invalid or incompatible SmolVLA history manifest')
    return copy.deepcopy(spec)


def bind_history(config, spec):
    if config.n_obs_steps != 3:
        raise ValueError('History checkpoint requires n_obs_steps=3')
    if tuple(config.image_features) != CAMERAS or config.empty_cameras:
        raise ValueError('History needs exactly canonical cameras 0, 1, 2')
    config.excavator_observation_history = validate_history(spec)
    return config


class ObservationHistory(PI05ObservationHistory):
    """Reuse PI05's consecutive-control-step addressing, cloning and episode padding."""
    def __init__(self, spec):
        validate_history(spec)
        super().__init__(make_history(spec['delta_indices'], spec['fps']))


def configure_dataset(dataset, spec):
    spec = validate_history(spec)
    if dataset.fps != spec['fps']:
        raise ValueError('Dataset and history control cadence differ')
    dataset.delta_timestamps = dict(dataset.delta_timestamps or {})
    dataset.delta_indices = dict(dataset.delta_indices or {})
    for key in ('observation.state', *CAMERAS):
        dataset.delta_timestamps[key] = [i / spec['fps'] for i in spec['delta_indices']]
        dataset.delta_indices[key] = list(spec['delta_indices'])
    return dataset


def policy_batch(batch):
    """Explicit input allowlist: evaluation phases/rewards/future observations never enter the model."""
    allowed = {'task', 'observation.state', *CAMERAS, MASK, 'action', 'action_is_pad', 'action_loss_weight'}
    allowed.update(f'{key}_is_pad' for key in ('observation.state', *CAMERAS))
    return {key: value for key, value in batch.items() if key in allowed}


def inference_batch(batch):
    return {key:value for key,value in policy_batch(batch).items()
            if key not in ('action','action_is_pad','action_loss_weight')}


class HistoryPreprocessor:
    def __init__(self, pipeline, spec):
        self.pipeline, self.spec = pipeline, validate_history(spec)

    def __call__(self, batch):
        batch = policy_batch(batch)
        batch[MASK] = window_mask(batch, self.spec)
        if not torch.isfinite(batch['observation.state']).all():
            raise ValueError('Nonfinite SmolVLA state')
        for key in CAMERAS:
            image = batch[key]
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError('SmolVLA expects finite RGB in [0,1]')
        return self.pipeline(batch)

    def save_pretrained(self, *args, **kwargs):
        return self.pipeline.save_pretrained(*args, **kwargs)


def policy_class():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    class HistorySmolVLAPolicy(SmolVLAPolicy):
        def __init__(self, config, **kwargs):
            self.history = validate_history(config.excavator_observation_history)
            bind_history(config, self.history)
            super().__init__(config, **kwargs)

        def prepare_images(self, batch):
            return temporal_images(batch, self.history, super().prepare_images)

        def forward(self, batch, noise=None, time=None, reduction='mean'):
            from train_smolvla import excavator_forward
            return excavator_forward(self, batch, noise=noise, time=time, reduction=reduction)

        def predict_action_chunk(self, batch, **kwargs):
            batch = {key:value for key,value in batch.items()
                     if key not in ('action','action_is_pad','action_loss_weight')}
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=next(self.parameters()).device.type == 'cuda'):
                return super().predict_action_chunk(batch, **kwargs)

        def _save_pretrained(self, save_directory):
            super()._save_pretrained(save_directory)
            (Path(save_directory) / MANIFEST).write_text(json.dumps(self.history, indent=2) + '\n')
            assets = Path(save_directory) / 'backbone_assets'
            self.model.vlm_with_expert.processor.save_pretrained(assets)
            self.model.vlm_with_expert.vlm.config.save_pretrained(assets)

    return HistorySmolVLAPolicy


def load_policy(checkpoint, config=None, *, initial_base=False, device=None):
    """Strict full-weight loading, with explicit base-to-history initialization only for training."""
    from lerobot.configs.policies import PreTrainedConfig
    checkpoint = Path(checkpoint).resolve()
    saved = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config = saved if config is None else config
    if initial_base:
        if saved.n_obs_steps != 1 or (checkpoint / MANIFEST).exists():
            raise ValueError('Initialization needs an unchanged single-timestamp SmolVLA base')
        spec = config.excavator_observation_history
    else:
        spec = validate_history(json.loads((checkpoint / MANIFEST).read_text()))
        if saved.n_obs_steps != 3:
            raise ValueError('Saved model and history manifest disagree')
        requested = getattr(config, 'excavator_observation_history', spec)
        if requested != spec:
            raise ValueError('Resume/loading cannot change temporal sampling')
    bind_history(config, spec)
    if device is not None:
        config.device = str(device)
    config.load_vlm_weights = False
    if (checkpoint / 'backbone_assets/config.json').is_file():
        config.vlm_model_name = str(checkpoint / 'backbone_assets')
    return policy_class().from_pretrained(checkpoint, config=config, strict=True, local_files_only=True)


def load_history(checkpoint, device='cpu'):
    from lerobot.policies.factory import make_pre_post_processors
    policy = load_policy(checkpoint, device=device)
    overrides = {'device_processor': {'device': str(device)}}
    if (Path(checkpoint) / 'backbone_assets/tokenizer_config.json').is_file():
        overrides['tokenizer_processor'] = {'tokenizer_name':str(Path(checkpoint) / 'backbone_assets')}
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides=overrides)
    return policy.eval(), HistoryPreprocessor(pre, policy.history), post
