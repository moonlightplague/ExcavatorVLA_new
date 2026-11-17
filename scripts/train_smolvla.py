"""Load the repository YAML configuration and run LeRobot SmolVLA training."""
import argparse
import copy
from datetime import datetime
import functools
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

BASE = Path(__file__).resolve().parents[1]


def resolve_path(value, root):
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def load_settings(path, run_output=None):
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text())
    if not isinstance(settings, dict):
        raise ValueError('Training YAML must contain a mapping of settings.')
    settings = copy.deepcopy(settings)
    launcher = settings.pop('launcher', {})
    if not isinstance(launcher, dict):
        raise ValueError('launcher must be a mapping.')
    unknown = set(launcher) - {'output_root', 'hf_home', 'gpu_ids', 'mixed_precision', 'micro_batch_size'}
    if unknown:
        raise ValueError(f'Unknown launcher settings: {sorted(unknown)}')
    gpu_ids = launcher.get('gpu_ids', [0])
    if (not isinstance(gpu_ids, list) or not gpu_ids
            or any(type(gpu) is not int or gpu < 0 for gpu in gpu_ids)
            or len(set(gpu_ids)) != len(gpu_ids)):
        raise ValueError('launcher.gpu_ids must be a nonempty list of unique nonnegative integers.')
    precision = launcher.get('mixed_precision', 'no')
    if precision not in ('no', 'fp16', 'bf16'):
        raise ValueError('launcher.mixed_precision must be "no", "fp16", or "bf16".')
    for section in ('dataset', 'policy'):
        if not isinstance(settings.get(section), dict):
            raise ValueError(f'{section} must be a mapping.')
    history = settings.pop('observation_history', None)
    prepared = settings.pop('prepared_data', None)
    micro_batch = launcher.get('micro_batch_size')
    if micro_batch is not None and (type(micro_batch) is not int or micro_batch < 1):
        raise ValueError('launcher.micro_batch_size must be positive')
    settings['dataset'].setdefault('root', str(BASE / 'data/repeat'))
    for section, key in [('dataset', 'root'), ('policy', 'path')]:
        if not settings[section].get(key):
            raise ValueError(f'{section}.{key} is required.')
        settings[section][key] = str(resolve_path(settings[section][key], path.parent))
    dataset_root = Path(settings['dataset']['root'])
    if not (dataset_root / 'meta/info.json').is_file() and (dataset_root / 'export_repeat/meta/info.json').is_file():
        settings['dataset']['root'] = str(dataset_root / 'export_repeat')
    launcher = {
        'gpu_ids': gpu_ids, 'mixed_precision': precision,
        'hf_home': str(resolve_path(launcher.get('hf_home', 'models/hf-cache'), path.parent)),
        'output_root': str(resolve_path(launcher.get('output_root', 'outputs/train'), path.parent)),
    }
    if history is not None:
        from prepare_smolvla_history import resolve_dataset
        from smolvla_history import history_spec
        if not isinstance(history, dict) or set(history) != {'delta_indices'}:
            raise ValueError('observation_history must specify delta_indices')
        if settings['policy'].get('n_obs_steps') != 3 or not prepared:
            raise ValueError('History training needs n_obs_steps=3 and prepared_data')
        prepared = resolve_path(prepared, path.parent)
        split = json.loads((prepared / 'split_manifest.json').read_text())
        if settings['dataset'].get('episodes') not in (None, split['train']):
            raise ValueError('Requested episodes differ from the frozen history training split')
        settings['dataset']['root'] = str(resolve_dataset(settings['dataset']['root']))
        settings['dataset']['episodes'] = split['train']
        launcher.update(observation_history=history_spec(history['delta_indices']),
                        prepared_data=str(prepared), micro_batch_size=micro_batch)
    elif settings['policy'].get('n_obs_steps', 1) != 1:
        raise ValueError('Multi-timestamp SmolVLA requires an explicit causal history contract')
    output = run_output or settings.get('output_dir')
    if output:
        settings['output_dir'] = str(resolve_path(output, path.parent))
    else:
        name = settings.get('job_name', 'excavator_smolvla')
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError('job_name must be a nonempty directory name.')
        suffix = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        settings['output_dir'] = str(Path(launcher['output_root']) / f'{name}_{suffix}_{os.getpid()}')
    return settings, launcher


def build_train_config(settings):
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    if 'stage_loss_weight' in SmolVLAConfig.__dataclass_fields__:
        raise ValueError('This environment has legacy stage patches. Use ExcavatorVLA-current.')
    settings = copy.deepcopy(settings)
    overrides = settings.pop('policy')
    model_path = Path(overrides.pop('path'))
    policy = json.loads((model_path / 'config.json').read_text())
    policy.update(overrides)
    policy['pretrained_path'] = str(model_path)
    info = json.loads((Path(settings['dataset']['root']) / 'meta/info.json').read_text())
    features = info['features']
    if features['observation.state']['shape'] != [28] or features['action']['shape'] != [4]:
        raise ValueError('Expected current-v4 28D state and 4D actions.')
    sys.path.insert(0, str(BASE / 'ExcavatorVLA-smolvla'))
    from excavator_common.deployment_contract import STATE_NAMES_28D, ACTION_NAMES_4D
    if features['observation.state']['names'] != list(STATE_NAMES_28D):
        raise ValueError('Dataset state names/order do not match the deployment contract.')
    if features['action']['names'] != list(ACTION_NAMES_4D):
        raise ValueError('Dataset action names/order do not match canonical command velocities.')
    inputs = {'observation.state': {'type': 'STATE', 'shape': [28]}}
    for name in ('0', '1', '2'):
        key = f'observation.images.{name}'
        height, width, channels = features[key]['shape']
        inputs[key] = {'type': 'VISUAL', 'shape': [channels, height, width]}
    # Replace the base model's camera names; do not merge extra legacy keys.
    policy['input_features'] = inputs
    policy['output_features'] = {'action': {'type': 'ACTION', 'shape': [4]}}
    settings['policy'] = policy
    cfg = draccus.decode(TrainPipelineConfig, settings)
    cfg.policy.device = policy['device']
    for name in ('steps', 'batch_size', 'save_freq', 'log_freq'):
        value = getattr(cfg, name)
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if cfg.num_workers < 0 or cfg.eval_freq < 0:
        raise ValueError('num_workers and eval_freq must be nonnegative.')
    if not 0 <= cfg.policy.scheduler_warmup_steps < cfg.policy.scheduler_decay_steps:
        raise ValueError('Scheduler warmup must be nonnegative and shorter than decay steps.')
    if cfg.policy.scheduler_decay_steps > cfg.steps:
        raise ValueError('policy.scheduler_decay_steps must not exceed steps.')
    return cfg


def install_trainer_adapters(trainer, cfg, history=None, stats=None):
    """Keep compatibility changes local to this process, not site-packages."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    make_dataset = trainer.make_dataset

    @functools.wraps(make_dataset)
    def make_excavator_dataset(*args, **kwargs):
        dataset = make_dataset(*args, **kwargs)
        # Recovery labels must refer to each FUTURE action, not just the
        # observation at the start of a chunk. LeRobot only chunks "action".
        if 'action_loss_weight' in dataset.features:
            dataset.delta_timestamps['action_loss_weight'] = list(dataset.delta_timestamps['action'])
            dataset.delta_indices['action_loss_weight'] = list(dataset.delta_indices['action'])
        if history is not None:
            import torch
            from smolvla_history import configure_dataset
            configure_dataset(dataset, history)
            dataset.meta.stats = {**dataset.meta.stats, **{
                key: {name: torch.tensor(value, dtype=torch.float32) for name, value in values.items()}
                for key, values in stats.items()}}
        return dataset

    trainer.make_dataset = make_excavator_dataset

    make_policy = trainer.make_policy

    @functools.wraps(make_policy)
    def make_excavator_policy(*args, **kwargs):
        if history is not None:
            from smolvla_history import bind_history, load_policy, MANIFEST
            bind_history(cfg.policy, history)
            checkpoint = Path(cfg.policy.pretrained_path)
            return load_policy(checkpoint, config=cfg.policy,
                initial_base=not (checkpoint / MANIFEST).exists() and not cfg.resume)
        # LeRobot otherwise classifies action_is_expert/action_loss_weight as
        # additional policy outputs because their names start with "action".
        meta = copy.copy(kwargs['ds_meta'])
        allowed = {*cfg.policy.input_features, 'action'}
        meta.info = {**meta.info, 'features': {
            key: value for key, value in meta.features.items() if key in allowed
        }}
        kwargs['ds_meta'] = meta
        return make_policy(*args, **kwargs)

    trainer.make_policy = make_excavator_policy
    make_processors = trainer.make_pre_post_processors

    @functools.wraps(make_processors)
    def make_excavator_processors(*args, **kwargs):
        overrides = kwargs.setdefault('preprocessor_overrides', {})
        overrides.setdefault('tokenizer_processor', {}).update(
            max_length=cfg.policy.tokenizer_max_length, padding=cfg.policy.pad_language_to,
        )
        pre, post = make_processors(*args, **kwargs)
        if history is not None:
            from smolvla_history import HistoryPreprocessor
            pre = HistoryPreprocessor(pre, history)
        return pre, post

    trainer.make_pre_post_processors = make_excavator_processors
    SmolVLAPolicy.forward = excavator_forward


def microbatch_update(metrics, policy, batch, optimizer, grad_clip_norm, *, accelerator,
                      lr_scheduler=None, lock=None, rabc_weights_provider=None, micro_batch_size=1):
    """Split one logical batch while preserving the existing valid-action loss denominator."""
    import contextlib
    import torch
    if rabc_weights_provider is not None or lock is not None:
        raise ValueError('History microbatching does not support RA-BC or an external optimizer lock')
    started = time.perf_counter()
    size = batch['action'].shape[0]
    weights = torch.ones_like(batch['action'][..., 0])
    if 'action_is_pad' in batch:
        weights = weights * ~batch['action_is_pad']
    if 'action_loss_weight' in batch:
        supervision = batch['action_loss_weight']
        if supervision.ndim == 3:
            supervision = supervision.squeeze(-1)
        weights = weights * supervision
    total = weights.sum().detach().clamp_min(1)
    value = 0.
    optimizer.zero_grad(set_to_none=True)
    policy.train()
    for start in range(0, size, micro_batch_size):
        end = min(size, start + micro_batch_size)
        part = {key: item[start:end] if (torch.is_tensor(item) and item.ndim and item.shape[0] == size)
                or (isinstance(item, list) and len(item) == size) else item for key, item in batch.items()}
        synchronization = accelerator.no_sync(policy) if end < size else contextlib.nullcontext()
        with synchronization:
            with accelerator.autocast():
                loss, _ = policy(part)
                scaled = loss * (weights[start:end].sum().detach() / total)
            if not torch.isfinite(scaled):
                raise FloatingPointError('Nonfinite history training loss')
            accelerator.backward(scaled)
        value += scaled.detach().item()
    norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm or float('inf'))
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite history gradient norm')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if lr_scheduler is not None:
        lr_scheduler.step()
    metrics.loss = value
    metrics.grad_norm = float(norm)
    metrics.lr = optimizer.param_groups[0]['lr']
    metrics.update_s = time.perf_counter() - started
    return metrics, {'loss': value, 'supervised_action_fraction': weights.mean().item()}


def excavator_forward(self, batch, noise=None, time=None, reduction='mean'):
    """Flow loss over valid, supervised timesteps and the four real joints.

    LeRobot 0.4.4 averages over max_action_dim (32), including 28 artificial
    channels, and counts masked episode padding in the mean's denominator.
    Keep the pretrained 32D heads; only change the training loss reduction.
    """
    import torch

    if self.config.adapt_to_pi_aloha:
        batch = {**batch, 'observation.state': self._pi_aloha_decode_state(batch['observation.state']),
                 'action': self._pi_aloha_encode_actions_inv(batch['action'])}
    images, image_masks = self.prepare_images(batch)
    losses = self.model.forward(
        images, image_masks, batch['observation.language.tokens'],
        batch['observation.language.attention_mask'], self.prepare_state(batch),
        self.prepare_action(batch), noise, time,
    )
    losses = losses[..., :self.config.action_feature.shape[0]]
    weights = torch.ones_like(losses[..., 0])
    padding = batch.get('action_is_pad', batch.get('actions_id_pad'))
    if padding is not None:
        weights = weights * (~padding).to(weights.dtype)
    supervision = batch.get('action_loss_weight')
    if supervision is not None:
        if supervision.ndim == 3 and supervision.shape[-1] == 1:
            supervision = supervision.squeeze(-1)
        if supervision.shape != weights.shape:
            raise ValueError('action_loss_weight must be chunked with the same timestamps as action')
        weights = weights * supervision
    numerator = (losses * weights.unsqueeze(-1)).sum(dim=(1, 2))
    denominator = weights.sum(dim=1) * losses.shape[-1]
    per_sample = numerator / denominator.clamp_min(1)
    loss = numerator.sum() / denominator.sum().clamp_min(1)
    metrics = {'loss': loss.detach().item(), 'supervised_action_fraction': weights.mean().detach().item()}
    if reduction == 'none':
        return per_sample, metrics
    if reduction != 'mean':
        raise ValueError(f'Unsupported loss reduction: {reduction}')
    return loss, metrics


def run_training(cfg, launcher, config_path):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from lerobot.scripts import lerobot_train as trainer

    accelerator = Accelerator(
        mixed_precision=launcher['mixed_precision'], cpu=cfg.policy.device == 'cpu',
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if cfg.policy.device == 'cpu' and len(launcher['gpu_ids']) > 1:
        raise ValueError('Distributed GPU training requires policy.device=cuda.')
    validate = cfg.validate

    def validate_and_record():
        validate()
        # All ranks validate the unused output path before rank 0 creates it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            cfg.output_dir.mkdir(parents=True, exist_ok=cfg.resume)
            if not cfg.resume:
                (cfg.output_dir / 'training_config.yaml').write_text(config_path.read_text())
            (cfg.output_dir / 'resolved_training_config.json').write_text(
                json.dumps({'launcher': launcher, 'training': cfg.to_dict()}, indent=2) + '\n',
            )
        accelerator.wait_for_everyone()

    cfg.validate = validate_and_record
    init_logging = trainer.init_logging

    def init_run_logging(*args, **kwargs):
        init_logging(*args, **kwargs)
        if accelerator.is_main_process:
            handler = logging.FileHandler(cfg.output_dir / 'train.log')
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logging.getLogger().addHandler(handler)

    trainer.init_logging = init_run_logging
    history = launcher.get('observation_history')
    stats = None
    if history is not None:
        prepared = Path(launcher['prepared_data'])
        stats = json.loads((prepared / 'normalization_stats.json').read_text())
        trainer.get_step_checkpoint_dir = lambda output_dir, total_steps, step: output_dir / 'checkpoints' / f'step_{step}'
        original_update = trainer.update_policy
        consumed_step = 0
        if cfg.resume:
            consumed_step = json.loads((cfg.checkpoint_path / 'training_state/training_step.json').read_text())['step']

        def update_and_record(*args, **kwargs):
            nonlocal consumed_step
            if launcher.get('micro_batch_size') is not None:
                result = microbatch_update(*args, **kwargs, micro_batch_size=launcher['micro_batch_size'])
            else:
                result = original_update(*args, **kwargs)
            consumed_step += 1
            if accelerator.is_main_process:
                record = {'step': consumed_step, 'loss': float(result[0].loss.val),
                          'grad_norm': float(result[0].grad_norm.val), 'lr': float(result[0].lr.val),
                          'update_seconds': float(result[0].update_s.val)}
                with (cfg.output_dir / 'losses.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                from success_eval_runtime import atomic_json
                atomic_json(cfg.output_dir / 'progress.json', {**record, 'status':'running', 'target_steps':cfg.steps})
            return result

        trainer.update_policy = update_and_record
    install_trainer_adapters(trainer, cfg, history, stats)
    trainer.train(cfg, accelerator)
    if accelerator.is_main_process:
        if history is not None:
            from success_eval_runtime import atomic_json
            atomic_json(cfg.output_dir / 'progress.json', {'status':'completed', 'step':consumed_step, 'target_steps':cfg.steps})
        print(f'Checkpoint: {cfg.output_dir / "checkpoints/last/pretrained_model"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE / 'training_config.yaml')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print applied settings without training.')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-output', help=argparse.SUPPRESS)
    parser.add_argument('--resume', type=Path, help='Resume optimizer/scheduler/RNG state from a complete checkpoint directory.')
    parser.add_argument('--config_path', help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings, launcher = load_settings(config_path, args.run_output)
    os.environ['HF_HOME'] = launcher['hf_home']
    if not args.worker:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, launcher['gpu_ids']))
    cfg = build_train_config(settings)
    if launcher.get('observation_history'):
        from smolvla_history import bind_history
        bind_history(cfg.policy, launcher['observation_history'])
    if args.resume:
        checkpoint = args.resume.resolve()
        saved_cfg = json.loads((checkpoint / 'pretrained_model/train_config.json').read_text())
        if saved_cfg['output_dir'] != str(cfg.output_dir):
            raise ValueError('Resume must use the original --run-output directory')
        current = cfg.to_dict()
        for key in ('batch_size', 'seed', 'steps'):
            if saved_cfg[key] != current[key]:
                raise ValueError(f'Resume cannot change {key}')
        if saved_cfg['dataset']['episodes'] != current['dataset']['episodes']:
            raise ValueError('Resume cannot change the frozen training episodes')
        for key in ('n_obs_steps','chunk_size','n_action_steps','normalization_mapping','optimizer_lr',
                    'scheduler_warmup_steps','scheduler_decay_steps','freeze_vision_encoder','train_expert_only'):
            if saved_cfg['policy'][key] != current['policy'][key]:
                raise ValueError(f'Resume cannot change policy.{key}')
        cfg.resume = True
        sys.argv.append('--config_path=' + str(checkpoint / 'pretrained_model/train_config.json'))
        cfg.validate()
    if args.dry_run:
        cfg.validate()
        print(json.dumps({'launcher': launcher, 'training': cfg.to_dict()}, indent=2))
        return
    if len(launcher['gpu_ids']) > 1 and not args.worker:
        cfg.validate()
        command = [
            sys.executable, '-m', 'accelerate.commands.launch', '--multi_gpu',
            f'--num_processes={len(launcher["gpu_ids"])}', '--num_machines=1',
            f'--mixed_precision={launcher["mixed_precision"]}', '--dynamo_backend=no',
            str(Path(__file__).resolve()), '--config', str(config_path),
            '--worker', '--run-output', str(cfg.output_dir),
        ]
        if args.resume:
            command += ['--resume', str(args.resume.resolve())]
        subprocess.run(command, check=True)
        return
    run_training(cfg, launcher, config_path)


if __name__ == '__main__':
    main()
