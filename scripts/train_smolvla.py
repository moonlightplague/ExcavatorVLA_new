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
    unknown = set(launcher) - {'output_root', 'hf_home', 'gpu_ids', 'mixed_precision'}
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
    for section, key in [('dataset', 'root'), ('policy', 'path')]:
        if not settings[section].get(key):
            raise ValueError(f'{section}.{key} is required.')
        settings[section][key] = str(resolve_path(settings[section][key], path.parent))
    launcher = {
        'gpu_ids': gpu_ids, 'mixed_precision': precision,
        'hf_home': str(resolve_path(launcher.get('hf_home', 'models/hf-cache'), path.parent)),
        'output_root': str(resolve_path(launcher.get('output_root', 'outputs/train'), path.parent)),
    }
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


def install_trainer_adapters(trainer, cfg):
    """Keep compatibility changes local to this process, not site-packages."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    make_policy = trainer.make_policy

    @functools.wraps(make_policy)
    def make_excavator_policy(*args, **kwargs):
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
        return make_processors(*args, **kwargs)

    trainer.make_pre_post_processors = make_excavator_processors
    forward = SmolVLAPolicy.forward

    @functools.wraps(forward)
    def forward_with_padding(self, batch, *args, **kwargs):
        if 'action_is_pad' in batch:
            batch = {**batch, 'actions_id_pad': batch['action_is_pad']}
        return forward(self, batch, *args, **kwargs)

    SmolVLAPolicy.forward = forward_with_padding


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
            cfg.output_dir.mkdir(parents=True, exist_ok=False)
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
    install_trainer_adapters(trainer, cfg)
    trainer.train(cfg, accelerator)
    if accelerator.is_main_process:
        print(f'Checkpoint: {cfg.output_dir / "checkpoints/last/pretrained_model"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE / 'training_config.yaml')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print applied settings without training.')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-output', help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings, launcher = load_settings(config_path, args.run_output)
    os.environ['HF_HOME'] = launcher['hf_home']
    if not args.worker:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, launcher['gpu_ids']))
    cfg = build_train_config(settings)
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
        subprocess.run(command, check=True)
        return
    run_training(cfg, launcher, config_path)


if __name__ == '__main__':
    main()
