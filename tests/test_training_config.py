import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('excavator_train', BASE / 'scripts/train_smolvla.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.source = yaml.safe_load((BASE / 'training_config.yaml').read_text())
        self.source['policy']['path'] = str(BASE / 'models/smolvla_base')
        self.source['dataset']['root'] = str(BASE / 'data/export')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'training.yaml'

    def load(self, source=None):
        self.path.write_text(yaml.safe_dump(source or self.source))
        return train.load_settings(self.path)

    def test_yaml_changes_reach_validated_optimizer_and_policy(self):
        self.source.update(steps=4000, batch_size=3, num_workers=2, save_freq=500, log_freq=7)
        self.source['policy'].update(optimizer_lr=0.00005, scheduler_warmup_steps=200,
                                     scheduler_decay_steps=4000, tokenizer_max_length=80,
                                     freeze_vision_encoder=False, train_expert_only=False)
        settings, _ = self.load()
        cfg = train.build_train_config(settings)
        with patch.object(sys, 'argv', ['train_smolvla.py']):
            cfg.validate()
        self.assertEqual((cfg.steps, cfg.batch_size, cfg.num_workers), (4000, 3, 2))
        self.assertEqual((cfg.save_freq, cfg.log_freq), (500, 7))
        self.assertEqual(cfg.optimizer.lr, 0.00005)
        self.assertEqual(cfg.scheduler.num_warmup_steps, 200)
        self.assertEqual(cfg.scheduler.num_decay_steps, 4000)
        self.assertEqual(cfg.policy.tokenizer_max_length, 80)
        self.assertFalse(cfg.policy.freeze_vision_encoder)
        self.assertFalse(cfg.policy.train_expert_only)
        self.assertEqual(set(cfg.policy.input_features), {
            'observation.state', 'observation.images.0', 'observation.images.1', 'observation.images.2',
        })
        self.assertEqual(set(cfg.policy.output_features), {'action'})

    def test_paths_are_relative_to_yaml_and_output_override_is_honored(self):
        self.source['output_dir'] = 'custom-run'
        settings, launcher = self.load()
        self.assertEqual(settings['output_dir'], str(self.path.parent / 'custom-run'))
        self.assertEqual(launcher['hf_home'], str(self.path.parent / 'models/hf-cache'))
        self.assertEqual(settings['dataset']['root'], str((BASE / 'data/export').resolve()))

    def test_invalid_gpu_ids_and_unknown_settings_fail(self):
        self.source['launcher']['gpu_ids'] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.load()
        self.source['launcher']['gpu_ids'] = [0]
        self.source['launcher']['unknown_setting'] = True
        with self.assertRaisesRegex(ValueError, 'Unknown launcher'):
            self.load()
        del self.source['launcher']['unknown_setting']
        self.source['steps_typo'] = 100
        settings, _ = self.load()
        with self.assertRaises(Exception) as error:
            train.build_train_config(settings)
        self.assertIn('steps_typo', str(error.exception))

    def test_inconsistent_schedule_fails_before_training(self):
        self.source['steps'] = 20
        settings, _ = self.load()
        with self.assertRaisesRegex(ValueError, 'must not exceed steps'):
            train.build_train_config(settings)

    def test_processor_overrides_are_applied(self):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        calls = {}

        def make_processors(*args, **kwargs):
            calls.update(kwargs)
            return 'preprocessor', 'postprocessor'

        trainer = SimpleNamespace(make_policy=lambda **kwargs: kwargs,
                                  make_dataset=lambda *args, **kwargs: None,
                                  make_pre_post_processors=make_processors)
        cfg = train.build_train_config(self.load()[0])
        original = SmolVLAPolicy.forward
        self.addCleanup(setattr, SmolVLAPolicy, 'forward', original)
        train.install_trainer_adapters(trainer, cfg)
        trainer.make_pre_post_processors(preprocessor_overrides={'device_processor': {'device': 'cpu'}})
        self.assertEqual(calls['preprocessor_overrides']['tokenizer_processor']['max_length'], 64)
        self.assertEqual(calls['preprocessor_overrides']['device_processor']['device'], 'cpu')

    def test_auxiliary_dataset_labels_do_not_become_policy_outputs(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        cfg = train.build_train_config(self.load()[0])
        meta = LeRobotDatasetMetadata('local/excavator', root=BASE / 'data/export')
        before = copy.deepcopy(meta.info)
        original = SmolVLAPolicy.forward
        self.addCleanup(setattr, SmolVLAPolicy, 'forward', original)
        trainer = SimpleNamespace(make_policy=lambda **kwargs: kwargs['ds_meta'],
                                  make_dataset=lambda *args, **kwargs: None,
                                  make_pre_post_processors=lambda **kwargs: None)
        train.install_trainer_adapters(trainer, cfg)
        filtered = trainer.make_policy(cfg=cfg.policy, ds_meta=meta)
        self.assertEqual(set(filtered.features), {*cfg.policy.input_features, 'action'})
        self.assertEqual(meta.info, before)

    def test_recovery_weights_follow_future_actions_and_episode_padding(self):
        from lerobot.datasets.factory import make_dataset
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        cfg = train.build_train_config(self.load()[0])
        cfg.dataset.episodes = [0]
        original = SmolVLAPolicy.forward
        self.addCleanup(setattr, SmolVLAPolicy, 'forward', original)
        trainer = SimpleNamespace(make_dataset=make_dataset, make_policy=lambda **kw: None,
                                  make_pre_post_processors=lambda **kw: None)
        train.install_trainer_adapters(trainer, cfg)
        dataset = trainer.make_dataset(cfg)
        last = dataset[len(dataset) - 1]
        self.assertEqual(tuple(last['action_loss_weight'].shape), (cfg.policy.chunk_size,))
        self.assertEqual(last['action_is_pad'].tolist(), [False] + [True] * (cfg.policy.chunk_size - 1))
        self.assertEqual(dataset.delta_indices['action'], dataset.delta_indices['action_loss_weight'])


class ExcavatorLossTests(unittest.TestCase):
    def forward(self, losses, **batch):
        import torch
        policy = SimpleNamespace(
            config=SimpleNamespace(adapt_to_pi_aloha=False, action_feature=SimpleNamespace(shape=[4])),
            prepare_images=lambda b: (None, None), prepare_state=lambda b: None,
            prepare_action=lambda b: None, model=SimpleNamespace(forward=lambda *args: losses),
        )
        return train.excavator_forward(policy, {
            'observation.language.tokens': None, 'observation.language.attention_mask': None, **batch,
        })[0]

    def test_artificial_channels_and_episode_padding_have_no_loss_or_gradient(self):
        import torch
        losses = torch.ones(2, 3, 32, requires_grad=True)
        pad = torch.tensor([[False, False, True], [False, True, True]])
        value = self.forward(losses, action_is_pad=pad)
        self.assertEqual(value.item(), 1)
        value.backward()
        self.assertEqual(losses.grad[..., 4:].abs().sum().item(), 0)
        self.assertEqual(losses.grad[pad].abs().sum().item(), 0)
        self.assertAlmostEqual(losses.grad.sum().item(), 1, places=6)

    def test_perturbations_mask_only_their_own_future_timesteps(self):
        import torch
        losses = torch.ones(1, 3, 32, requires_grad=True)
        value = self.forward(losses, action_loss_weight=torch.tensor([[1., 0., 1.]]))
        value.backward()
        self.assertEqual(value.item(), 1)
        self.assertEqual(losses.grad[:, 1].abs().sum().item(), 0)
        self.assertGreater(losses.grad[:, 2].sum().item(), 0)

    def test_fully_masked_batch_is_finite_and_differentiable(self):
        import torch
        losses = torch.ones(1, 3, 32, requires_grad=True)
        value = self.forward(losses, action_loss_weight=torch.zeros(1, 3))
        self.assertEqual(value.item(), 0)
        value.backward()
        self.assertEqual(losses.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
