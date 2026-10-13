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

    def test_processor_overrides_and_padding_alias_are_applied(self):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        calls = {}

        def make_processors(*args, **kwargs):
            calls.update(kwargs)
            return 'preprocessor', 'postprocessor'

        trainer = SimpleNamespace(make_policy=lambda **kwargs: kwargs,
                                  make_pre_post_processors=make_processors)
        cfg = train.build_train_config(self.load()[0])
        original = SmolVLAPolicy.forward
        seen = {}

        def forward(self, batch, *args, **kwargs):
            seen.update(batch)
            return 'loss'

        self.addCleanup(setattr, SmolVLAPolicy, 'forward', original)
        SmolVLAPolicy.forward = forward
        train.install_trainer_adapters(trainer, cfg)
        trainer.make_pre_post_processors(preprocessor_overrides={'device_processor': {'device': 'cpu'}})
        self.assertEqual(calls['preprocessor_overrides']['tokenizer_processor']['max_length'], 64)
        self.assertEqual(calls['preprocessor_overrides']['device_processor']['device'], 'cpu')
        batch = {'action_is_pad': [False, True]}
        self.assertEqual(SmolVLAPolicy.forward(None, batch), 'loss')
        self.assertEqual(seen['actions_id_pad'], batch['action_is_pad'])
        self.assertNotIn('actions_id_pad', batch)

    def test_auxiliary_dataset_labels_do_not_become_policy_outputs(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        cfg = train.build_train_config(self.load()[0])
        meta = LeRobotDatasetMetadata('local/excavator', root=BASE / 'data/export')
        before = copy.deepcopy(meta.info)
        original = SmolVLAPolicy.forward
        self.addCleanup(setattr, SmolVLAPolicy, 'forward', original)
        trainer = SimpleNamespace(make_policy=lambda **kwargs: kwargs['ds_meta'],
                                  make_pre_post_processors=lambda **kwargs: None)
        train.install_trainer_adapters(trainer, cfg)
        filtered = trainer.make_policy(cfg=cfg.policy, ds_meta=meta)
        self.assertEqual(set(filtered.features), {*cfg.policy.input_features, 'action'})
        self.assertEqual(meta.info, before)


if __name__ == '__main__':
    unittest.main()
