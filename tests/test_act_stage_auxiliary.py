"""Auxiliary labels cannot leak future actions or alter the ACT inference schema."""
import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'ExcavatorVLA-vlm')]

import torch
from act_stage_auxiliary import CurrentObservationFeatures, StageAuxiliaryHead, auxiliary_loss, stage_labels
from excavator_vlm.policies.act.adapter import ExcavatorACTPolicy


def policy_and_batch():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    config = ACTConfig(device='cpu', pretrained_backbone_weights=None,
        dim_model=32, n_heads=4, dim_feedforward=64, n_encoder_layers=1,
        n_decoder_layers=1, n_vae_encoder_layers=1, latent_dim=4,
        chunk_size=3, n_action_steps=1, dropout=0.,
        input_features={'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(28,)),
            'observation.images.0': PolicyFeature(type=FeatureType.VISUAL, shape=(3,64,64))},
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(4,))})
    policy = ExcavatorACTPolicy(config)
    batch = {'observation.state': torch.randn(2,28), 'observation.images.0':torch.randn(2,3,64,64),
             'action':torch.randn(2,3,4), 'action_is_pad':torch.zeros(2,3,dtype=torch.bool),
             'action_loss_weight':torch.ones(2,3,1)}
    return policy, batch


class StageAuxiliaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_current_features_do_not_see_future_labels_or_posterior_latent(self):
        policy, batch = policy_and_batch()
        policy.train()
        with CurrentObservationFeatures(policy) as captured:
            policy(batch)
            first = captured.tensor().detach().clone()
            captured.reset()
            changed = {**batch,'action':batch['action'] + 100}
            torch.manual_seed(9876)
            policy(changed)
            torch.testing.assert_close(captured.tensor(),first,rtol=0,atol=0)

    def test_auxiliary_gradient_reaches_observation_features_and_not_future_action_encoder(self):
        policy, batch = policy_and_batch()
        head = StageAuxiliaryHead(32,1)
        named = dict(policy.named_parameters())
        keys = ['model.encoder_robot_state_input_proj.weight',
                'model.encoder_img_feat_input_proj.weight',
                'model.backbone.conv1.weight',
                'model.vae_encoder_action_input_proj.weight',
                'model.decoder.layers.0.linear1.weight']
        with CurrentObservationFeatures(policy) as captured:
            policy(batch)
            loss, _ = auxiliary_loss(head,captured,torch.tensor([5,6]),torch.ones(10))
            grads = torch.autograd.grad(loss,[named[k] for k in keys],allow_unused=True)
        for gradient in grads[:3]:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()),0)
        self.assertEqual(grads[3:],(None,None))

    def test_zero_coefficient_preserves_policy_loss_gradient_and_export_keys(self):
        policy, batch = policy_and_batch()
        baseline = copy.deepcopy(policy)
        keys = set(policy.state_dict())
        torch.manual_seed(42)
        ordinary, _ = baseline(batch)
        ordinary.backward()
        head = StageAuxiliaryHead(32,1)
        torch.manual_seed(42)
        with CurrentObservationFeatures(policy) as captured:
            original, _ = policy(batch)
            auxiliary, _ = auxiliary_loss(head,captured,torch.tensor([5,6]),torch.ones(10))
            combined = original + 0 * auxiliary
            torch.testing.assert_close(combined,ordinary,rtol=0,atol=0)
            combined.backward()
        for (name, parameter),(other_name, other) in zip(policy.named_parameters(),baseline.named_parameters(),strict=True):
            self.assertEqual(name,other_name)
            torch.testing.assert_close(parameter.grad,other.grad,rtol=0,atol=0)
        self.assertEqual(keys,set(policy.state_dict()))
        policy.eval()
        with torch.inference_mode():
            prediction = policy.predict_action_chunk({k:v for k,v in batch.items() if k.startswith('observation.')})
        self.assertEqual(tuple(prediction.shape),(2,3,4))

    def test_labels_join_by_identity_and_reject_heldout_frames(self):
        batch = {'episode_index':torch.tensor([2,1,2]),'frame_index':torch.tensor([5,0,5])}
        mapping = {(1,0):5,(2,5):6,(3,0):9}
        self.assertEqual(stage_labels(batch,mapping,[1,2]).tolist(),[6,5,6])
        for bad in ({'episode_index':torch.tensor([3]),'frame_index':torch.tensor([0])},
                    {'episode_index':torch.tensor([1]),'frame_index':torch.tensor([99])}):
            with self.assertRaises(ValueError):
                stage_labels(bad,mapping,[1,2])


if __name__ == '__main__':
    unittest.main()
