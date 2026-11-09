"""Check the loss partition and gradient measurements that decide ACT's gate."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ExcavatorVLA-vlm'))
sys.path.insert(0, str(ROOT / 'scripts'))

import torch
from excavator_vlm.policies.act.adapter import ExcavatorACTPolicy
from run_act_gradient_diagnosis import component_gradients, gradient_statistics, split_losses


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prediction = torch.nn.Parameter(torch.arange(1., 25.).reshape(2, 3, 4) / 10)
        self.mu = torch.nn.Parameter(torch.tensor([[.1, .2], [-.3, .4]]))
        self.logvar = torch.nn.Parameter(torch.zeros(2, 2))

    def forward(self, batch):
        return self.prediction, (self.mu, self.logvar)


class GradientDiagnosisTests(unittest.TestCase):
    def test_future_stage_partition_preserves_original_loss_and_gradient_with_padding(self):
        policy = SimpleNamespace(config=SimpleNamespace(image_features=[], use_vae=True, kl_weight=10.),
                                 model=ToyModel())
        batch = {'action': torch.zeros(2, 3, 4),
                 'action_is_pad': torch.tensor([[False, False, True], [False, False, False]]),
                 'action_loss_weight': torch.tensor([[[2.], [0.], [5.]], [[1.], [3.], [2.]]])}
        stages = torch.tensor([[6, 7, 6], [5, 6, 9]])
        named = list(policy.model.named_parameters())
        original, _ = ExcavatorACTPolicy.forward(policy, batch)
        expected = torch.autograd.grad(original, [p for _, p in named])
        hold, other, kl, parts = split_losses(policy, batch, stages)
        torch.testing.assert_close(hold + other + kl, original)
        self.assertEqual(parts['denominator'], 32.)
        self.assertEqual(parts['effective_hold_weight'], 5.)
        self.assertEqual(parts['effective_other_weight'], 3.)
        components = component_gradients((hold, other, kl), named)
        for index, (_, parameter) in enumerate(named):
            combined = sum((values[index] for values in components if values[index] is not None),
                           torch.zeros_like(parameter))
            torch.testing.assert_close(combined, expected[index], rtol=1e-6, atol=1e-7)
            self.assertIsNone(parameter.grad)
        # Neither padded labels nor zero-weight labels can contribute gradients.
        hold_prediction, other_prediction = components[0][0], components[1][0]
        self.assertEqual(int(torch.count_nonzero(hold_prediction[0, 1:])), 0)
        self.assertEqual(int(torch.count_nonzero(other_prediction[0])), 0)
        self.assertEqual(int(torch.count_nonzero(hold_prediction[1, [0, 2]])), 0)
        self.assertEqual(int(torch.count_nonzero(other_prediction[1, 1])), 0)

    def test_group_reduction_matches_explicit_full_vectors_with_unused_kl_parameters(self):
        named = [('model.action_head.weight', torch.nn.Parameter(torch.zeros(2))),
                 ('model.encoder_robot_state_input_proj.weight', torch.nn.Parameter(torch.zeros(2))),
                 ('model.vae_encoder_latent_output_proj.weight', torch.nn.Parameter(torch.zeros(1)))]
        gradients = ((torch.tensor([1., -2.]), torch.tensor([.5, 1.]), torch.tensor([.2])),
                     (torch.tensor([-3., 4.]), torch.tensor([2., -1.]), torch.tensor([-.4])),
                     (None, None, torch.tensor([2.])))
        measured = gradient_statistics(named, gradients)
        flat = [torch.cat([value if value is not None else torch.zeros_like(parameter)
                           for value, (_, parameter) in zip(vector, named)]) for vector in gradients]
        hold, other, kl = flat
        keep = other + kl
        expected_cosine = float(torch.dot(hold, other) / (hold.norm() * other.norm()))
        global_stats = measured['global_parameters']
        self.assertAlmostEqual(global_stats['hold_other_dot'], float(torch.dot(hold, other)), places=5)
        self.assertAlmostEqual(global_stats['hold_other_cosine'], expected_cosine, places=6)
        self.assertAlmostEqual(global_stats['keep_norm'], float(keep.norm()), places=6)
        self.assertAlmostEqual(global_stats['hold_keep_dot'], float(torch.dot(hold, keep)), places=5)
        self.assertAlmostEqual(global_stats['projection_coefficient'],
                               float(torch.dot(hold, keep) / torch.dot(keep, keep)), places=6)
        self.assertEqual(len(measured['parameter_groups']['action_head'][
            'missing_gradient_parameters']['weighted_kl']), 1)

    def test_zero_gradient_has_no_cosine_or_projection(self):
        named = [('model.action_head.weight', torch.nn.Parameter(torch.zeros(2)))]
        measured = gradient_statistics(named, ((torch.zeros(2),), (None,), (None,)))['global_parameters']
        self.assertIsNone(measured['hold_other_cosine'])
        self.assertIsNone(measured['hold_keep_cosine'])
        self.assertEqual(measured['projection_coefficient'], 0.)


if __name__ == '__main__':
    unittest.main()
