"""Loss changes must preserve original masking/KL and suppress only small residual gradients."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'ExcavatorVLA-vlm')]
import torch
from act_huber_objective import reconstruction_loss, policy_objective
from excavator_vlm.policies.act.adapter import ExcavatorACTPolicy


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prediction = torch.nn.Parameter(torch.arange(1.,25.).reshape(2,3,4)/10)
        self.mu = torch.nn.Parameter(torch.tensor([[.1,.2],[-.3,.4]]))
        self.logvar = torch.nn.Parameter(torch.zeros(2,2))

    def forward(self,batch):
        return self.prediction,(self.mu,self.logvar)


class HuberObjectiveTests(unittest.TestCase):
    def test_control_is_exact_original_loss_and_all_parameter_gradients(self):
        policy = SimpleNamespace(config=SimpleNamespace(image_features=[],use_vae=True,kl_weight=10.),model=ToyModel())
        batch = dict(action=torch.zeros(2,3,4),action_is_pad=torch.tensor([[False,False,True],[False,False,False]]),
                     action_loss_weight=torch.tensor([[[2.],[0.],[5.]],[[1.],[3.],[2.]]]))
        original,_ = ExcavatorACTPolicy.forward(policy,batch)
        expected = torch.autograd.grad(original,list(policy.model.parameters()))
        actual,_ = policy_objective(policy,batch,'l1')
        measured = torch.autograd.grad(actual,list(policy.model.parameters()))
        torch.testing.assert_close(actual,original,rtol=0,atol=0)
        for a,b in zip(measured,expected,strict=True):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_small_residual_gradient_reduces_and_large_residual_gradient_is_preserved(self):
        prediction=torch.tensor([[[.001,.01,.1,1.]]],requires_grad=True)
        loss,_,_=reconstruction_loss(prediction,torch.zeros_like(prediction),torch.zeros(1,1,dtype=torch.bool),torch.ones(1,1),'smooth_l1',.1)
        gradient=torch.autograd.grad(loss,prediction)[0]
        torch.testing.assert_close(gradient,torch.tensor([[[.0025,.025,.25,.25]]]))

    def test_padding_zero_weights_and_all_masked_loss_have_zero_gradients(self):
        prediction=torch.ones(1,3,4,requires_grad=True)
        padding=torch.tensor([[False,True,False]])
        loss,_,_=reconstruction_loss(prediction,torch.zeros_like(prediction),padding,torch.tensor([[2.,5.,0.]]),'smooth_l1')
        gradient=torch.autograd.grad(loss,prediction)[0]
        torch.testing.assert_close(gradient[:,0],torch.full((1,4),.25))
        self.assertEqual(int(torch.count_nonzero(gradient[:,1:])),0)
        zero,_,_=reconstruction_loss(prediction,torch.zeros_like(prediction),torch.ones_like(padding),torch.ones(1,3),'smooth_l1')
        self.assertEqual(float(zero.detach()),0.)
        self.assertEqual(int(torch.count_nonzero(torch.autograd.grad(zero,prediction)[0])),0)

    def test_beta_validation(self):
        for beta in (0,-1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):
                reconstruction_loss(torch.zeros(1,1,4),torch.zeros(1,1,4),torch.zeros(1,1,dtype=torch.bool),torch.ones(1,1),'smooth_l1',beta)

    def test_fractional_weights_preserve_global_scale_invariance(self):
        prediction=torch.tensor([[[.01,.05,.2,1.]]],requires_grad=True)
        values=[]
        for scale in (1.,.001):
            loss,_,_=reconstruction_loss(prediction,torch.zeros_like(prediction),torch.zeros(1,1,dtype=torch.bool),
                                         torch.full((1,1),scale),'smooth_l1')
            values.append(loss)
        torch.testing.assert_close(values[0],values[1])

    def test_actual_execution_prefix_keeps_original_stop_gradients_and_tail_is_smooth(self):
        prediction=torch.full((1,5,4),.01,requires_grad=True)
        loss,_,_=reconstruction_loss(prediction,torch.zeros_like(prediction),torch.zeros(1,5,dtype=torch.bool),
            torch.ones(1,5),'executed_prefix_l1',.1,executed_steps=3)
        gradient=torch.autograd.grad(loss,prediction)[0]
        torch.testing.assert_close(gradient[:,:3],torch.full((1,3,4),.05))
        torch.testing.assert_close(gradient[:,3:],torch.full((1,2,4),.005))

    def test_execution_prefix_length_is_validated(self):
        for count in (0,4,True):
            with self.assertRaises(ValueError):
                reconstruction_loss(torch.zeros(1,3,4),torch.zeros(1,3,4),torch.zeros(1,3,dtype=torch.bool),
                    torch.ones(1,3),'executed_prefix_l1',.1,count)


if __name__=='__main__':unittest.main()
