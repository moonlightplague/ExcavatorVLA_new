"""Causal transport, native SmolVLA image/state processing and logical-batch equivalence."""
import contextlib
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'ExcavatorVLA-smolvla')]
from smolvla_history import (CAMERAS, MASK, ObservationHistory, configure_dataset, history_spec, inference_batch,
    policy_batch, policy_class, validate_history)
from train_smolvla import microbatch_update


def frame(step, episode=0):
    return {'observation.state':torch.full((1,28),float(100*episode+step)),
        **{key:torch.full((1,3,256,256),(10*episode+3*step+i)/100.) for i,key in enumerate(CAMERAS)},
        'task':['Dig consecutive scoops.']}


class HistoryTests(unittest.TestCase):
    def test_every_control_step_updates_the_window_including_between_replans(self):
        history = ObservationHistory(history_spec())
        for step in range(21):
            current = frame(step)
            history.append(current,step)
            if step in (0,1,3,6,7,20):
                batch = history.batch()
                indices=[max(0,step+offset) for offset in (-6,-3,0)]
                self.assertEqual(batch['observation.state'][0,:,0].tolist(),indices)
                self.assertEqual(batch[MASK].tolist(),[[step+offset<0 for offset in (-6,-3,0)]])
                for i,key in enumerate(CAMERAS):
                    torch.testing.assert_close(batch[key][0,:,0,0,0],torch.tensor([(3*j+i)/100 for j in indices]))
        self.assertEqual(indices,[14,17,20])

    def test_episode_reset_cannot_read_previous_episode_and_input_is_cloned(self):
        history=ObservationHistory(history_spec())
        first=frame(0)
        history.append(first,0)
        first['observation.state'].fill_(999)
        self.assertEqual(history.batch()['observation.state'].max().item(),0)
        history.reset()
        history.append(frame(0,episode=1),0)
        self.assertEqual(history.batch()['observation.state'].min().item(),100)
        self.assertEqual(history.batch()[MASK].tolist(),[[True,True,False]])

    def test_noncausal_offsets_skipped_steps_and_bad_manifests_fail(self):
        for offsets in ((-3,0,3),(-3,-6,0),(-6,0),(-6,-6,0)):
            with self.subTest(offsets=offsets),self.assertRaises(ValueError):
                history_spec(offsets)
        history=ObservationHistory(history_spec())
        with self.assertRaises(ValueError): history.append(frame(1),1)
        spec=history_spec();spec['fps']=20
        with self.assertRaises(ValueError):validate_history(spec)

    def test_dataset_history_offsets_never_change_action_targets(self):
        dataset=SimpleNamespace(fps=10,delta_timestamps={'action':[i/10 for i in range(10)]},
            delta_indices={'action':list(range(10))})
        expected=copy.deepcopy(dataset.delta_indices['action'])
        configure_dataset(dataset,history_spec())
        for key in ('observation.state',*CAMERAS):
            self.assertEqual(dataset.delta_indices[key],[-6,-3,0])
            self.assertEqual(dataset.delta_timestamps[key],[-.6,-.3,0])
        self.assertEqual(dataset.delta_indices['action'],expected)

    def test_phase_rewards_and_other_annotations_are_excluded(self):
        batch={**frame(0),'reward':999,'observation.stage_current_id':9,
               'observation.return_to_pile':1,'action_is_expert':1,'next.observation.state':torch.ones(28)}
        self.assertEqual(set(policy_batch(batch)),{'observation.state',*CAMERAS,'task'})
        batch['action']=torch.ones(1,10,4)
        batch['action_loss_weight']=torch.ones(1,10)
        self.assertNotIn('action',inference_batch(batch))
        self.assertNotIn('action_loss_weight',inference_batch(batch))


class NativeSmolVLAProcessingTests(unittest.TestCase):
    def policy(self):
        cls=policy_class()
        policy=cls.__new__(cls)
        torch.nn.Module.__init__(policy)
        policy.config=SimpleNamespace(image_features={key:None for key in CAMERAS},
            resize_imgs_with_padding=None,empty_cameras=0,max_state_dim=32,max_action_dim=32)
        policy.history=history_spec()
        return policy

    def batch(self):
        return {'observation.state':torch.stack([torch.full((1,28),float(i)) for i in range(3)],dim=1),
            **{key:torch.stack([torch.full((1,3,256,256),(3*t+c)/10.) for t in range(3)],dim=1)
               for c,key in enumerate(CAMERAS)},MASK:torch.zeros((1,3),dtype=torch.bool)}

    def test_nine_images_preserve_time_camera_identity_and_native_scaling(self):
        images,masks=self.policy().prepare_images(self.batch())
        self.assertEqual(len(images),9)
        for i,(image,mask) in enumerate(zip(images,masks)):
            self.assertEqual(image.shape,(1,3,256,256))
            self.assertAlmostEqual(image[0,0,0,0].item(),i/5-1,places=6)
            self.assertEqual(mask.tolist(),[True])

    def test_padding_masks_whole_old_camera_triplets(self):
        batch=self.batch();batch[MASK][0,:2]=True
        _,masks=self.policy().prepare_images(batch)
        self.assertEqual([m.item() for m in masks],[False]*6+[True]*3)
        batch[f'{CAMERAS[0]}_is_pad']=torch.tensor([[False,False,False]])
        with self.assertRaisesRegex(ValueError,'agree'):self.policy().prepare_images(batch)

    def test_current_state_and_action_heads_keep_native_representation(self):
        policy=self.policy();batch=self.batch();batch['action']=torch.randn(1,10,4)
        state=policy.prepare_state(batch)
        self.assertEqual(state.shape,(1,32))
        torch.testing.assert_close(state[:,:28],batch['observation.state'][:,-1])
        action=policy.prepare_action(batch)
        self.assertEqual(action.shape,(1,10,32))
        torch.testing.assert_close(action[:,:,:4],batch['action'])
        self.assertEqual(action[:,:,4:].abs().sum().item(),0)


class LogicalBatchTests(unittest.TestCase):
    def test_microbatches_preserve_weighted_masked_batch_gradient(self):
        class Policy(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3))
            def forward(self,batch):
                weights=(~batch['action_is_pad']).float()*batch['action_loss_weight']
                losses=(self.weight-batch['action'])**2
                return (losses*weights.unsqueeze(-1)).sum()/(4*weights.sum()).clamp_min(1),{}
        batch={'action':torch.arange(5*3*4,dtype=torch.float32).reshape(5,3,4)/10,
            'action_is_pad':torch.tensor([[False,False,False],[False,False,True],[False,True,True],
                                        [False,False,False],[False,False,True]]),
            'action_loss_weight':torch.tensor([[1.,2.,0.],[.2,.8,0.],[1.,0.,0.],[1.,1.,1.],[1.,1.,0.]])}
        expected=Policy();optimizer=torch.optim.SGD(expected.parameters(),lr=.01)
        expected(batch)[0].backward();optimizer.step()
        accelerator=SimpleNamespace(autocast=contextlib.nullcontext,no_sync=lambda p:contextlib.nullcontext(),
            backward=lambda loss:loss.backward(),clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
        for size in (1,2,5):
            actual=Policy();optimizer=torch.optim.SGD(actual.parameters(),lr=.01)
            microbatch_update(SimpleNamespace(),actual,batch,optimizer,1000,accelerator=accelerator,micro_batch_size=size)
            torch.testing.assert_close(expected.weight,actual.weight,atol=1e-7,rtol=1e-7)


if __name__=='__main__':unittest.main()
