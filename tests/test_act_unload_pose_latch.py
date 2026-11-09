from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from act_unload_pose_latch import UnloadPoseLatch,patch_controller,UnloadHandoffWait,patch_wait_controller


class UnloadLatchTests(unittest.TestCase):
    def setUp(self):
        self.state=np.zeros(28);self.state[:4]=[.1,.5,-.8,-2.]
        self.state[26]=.03
        self.command=self.state[:4].copy();self.latch=UnloadPoseLatch(.1,[6.,2.5,2.5,6.])

    def test_verified_handoff_holds_before_act_opening_without_changing_bucket(self):
        action=np.array([.4,.2,-.2,.03])
        chosen,t=self.latch.update(self.state,action,action,self.command,True,False,True)
        self.assertTrue(t['active']);self.assertEqual(t['phase'],'wait_for_act_opening')
        np.testing.assert_allclose(chosen[:3],0);self.assertEqual(chosen[3],action[3])

    def test_target_does_not_follow_drifting_measured_pose(self):
        action=np.array([.2,.2,.2,.3])
        self.latch.update(self.state,action,action,self.command,True,False)
        goal=self.state[:3].copy();self.state[:3]+=.05;self.command[:3]+=.03
        chosen,t=self.latch.update(self.state,action,action,self.command,True,False)
        np.testing.assert_allclose(t['latched_parent_goal'],goal)
        np.testing.assert_allclose(chosen[:3],-.3)
        self.assertEqual(chosen[3],.3)

    def test_no_handoff_and_no_intent_does_not_interrupt_act(self):
        action=np.array([.2,.2,.2,.03])
        chosen,t=self.latch.update(self.state,action,action,self.command,True,False)
        self.assertFalse(t['active']);np.testing.assert_array_equal(chosen,action)
        _,t=self.latch.update(self.state,np.array([0.,0.,0.,.4]),action,self.command,False,False)
        self.assertFalse(t['active'])

    def test_wait_timeout_is_bounded_and_opening_has_its_own_original_budget(self):
        a=np.array([.2,.2,.2,0.])
        for i in range(20):
            _,t=self.latch.update(self.state,a,a,self.command,True,False,i==0)
            self.assertTrue(t['active'])
        _,t=self.latch.update(self.state,a,a,self.command,True,False)
        self.assertFalse(t['active']);self.assertEqual(t['release_reason'],'act_opening_wait_timeout')
        latch=UnloadPoseLatch(.1,[6.,2.5,2.5,6.])
        for i in range(19):latch.update(self.state,a,a,self.command,True,False,i==0)
        a[3]=.3
        for i in range(50):
            _,t=latch.update(self.state,a,a,self.command,True,False)
            self.assertTrue(t['active']);self.assertEqual(t['opening_steps'],i+1)
        _,t=latch.update(self.state,a,a,self.command,True,False)
        self.assertFalse(t['active']);self.assertEqual(t['release_reason'],'opening_hold_timeout')

    def test_three_empty_observations_release_latch(self):
        a=np.array([0.,0.,0.,.4])
        self.latch.update(self.state,a,a,self.command,True,False)
        self.state[26]=0.
        for _ in range(3):_,t=self.latch.update(self.state,a,a,self.command,True,False)
        self.assertFalse(t['active']);self.assertEqual(t['release_reason'],'payload_empty')

    def test_patch_only_replaces_hold_and_compiles_frozen_controller(self):
        source=ROOT/'outputs/experiments/act_combined_random40_20261106_01/frozen_sources/scripts/act_stall_recovery.py'
        if not source.exists():self.skipTest('Frozen source unavailable')
        original=source.read_text();patched=patch_controller(original)
        self.assertEqual(original[:original.index('def make_stall_guard_class')].split('import numpy as np')[1],
                         patched[:patched.index('def make_stall_guard_class')].split('from act_unload_pose_latch import UnloadPoseLatch')[1])
        compile(patched,'latched_controller','exec')
        with self.assertRaises(ValueError):patch_controller(patched)

    def test_wait_only_passes_legacy_hold_action_through_byte_for_byte(self):
        waiter=UnloadHandoffWait(.1,[6.,2.5,2.5,6.])
        action=np.array([.12,-.03,.04,.5],dtype=np.float32)
        chosen,trace=waiter.update(self.state,action,self.command,True,False,True,True)
        self.assertIs(chosen,action);self.assertFalse(trace['active'])
        chosen,trace=waiter.update(self.state,action,self.command,True,False,False,True)
        self.assertIs(chosen,action)

    def test_wait_only_relinquishes_to_legacy_hold_after_anchoring(self):
        waiter=UnloadHandoffWait(.1,[6.,2.5,2.5,6.]);action=np.array([.3,-.2,.1,-.03])
        chosen,trace=waiter.update(self.state,action,self.command,True,False,True,False)
        self.assertTrue(trace['active']);np.testing.assert_allclose(chosen[:3],0)
        action[3]=.4
        chosen,trace=waiter.update(self.state,action,self.command,True,False,False,True)
        self.assertIs(chosen,action);self.assertEqual(trace['release_reason'],'legacy_unloading_hold_started')

    def test_wait_only_has_bounded_budget_and_releases_when_empty_or_recovery_resumes(self):
        action=np.array([.3,-.2,.1,-.03])
        waiter=UnloadHandoffWait(.1,[6.,2.5,2.5,6.])
        for index in range(20):
            _,trace=waiter.update(self.state,action,self.command,True,False,index==0,False)
            self.assertTrue(trace['active'])
        chosen,trace=waiter.update(self.state,action,self.command,True,False,False,False)
        self.assertIs(chosen,action);self.assertEqual(trace['release_reason'],'act_opening_wait_timeout')
        _,trace=waiter.update(self.state,action,self.command,True,False,False,False)
        self.assertFalse(trace['active'])
        waiter=UnloadHandoffWait(.1,[6.,2.5,2.5,6.])
        waiter.update(self.state,action,self.command,True,False,True,False)
        self.state[26]=0
        for _ in range(3):_,trace=waiter.update(self.state,action,self.command,True,False,False,False)
        self.assertFalse(trace['active']);self.assertEqual(trace['release_reason'],'payload_empty')
        self.state[26]=.03
        waiter=UnloadHandoffWait(.1,[6.,2.5,2.5,6.])
        waiter.update(self.state,action,self.command,True,False,True,False)
        chosen,trace=waiter.update(self.state,action,self.command,True,True,False,False)
        self.assertIs(chosen,action);self.assertEqual(trace['release_reason'],'recovery_in_charge')

    def test_wait_patch_preserves_legacy_hold_block_verbatim(self):
        source=ROOT/'outputs/experiments/act_combined_random40_20261106_01/frozen_sources/scripts/act_stall_recovery.py'
        if not source.exists():self.skipTest('Frozen source unavailable')
        original=source.read_text();patched=patch_wait_controller(original)
        begin='            loaded = bool(state[26]';end="            closure_phase = trace.get('phase')"
        legacy=original[original.index(begin):original.index(end)]
        self.assertIn(legacy,patched)
        compile(patched,'wait_controller','exec')


if __name__=='__main__':unittest.main()
