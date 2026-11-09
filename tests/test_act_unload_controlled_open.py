from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from act_unload_controlled_open import allow_controlled_open,bucket_open_velocity,patch_controller


class ControlledUnloadTests(unittest.TestCase):
    def setUp(self):
        self.event=dict(handback=True,phase='completed_secure_align',contained_opening_arc_residual_m=.01)

    def test_only_verified_loaded_contained_handoff_can_start(self):
        self.assertTrue(allow_controlled_open(self.event,True,True,False))
        for loaded,over,active,enabled in ((False,True,False,True),(True,False,False,True),
            (True,True,True,True),(True,True,False,False)):
            self.assertFalse(allow_controlled_open(self.event,loaded,over,active,enabled))
        for event in ({},dict(self.event,handback=False),dict(self.event,phase='aborted_budget'),
            dict(self.event,contained_opening_arc_residual_m=.016),dict(self.event,contained_opening_arc_residual_m=float('nan'))):
            self.assertFalse(allow_controlled_open(event,True,True,False))

    def test_opening_respects_speed_original_limit_and_endpoint(self):
        self.assertEqual(bucket_open_velocity(-2.,.1,6.),.8)
        self.assertEqual(bucket_open_velocity(-2.,.1,.5),.5)
        self.assertAlmostEqual(bucket_open_velocity(-.42,.1,6.),.2)
        self.assertEqual(bucket_open_velocity(-.4,.1,6.),0.)
        self.assertEqual(bucket_open_velocity(.1,.1,6.),0.)
        with self.assertRaises(ValueError):bucket_open_velocity(-2.,0.,6.)

    def test_patch_is_fail_closed(self):
        with self.assertRaises(ValueError):patch_controller('unknown controller')
        source=ROOT/'outputs/experiments/act_unload_wait_20261106_01/frozen_sources/scripts/act_stall_recovery.py'
        if not source.exists():self.skipTest('Frozen controller absent')
        code=source.read_text();patched=patch_controller(code)
        compile(patched,'controlled_unload','exec')
        with self.assertRaises(ValueError):patch_controller(patched)
        # Recovery planning and the original measured parent target survive.
        self.assertEqual(code[:code.index('def make_stall_guard_class')].split('import numpy as np\n')[1],
            patched[:patched.index('def make_stall_guard_class')].split('from act_unload_controlled_open import allow_controlled_open, bucket_open_velocity\n')[1])
        target='chosen[:3] = np.clip((np.asarray(state[:3]) - np.asarray(command_positions[:3])) / self.dt,'
        self.assertIn(target,patched)


if __name__=='__main__':unittest.main()
