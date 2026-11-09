from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from act_unload_geometry import aperture_from_mesh,ApertureAlignment,GeometryUnload,patch_controller
from excavator_reachability import TruckBedColumn


class FakeGeometry:
    def __init__(self):
        self.chain=self;self.allowed=True;self.fail=False
        self.bounds=np.array([[-3.,3.]]*4);self.parent_delta=np.array([.12,.12,.16])
    def point(self,q,key):return np.zeros(3)
    def carry_pitch(self,q):return np.radians(-135)
    def plan(self,q):
        if self.fail:raise ValueError('No local endpoint')
        goal=np.asarray(q).copy();goal[0]=.1
        return goal,{'already_contained':False}
    def safe(self,q):return bool(self.allowed and q[0]>=.099)
    def clearance(self,q):return dict(minimum_xy_m=.15 if self.safe(q) else -.1,minimum_height_m=1.)


class GeometryUnloadTests(unittest.TestCase):
    def setUp(self):
        self.fake=FakeGeometry()
        with patch('act_unload_geometry.ApertureAlignment',return_value=self.fake):
            self.control=GeometryUnload({}, {}, .1,[6.,2.5,2.5,6.])
        self.state=np.zeros(28);self.state[3]=-2.;self.state[26]=.02
        self.command=self.state[:4].copy();self.action=np.array([.3,.2,-.1,-.03])
        self.points={k:[0.,0.,0.] for k in ('tip','load','pour')}
        self.event=dict(handback=True,phase='completed_secure_align',contained_opening_arc_residual_m=0.)
    def update(self,event=None,legacy=False,step=200):
        return self.control.update(self.state,self.action,self.command,self.points,
            self.event if event is None else event,legacy,step)

    def test_closed_until_measured_alignment_is_stable_then_opens(self):
        chosen,t=self.update();self.assertTrue(t['active']);self.assertEqual(t['phase'],'align_closed')
        self.assertEqual(chosen[3],0.);self.assertGreater(chosen[0],0)
        self.state[0]=self.command[0]=.1
        _,t=self.update({});self.assertEqual(t['phase'],'align_closed')
        chosen,t=self.update({});self.assertEqual(t['phase'],'open_contained');self.assertEqual(chosen[3],.8)
        self.fake.allowed=False
        chosen,t=self.update({});self.assertEqual(chosen[3],0.);self.assertTrue(t['opening_blocked'])

    def test_legacy_hold_and_late_entry_pass_original_action_through(self):
        chosen,t=self.update(legacy=True);self.assertIs(chosen,self.action);self.assertFalse(t['active'])
        chosen,t=self.update(step=451);self.assertIs(chosen,self.action)
        self.assertEqual(t['release_reason'],'insufficient_episode_budget')

    def test_unresolved_plan_and_empty_release_do_not_force_actions(self):
        self.fake.fail=True
        chosen,t=self.update();self.assertIs(chosen,self.action);self.assertEqual(t['release_reason'],'local_plan_unresolved')
        self.setUp();self.update();self.state[26]=0
        for _ in range(3):chosen,t=self.update({})
        self.assertFalse(t['active']);self.assertTrue(t['handback']);self.assertEqual(t['release_reason'],'payload_empty')

    def test_alignment_and_total_budgets_are_bounded(self):
        for i in range(20):chosen,t=self.update(self.event if i==0 else {})
        self.assertFalse(t['active']);self.assertEqual(t['release_reason'],'alignment_timeout')
        self.setUp();self.update();self.state[0]=self.command[0]=.1
        for _ in range(49):chosen,t=self.update({})
        self.assertFalse(t['active']);self.assertEqual(t['release_reason'],'step_timeout');self.assertEqual(t['steps'],50)

    def test_measured_feedback_corrects_static_servo_offset(self):
        self.state[0]=.084;self.command[0]=.1
        chosen,t=self.update()
        self.assertTrue(t['active'])
        self.assertGreater(chosen[0],0.,'A commanded target alone must not hide measured tracking error')
        for _ in range(15):
            chosen,t=self.update({});self.command+=chosen*.1
            self.state[0]=self.command[0]-.016
        self.assertGreater(self.state[0],.099)

    def test_outside_command_bounds_does_not_force_a_large_correction(self):
        self.command[0]=.5
        chosen,t=self.update()
        self.assertIs(chosen,self.action);self.assertFalse(t['active'])
        self.assertEqual(t['release_reason'],'local_plan_unresolved')


class ApertureTests(unittest.TestCase):
    def test_asset_width_follows_hinge_axis(self):
        profile=[[.6+.6*np.cos(a),.6*np.sin(a)] for a in np.linspace(0,np.pi,13)]
        vertices=[[x,y,z] for x,y in profile for z in (-.475,.475)]
        corners,info=aperture_from_mesh(dict(bucket_collision_vertices=vertices,joints=[{}, {}, {},dict(axis='Z')]))
        self.assertAlmostEqual(info['width_m'],.95);self.assertAlmostEqual(info['length_m'],1.2)
        np.testing.assert_allclose(np.abs(corners[:,2]),.475)

    def test_point_inside_does_not_imply_full_aperture_inside(self):
        class Chain:
            def bucket_transform(self,q):
                t=np.repeat(np.eye(4)[None,:,:],len(q),axis=0);t[:,0,3]=q[:,0];t[:,2,3]=2.
                return t
        a=ApertureAlignment.__new__(ApertureAlignment);a.chain=Chain()
        a.bed=TruckBedColumn(dict(truck_counting_volume=dict(local_from_world=np.eye(4).tolist(),local_min=[-1,-.5,0],local_max=[1,.5,1])),.15)
        a.corners=np.array([[-.5,-.6,0],[.5,-.6,0],[.5,.6,0],[-.5,.6,0]])
        self.assertFalse(a.safe(np.array([0.,0.,0.,-2.])))
        a.corners[:,1]*=.5;self.assertTrue(a.safe(np.array([0.,0.,0.,-2.])))

    def test_patch_keeps_recovery_and_rejects_unknown_source(self):
        with self.assertRaises(ValueError):patch_controller('unknown')
        p=ROOT/'outputs/experiments/act_unload_controlled_open_20261106_01/frozen_sources/scripts/act_stall_recovery.py'
        if not p.exists():self.skipTest('Selected source absent')
        code=p.read_text();new=patch_controller(code);compile(new,'aperture_controller','exec')
        self.assertIn('self.unload_geometry.active',new)
        self.assertEqual(code[:code.index('def make_stall_guard_class')].split('import numpy as np\n')[1],
            new[:new.index('def make_stall_guard_class')].split('from act_unload_geometry import GeometryUnload\n')[1])


if __name__=='__main__':unittest.main()
