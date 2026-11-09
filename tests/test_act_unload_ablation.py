import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from act_unload_snapshot import array_hash, capture, patch_simulator
from run_act_unload_ablation import geometry_metrics, summarize
from analyze_act_unload_ablation import alignment


class SnapshotValidationTests(unittest.TestCase):
    def test_capture_rejects_missing_particle_velocities(self):
        class Robot:
            def get_world_pose(self):return np.zeros(3),np.array([1.,0.,0.,0.])
            def get_joint_positions(self):return np.zeros(4)
            def get_joint_velocities(self):return np.zeros(4)
            def get_linear_velocity(self):return np.zeros(3)
            def get_angular_velocity(self):return np.zeros(3)
        class Attr:
            def __init__(self,value):self.value=value
            def Get(self):return self.value
        class Particles:
            def GetPointsAttr(self):return Attr(np.ones((2,3)))
            def GetVelocitiesAttr(self):return Attr(np.ones((1,3)))
        with self.assertRaisesRegex(ValueError,'velocities'):
            capture(Robot(),Particles())
        self.assertNotEqual(array_hash(np.zeros((2,3),np.float32)),array_hash(np.ones((2,3),np.float32)))

    def test_patch_is_fail_closed_and_compiles_real_archived_simulator(self):
        with self.assertRaises(ValueError):patch_simulator('unrecognized simulator')
        source=ROOT/'outputs/baselines/act_phw40k_carry275_20261105/frozen_sources/ExcavatorVLA-smolvla/run_simulation.py'
        if not source.exists():self.skipTest('Archived runtime unavailable')
        patched=patch_simulator(source.read_text())
        compile(patched,'patched_simulator','exec')
        with self.assertRaises(ValueError):patch_simulator(patched)


class UnloadMeasurementTests(unittest.TestCase):
    def test_opening_drift_excludes_departure_after_three_empty_observations(self):
        rows=[]
        for index in range(60):
            state=[0.]*28;state[26]=.01 if index<2 else 0.
            rows.append(dict(state=state,geometry=dict(hinge_xy_drift_m=.1 if index<5 else 5.,
                opening_pour_bed_gap_m=.2,parent_abs_drift_rad=[.01,.02,.03]),carry_guard={}))
        result=alignment(rows)
        self.assertEqual(result['first_opening_steps'],5)
        self.assertEqual(result['first_opening_hinge_xy_drift_m'],.1)
        self.assertEqual(result['first_five_seconds_hinge_xy_drift_m'],5.)

    def test_bucket_rotation_does_not_create_false_hinge_drift(self):
        def landmarks(angle,translation=(0.,0.,0.)):
            x=np.array([np.cos(angle),0.,np.sin(angle)])
            z=np.array([0.,1.,0.]);p=np.asarray(translation)
            return {k:(p+a*x+b*z).tolist() for k,a,b in
                [('tip',.75,-.18),('load',.35,.08),('pour',.85,.30)]}
        scene={'truck_counting_volume':dict(local_from_world=np.eye(4).tolist(),
            local_min=[-2.,-2.,0.],local_max=[2.,2.,1.])}
        s=np.zeros(28);s[3]=-.8
        before=landmarks(0.)
        rotated=geometry_metrics(s,landmarks(.6),scene,s,before)
        self.assertLess(rotated['hinge_xy_drift_m'],1e-12)
        shifted=geometry_metrics(s,landmarks(.6,(.3,.4,0.)),scene,s,before)
        self.assertAlmostEqual(shifted['hinge_xy_drift_m'],.5)

    def test_repeatability_failure_blocks_apparent_treatment_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            protocol=dict(scenes=[29],advance_gate=dict(minimum_matched_pairs=1,minimum_reached_scenes=1,
                minimum_mean_truck_fraction_gain=.1,minimum_positive_scene_count=1,
                maximum_mean_outside_fraction_increase=.02,maximum_identical_off_truck_fraction_difference=.05))
            (out/'protocol.json').write_text(json.dumps(protocol))
            folder=out/'scene29_repeat1';folder.mkdir()
            common=dict(outside_fraction=.1,original_task_completed=True,max_hinge_xy_drift_m=.1)
            result=dict(status='completed',scene=29,repetition=1,cohort_count=200,branches=dict(
                off=dict(common,truck_fraction=.3),on=dict(common,truck_fraction=.8),
                off_repeat=dict(common,truck_fraction=.6)))
            (folder/'result.json').write_text(json.dumps(result))
            summary=summarize(out)
            self.assertTrue(summary['checks']['truck_gain'])
            self.assertFalse(summary['checks']['repeatability'])
            self.assertFalse(summary['advance'])
            self.assertFalse(summary['baseline_replaced'])

    def test_control_material_loss_blocks_success_preserving_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            protocol=dict(comparison='controlled_open',scenes=[10,8],problem_scenes=[10],success_controls=[8],
                advance_gate=dict(minimum_matched_pairs=2,minimum_problem_pairs=1,minimum_control_pairs=1,
                    minimum_problem_truck_gain=.1,minimum_problem_task_success_gain=1,
                    maximum_mean_outside_fraction_increase=.02,maximum_identical_off_truck_fraction_difference=.05,
                    minimum_branch_steps_for_coverage=50,maximum_control_truck_fraction_loss=.05))
            (out/'protocol.json').write_text(json.dumps(protocol))
            for scene,off,on,old_success in ((10,.1,.9,False),(8,.9,.6,True)):
                folder=out/f'scene{scene}_repeat1';folder.mkdir()
                def branch(fraction,success):
                    return dict(truck_fraction=fraction,outside_fraction=1-fraction,
                        original_task_completed=success,max_hinge_xy_drift_m=.1,branch_steps=100)
                result=dict(status='completed',scene=scene,repetition=1,cohort_count=200,branches=dict(
                    off=branch(off,old_success),on=branch(on,True),off_repeat=branch(off,old_success)))
                (folder/'result.json').write_text(json.dumps(result))
            summary=summarize(out)
            self.assertTrue(summary['checks']['problem_task_success'])
            self.assertTrue(summary['checks']['success_controls'])
            self.assertFalse(summary['checks']['control_material'])
            self.assertFalse(summary['advance'])


if __name__=='__main__':unittest.main()
