"""Collection/evaluation parity for avalanched Newton material IDs."""
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ExcavatorVLA-smolvla'),str(ROOT/'scripts')]
from excavator_common.newton_source_identity import SOURCE_IDENTITY, initial_source_mask
from baseline.tuning import install_source_identity
from excavator_common.evaluation_metrics import ParticleTransferTracker


class NewtonIdentityTests(unittest.TestCase):
    def test_avalanched_points_match_collection_and_exclude_initial_bucket_truck(self):
        points=np.array([[0.,0.,.5],[4.,0.,.2],[6.,0.,.1],[8.,0.,.1]])
        bucket=np.array([False,False,True,False]);truck=np.array([False,False,False,True])
        rt=SimpleNamespace(os=SimpleNamespace(environ={'EXCAVATOR_SOIL_BACKEND':'newton_mpm'}),STATE={},
            capture_initial_pile_particle_ids=lambda:None,sand_particle_positions=lambda:points,
            sand_region_masks=lambda pts:(np.array([True,False,False,False]),bucket,truck))
        install_source_identity(rt)
        rt.capture_initial_pile_particle_ids()
        expected=initial_source_mask(points,[2],[3])
        np.testing.assert_array_equal(expected,[True,True,False,False])
        np.testing.assert_array_equal(expected,rt.STATE['dataset_initial_pile_particle_mask'])
        self.assertEqual(rt.STATE['repeat_source_identity'],SOURCE_IDENTITY)

    def test_47_loaded_points_remain_47_eligible_source_ids_after_deposit(self):
        points=np.array([[0.,0.,.5]]+[[4.+i*.01,0.,.2] for i in range(46)])
        mask=initial_source_mask(points,[],[])
        ids=np.flatnonzero(mask).tolist()
        self.assertEqual(len(ids),47)
        tracker=ParticleTransferTracker();tracker.observe(47,[],[])
        tracker.observe(47,ids,[])
        result=tracker.observe(47,[],ids)
        self.assertEqual(result['payload_particles'],47)
        self.assertEqual(result['spillage_particles'],0)

    def test_production_capture_uses_full_material_probe_and_restores_tracker(self):
        tree=ast.parse((ROOT/'ExcavatorVLA-smolvla/run_simulation.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='capture_initial_pile_source_mask')
        points=np.array([[0.,0.,.5],[4.,0.,.2],[6.,0.,.1],[8.,0.,.1]],dtype=np.float32)
        original=np.array([True,False,False,False]);tracker={'available':True,'mask':original,'source_count':1}
        def bucket_probe(required,ids):
            self.assertTrue(tracker['mask'].all());ids.append(2)
            return {'source_tracking_valid':True,'particle_count':4}
        def truck_probe(ids):
            self.assertTrue(tracker['mask'].all());ids.append(3)
            return {'available':True,'source_tracking_valid':True,'particle_count':4}
        api={'soil_backend':'newton_mpm','particle_positions_fn':lambda:points}
        ns=dict(np=np,os=os,builtins=SimpleNamespace(_SAND_SITE=api),sand_api=api,
                bucket_source_tracker=tracker,estimate_bucket_load_particles=bucket_probe,
                estimate_truck_bed_load_particles=truck_probe)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<production-source-capture>','exec'),ns)
        with patch.dict(os.environ,{'EXCAVATOR_MPM_SOURCE_IDENTITY':SOURCE_IDENTITY}):
            result=ns['capture_initial_pile_source_mask']()
        self.assertTrue(result['available'],result)
        self.assertEqual(result['source_identity'],SOURCE_IDENTITY)
        self.assertEqual(result['source_count'],2)
        np.testing.assert_array_equal(result['mask'],[True,True,False,False])
        self.assertIs(tracker['mask'],original)
        self.assertEqual(tracker['source_count'],1)

    def test_invalid_initial_source_geometry_or_ids_fail(self):
        for points,ids in [(np.array([[np.nan,0,0]]),[]),(np.zeros((2,3)),[2]),(np.zeros((2,3)),[-1])]:
            with self.assertRaises(ValueError):initial_source_mask(points,ids,[])


if __name__=='__main__':unittest.main()
