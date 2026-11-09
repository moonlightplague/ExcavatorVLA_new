import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from act_transport_diagnostics import analyze_transport


class TransportDiagnosticsTests(unittest.TestCase):
    def fixture(self,folder,retained,payload=0):
        scene=dict(truck_counting_volume=dict(local_from_world=np.eye(4).tolist(),
            local_min=[-1,-2,0],local_max=[1,2,2]),
            observation_context=dict(unload_landing_xyz=[0,0,0]))
        (folder/'scene_context.json').write_text(json.dumps(scene))
        (folder/'episode_status.json').write_text(json.dumps(dict(scene_seed=4,success=False)))
        rows=[]
        for i,n in enumerate(retained):
            s=[0.]*28;s[3]=-2;s[4]=.2;s[16]=1;s[20]=-1;s[21]=-5;s[23]=-3
            rows.append(dict(step=i,state=s,particle_transfer_metrics=dict(
                payload_particles=payload,retained_particles=n,scooped_particles=max(120,n),
                spillage_particles=max(120,n)-n)))
        (folder/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_rescooping_after_emptying_does_not_mask_the_first_transport_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);self.fixture(p,[120,0,0,0]+[200]*90)
            r=analyze_transport(p)
            self.assertEqual(r['pretruck_empty_step'],1)
            self.assertEqual(r['checkpoints']['8']['retained_particles'],0)
            self.assertTrue(r['checkpoints']['8']['truncated_by_transport_end'])

    def test_small_partial_deposit_does_not_hide_losing_the_rest_before_truck(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);self.fixture(p,[120,0,0,0],payload=50)
            self.assertEqual(analyze_transport(p)['pretruck_empty_step'],1)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);self.fixture(p,[120,0,0,0],payload=100)
            self.assertIsNone(analyze_transport(p)['pretruck_empty_step'])

    def test_parking_with_material_has_zero_effective_loaded_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);self.fixture(p,[120]*90)
            r=analyze_transport(p)
            self.assertIsNone(r['pretruck_empty_step'])
            self.assertEqual(r['loaded_bed_gap_reduction_m'],0)


if __name__=='__main__':
    unittest.main()
