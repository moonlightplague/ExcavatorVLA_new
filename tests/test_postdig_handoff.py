import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_postdig_handoff import audit


class HandoffAuditTests(unittest.TestCase):
    def record(self, folder, autonomous_progress):
        matrix = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        scene = {'truck_counting_volume': {'local_from_world': matrix,
                 'local_min': [0., 0., 0.], 'local_max': [2., 2., 2.]}}
        (folder / 'scene_context.json').write_text(json.dumps(scene))
        rows = []
        for step in range(85):
            if step == 0:
                x = 15.
            elif autonomous_progress:
                x = max(2.5, 10. - .1 * (step - 2))
            else:
                x = 2.5
            trace = {'override': step == 0, 'handback': step == 1,
                     'phase': 'completed_secure_lift' if step == 1 else 'idle'}
            rows.append({'step': step, 'carry_guard': {'stall_recovery': trace},
                'input_bucket_landmarks_world': {'load': [x, 1., 3.]},
                'particle_transfer_metrics': {'retained_particles': 100 + step, 'payload_particles': 0}})
        (folder / 'steps.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def test_only_post_handoff_motion_counts_and_particles_are_causally_aligned(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.record(folder, False)
            result = audit(folder)
            self.assertFalse(result['sustained_autonomous_loaded_transport'])
            self.assertAlmostEqual(result['loaded_bed_gap_reduction_m'], 0.)
            self.assertEqual(result['start']['step'], 2)
            self.assertEqual(result['start']['retained_particles'], 101)
            self.record(folder, True)
            result = audit(folder)
            self.assertTrue(result['sustained_autonomous_loaded_transport'])
            self.assertGreater(result['eight_second_loaded_bed_gap_reduction_m'], 3.)


if __name__ == '__main__':
    unittest.main()
