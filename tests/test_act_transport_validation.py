import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_act_transport_validation import empty_event_followup


def row(payload=0):
    return dict(particle_transfer_metrics=dict(payload_particles=payload))


class EmptyFollowupTests(unittest.TestCase):
    def test_incoming_material_within_three_seconds_is_not_far_loss(self):
        transport=dict(pretruck_empty_step=1,at_pretruck_empty=dict(bed_gap_m=4.))
        rows=[row() for _ in range(33)];rows[31]=row(150)
        with tempfile.TemporaryDirectory() as directory:
            result=empty_event_followup(transport,rows,Path(directory))
        self.assertEqual(result['observations'],31)
        self.assertTrue(result['deposit_during_followup'])
        self.assertFalse(result['far_empty_no_early_deposit'])

    def test_tail_extends_observation_without_rewriting_scored_rows(self):
        transport=dict(pretruck_empty_step=1,at_pretruck_empty=dict(bed_gap_m=4.))
        rows=[row() for _ in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);tail=[row() for _ in range(31)];tail[10]=row(120)
            (p/'post_rollout_steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tail))
            result=empty_event_followup(transport,rows,p)
        self.assertEqual(len(rows),5)
        self.assertEqual(result['observations'],31)
        self.assertTrue(result['deposit_during_followup'])
        self.assertFalse(result['far_empty_no_early_deposit'])

    def test_near_truck_or_short_observation_is_not_classified_as_far_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            for gap,length,expected in [(1.,35,False),(4.,10,False),(4.,35,True)]:
                transport=dict(pretruck_empty_step=1,at_pretruck_empty=dict(bed_gap_m=gap))
                result=empty_event_followup(transport,[row() for _ in range(length)],p)
                self.assertEqual(result['far_empty_no_early_deposit'],expected)


if __name__=='__main__':
    unittest.main()
