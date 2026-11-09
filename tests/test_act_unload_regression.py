from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from assess_act_unload_regression import assess


class RegressionAssessmentTests(unittest.TestCase):
    def build(self,out):
        def write(name,value):(out/name).write_text(json.dumps(value))
        write('status.json',dict(status='completed'))
        write('regression_protocol.json',dict(repetitions=4,minimum_matched_pairs=2,minimum_affected_pairs=2,
            minimum_affected_current_success_pairs=1,minimum_branch_steps=50,
            maximum_truck_fraction_loss_per_pair=.05,maximum_repeated_current_difference=.05))
        records={}
        for r in range(1,5):
            common=dict(branch_steps=100,original_task_completed=True,truck_fraction=.99,geometry_feedback_active_steps=0)
            records[str(r)]=dict(status='completed',scene=9,repetition=r,cohort_count=200,branches=dict(
                off=dict(common),off_repeat=dict(common),on=dict(common,truck_fraction=1.,geometry_feedback_active_steps=30)))
        write('results.json',dict(results=records,not_reached=0))
        return records

    def test_any_lost_success_rejects_even_with_other_successful_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);records=self.build(out)
            self.assertTrue(assess(out)['regression_passed'])
            records['1']['branches']['on']['original_task_completed']=False
            (out/'results.json').write_text(json.dumps(dict(results=records,not_reached=0)))
            result=assess(out)
            self.assertFalse(result['checks']['task_preservation']);self.assertFalse(result['regression_passed'])

    def test_missing_planned_repetition_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);records=self.build(out);records.pop('4')
            (out/'results.json').write_text(json.dumps(dict(results=records,not_reached=0)))
            with self.assertRaises(AssertionError):assess(out)


if __name__=='__main__':unittest.main()
