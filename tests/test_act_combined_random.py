import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_act_combined_random import aggregate,make_command,controller_metrics


class CombinedRandomTests(unittest.TestCase):
    def test_random_command_has_fresh_seeds_and_no_replayed_scenes(self):
        source=ROOT/'outputs/baselines/act_phw40k_carry275_20261105/evidence/protocol.json'
        if not source.exists():self.skipTest('Retained evaluation protocol absent')
        template=json.loads(source.read_text())['command']
        command=make_command(template,Path('/frozen'),Path('/checkpoint'),Path('/output'),40,4100000)
        for flag,value in [('--episodes','40'),('--seed','4100000'),('--steps','500'),('--workers','2')]:
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertNotIn('--reference-episode',command)
        self.assertNotIn('--replay-reference',command)
        with self.assertRaises(ValueError):
            make_command(template+['--reference-episode','old_scene'],Path('/frozen'),Path('/checkpoint'),Path('/output'),40,4100000)

    def test_success_denominator_waits_for_all_planned_episodes(self):
        def record(seed,success):
            return dict(status='completed',scene_seed=seed,success=success,recovery_triggered=False,unload_hold_steps=0)
        records={'1':record(10,True),'2':record(11,False)}
        pending=aggregate(records,4)
        self.assertFalse(pending['result_valid'])
        self.assertIsNone(pending['success_rate'])
        records.update({'3':record(12,True),'4':dict(status='infrastructure_error')})
        self.assertIsNone(aggregate(records,4)['success_rate'])
        records['4']=record(13,False)
        self.assertEqual(aggregate(records,4)['success_rate'],.5)
        records['5']=record(14,True)
        with self.assertRaises(ValueError):aggregate(records,4)

    def test_wait_is_not_counted_as_recovery_and_timeout_is_observed_once(self):
        def row(step,hold=False,wait=False,started=False,reason=None,verified=False):
            trace=dict(override=True,handback=verified,phase='completed_secure_align' if verified else None,
                unload_pose_hold=dict(active=hold,started=False),
                unload_handoff_wait=dict(active=wait,started=started,steps=20 if reason else int(wait),release_reason=reason))
            return dict(step=step,carry_guard=dict(stall_recovery=trace))
        rows=[row(0),row(1,wait=True,started=True,verified=True),row(2,wait=True),
            row(3,reason='act_opening_wait_timeout'),row(4,hold=True)]
        result=controller_metrics(rows,require_wait=True)
        self.assertEqual(result['recovery_steps'],2)
        self.assertEqual(result['handoff_wait_steps'],2)
        self.assertEqual(result['handoff_wait_timeout_events'],1)
        self.assertEqual(result['first_verified_alignment_step'],1)
        self.assertFalse(result['episode_ended_waiting'])
        rows[1]['carry_guard']['stall_recovery']['handback']=False
        with self.assertRaises(ValueError):controller_metrics(rows,require_wait=True)

    def test_enabled_wait_requires_full_traces_and_original_budget(self):
        rows=[dict(step=0,carry_guard=dict(stall_recovery=dict(unload_pose_hold=dict(active=False))))]
        with self.assertRaises(ValueError):controller_metrics(rows,require_wait=True)
        rows[0]['carry_guard']['stall_recovery']['unload_handoff_wait']=dict(steps=21)
        with self.assertRaises(ValueError):controller_metrics(rows,require_wait=True)


if __name__=='__main__':unittest.main()
