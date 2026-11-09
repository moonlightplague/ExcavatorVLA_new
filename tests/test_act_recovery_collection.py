"""Expert provenance and feedback limits must survive collection/export."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_act_recovery_collection import accepted_supervision,takeover_valid
from act_recovery_expert import RecoveryExpert


class RecoveryCollectionTests(unittest.TestCase):
    def test_only_successful_new_expert_actions_get_supervision(self):
        sources=['model_prefix_replay','model_prefix_replay','online_geometry_expert','online_geometry_expert']
        self.assertEqual(accepted_supervision(sources,True),[0,0,1,1])
        self.assertEqual(accepted_supervision(sources,False),[0,0,0,0])

    def test_takeover_requires_real_loaded_failure_class(self):
        state=np.zeros(28);state[26]=.02;state[3]=-1.6
        self.assertTrue(takeover_valid(1,state,dict(retained_particles=120)))
        self.assertFalse(takeover_valid(1,state,dict(retained_particles=20)))
        self.assertFalse(takeover_valid(9,state,dict(retained_particles=120)))
        state[3]=-2.0
        self.assertTrue(takeover_valid(9,state,dict(retained_particles=120)))
        state[3]=.1
        self.assertTrue(takeover_valid(20,state,dict(retained_particles=120)))

    def test_feedback_commands_target_integrated_command_and_respect_speeds(self):
        expert=RecoveryExpert.__new__(RecoveryExpert);expert.dt=.1
        goal=np.array([1.,.4,-.8,-2.094]);command=np.array([0.,0.,0.,-2.094]);speeds=np.array([.28,.24,.24,.4])
        action=expert.velocity(goal,command,speeds)
        self.assertTrue(np.all(np.abs(action)<=speeds+1e-12))
        self.assertEqual(float(action[3]),0.)
        self.assertTrue(np.all(np.abs(goal-(command+.1*action))<=np.abs(goal-command)+1e-12))

    def test_open_bucket_recovery_path_stays_above_ground_and_finishes_closed(self):
        import json,math
        model_path=ROOT/'outputs/analysis/act_reachability_20261103_01/robot_and_truck_model.json'
        collision=ROOT/'outputs/analysis/act_bucket_collision_20261105_01.json'
        cases=ROOT/'outputs/experiments/act_recovery_data_prepare_20261105_01/protocol.json'
        if not all(p.exists() for p in (model_path,collision,cases)):
            self.skipTest('Authored recovery geometry unavailable')
        model=json.loads(model_path.read_text());model['bucket_collision_vertices']=json.loads(collision.read_text())['vertices']
        case=json.loads(cases.read_text())['cases'][-1]
        scene=json.loads((Path(case['source_episode'])/'scene_context.json').read_text())
        expert=RecoveryExpert(model,scene,case['recorded_state'])
        self.assertEqual(expert.phase,'unwind')
        self.assertLessEqual(expert.chain.carry_pitch(expert.curl_path[-1]),math.radians(-130))
        previous=np.asarray(case['recorded_state'][:4])
        for goal in expert.curl_path:
            for q in previous+(goal-previous)*np.linspace(0,1,7)[:,None]:
                self.assertGreaterEqual(expert.minimum_bucket_z(q),.005)
            previous=goal


if __name__=='__main__':unittest.main()
