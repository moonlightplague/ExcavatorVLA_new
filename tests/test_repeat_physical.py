"""Regression coverage for cycle counting independent of successful delivery."""
import json
from pathlib import Path
import sys
import unittest
import copy
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'ExcavatorVLA-smolvla')]
from repeat_eval import PHYSICAL_MASS_STANDARD, make_repeated_task


def spec(cycles=3):
    return dict(version=PHYSICAL_MASS_STANDARD, cycles=cycles, fps=10, cycle_step_limit=0,
                metric_point_mass_kg=5., expected_metric_points=1000,
                minimum_per_cycle_kg=50., minimum_retention_ratio=.8,
                min_payload_particles=10, min_scooped_particles=10, strict_min_payload_particles=100,
                pile_polygon_xy=[[-1.,-1.], [1.,-1.], [1.,1.], [-1.,1.]],
                pile_center_xy=[0.,0.], truck_center_xy=[8.,0.], ground_top_z=0.,
                dig_max_height_m=2.5, carry_min_height_m=1.2, carry_bucket_closed_rad=-1.,
                return_height_above_ground_m=[.15,2.5], return_empty_bucket_particles=5,
                return_consecutive_observations=3, physical_event_observations=3,
                goals=[dict(dig_target_xyz=[-1.,-1.,.12], unload_landing_xyz=[8.,0.,4.]) for _ in range(cycles)])


def observation(tip=(0.,0.,1.), angle=0., bucket=(), truck=()):
    state = [0.]*28
    state[3] = angle
    return dict(evaluation_particle_ids=dict(particle_count=1000, metric_point_mass_kg=5.,
                soil_backend='newton_mpm', bucket=list(bucket), truck=list(truck)),
                observation_state_28d=state, bucket_load_metrics={'count':len(bucket)},
                bucket_landmarks_world={'tip':list(tip), 'load':list(tip), 'pour':list(tip)})


class PhysicalCycleTests(unittest.TestCase):
    def run_cycles(self, outcomes):
        m = make_repeated_task(spec(len(outcomes)), observation(), started_wall=0.)
        truck, step = [], 0
        for cycle, success in enumerate(outcomes):
            ids = list(range(cycle*10, cycle*10+10))
            observations = [observation(bucket=ids,truck=truck)]*3
            observations += [observation(tip=(7.,0.,4.),angle=-2.,bucket=ids,truck=truck)]
            new_truck = truck+ids if success else truck
            observations += [observation(tip=(7.,0.,4.),truck=new_truck)]*3
            # This valid pile return is >1.25 m from the specific task dig point.
            observations += [observation(tip=(.9,.9,1.),truck=new_truck)]*3
            for reply in observations:
                event = m.observe(reply, step, step/10)
                step += 1
            self.assertEqual(len(m.cycles), cycle+1)
            truck = new_truck
            if cycle < len(outcomes)-1:
                self.assertEqual(event, 'next_cycle')
                m.advance(reply, step/10)
        self.assertEqual(event, 'finished')
        return m.result(final=True, wall=step/10)

    def test_three_physical_rounds_and_two_successful_deposits_are_distinct(self):
        result = self.run_cycles([True,False,True])
        self.assertEqual(result['repeat_cycles_completed'], 3)
        self.assertEqual(result['repeat_returns_completed'], 3)
        self.assertEqual(result['repeat_cycles_successful'], 2)
        self.assertEqual([r['physical_cycle_completed'] for r in result['repeat_cycles']], [True]*3)
        self.assertEqual(result['repeat_cycles'][1]['failure_mode'], 'low_delivered_mass_kg')
        self.assertFalse(result['repeat_collection_success'])

    def test_ten_successful_physical_rounds_pass_full_task(self):
        result = self.run_cycles([True]*10)
        self.assertTrue(result['repeat_collection_success'])
        self.assertEqual(result['repeat_cycles_completed'], 10)
        self.assertEqual(result['repeat_cycles_successful'], 10)
        self.assertEqual(result['repeat_final_payload_mass_kg'], 500.)

    def test_idle_opening_in_pile_cannot_create_cycles_or_trigger_500_step_timeout(self):
        m = make_repeated_task(spec(), observation(), started_wall=0.)
        for step in range(650):
            self.assertIsNone(m.observe(observation(), step, step/10))
        self.assertFalse(m.finished)
        self.assertEqual(m.result()['repeat_cycles_completed'], 0)
        result = m.result(final=True, wall=65.)
        self.assertEqual(result['repeat_failure_mode'], 'episode_step_limit')

    def test_collection_source_radius_includes_avalanched_return_outside_authored_polygon(self):
        settings=spec(1)
        settings['pile_radius_m']=3.
        m=make_repeated_task(settings,observation(),started_wall=0.)
        ids=list(range(10))
        sequence=[observation(bucket=ids)]*3+[observation(tip=(7.,0.,4.),angle=-2.,bucket=ids)]
        sequence += [observation(tip=(7.,0.,4.),truck=ids)]*3
        sequence += [observation(tip=(-.9,2.4,1.),truck=ids)]*3
        for i,r in enumerate(sequence):event=m.observe(r,i,i/10)
        self.assertEqual(event,'finished')
        self.assertEqual(m.result()['repeat_cycles_completed'],1)
        self.assertTrue(m.result()['repeat_collection_success'])

    def test_recorded_episode15_motion_and_mass_regression(self):
        fixture = json.loads((ROOT/'tests/fixtures/repeat_physical_episode15.json').read_text())
        m = make_repeated_task(fixture['spec'], fixture['initial_reply'], started_wall=0.)
        for step, observed in enumerate(fixture['observations']):
            reply = observed['reply']
            if m.observe(reply,step,step/10) == 'next_cycle':
                m.advance(reply, step/10)
        result = m.result(final=True,wall=len(fixture['observations'])/10)
        self.assertEqual(result['repeat_cycles_completed'], 3)
        self.assertEqual(result['repeat_cycles_successful'], 2)
        self.assertAlmostEqual(result['repeat_cycles'][0]['delivered_mass_kg'], 239.65724352278224)
        self.assertEqual(result['repeat_cycles'][1]['delivered_mass_kg'], 0.)
        self.assertAlmostEqual(result['repeat_cycles'][2]['delivered_mass_kg'], 287.5886922273387)
        self.assertAlmostEqual(result['repeat_final_payload_mass_kg'], 527.2459357501209)

    def test_shared_model_loop_advances_failed_cycle_and_preserves_history_and_audit(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        import torch
        import eval_smolvla_success as evaluation
        from excavator_common.task_goal_update import update_task_goals
        from excavator_common.evaluation_metrics import ParticleTransferTracker
        from smolvla_history import history_spec

        settings, history = spec(), history_spec()
        for i, goal in enumerate(settings['goals']):
            goal['dig_target_xyz'][0] += .1*i
            goal['unload_landing_xyz'][0] += .1*i

        class Model:
            config = SimpleNamespace(n_obs_steps=3, excavator_observation_history=history)
            def reset(self): self.calls = []
            def predict_action_chunk(self, batch):
                self.calls.append(batch['observation.state'].clone())
                return torch.zeros(1,10,4)
        model, global_ids = Model(), ParticleTransferTracker()
        model.history = history
        context = dict(allow_task_goal_updates=True, goal_revision=0, initial_heading_rad=0.,
                       phase_mode='auto', task_text='Dig three scoops.', **settings['goals'][0])
        scripted, truck = [], []
        for i, success in enumerate([True,False,True]):
            ids = list(range(i*10, i*10+10))
            scripted += [observation(bucket=ids,truck=truck)]*3
            scripted += [observation(tip=(7.,0.,4.),angle=-2.,bucket=ids,truck=truck)]
            truck = truck+ids if success else truck
            scripted += [observation(tip=(7.,0.,4.),truck=truck)]*3
            scripted += [observation(tip=(.9,.9,1.),truck=truck)]*3
        latest, position = None, 0

        def enrich(reply):
            r = copy.deepcopy(reply)
            ids, state = r['evaluation_particle_ids'], r['observation_state_28d']
            tip = r['bucket_landmarks_world']['tip']
            state[18:24] = [a-b for a,b in zip(context['dig_target_xyz'],tip)] + [a-b for a,b in zip(context['unload_landing_xyz'],tip)]
            r.update(observation_state=list(state), observation_32d_ready=True,
                     observation_context=dict(context), joint_positions=state[:4], joint_command_positions=state[:4],
                     task_text=context['task_text'], cameras={}, bridge_timing={'physics_ticks':6,'simulated_seconds':.1,'physics_time_advanced_seconds':.1},
                     observation_contract={'observation.state':{'names':list(evaluation.STATE_NAMES_28D)}},
                     particle_transfer_metrics=global_ids.observe(ids['particle_count'],ids['bucket'],ids['truck']),
                     truck_bed_load_metrics={'available':True,'source_tracking_valid':True,
                        'method':evaluation.TRUCK_COUNT_METHOD,'scene_seed':2,'count':len(ids['truck'])})
            return r

        scene = dict(scene_seed=2,simulator_pid=123,simulation_settings={'soil_backend':'newton_mpm'},
                     observation_context=context,canonical_joint_limits_rad=[[-3.2,3.2],[-1.3,1.3],[-1.7,1.7],[-2.1,1.6]])
        def exchange(sock, command):
            nonlocal context,latest,position
            if command.get('type') == 'scene_context': return scene
            if command.get('type') == 'handshake':
                return dict(type='handshake_ack',timing_mode='deterministic_lockstep',state_names=list(evaluation.STATE_NAMES_28D),training_fps=10)
            if command.get('type') == 'set_task_goals':
                context,latest = update_task_goals(context,latest,command,0.)
                return latest
            if latest is None: latest = enrich(observation())
            else:
                latest = enrich(scripted[position])
                position += 1
            return latest

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'workers/worker_00').mkdir(parents=True)
            log=output/'simulator.log';log.touch()
            session = SimpleNamespace(command=[],log_path=log,sock=None,prepare=lambda seed:{},close=lambda:None)
            args = SimpleNamespace(output=output,episodes=1,worker_index=0,worker_count=1,
                policy_type='smolvla',gpu_label='fixture',seed=2,min_particles=100,video=False,
                steps=5000,replan_interval=3,save_trajectory=False,progress_interval=0.,
                replay_reference=False,fresh_simulator=True,post_rollout_observe_steps=0)
            config = dict(execution_source='model',evaluation_standard=PHYSICAL_MASS_STANDARD,
                evaluation_profile='diagnostic',evaluation_id='fixture',checkpoint_sha256='fixture',
                simulation_settings=scene['simulation_settings'],repeat_spec=settings,observation_history=history)
            (output/'config.json').write_text(json.dumps(config))
            frames = {str(i):np.zeros((256,256,3),dtype=np.uint8) for i in range(3)}
            with patch.object(evaluation,'exchange',side_effect=exchange), patch.object(evaluation,'decoded_frames',return_value=frames):
                evaluation.run_episodes(args,{'fps':10},None,np.array([6.,2.5,2.5,6.]),
                    model,lambda batch:batch,lambda action:action,'fixture',config,session)
            folder=output/'episodes/episode_0001'
            result=json.loads((folder/'episode_status.json').read_text())
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['repeat_cycles_completed'],3)
            self.assertEqual(result['repeat_cycles_successful'],2)
            self.assertFalse(result['success'])
            self.assertEqual(result['policy_steps_executed'],30)
            self.assertEqual(len(json.loads((folder/'goal_updates.json').read_text())),2)
            self.assertEqual(json.loads((folder/'repeat_audit.json').read_text())['status'],'passed')


if __name__ == '__main__':
    unittest.main()
