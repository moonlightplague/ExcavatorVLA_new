"""Synthetic five-cycle regressions; not model-performance measurements."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'ExcavatorVLA-smolvla')]
from excavator_common.task_goal_update import update_task_goals
from repeat_eval import RepeatedTask, verify_goal_ack, audit_repeat_trace, load_repeat_spec


def spec():
    return {'cycles': 5, 'fps': 10, 'cycle_step_limit': 30, 'min_payload_particles': 3,
            'min_scooped_particles': 5, 'strict_min_payload_particles': 100,
            'return_xy_radius_m': 1.25, 'return_height_above_target_m': [.15, 2.5],
            'return_empty_bucket_particles': 5, 'return_consecutive_observations': 3,
            'goals': [{'dig_target_xyz': [float(i), 7., .3],
                       'unload_landing_xyz': [-7. - i, -4., 4.]} for i in range(5)]}


def observation(bucket=(), truck=(), near=False, goal=None):
    goal = goal or spec()['goals'][0]
    tip = [goal['dig_target_xyz'][0] + (.2 if near else 4.), 7., 1.3]
    load = [*tip[:2], 1.5]
    state = [0.] * 28
    state[3] = -1.2 if bucket else 0.
    state[18:21] = [a-b for a,b in zip(goal['dig_target_xyz'], tip)]
    state[21:24] = [a-b for a,b in zip(goal['unload_landing_xyz'], load)]
    return {'evaluation_particle_ids': {'particle_count': 1000, 'bucket': list(bucket), 'truck': list(truck)},
            'observation_state_28d': state, 'observation_state': list(state),
            'bucket_load_metrics': {'count': len(bucket)},
            'bucket_landmarks_world': {'tip': tip, 'load': load, 'pour': [load[0]+.2, load[1], load[2]]},
            'observation_32d_ready': True, 'joint_positions': state[:4],
            'joint_command_positions': state[:4], 'task_text': 'Dig five scoops.',
            'cameras': {'0': {'rgb_zlib_b64': 'same_pixels'}},
            'evaluation_video_frame': {'rgb_zlib_b64': 'same_overview'},
            'bridge_timing': {'physics_ticks': 6, 'simulated_seconds': .1,
                              'physics_time_advanced_seconds': .1}}


class GoalUpdateTests(unittest.TestCase):
    def context(self):
        return {'allow_task_goal_updates': True, 'goal_revision': 0, 'task_text': 'Dig five scoops.',
                **spec()['goals'][0], 'initial_heading_rad': 0.}

    def test_ack_changes_only_goal_geometry_not_physics_or_previous_action(self):
        before = observation()
        original = copy.deepcopy(before)
        command = {'type': 'set_task_goals', 'revision': 1, **spec()['goals'][1]}
        context, after = update_task_goals(self.context(), before, command, 0.)
        verify_goal_ack(before, after, spec()['goals'][1], 1)
        self.assertEqual(before, original)
        self.assertEqual(context['goal_revision'], 1)
        self.assertNotEqual(before['observation_state_28d'][18:24], after['observation_state_28d'][18:24])
        self.assertEqual(after['bridge_timing']['physics_ticks'], 0)

    def test_legacy_clients_bad_revisions_and_action_injection_are_rejected(self):
        command = {'type': 'set_task_goals', 'revision': 1, **spec()['goals'][1]}
        for context, mutation in [({}, {}), (self.context(), {'revision': 0}),
                (self.context(), {'revision': True}),
                (self.context(), {'dig_target_xyz': [float('nan'), 1., 2.]}),
                (self.context(), {'joint_velocities': [0.] * 4}),
                (self.context(), {'phase_index': 2})]:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                update_task_goals(context, observation(), {**command, **mutation}, 0.)


class RepeatedTaskTests(unittest.TestCase):
    def test_old_truck_material_and_initial_bucket_residue_do_not_complete_another_cycle(self):
        initial = observation(bucket=[5, 6], truck=list(range(5)))
        machine = RepeatedTask(spec(), initial, started_wall=0.)
        for step in range(5):
            machine.observe(observation(truck=list(range(7)), near=True), step, step / 10)
        self.assertEqual(machine.transfer['payload_particles'], 0)
        self.assertEqual(machine.transfer['scooped_particles'], 0)
        self.assertEqual(machine.result()['repeat_cycles_completed'], 0)

    def test_dump_without_physical_return_does_not_advance(self):
        machine = RepeatedTask(spec(), observation(), started_wall=0.)
        machine.observe(observation(bucket=range(5)), 0, 0.)
        for step in range(1, 30):
            machine.observe(observation(truck=range(5)), step, step / 10)
        result = machine.result(final=True, wall=3.)
        self.assertEqual(result['repeat_cycles_completed'], 0)
        self.assertFalse(result['repeat_collection_success'])
        self.assertEqual(result['repeat_failure_mode'], 'return_timeout')
        self.assertEqual([row['status'] for row in result['repeat_cycles']], ['failed'] + ['not_attempted'] * 4)

    def sequence(self):
        settings = spec()
        initial = observation()
        machine = RepeatedTask(settings, initial, started_wall=0.)
        rows, updates, truck = [], [], []
        for cycle in range(5):
            current = list(range(cycle * 5, cycle * 5 + 5))
            goal = settings['goals'][cycle]
            replies = [observation(bucket=current, truck=truck, goal=goal)]
            replies += [observation(truck=truck + current, goal=goal) for _ in range(3)]
            replies += [observation(truck=truck + current, near=True, goal=goal) for _ in range(3)]
            for reply in replies:
                step = len(rows)
                event = machine.observe(reply, step, step / 10)
                row = {'step': step, 'repeat_cycle_index': cycle, 'state': reply['observation_state_28d'],
                       'evaluation_particle_ids': reply['evaluation_particle_ids'],
                       'repeat_bucket_particles': reply['bucket_load_metrics']['count'],
                       'bucket_landmarks_world': reply['bucket_landmarks_world'], 'action_source': 'model',
                       'camera_sha256': {'0': 'same'}, 'joint_command_positions': reply['joint_command_positions'],
                       'input_history_steps': [max(0, step+d) for d in (-6, -3, 0)],
                       'input_history_is_pad': [step+d < 0 for d in (-6, -3, 0)]}
                rows.append(row)
            truck += current
            if cycle < 4:
                self.assertEqual(event, 'next_cycle')
                updates.append({'after_step': step, 'command': {'type': 'set_task_goals', 'revision': cycle + 1,
                    **settings['goals'][cycle+1]}, 'physics_ticks': 0, 'simulated_seconds': 0.,
                    'camera_sha256': row['camera_sha256'], 'joint_command_positions': row['joint_command_positions']})
                machine.advance(reply, step / 10)
            else:
                self.assertEqual(event, 'finished')
        return initial, machine.result(final=True, wall=len(rows)/10), rows, updates

    def test_all_five_returns_and_per_cycle_ids_are_required(self):
        _initial, result, _rows, _updates = self.sequence()
        self.assertTrue(result['repeat_collection_success'])
        self.assertFalse(result['repeat_strict_success'])
        self.assertEqual(result['repeat_cycles_completed'], 5)
        self.assertEqual(result['repeat_returns_completed'], 5)
        self.assertEqual([row['payload_particles'] for row in result['repeat_cycles']], [5] * 5)

    def test_independent_trace_audit_checks_goals_history_and_zero_time_transitions(self):
        initial, result, rows, updates = self.sequence()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'episodes/episode_0001'
            folder.mkdir(parents=True)
            (root/'config.json').write_text(json.dumps({'observation_history': {'delta_indices': [-6,-3,0]}}))
            (folder/'initial_observation.json').write_text(json.dumps({
                'evaluation_particle_ids': initial['evaluation_particle_ids'],
                'observation_context': {'initial_heading_rad': 0.}}))
            def publish():
                (folder/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                (folder/'goal_updates.json').write_text(json.dumps(updates))
            publish()
            self.assertEqual(audit_repeat_trace(folder, spec(), result)['status'], 'passed')
            rows[7]['input_history_steps'] = [0, 0, 0]
            publish()
            with self.assertRaisesRegex(ValueError, 'history'):
                audit_repeat_trace(folder, spec(), result)
            rows[7]['input_history_steps'] = [1, 4, 7]
            updates[0]['physics_ticks'] = 6
            publish()
            with self.assertRaisesRegex(ValueError, 'transition'):
                audit_repeat_trace(folder, spec(), result)


class SharedRolloutIntegrationTests(unittest.TestCase):
    def test_model_loop_completes_five_returns_replans_goals_and_audits_continuous_history(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        import torch
        sys.path.insert(0, str(ROOT / 'ExcavatorVLA-vlm'))
        import eval_smolvla_success as evaluation
        from excavator_common.evaluation_metrics import ParticleTransferTracker
        from excavator_vlm.policies.pi05.history import make_history

        class Model:
            config = SimpleNamespace(n_obs_steps=3, excavator_observation_history=make_history([-6,-3,0]))
            def reset(self): self.calls = []
            def predict_action_chunk(self, batch):
                self.calls.append(batch['observation.state'].clone())
                return torch.zeros(1, 10, 4)

        settings = spec()
        model, global_ids = Model(), ParticleTransferTracker()
        context = {'allow_task_goal_updates': True, 'goal_revision': 0, 'initial_heading_rad': 0.,
                   'phase_mode': 'auto', 'task_text': 'Dig five scoops.', **settings['goals'][0]}
        scripted, truck = [], []
        for cycle, goal in enumerate(settings['goals']):
            bucket = list(range(cycle*5, cycle*5+5))
            scripted += [observation(bucket=bucket, truck=truck, goal=goal)]
            scripted += [observation(truck=truck+bucket, goal=goal) for _ in range(3)]
            scripted += [observation(truck=truck+bucket, near=True, goal=goal) for _ in range(3)]
            truck = truck + bucket
        initialized, position, latest = False, 0, None

        def enrich(reply):
            reply = copy.deepcopy(reply)
            ids = reply['evaluation_particle_ids']
            reply['particle_transfer_metrics'] = global_ids.observe(ids['particle_count'], ids['bucket'], ids['truck'])
            reply['observation_context'] = dict(context)
            reply['observation_contract'] = {'observation.state': {'names': list(evaluation.STATE_NAMES_28D)}}
            reply['truck_bed_load_metrics'] = {'available': True, 'source_tracking_valid': True,
                'method': evaluation.TRUCK_COUNT_METHOD, 'scene_seed': 2, 'count': len(ids['truck'])}
            return reply

        scene = {'scene_seed': 2, 'simulator_pid': 123,
                 'simulation_settings': {'soil_backend': 'physx'}, 'observation_context': context,
                 'canonical_joint_limits_rad': [[-3.2,3.2],[-1.3,1.3],[-1.7,1.7],[-2.1,1.6]]}

        def exchange(sock, command):
            nonlocal initialized, position, latest, context
            if command.get('type') == 'scene_context': return scene
            if command.get('type') == 'handshake':
                return {'type': 'handshake_ack', 'timing_mode': 'deterministic_lockstep',
                        'state_names': list(evaluation.STATE_NAMES_28D), 'training_fps': 10}
            if command.get('type') == 'set_task_goals':
                context, latest = update_task_goals(context, latest, command, 0.)
                return latest
            if not initialized:
                initialized, latest = True, enrich(observation())
            else:
                latest = enrich(scripted[position])
                position += 1
            return latest

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'workers/worker_00').mkdir(parents=True)
            log = output/'simulator.log'
            log.touch()
            session = SimpleNamespace(command=[], log_path=log, sock=None,
                prepare=lambda seed: {}, close=lambda: None)
            args = SimpleNamespace(output=output, episodes=1, worker_index=0, worker_count=1,
                policy_type='pi05', gpu_label='fixture', seed=2, min_particles=100, video=False,
                steps=150, replan_interval=3, save_trajectory=False, progress_interval=0.,
                replay_reference=False, fresh_simulator=True, post_rollout_observe_steps=0)
            config = {'execution_source': 'model', 'evaluation_standard': 'excavator_repeat_reference_v1',
                'evaluation_profile': 'diagnostic', 'evaluation_id': 'fixture', 'checkpoint_sha256': 'fixture',
                'simulation_settings': scene['simulation_settings'], 'repeat_spec': settings,
                'observation_history': make_history([-6,-3,0])}
            (output/'config.json').write_text(json.dumps(config))
            frames = {str(i): np.zeros((256,256,3), dtype=np.uint8) for i in range(3)}
            with patch.object(evaluation, 'exchange', side_effect=exchange), \
                    patch.object(evaluation, 'decoded_frames', return_value=frames):
                evaluation.run_episodes(args, {'fps': 10}, None, np.array([6.,2.5,2.5,20.]),
                    model, lambda batch: batch, lambda action: action, 'fixture', config, session)
            result = json.loads((output/'episodes/episode_0001/episode_status.json').read_text())
            self.assertEqual(result['status'], 'completed')
            self.assertTrue(result['success'])
            self.assertEqual(result['repeat_cycles_completed'], 5)
            self.assertEqual(result['policy_steps_executed'], 35)
            self.assertEqual(len(model.calls), 15)  # Cached old-goal actions are discarded after every return.
            self.assertEqual(json.loads((output/'episodes/episode_0001/repeat_audit.json').read_text())['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
