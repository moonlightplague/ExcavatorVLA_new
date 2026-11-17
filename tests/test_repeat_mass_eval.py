"""Mass and retention regressions; synthetic observations are never success data."""
import copy
import json
from pathlib import Path
import sys
import unittest
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts'),
               str(Path(__file__).resolve().parents[1] / 'ExcavatorVLA-smolvla')]
from repeat_eval import MASS_STANDARD, RepeatedTask
from test_repeat_eval import observation, spec
from excavator_diagnostics import reference_setup
from excavator_common.vla_observation_contract import STATE_NAMES_28D, ACTION_NAMES_4D


def mass_spec():
    result = spec()
    result.update(version=MASS_STANDARD, cycles=10, metric_point_mass_kg=5., expected_metric_points=1000,
                  min_payload_particles=10, min_scooped_particles=10,
                  minimum_per_cycle_kg=50., minimum_retention_ratio=.8)
    result['goals'] *= 2
    return result


def reply(**kwargs):
    result = observation(**kwargs)
    result['evaluation_particle_ids'].update(metric_point_mass_kg=5., soil_backend='newton_mpm')
    return result


class MassRepeatTests(unittest.TestCase):
    def sequence(self, lose_old_on_last=False):
        settings = mass_spec()
        machine = RepeatedTask(settings, reply(), started_wall=0.)
        truck, step = [], 0
        for cycle in range(10):
            ids = list(range(10*cycle, 10*(cycle+1)))
            goal = settings['goals'][cycle]
            if lose_old_on_last and cycle == 9:
                truck = truck[50:]
            observations = [reply(bucket=ids, truck=truck, goal=goal)]
            observations += [reply(truck=truck+ids, goal=goal) for _ in range(3)]
            observations += [reply(truck=truck+ids, near=True, goal=goal) for _ in range(3)]
            for value in observations:
                event = machine.observe(value, step, step/10)
                step += 1
            truck += ids
            if cycle < 9:
                self.assertEqual(event, 'next_cycle')
                machine.advance(value, step/10)
        return machine.result(final=True, wall=step/10)

    def test_ten_distinct_deposits_and_final_returns_pass_mass_policy(self):
        result = self.sequence()
        self.assertTrue(result['repeat_collection_success'])
        self.assertEqual(result['repeat_cycles_completed'], 10)
        self.assertEqual(result['repeat_returns_completed'], 10)
        self.assertEqual(result['repeat_final_payload_mass_kg'], 500.)
        self.assertEqual([r['delivered_mass_kg'] for r in result['repeat_cycles']], [50.]*10)
        self.assertEqual(result['repeat_deposit_retention_ratio'], 1.)

    def test_losing_previous_deposits_fails_even_if_all_ten_cycles_finish(self):
        result = self.sequence(lose_old_on_last=True)
        self.assertEqual(result['repeat_cycles_completed'], 10)
        self.assertFalse(result['repeat_collection_success'])
        self.assertEqual(result['repeat_deposit_retention_ratio'], .5)
        self.assertEqual(result['repeat_failure_mode'], 'insufficient_final_mass_or_deposit_retention')

    def test_mass_or_backend_drift_is_infrastructure_error(self):
        initial = reply()
        for key, value in [('metric_point_mass_kg', .005), ('soil_backend', 'physx'), ('particle_count', 999)]:
            bad = copy.deepcopy(initial)
            bad['evaluation_particle_ids'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Live material'):
                RepeatedTask(mass_spec(), bad)

    def test_subthreshold_deposit_and_old_truck_ids_cannot_complete_cycle(self):
        machine = RepeatedTask(mass_spec(), reply(truck=range(50)), started_wall=0.)
        machine.observe(reply(bucket=range(50, 59), truck=range(50)), 0, 0.)
        for step in range(1, 30):
            machine.observe(reply(truck=range(59), near=True), step, step/10)
        self.assertFalse(machine.result(final=True)['repeat_collection_success'])
        self.assertEqual(machine.result(final=True)['repeat_cycle_payload_particles'], 9)


class SceneReferenceTests(unittest.TestCase):
    def test_scene_only_reference_has_no_expert_actions_and_rejects_action_injection(self):
        data = dict(schema_version='excavator_scene_setup_v2', fps=10, action_units='rad/s',
            state_names=list(STATE_NAMES_28D), action_names=list(ACTION_NAMES_4D),
            provenance={'source_export_episode':2}, task='Dig ten consecutive scoops.',
            profile=dict(sand_xy=[0., 6.], truck_translation_xyz=[-5., -5., 0.],
                         truck_yaw_deg=-70., unload_landing_xyz=[-7., -4., 4.]),
            initial_joints_deg=[0.]*4, dig_target_xyz=[0., 6., .12],
            sand_amount=9., expected_initial_state=[0.]*28)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.json'
            path.write_text(json.dumps(data))
            loaded = reference_setup(path, 10)
            self.assertEqual(loaded['actions'].shape, (0, 4))
            data['actions'] = [[1., 0., 0., 0.]]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'must not contain expert actions'):
                reference_setup(path, 10)


if __name__ == '__main__':
    unittest.main()
