"""Verify physical accounting, timing, rollout integration and persisted reports."""
import ast
from contextlib import nullcontext
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(BASE / 'ExcavatorVLA-smolvla'))
sys.path.insert(0, str(BASE / 'ExcavatorVLA-smolvla/scripts/evaluation'))
from excavator_common.evaluation_metrics import (
    EpisodeMetrics, METRIC_FIELDS, ParticleTransferTracker, failure_mode, numeric_summary,
)
from excavator_common import vla_observation_contract
from success_eval_runtime import atomic_json, episode_indices
from summarize_isaacsim_success_rate import build_summary, load_episode_rows, write_outputs


def extract_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def reply(tracker, bucket=(), truck=(), seconds=.1):
    transfer = tracker.observe(20, bucket, truck)
    return {
        'particle_transfer_metrics': transfer,
        'bridge_timing': {'simulated_seconds': seconds},
        'truck_bed_load_metrics': {'count': len(truck)},
        'bucket_load_metrics': {'count': len(bucket)},
        'joint_positions': [0., 0., 0., 0.], 'observation_state_28d': [0.] * 28,
        'joint_command_positions': [0.] * 4, 'task_text': 'Excavate.',
    }


class ParticleAccountingTests(unittest.TestCase):
    def test_deposit_is_not_spillage_and_retained_material_is_not_spilled(self):
        tracker = ParticleTransferTracker()
        first = tracker.observe(10, [], [9])
        self.assertEqual(first['payload_particles'], 0)
        self.assertIsNone(first['spillage_fraction'])
        tracker.observe(10, [0, 1, 2, 3], [9])
        final = tracker.observe(10, [3], [0, 1, 9])
        self.assertEqual(final['payload_particles'], 2)
        self.assertEqual(final['scooped_particles'], 4)
        self.assertEqual(final['retained_particles'], 1)
        self.assertEqual(final['spillage_particles'], 1)
        self.assertEqual(final['spillage_fraction'], .25)

    def test_recovery_rescoop_and_repeated_observations_do_not_double_count(self):
        tracker = ParticleTransferTracker()
        tracker.observe(10, [1, 2], [])
        self.assertEqual(tracker.observe(10, [], [])['spillage_particles'], 2)
        for _ in range(3):
            recovered = tracker.observe(10, [2], [1])
            self.assertEqual(recovered['scooped_particles'], 2)
            self.assertEqual(recovered['spillage_particles'], 0)
        self.assertEqual(tracker.observe(10, [], [])['spillage_particles'], 2)

    def test_initial_deposits_are_excluded_even_if_they_enter_the_bucket(self):
        tracker = ParticleTransferTracker()
        tracker.observe(10, [], [9])
        result = tracker.observe(10, [9], [1])
        self.assertEqual(result['scooped_particles'], 0)
        self.assertEqual(result['payload_particles'], 1)

    def test_invalid_identity_data_is_rejected(self):
        for count, bucket, truck in [(0, [], []), (True, [], []), (10, [-1], []),
                                     (10, [10], []), (10, [1.5], []), (10, [1], [1])]:
            with self.subTest(count=count, bucket=bucket, truck=truck), self.assertRaises(ValueError):
                ParticleTransferTracker().observe(count, bucket, truck)
        tracker = ParticleTransferTracker()
        tracker.observe(10, [], [])
        with self.assertRaisesRegex(ValueError, 'changed'):
            tracker.observe(11, [], [])


class EpisodeMetricTests(unittest.TestCase):
    def test_actual_simulation_durations_latency_percentiles_and_wall_fps(self):
        metrics, tracker = EpisodeMetrics(), ParticleTransferTracker()
        metrics.observe(reply(tracker, seconds=123))  # Initial no-op must not count.
        for seconds in (6 / 60, 7 / 60, 6 / 60):
            metrics.observe(reply(tracker, [0, 1], seconds=seconds), action_step=True)
        metrics.policy_ms = [10, 30]
        metrics.bridge_ms = [100, 200, 300]
        metrics.step_ms = [110, 200, 330]
        status = dict(status='completed', success=True, min_particles=1,
                      task_completed_success_and_bucket_open=True,
                      task_completion_time_from_policy_start_seconds=1.2)
        metrics.finalize(status, 1.5)
        self.assertAlmostEqual(status['cycle_time_seconds'], 19 / 60)
        self.assertEqual(status['cycle_time_wall_seconds'], 1.2)
        self.assertEqual(status['fps'], 2.)
        self.assertEqual(status['policy_inference_calls'], 2)
        self.assertEqual(status['policy_inference_time_seconds'], .04)
        self.assertEqual(status['policy_latency_ms_mean'], 20)
        self.assertEqual(status['policy_latency_ms_p50'], 20)
        self.assertEqual(status['policy_latency_ms_p95'], 30)
        self.assertEqual(status['bridge_latency_ms_mean'], 200)
        self.assertEqual(status['failure_mode'], 'none')

    def test_empty_and_error_episodes_have_null_measurements(self):
        status = dict(status='error', success=False)
        EpisodeMetrics().finalize(status, None)
        for field in ('payload_particles', 'spillage_particles', 'cycle_time_seconds',
                      'policy_latency_ms_mean', 'fps', 'simulated_rollout_time_seconds'):
            self.assertIsNone(status[field], field)
        self.assertEqual(status['failure_mode'], 'infrastructure_error')
        self.assertEqual(numeric_summary([None, float('nan'), float('inf')])['count'], 0)

    def test_missing_invalid_and_inconsistent_measurements_fail_closed(self):
        baseline = reply(ParticleTransferTracker())
        for key, value in [('available', False), ('method', 'wrong'), ('payload_particles', -1),
                           ('spillage_fraction', .2), ('retained_particles', 1)]:
            invalid = copy.deepcopy(baseline)
            invalid['particle_transfer_metrics'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                EpisodeMetrics().observe(invalid)
        for duration in (0, -1, float('nan'), float('inf')):
            invalid = copy.deepcopy(baseline)
            invalid['bridge_timing']['simulated_seconds'] = duration
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                EpisodeMetrics().observe(invalid, action_step=True)

    def test_failed_cycles_have_elapsed_time_but_no_completion_time(self):
        metrics, tracker = EpisodeMetrics(), ParticleTransferTracker()
        metrics.observe(reply(tracker))
        metrics.observe(reply(tracker), action_step=True)
        status = dict(status='completed', success=False, min_particles=2)
        metrics.finalize(status, 2)
        self.assertIsNone(status['cycle_time_seconds'])
        self.assertEqual(status['simulated_rollout_time_seconds'], .1)
        self.assertEqual(status['fps'], .5)
        self.assertIsNone(status['policy_latency_ms_mean'])  # Expert replay.

    def test_failure_classifications(self):
        baseline = dict(status='completed', success=False, min_particles=2,
                        payload_particles=0, retained_particles=0, spillage_particles=0)
        cases = [({}, 'no_payload'), ({'retained_particles': 3}, 'no_payload'),
                 ({'spillage_particles': 3}, 'lost_payload'),
                 ({'payload_particles': 1}, 'insufficient_payload'),
                 ({'payload_particles': 2}, 'unstable_deposit'),
                 ({'success': True}, 'bucket_not_open'),
                 ({'task_completed_success_and_bucket_open': True}, 'none'),
                 ({'status': 'error'}, 'infrastructure_error'),
                 ({'status': 'validation_error'}, 'validation_error'),
                 ({'payload_particles': None}, 'unclassified')]
        for updates, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(failure_mode({**baseline, **updates}), expected)


class SimulatorParticleIndicesTests(unittest.TestCase):
    def test_source_ids_match_bucket_geometry_and_truck_excludes_bucket(self):
        class Identity:
            def Transform(self, value):
                return value

            def GetInverse(self):
                return self

        points = np.array([[.2, .2, .2], [.4, .4, .4], [5, 5, 5], [6, 6, 6], [7, 7, 7]])
        api = {'particle_positions_fn': lambda: points}
        namespace = dict(
            np=np, time=SimpleNamespace(perf_counter=lambda: 0), sand_api=api,
            builtins=SimpleNamespace(_SAND_SITE=api),
            stage=SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: True)),
            UsdGeom=SimpleNamespace(Xformable=lambda prim: SimpleNamespace(
                ComputeLocalToWorldTransform=lambda time: Identity())),
            Usd=SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: None)),
            Gf=SimpleNamespace(Vec3d=lambda *values: values),
            bucket_source_tracker={'available': True, 'mask': np.array([True, False, True, True, False])},
            bucket_volume_topology={'available': True, 'source': 'mesh', 'reason': 'ok',
                                    'local_min': [0, 0, 0], 'local_max': [1, 1, 1]},
            truck_bed_counting_volume=dict(available=True, source='bed', reason='ok',
                local_from_world=Identity(), world_min=[0, 0, 0], world_max=[6, 6, 6],
                local_min=[0, 0, 0], local_max=[6, 6, 6], horizontal_margin=0, lower_margin=0, upper_extra=0),
            points_in_closed_bucket_mesh=lambda values: np.all((values >= 0) & (values <= 1), axis=1),
            vla_observation_contract=vla_observation_contract,
        )
        extract_functions(BASE / 'ExcavatorVLA-smolvla/run_simulation.py',
                          {'estimate_bucket_load_particles', 'estimate_truck_bed_load_particles'}, namespace)
        bucket, truck = [], []
        bucket_metric = namespace['estimate_bucket_load_particles'](True, bucket)
        truck_metric = namespace['estimate_truck_bed_load_particles'](truck)
        self.assertEqual(bucket, [0])  # Non-source particle 1 is excluded.
        self.assertEqual(bucket_metric['count'], 1)
        self.assertEqual(truck, [2, 3])
        self.assertEqual(truck_metric['count'], 2)
        self.assertEqual(truck_metric['bucket_overlap_count'], 1)
        # The policy's observation fallback may count non-source material. The
        # evaluation must still track only IDs in the strict initial-pile mask.
        points = np.array([[.2, .2, .2]] * 40 + [[5, 5, 5], [6, 6, 6]])
        namespace['bucket_source_tracker']['mask'] = np.array([True] + [False] * 39 + [True, True])
        bucket, truck = [], []
        bucket_metric = namespace['estimate_bucket_load_particles'](True, bucket)
        namespace['estimate_truck_bed_load_particles'](truck)
        self.assertEqual(bucket_metric['count'], 40)
        self.assertEqual(bucket_metric['source_tracking'], 'bucket_region_fallback')
        self.assertEqual(bucket, [0])
        self.assertEqual(truck, [40, 41])


class RolloutIntegrationTests(unittest.TestCase):
    def run_rollout(self, directory, *, video=True, steps=8, replay=False, fail=False, video_error=False, episodes=1):
        clock = SimpleNamespace(now=0.)
        clock.monotonic = clock.perf_counter = lambda: clock.now

        def advance(seconds):
            clock.now += seconds

        class Chunk:
            shape = (1, 3, 4)

            def __getitem__(self, index):
                return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros((1, 4))))

        policy = Mock()

        def infer(batch):
            advance(.4)
            return Chunk()

        policy.predict_action_chunk.side_effect = infer
        session = Mock(log_path=Path(directory) / 'simulator.log')
        sequences = []

        def prepare(seed):
            tracker = ParticleTransferTracker()
            sequence = [reply(tracker, truck=[9]), reply(tracker, [0, 1, 2, 3], [9])]
            sequence.extend(reply(tracker, [3], [0, 1, 9]) for _ in range(7))
            sequences[:] = sequence
            return {}

        session.prepare.side_effect = prepare

        def exchange(sock, command):
            if command.get('type') == 'scene_context':
                return {'scene_seed': session.prepare.call_args.args[0], 'simulator_pid': 123,
                        'observation_context': {}}
            if command.get('type') == 'handshake':
                return {'type': 'handshake_ack', 'timing_mode': 'deterministic_lockstep',
                        'state_names': list(range(28)), 'training_fps': 10}
            if fail:
                raise RuntimeError('bridge broke')
            advance(.25)
            return sequences.pop(0)

        class Video:
            frame_count = 0

            def __init__(self, path, *args):
                self.path = path

            def write(self, *args):
                advance(.5)
                self.frame_count += 1

            def finish(self, status):
                advance(99)
                if video_error:
                    raise RuntimeError('encoder broke')

        namespace = dict(
            np=np, json=json, time=clock, torch=SimpleNamespace(manual_seed=lambda seed: None, inference_mode=nullcontext),
            EpisodeVideo=Video, EpisodeMetrics=EpisodeMetrics, failure_mode=failure_mode,
            episode_indices=episode_indices, atomic_json=atomic_json, exchange=exchange,
            STATE_NAMES_28D=tuple(range(28)), OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT='v4',
            build_client_contract=lambda *args: {'type': 'handshake'},
            validate_state=lambda result: np.zeros(28), observation=lambda result: {},
            deposited_count=lambda result: result['truck_bed_load_metrics']['count'],
        )
        extract_functions(BASE / 'scripts/eval_smolvla_success.py', {'update_success', 'run_episodes'}, namespace)
        args = SimpleNamespace(episodes=episodes, worker_index=0, worker_count=1, output=Path(directory),
                               gpu_label='test', seed=2, video=video, steps=steps, min_particles=2,
                               replay_reference=replay, replan_interval=3, fresh_simulator=False)
        reference = {'actions': [np.zeros(4)] * 2, 'expected_initial_state': np.zeros(28)} if replay else None
        config = {'execution_source': 'expert_replay' if replay else 'model'}
        call = lambda: namespace['run_episodes'](args, {'fps': 10}, reference, np.ones(4),
                    None if replay else policy, lambda value: value, lambda value: value, '', config, session)
        if fail or video_error:
            with self.assertRaises(SystemExit):
                call()
        else:
            call()
        return load_episode_rows(Path(directory) / 'episodes'), policy

    def test_success_persists_all_metrics_and_excludes_final_video_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, policy = self.run_rollout(directory, episodes=2)
            for row in rows:
                self.assertTrue(row['success'])
                self.assertEqual(row['policy_steps_executed'], 5)
                self.assertEqual(row['payload_particles'], 2)
                self.assertEqual(row['spillage_particles'], 1)
                self.assertEqual(row['spillage_fraction'], .25)
                self.assertAlmostEqual(row['cycle_time_seconds'], .5)
                self.assertAlmostEqual(row['cycle_time_wall_seconds'], 4.05)
                self.assertAlmostEqual(row['fps'], 5 / 4.55)
                self.assertAlmostEqual(row['policy_latency_ms_mean'], 400)
                self.assertEqual(row['policy_inference_calls'], 2)
                self.assertEqual(row['failure_mode'], 'none')
            self.assertEqual(policy.predict_action_chunk.call_count, 4)
            trace = [json.loads(line) for line in (Path(directory) / 'episodes/episode_0001/steps.jsonl').read_text().splitlines()]
            self.assertEqual(sum(row['policy_latency_ms'] is not None for row in trace), 2)
            self.assertAlmostEqual(trace[0]['bridge_latency_ms'], 250)
            write_outputs(Path(directory), rows, build_summary(rows, 2))
            summary = json.loads((Path(directory) / 'success_rate_summary.json').read_text())
            self.assertEqual(summary['metrics']['payload_particles']['mean'], 2)
            self.assertEqual(summary['failure_mode_counts'], {'none': 2})
            with (Path(directory) / 'episode_results.csv').open() as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertTrue(set(METRIC_FIELDS).issubset(csv_rows[0]))
            self.assertEqual(csv_rows[0]['spillage_particles'], '1')

    def test_step_limit_and_reference_exhaustion_are_incomplete_cycles(self):
        for replay in (False, True):
            with self.subTest(replay=replay), tempfile.TemporaryDirectory() as directory:
                rows, _ = self.run_rollout(directory, video=False, steps=2 if not replay else 8, replay=replay)
                row = rows[0]
                self.assertFalse(row['success'])
                self.assertEqual(row['failure_mode'], 'unstable_deposit')
                self.assertIsNone(row['cycle_time_seconds'])
                self.assertAlmostEqual(row['simulated_rollout_time_seconds'], .2)
                if replay:
                    self.assertEqual(row['termination_reason'], 'reference_finished')
                    self.assertIsNone(row['policy_latency_ms_mean'])

    def test_bridge_and_video_errors_are_infrastructure_failures(self):
        for fail, video_error in ((True, False), (False, True)):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                rows, _ = self.run_rollout(directory, fail=fail, video_error=video_error)
                self.assertEqual(rows[0]['status'], 'error')
                self.assertEqual(rows[0]['failure_mode'], 'infrastructure_error')
                self.assertIsNone(rows[0]['cycle_time_seconds'])


class HistoricalAndValidationTests(unittest.TestCase):
    def test_legacy_files_remain_readable_and_strict_cli_detects_validation_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / 'episodes/episode_0001'
            episode.mkdir(parents=True)
            payload = dict(schema_version='excavator_success_rate_episode_v1', status='completed',
                           success=False, early_terminated=False)
            atomic_json(episode / 'episode_status.json', payload)
            atomic_json(episode / 'rollout_validation.json', {'status': 'passed'})
            rows = load_episode_rows(root / 'episodes')
            self.assertEqual(rows[0]['failure_mode'], 'unclassified')
            summary = build_summary(rows, 1)
            self.assertEqual(summary['metrics']['spillage_particles']['count'], 0)
            self.assertEqual(summary['success_rate_over_completed_episodes'], 0)
            payload.update(payload_particles=2, min_particles=1, failure_mode='none',
                           cycle_time_seconds=.5, cycle_time_wall_seconds=2)
            atomic_json(episode / 'episode_status.json', payload)
            atomic_json(episode / 'rollout_validation.json', {'status': 'failed'})
            rows = load_episode_rows(root / 'episodes')
            self.assertEqual(rows[0]['failure_mode'], 'validation_error')
            self.assertIsNone(rows[0]['cycle_time_seconds'])
            summary = build_summary(rows, 1)
            self.assertEqual(summary['metrics']['payload_particles']['count'], 0)
            command = [sys.executable, str(BASE / 'ExcavatorVLA-smolvla/scripts/evaluation/summarize_isaacsim_success_rate.py'),
                       '--batch-dir', directory, '--expected-episodes', '1', '--strict']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Batch is incomplete', result.stderr)
            summary = json.loads((root / 'success_rate_summary.json').read_text())
            self.assertEqual(summary['failure_mode_counts'], {'validation_error': 1})


if __name__ == '__main__':
    unittest.main()
