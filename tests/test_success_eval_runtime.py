"""Test GPU isolation, episode assignment and process cleanup without Isaac."""
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import success_eval_runtime as runtime


class GpuAssignmentTests(unittest.TestCase):
    def resolve(self, requested=None, mask=None):
        env = {} if mask is None else {'CUDA_VISIBLE_DEVICES': mask}
        with patch.object(runtime.subprocess, 'run', return_value=SimpleNamespace(stdout='GPU-aaa\nGPU-bbb\n')):
            return runtime.resolve_gpu_ids(requested, env)

    def test_auto_uses_every_visible_gpu_in_mask_order(self):
        self.assertEqual(self.resolve(), ['GPU-aaa', 'GPU-bbb'])
        self.assertEqual(self.resolve(mask='1,0'), ['GPU-bbb', 'GPU-aaa'])
        self.assertEqual(self.resolve(mask='GPU-bbb'), ['GPU-bbb'])

    def test_explicit_selection_cannot_escape_mask(self):
        self.assertEqual(self.resolve('1', '1'), ['GPU-bbb'])
        with self.assertRaisesRegex(ValueError, 'CUDA_VISIBLE_DEVICES'):
            self.resolve('0', '1')

    def test_invalid_empty_duplicate_and_ambiguous_devices_fail(self):
        for requested, mask in [('0,0', None), ('2', None), ('GPU-', None), (None, ''), (None, '-1')]:
            with self.subTest(requested=requested, mask=mask), self.assertRaises(ValueError):
                self.resolve(requested, mask)

    def test_workers_cover_all_episodes_once_and_keep_global_seed_order(self):
        for count in (1, 2, 3, 8):
            assignments = [list(runtime.episode_indices(100, worker, count)) for worker in range(count)]
            flat = [index for shard in assignments for index in shard]
            self.assertEqual(sorted(flat), list(range(1, 101)))
            self.assertEqual(len(flat), len(set(flat)))
            self.assertEqual(sorted(2 + index - 1 for index in flat), list(range(2, 102)))

    def test_shader_caches_are_independent_and_seeded_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'isaacsim/kit/cache/nv_shadercache'
            source.mkdir(parents=True)
            (source / 'compiled-shader').write_text('warm')
            env = {'EXCAVATOR_EVAL_CACHE': str(root / 'cache'), 'EXCAVATOR_ISAAC_DIR': str(root / 'isaacsim')}
            first = runtime.prepare_shader_cache(root, 'GPU-a', env)
            second = runtime.prepare_shader_cache(root, 'GPU-b', env)
            self.assertNotEqual(first, second)
            self.assertEqual((second / 'driver/compiled-shader').read_text(), 'warm')
            (first / 'driver/compiled-shader').write_text('updated')
            runtime.prepare_shader_cache(root, 'GPU-a', env)
            self.assertEqual((first / 'driver/compiled-shader').read_text(), 'updated')


class SimulatorLifecycleTests(unittest.TestCase):
    def test_second_episode_resets_same_process_and_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Mock(pid=123)
            process.poll.return_value = None
            sock = Mock()
            exchange = Mock(return_value={'type': 'reset_episode_ack', 'scene_seed': 9})
            session = runtime.SimulatorSession(['isaac', 'simulation.py'], Path(directory),
                                               Path(directory) / 'simulator.log', False, exchange)
            with patch.object(runtime.subprocess, 'Popen', return_value=process) as launch, \
                    patch.object(runtime, 'connect_simulator', return_value=sock), \
                    patch.object(runtime, 'stop_simulator') as stop:
                first = session.prepare(2)
                second = session.prepare(9)
                self.assertFalse(first['simulator_reused'])
                self.assertTrue(second['simulator_reused'])
                self.assertEqual(first['simulator_pid'], second['simulator_pid'])
                launch.assert_called_once()
                self.assertEqual(launch.call_args.kwargs['env']['EXCAVATOR_EVAL_VIDEO'], '0')
                self.assertEqual(launch.call_args.kwargs['env']['EXCAVATOR_EVAL_METRICS'], '1')
                exchange.assert_called_once_with(sock, {'type': 'reset_episode', 'scene_seed': 9})
                session.close()
                session.close()
                stop.assert_called_once_with(process)
                sock.close.assert_called_once()

    def test_failed_startup_is_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Mock(pid=123)
            session = runtime.SimulatorSession(['isaac'], Path(directory),
                                               Path(directory) / 'simulator.log', True, Mock())
            with patch.object(runtime.subprocess, 'Popen', return_value=process), \
                    patch.object(runtime, 'connect_simulator', side_effect=TimeoutError('startup')), \
                    patch.object(runtime, 'stop_simulator') as stop:
                with self.assertRaises(TimeoutError):
                    try:
                        session.prepare(2)
                    finally:
                        session.close()
                stop.assert_called_once_with(process)
                self.assertIsNone(session.log)

    def test_wrong_reset_seed_is_rejected(self):
        session = runtime.SimulatorSession([], Path('.'), Path('unused'), False,
                                           Mock(return_value={'type': 'reset_episode_ack', 'scene_seed': 7}))
        session.process, session.sock = Mock(pid=1), Mock()
        with self.assertRaisesRegex(ValueError, 'reset acknowledgment'):
            session.prepare(8)

    def test_stuck_simulator_is_killed_and_reaped(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('isaac', 15), 0]
        with patch.object(runtime.os, 'killpg') as kill:
            runtime.stop_simulator(process)
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(123, signal.SIGTERM), (123, signal.SIGKILL)])
        self.assertEqual(process.wait.call_count, 2)


class WorkerLifecycleTests(unittest.TestCase):
    def test_parent_reports_new_metrics_from_completed_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(episodes=1, workers=1, cpu_threads=2, output=Path(directory))
            process = Mock(returncode=0)
            process.poll.return_value = 0
            row = dict(episode_index=1, status='completed', success=True, error=None,
                       payload_particles=50, spillage_particles=5, cycle_time_seconds=10,
                       failure_mode='none', policy_latency_ms_mean=20, fps=3)
            output = StringIO()
            with patch.object(runtime.subprocess, 'Popen', return_value=process), redirect_stdout(output):
                code = runtime.run_workers(args, Path('evaluate.py'), ['GPU-a'], lambda: [row])
            self.assertEqual(code, 0)
            self.assertIn('payload=50, spillage=5, cycle_s=10', output.getvalue())
            self.assertIn('failure_mode=none, policy_latency_ms=20, fps=3', output.getvalue())

    def test_worker_error_stops_other_workers_and_records_incomplete_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(episodes=4, workers=0, cpu_threads=2, output=Path(directory))
            running = Mock(returncode=-15)
            running.poll.return_value = None
            failed = Mock(returncode=1)
            failed.poll.return_value = 1
            with patch.object(runtime.subprocess, 'Popen', side_effect=[running, failed]) as launch:
                code = runtime.run_workers(args, Path('evaluate.py'), ['GPU-a', 'GPU-b'], lambda: [])
            self.assertEqual(code, 1)
            running.terminate.assert_called_once()
            failed.terminate.assert_not_called()
            for process in (running, failed):
                process.wait.assert_called_once()
            self.assertEqual(launch.call_args_list[0].kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-a')
            self.assertEqual(launch.call_args_list[1].kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-b')
            self.assertTrue((Path(directory) / 'runtime.json').is_file())


if __name__ == '__main__':
    unittest.main()
