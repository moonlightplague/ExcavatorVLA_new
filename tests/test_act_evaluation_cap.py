"""The cap must retain a fixed paired prefix, never choose episodes by outcome."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import time
import unittest

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cap_act_evaluation import CapMonitor, prefix_rows, pair_rows, validate_evaluator


class EpisodeCapTests(unittest.TestCase):
    def rows(self, count):
        return [{'episode_index': i, 'scene_seed': 1000000 + i - 1,
                 'status': 'completed', 'error': None, 'success': i % 3 == 0}
                for i in range(1, count + 1)]

    def test_uses_exact_prefix_and_does_not_include_later_successes_or_errors(self):
        rows = self.rows(52)
        rows[50].update(status='error', success=True)
        selected = prefix_rows(list(reversed(rows)), 50, True)
        self.assertEqual([r['episode_index'] for r in selected], list(range(1, 51)))
        self.assertEqual(sum(r['success'] for r in selected), 16)
        with self.assertRaises(ValueError):
            prefix_rows(rows[1:], 50, True)
        with self.assertRaises(ValueError):
            prefix_rows([*rows, rows[0]], 50)

    def test_rejects_errors_in_scope_and_unpaired_seeds(self):
        rows = self.rows(50)
        other = self.rows(50)
        other[-1]['scene_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'unpaired'):
            pair_rows(rows, other, 50)
        rows[0]['status'] = 'error'
        with self.assertRaises(ValueError):
            prefix_rows(rows, 50, True)

    def test_process_validation_rejects_other_jobs_before_any_signal(self):
        command = [sys.executable, '-c', 'import signal; signal.pause()', '--output', '/tmp/act-cap-test-output']
        process = subprocess.Popen(command)
        try:
            recorded = {'command': command, 'output': '/tmp/act-cap-test-output'}
            validate_evaluator(psutil.Process(process.pid), recorded)
            with self.assertRaises(RuntimeError):
                validate_evaluator(psutil.Process(process.pid), {**recorded, 'output': '/tmp/other-job'})
            self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_controller_takeover_keeps_both_evaluation_children_running(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            script = out / 'frozen_sources/scripts/run_act_optimization.py'
            script.parent.mkdir(parents=True)
            script.write_text('''import json, os, signal, subprocess, sys
from pathlib import Path
out=Path(sys.argv[1])
tasks=[]
for name in ('heldout_baseline','heldout_temporal'):
    folder=out/name
    folder.mkdir()
    (folder/'config.json').write_text(json.dumps({'evaluation_id':name}))
    (folder/'provenance.json').write_text('{}')
    command=[sys.executable,'-c','import signal; signal.pause()','--output',str(folder)]
    child=subprocess.Popen(command,start_new_session=True)
    tasks.append({'name':name,'kind':'evaluation','pid':child.pid,'command':command,
                  'output':str(folder),'status':'running','episodes':100})
(out/'status.json').write_text(json.dumps({'status':'running','pid':os.getpid(),'tasks':tasks}))
signal.pause()
''')
            (out / 'heldout_protocol.json').write_text('{"episodes":100}')
            (out / 'results.json').write_text('{}')
            (out / 'REPORT.md').write_text('Original report')
            controller = subprocess.Popen([sys.executable, str(script), str(out)])
            monitor = None
            children = []
            try:
                deadline = time.monotonic() + 5
                while not (out / 'status.json').exists():
                    if time.monotonic() > deadline:
                        self.fail('Fixture controller did not start')
                    time.sleep(.01)
                children = psutil.Process(controller.pid).children()
                monitor = CapMonitor(out, 50)
                monitor.takeover()
                controller.wait(timeout=5)
                self.assertTrue(all(child.is_running() for child in children))
                state = json.loads((out / 'status.json').read_text())
                self.assertEqual(state['final_requested_episodes'], 50)
                self.assertTrue(all(task['episodes'] == 50 for task in state['tasks']))
                self.assertTrue((out / 'before_episode_cap.json').is_file())
            finally:
                if controller.poll() is None:
                    controller.kill()
                    controller.wait(timeout=5)
                for child in children:
                    if child.is_running():
                        child.kill()
                if monitor is not None:
                    monitor.guard.close()
                    if hasattr(monitor, 'controller_lock'):
                        monitor.controller_lock.close()


if __name__ == '__main__':
    unittest.main()
