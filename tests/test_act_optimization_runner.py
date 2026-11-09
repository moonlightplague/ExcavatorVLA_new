"""Check controller completion persistence and cleanup of its own subprocesses."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import psutil

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'scripts'))
from run_act_optimization import Runner


class ControllerProcessTests(unittest.TestCase):
    def runner(self, directory):
        runner = object.__new__(Runner)
        runner.out = runner.frozen = Path(directory)
        runner.state = {'status': 'prepared', 'phase': 'prepared', 'train_steps': 5000, 'tasks': []}
        runner.children = []
        runner.verify = lambda: None
        return runner

    def task(self, name, directory, code):
        return {'name': name, 'kind': 'offline', 'output': str(Path(directory) / name),
                'command': [sys.executable, '-c', code]}

    def test_completed_group_persists_success_and_can_be_read_after_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            task = self.task('finished', directory, 'pass')
            runner.execute_group([task], 'offline_validation')
            saved = json.loads((Path(directory) / 'status.json').read_text())
            self.assertEqual(saved['tasks'][0]['status'], 'completed')
            self.assertEqual(saved['tasks'][0]['exit_code'], 0)
            self.assertTrue((Path(directory) / 'REPORT.md').is_file())

    def test_failed_group_terminates_its_stalled_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            stalled = self.task('stalled', directory, 'import signal; signal.pause()')
            failed = self.task('failed', directory, 'raise SystemExit(7)')
            with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                runner.execute_group([stalled, failed], 'offline_validation')
            self.assertFalse(psutil.pid_exists(stalled['pid']))
            self.assertEqual(stalled['status'], 'failed_or_cancelled')
            self.assertEqual(runner.children, [])


if __name__ == '__main__':
    unittest.main()
