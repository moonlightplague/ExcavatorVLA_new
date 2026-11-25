"""Direct evaluator entrypoints must find local policy code without PYTHONPATH."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EvaluatorEntrypointTests(unittest.TestCase):
    def test_local_policy_contract_import_from_unrelated_working_directory(self):
        script = ROOT / 'scripts/eval_smolvla_success.py'
        code = ('import sys,runpy; '
                'from pathlib import Path; '
                'script=Path(sys.argv[1]); '
                'sys.path.insert(0,str(script.parent)); '
                'runpy.run_path(str(script),run_name="entrypoint_probe"); '
                'from excavator_vlm.data.action_limits import action_contract; '
                'from excavator_vlm.data.control_repair import require_control_repair_gate; '
                'print("local policy contracts available")')
        env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
        env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', '-c', code, str(script)],
                                    cwd=directory, env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('local policy contracts available', result.stdout)


if __name__ == '__main__':
    unittest.main()
