"""The control audit must distinguish hidden target resets from parent boundaries."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_act_return import control_target_audit


def episode(eid, targets, actions):
    state = np.zeros((len(targets), 28), dtype=np.float32)
    # Preserve measured joints: the reset is in the recorded tracking error.
    state[:, 11] = targets
    action = np.zeros((len(targets), 4), dtype=np.float32)
    action[:, 3] = actions
    return pd.DataFrame(dict(episode_index=eid, frame_index=range(len(targets)),
        action=list(action), **{'observation.state': list(state),
        'observation.return_to_pile': [0] * len(state),
        'observation.scoop_index': [0] * len(state)}))


class ControlAuditTests(unittest.TestCase):
    def test_parent_boundary_is_never_an_action(self):
        frame = pd.concat([episode(0, [0, .1], [1, 0]),
                           episode(1, [-1, -.9], [1, 0])], ignore_index=True)
        result = control_target_audit(frame)
        self.assertEqual(result['transitions'], 2)
        self.assertEqual(result['mismatched_transitions'], 0)

    def test_zero_label_does_not_hide_tracking_target_reset(self):
        result = control_target_audit(episode(0, [.8, -.2], [0, 0]))
        self.assertEqual(result['mismatched_transitions'], 1)
        self.assertAlmostEqual(result['max_required_abs_rad_s'][3], 10, places=5)

    def test_source_gaps_are_not_interpolated(self):
        frame = episode(0, [0, .1], [1, 0])
        frame['frame_index'] = [0, 2]
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            control_target_audit(frame)


if __name__ == '__main__':
    unittest.main()
