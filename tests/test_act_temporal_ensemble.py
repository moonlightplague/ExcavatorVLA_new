import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from act_temporal_ensemble import TemporalActionEnsemble


class TemporalEnsembleTests(unittest.TestCase):
    def test_blends_same_timestamp_instead_of_previous_executed_action(self):
        ensemble = TemporalActionEnsemble(window=2, decay=0)
        first = np.array([[1.] * 4, [10.] * 4, [100.] * 4])
        second = np.array([[20.] * 4, [200.] * 4, [2000.] * 4])
        np.testing.assert_array_equal(ensemble.update(first), [1.] * 4)
        np.testing.assert_array_equal(ensemble.update(second), [15.] * 4)
        np.testing.assert_array_equal(ensemble.update(np.full((3, 4), 300.)), [250.] * 4)

    def test_short_chunks_expire_and_newer_predictions_weigh_more(self):
        ensemble = TemporalActionEnsemble(window=3, decay=np.log(2))
        ensemble.update(np.array([[0.] * 4, [10.] * 4]))
        np.testing.assert_allclose(ensemble.update(np.array([[20.] * 4])), [50 / 3] * 4)
        np.testing.assert_array_equal(ensemble.update(np.array([[30.] * 4])), [30.] * 4)
        np.testing.assert_array_equal(TemporalActionEnsemble().update(np.zeros((2, 4))), np.zeros(4))

    def test_rejects_invalid_actions_and_settings(self):
        for window, decay in ((0, .5), (1, -1), (2, float('nan'))):
            with self.assertRaises(ValueError):
                TemporalActionEnsemble(window, decay)
        for chunk in (np.zeros((4,)), np.zeros((2, 5)), np.full((1, 4), float('nan'))):
            with self.assertRaises(ValueError):
                TemporalActionEnsemble().update(chunk)


if __name__ == '__main__':
    unittest.main()
