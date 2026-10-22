import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from towel_act.sync import interpolate_stream, nearest_indices


class SyncTest(unittest.TestCase):
    def test_nearest_indices(self):
        source = np.array([0, 10, 20, 30], dtype=np.int64)
        query = np.array([1, 19, 28], dtype=np.int64)
        np.testing.assert_array_equal(nearest_indices(source, query, 3, "test"), [0, 2, 3])

    def test_linear_interpolation(self):
        source = np.array([0, 10, 20], dtype=np.int64)
        values = np.array([[0.0], [1.0], [2.0]])
        query = np.array([5, 15], dtype=np.int64)
        result = interpolate_stream(source, values, query, 5, "test")
        np.testing.assert_allclose(result[:, 0], [0.5, 1.5])

    def test_skew_rejected(self):
        with self.assertRaises(ValueError):
            nearest_indices(np.array([0, 100]), np.array([50]), 20, "test")


if __name__ == "__main__":
    unittest.main()
