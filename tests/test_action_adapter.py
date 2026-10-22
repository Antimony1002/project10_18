import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "deployment"))

from action_adapter import unpack_action


class ActionAdapterTest(unittest.TestCase):
    def test_unpack(self):
        action = np.arange(14, dtype=np.float32) / 13
        command = unpack_action(action)
        np.testing.assert_array_equal(command.left_joints_rad, action[:6])
        np.testing.assert_array_equal(command.right_joints_rad, action[7:13])
        self.assertEqual(command.left_gripper_opening, float(action[6]))
        self.assertEqual(command.right_gripper_opening, float(action[13]))

    def test_bad_gripper_rejected(self):
        action = np.zeros(14, dtype=np.float32)
        action[6] = 2.0
        with self.assertRaises(ValueError):
            unpack_action(action)


if __name__ == "__main__":
    unittest.main()
