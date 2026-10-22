from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_NAMES = tuple(
    [f"left_joint_{index}" for index in range(6)]
    + ["left_gripper"]
    + [f"right_joint_{index}" for index in range(6)]
    + ["right_gripper"]
)


@dataclass(frozen=True)
class BimanualCommand:
    left_joints_rad: np.ndarray
    left_gripper_opening: float
    right_joints_rad: np.ndarray
    right_gripper_opening: float


def unpack_action(action: np.ndarray) -> BimanualCommand:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (14,):
        raise ValueError(f"Expected a 14-D action, got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Action contains NaN or Inf")
    if not 0.0 <= float(action[6]) <= 1.0 or not 0.0 <= float(action[13]) <= 1.0:
        raise ValueError("Gripper outputs must be normalized to [0, 1]")
    return BimanualCommand(
        left_joints_rad=action[:6].copy(),
        left_gripper_opening=float(action[6]),
        right_joints_rad=action[7:13].copy(),
        right_gripper_opening=float(action[13]),
    )
