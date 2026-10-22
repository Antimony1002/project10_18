from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    repo_id: str = "local/towel_folding"
    robot_type: str = "i2rt_yam_bimanual"
    fps: int = 30
    task_override: str = "Fold the towel in half."
    required_arm_sides: tuple[str, ...] = ("left", "right")
    required_cameras: tuple[str, ...] = ("head", "left", "right")
    reference_camera: str = "head"
    action_source: str = "leader"
    action_time_offset_ms: float = 0.0
    min_episode_seconds: float = 3.0
    min_rate_ratio: float = 0.8
    max_camera_skew_ms: float = 25.0
    max_joint_skew_ms: float = 25.0
    max_gripper_skew_ms: float = 60.0
    expected_image_height: int = 480
    expected_image_width: int = 640
    shuffle_seed: int = 42

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.action_source != "leader":
            raise ValueError("Only action_source='leader' is currently supported")
        if self.reference_camera not in self.required_cameras:
            raise ValueError("reference_camera must be one of required_cameras")
        if not 0 < self.min_rate_ratio <= 1:
            raise ValueError("min_rate_ratio must be in (0, 1]")


def load_project_config(path: str | Path) -> ProjectConfig:
    payload = json.loads(Path(path).read_text())
    valid_fields = {f.name for f in fields(ProjectConfig)}
    unknown = sorted(set(payload) - valid_fields)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")
    for key in ("required_arm_sides", "required_cameras"):
        if key in payload:
            payload[key] = tuple(payload[key])
    return ProjectConfig(**payload)
