import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from towel_act.config import ProjectConfig
from towel_act.nedf3 import inspect_episode


class Nedf3InspectionTest(unittest.TestCase):
    def make_episode(self, root: Path) -> Path:
        episode = root / "episode_0001"
        for subdir in ("exports/joints", "exports/grippers", "exports/videos"):
            (episode / subdir).mkdir(parents=True, exist_ok=True)
        start = 1_000_000_000_000
        duration_s = 3.0
        end = start + int(duration_s * 1e9)
        metadata = {
            "data_format": "NEDF3",
            "task_description": "Fold the towel in half.",
            "camera_info": {"camera_0": "right_wrist", "camera_1": "left_wrist", "camera_2": "head"},
            "extra": {
                "contract_version": "i2rt-nedf3-v2",
                "active_arm_sides": ["left", "right"],
                "synthetic_black_cameras": [],
            },
        }
        stats = {
            "contract_version": "i2rt-nedf3-v2",
            "duration_s": duration_s,
            "started_at_ns": start,
            "ended_at_ns": end,
        }
        (episode / "metadata.json").write_text(json.dumps(metadata))
        (episode / "episode_stats.json").write_text(json.dumps(stats))
        joint_ts = np.linspace(start, end, 271, dtype=np.int64)
        gripper_ts = np.linspace(start, end, 91, dtype=np.int64)
        for role in ("leader", "follower"):
            for side in ("left", "right"):
                np.savez_compressed(
                    episode / "exports/joints" / f"{role}_{side}.npz",
                    timestamps_ns=joint_ts,
                    position_rad=np.zeros((joint_ts.size, 6)),
                )
                np.savez_compressed(
                    episode / "exports/grippers" / f"{role}_{side}.npz",
                    timestamps_ns=gripper_ts,
                    opening_normalized=np.full((gripper_ts.size, 1), 0.5),
                )
        camera_ts = np.linspace(start, end, 91, dtype=np.int64)
        for camera in ("head", "left", "right"):
            video_path = episode / "exports/videos" / f"{camera}.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            for _ in camera_ts:
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            np.save(episode / "exports/videos" / f"{camera}_timestamps_ns.npy", camera_ts)
        return episode

    def test_valid_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = self.make_episode(Path(tmp))
            config = ProjectConfig(expected_image_height=48, expected_image_width=64)
            report = inspect_episode(episode, config)
            self.assertTrue(report.valid, [issue.message for issue in report.issues])

    def test_missing_stream_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = self.make_episode(Path(tmp))
            (episode / "exports/joints/leader_right.npz").unlink()
            config = ProjectConfig(expected_image_height=48, expected_image_width=64)
            report = inspect_episode(episode, config)
            self.assertFalse(report.valid)


if __name__ == "__main__":
    unittest.main()
