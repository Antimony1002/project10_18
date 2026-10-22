from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ProjectConfig
from .nedf3 import discover_episodes, find_camera_export, inspect_episode, load_npz_stream
from .sync import interpolate_stream, nearest_indices


@dataclass
class PreparedEpisode:
    source_name: str
    source_path: Path
    task: str
    timestamps_ns: np.ndarray
    state: np.ndarray
    action: np.ndarray
    camera_paths: dict[str, Path]
    camera_indices: dict[str, np.ndarray]

    @property
    def frame_count(self) -> int:
        return int(self.timestamps_ns.size)


class SequentialVideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video {path}")
        self.index = -1
        self.frame_rgb: np.ndarray | None = None

    def read(self, target_index: int) -> np.ndarray:
        if target_index < self.index:
            raise ValueError(f"Video indices must be monotonic for {self.path}")
        while self.index < target_index:
            ok, frame_bgr = self.capture.read()
            if not ok:
                raise RuntimeError(f"Video {self.path} ended before frame {target_index}")
            self.index += 1
            self.frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.frame_rgb is None:
            raise RuntimeError(f"No frame decoded from {self.path}")
        return self.frame_rgb

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "SequentialVideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _load_metadata(episode_dir: Path) -> dict[str, Any]:
    return json.loads((episode_dir / "metadata.json").read_text())


def _load_arm_streams(
    episode_dir: Path, role: str, side: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joint_ts, joints = load_npz_stream(
        episode_dir / "exports" / "joints" / f"{role}_{side}.npz", "position_rad"
    )
    gripper_ts, gripper = load_npz_stream(
        episode_dir / "exports" / "grippers" / f"{role}_{side}.npz", "opening_normalized"
    )
    return joint_ts, joints, gripper_ts, gripper


def prepare_episode(episode_dir: str | Path, config: ProjectConfig) -> PreparedEpisode:
    episode_dir = Path(episode_dir).resolve()
    inspection = inspect_episode(episode_dir, config)
    if not inspection.valid:
        messages = "; ".join(issue.message for issue in inspection.issues if issue.severity == "error")
        raise ValueError(f"{episode_dir.name} failed validation: {messages}")

    metadata = _load_metadata(episode_dir)
    camera_paths: dict[str, Path] = {}
    camera_timestamps: dict[str, np.ndarray] = {}
    for camera in config.required_cameras:
        video_path, timestamp_path = find_camera_export(episode_dir, metadata, camera)
        camera_paths[camera] = video_path
        camera_timestamps[camera] = np.asarray(np.load(timestamp_path, allow_pickle=False), dtype=np.int64)

    reference = camera_timestamps[config.reference_camera]
    stream_ranges: list[tuple[int, int]] = [(int(ts[0]), int(ts[-1])) for ts in camera_timestamps.values()]
    loaded_arms: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for role in ("leader", "follower"):
        for side in config.required_arm_sides:
            streams = _load_arm_streams(episode_dir, role, side)
            loaded_arms[(role, side)] = streams
            stream_ranges.extend(
                [
                    (int(streams[0][0]), int(streams[0][-1])),
                    (int(streams[2][0]), int(streams[2][-1])),
                ]
            )

    action_offset_ns = int(config.action_time_offset_ms * 1e6)
    lower_bound = max(start for start, _ in stream_ranges)
    upper_bound = min(end for _, end in stream_ranges)
    query = reference[(reference >= lower_bound) & (reference <= upper_bound)]
    if action_offset_ns:
        action_ranges = [
            (int(loaded_arms[("leader", side)][0][0]), int(loaded_arms[("leader", side)][0][-1]))
            for side in config.required_arm_sides
        ] + [
            (int(loaded_arms[("leader", side)][2][0]), int(loaded_arms[("leader", side)][2][-1]))
            for side in config.required_arm_sides
        ]
        action_lower = max(start for start, _ in action_ranges) - action_offset_ns
        action_upper = min(end for _, end in action_ranges) - action_offset_ns
        query = query[(query >= action_lower) & (query <= action_upper)]
    if query.size < int(config.min_episode_seconds * config.fps):
        raise ValueError(f"{episode_dir.name}: too few synchronized frames ({query.size})")

    joint_skew_ns = int(config.max_joint_skew_ms * 1e6)
    gripper_skew_ns = int(config.max_gripper_skew_ms * 1e6)

    def build_vector(role: str, timestamps: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        for side in config.required_arm_sides:
            joint_ts, joints, gripper_ts, gripper = loaded_arms[(role, side)]
            parts.append(interpolate_stream(joint_ts, joints, timestamps, joint_skew_ns, f"{role}.{side}.joints"))
            parts.append(
                interpolate_stream(gripper_ts, gripper, timestamps, gripper_skew_ns, f"{role}.{side}.gripper")
            )
        return np.concatenate(parts, axis=1).astype(np.float32)

    state = build_vector("follower", query)
    action = build_vector(config.action_source, query + action_offset_ns)
    camera_indices = {
        camera: nearest_indices(
            timestamps,
            query,
            int(config.max_camera_skew_ms * 1e6),
            f"camera.{camera}",
        )
        for camera, timestamps in camera_timestamps.items()
    }
    task = config.task_override or inspection.task
    return PreparedEpisode(
        source_name=episode_dir.name,
        source_path=episode_dir,
        task=task,
        timestamps_ns=query,
        state=state,
        action=action,
        camera_paths=camera_paths,
        camera_indices=camera_indices,
    )


def _axis_names(sides: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for side in sides:
        names.extend([f"{side}_joint_{index}" for index in range(6)])
        names.append(f"{side}_gripper")
    return names


def _features(config: ProjectConfig) -> dict[str, dict[str, Any]]:
    axes = _axis_names(config.required_arm_sides)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(axes),),
            "names": {"axes": axes},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(axes),),
            "names": {"axes": axes},
        },
    }
    output_names = {"left": "left_wrist", "right": "right_wrist", "head": "head"}
    for camera in config.required_cameras:
        features[f"observation.images.{output_names[camera]}"] = {
            "dtype": "video",
            "shape": (config.expected_image_height, config.expected_image_width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _write_episode(dataset: Any, episode: PreparedEpisode, config: ProjectConfig) -> None:
    output_names = {"left": "left_wrist", "right": "right_wrist", "head": "head"}
    readers = {camera: SequentialVideoReader(path) for camera, path in episode.camera_paths.items()}
    try:
        for frame_index in range(episode.frame_count):
            frame: dict[str, Any] = {
                "observation.state": episode.state[frame_index],
                "action": episode.action[frame_index],
                "task": episode.task,
            }
            for camera in config.required_cameras:
                image = readers[camera].read(int(episode.camera_indices[camera][frame_index]))
                expected_shape = (config.expected_image_height, config.expected_image_width, 3)
                if image.shape != expected_shape:
                    raise ValueError(
                        f"{episode.source_name}/{camera}: expected image {expected_shape}, got {image.shape}"
                    )
                frame[f"observation.images.{output_names[camera]}"] = image
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=True)
    finally:
        for reader in readers.values():
            reader.close()


def convert_dataset(
    raw_root: str | Path,
    output_root: str | Path,
    config: ProjectConfig,
    skip_invalid: bool = False,
    episode_names: set[str] | None = None,
) -> dict[str, Any]:
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    staging_root = output_root.with_name(f"{output_root.name}.building")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    if staging_root.exists():
        raise FileExistsError(f"Previous staging output exists: {staging_root}")

    episodes = discover_episodes(raw_root)
    if episode_names is not None:
        episodes = [path for path in episodes if path.name in episode_names]
        missing = sorted(episode_names - {path.name for path in episodes})
        if missing:
            raise FileNotFoundError(f"Requested episodes not found: {missing}")
    if not episodes:
        raise FileNotFoundError(f"No completed episode_* directories under {raw_root}")

    inspections = [inspect_episode(path, config) for path in episodes]
    invalid = [report for report in inspections if not report.valid]
    if invalid and not skip_invalid:
        details = "\n".join(
            f"- {report.episode}: " + "; ".join(issue.message for issue in report.issues)
            for report in invalid
        )
        raise ValueError(f"Raw dataset preflight failed:\n{details}")
    valid_paths = [Path(report.path) for report in inspections if report.valid]
    if not valid_paths:
        raise ValueError("No valid episodes remain after preflight")
    random.Random(config.shuffle_seed).shuffle(valid_paths)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not importable. Run this from the LeRobot environment on the Ubuntu training host."
        ) from exc

    staging_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=config.repo_id,
        root=staging_root,
        fps=config.fps,
        robot_type=config.robot_type,
        features=_features(config),
        use_videos=True,
        image_writer_threads=max(3, len(config.required_cameras)),
    )
    converted: list[dict[str, Any]] = []
    try:
        for output_index, source_path in enumerate(valid_paths):
            prepared = prepare_episode(source_path, config)
            _write_episode(dataset, prepared, config)
            converted.append(
                {
                    "dataset_episode_index": output_index,
                    "source_episode": prepared.source_name,
                    "source_path": str(prepared.source_path),
                    "frames": prepared.frame_count,
                    "task": prepared.task,
                    "raw_start_ns": int(prepared.timestamps_ns[0]),
                    "raw_end_ns": int(prepared.timestamps_ns[-1]),
                }
            )
    finally:
        dataset.finalize()

    staging_root.rename(output_root)
    report = {
        "repo_id": config.repo_id,
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "config": asdict(config),
        "state_semantics": "follower left joints+gripper, then follower right joints+gripper",
        "action_semantics": "leader left joints+gripper, then leader right joints+gripper",
        "converted_episodes": converted,
        "skipped_invalid_episodes": [report.to_dict() for report in invalid],
        "total_frames": sum(item["frames"] for item in converted),
    }
    (output_root / "conversion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
