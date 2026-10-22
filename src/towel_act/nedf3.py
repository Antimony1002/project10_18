from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ProjectConfig


CAMERA_ALIASES = {
    "left": ("left", "left_wrist", "wrist_left"),
    "right": ("right", "right_wrist", "wrist_right"),
    "head": ("head", "top", "overhead"),
}


@dataclass
class Issue:
    severity: str
    message: str


@dataclass
class EpisodeInspection:
    episode: str
    path: str
    valid: bool
    duration_s: float
    task: str
    data_format: str
    contract_version: str
    active_arm_sides: list[str]
    synthetic_black_cameras: list[str]
    streams: dict[str, dict[str, Any]] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _camera_candidates(metadata: dict[str, Any], camera: str) -> list[str]:
    candidates = list(CAMERA_ALIASES[camera])
    for camera_id, description in metadata.get("camera_info", {}).items():
        normalized = str(description).lower()
        if any(alias in normalized for alias in CAMERA_ALIASES[camera]):
            candidates.append(camera_id)
    return list(dict.fromkeys(candidates))


def find_camera_export(episode_dir: Path, metadata: dict[str, Any], camera: str) -> tuple[Path, Path]:
    video_dir = episode_dir / "exports" / "videos"
    for stem in _camera_candidates(metadata, camera):
        video = video_dir / f"{stem}.mp4"
        timestamps = video_dir / f"{stem}_timestamps_ns.npy"
        if video.is_file() and timestamps.is_file():
            return video, timestamps
    raise FileNotFoundError(f"No video/timestamp export found for camera '{camera}'")


def load_npz_stream(path: Path, value_key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if "timestamps_ns" not in archive or value_key not in archive:
            raise ValueError(f"{path}: expected timestamps_ns and {value_key}")
        timestamps = np.asarray(archive["timestamps_ns"], dtype=np.int64)
        values = np.asarray(archive[value_key])
    return timestamps, values


def _check_numeric_stream(
    report: EpisodeInspection,
    label: str,
    path: Path,
    value_key: str,
    expected_hz: float,
    expected_dims: int,
    config: ProjectConfig,
    started_at_ns: int,
    ended_at_ns: int,
) -> None:
    try:
        timestamps, values = load_npz_stream(path, value_key)
        sample_count = int(timestamps.size)
        report.streams[label] = {
            "path": str(path),
            "samples": sample_count,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "start_ns": int(timestamps[0]) if sample_count else None,
            "end_ns": int(timestamps[-1]) if sample_count else None,
        }
        if values.shape != (sample_count, expected_dims):
            report.issues.append(Issue("error", f"{label}: expected shape (N,{expected_dims}), got {values.shape}"))
        if sample_count and (not np.isfinite(values).all()):
            report.issues.append(Issue("error", f"{label}: contains NaN or Inf"))
        if sample_count > 1 and np.any(np.diff(timestamps) <= 0):
            report.issues.append(Issue("error", f"{label}: timestamps are not strictly increasing"))
        minimum_samples = int(report.duration_s * expected_hz * config.min_rate_ratio)
        if sample_count < minimum_samples:
            report.issues.append(
                Issue("error", f"{label}: {sample_count} samples, expected at least {minimum_samples}")
            )
        slack_ns = int(1e9)
        if sample_count and (timestamps[-1] < started_at_ns - slack_ns or timestamps[0] > ended_at_ns + slack_ns):
            report.issues.append(Issue("error", f"{label}: timestamps do not overlap the episode interval"))
    except Exception as exc:
        report.issues.append(Issue("error", f"{label}: {exc}"))


def _check_camera(
    report: EpisodeInspection,
    episode_dir: Path,
    metadata: dict[str, Any],
    camera: str,
    config: ProjectConfig,
    started_at_ns: int,
    ended_at_ns: int,
) -> None:
    label = f"camera.{camera}"
    try:
        video_path, timestamps_path = find_camera_export(episode_dir, metadata, camera)
        timestamps = np.asarray(np.load(timestamps_path, allow_pickle=False), dtype=np.int64)
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        report.streams[label] = {
            "video_path": str(video_path),
            "timestamps_path": str(timestamps_path),
            "samples": int(timestamps.size),
            "video_frames": frame_count,
            "shape": [height, width, 3],
            "start_ns": int(timestamps[0]) if timestamps.size else None,
            "end_ns": int(timestamps[-1]) if timestamps.size else None,
        }
        minimum_samples = int(report.duration_s * config.fps * config.min_rate_ratio)
        if timestamps.size < minimum_samples:
            report.issues.append(
                Issue("error", f"{label}: {timestamps.size} frames, expected at least {minimum_samples}")
            )
        if frame_count != timestamps.size:
            report.issues.append(
                Issue("error", f"{label}: video has {frame_count} frames but timestamps has {timestamps.size}")
            )
        if (height, width) != (config.expected_image_height, config.expected_image_width):
            report.issues.append(Issue("error", f"{label}: unexpected image size {height}x{width}"))
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
            report.issues.append(Issue("error", f"{label}: timestamps are not strictly increasing"))
        slack_ns = int(1e9)
        if timestamps.size and (timestamps[-1] < started_at_ns - slack_ns or timestamps[0] > ended_at_ns + slack_ns):
            report.issues.append(Issue("error", f"{label}: timestamps do not overlap the episode interval"))
    except Exception as exc:
        report.issues.append(Issue("error", f"{label}: {exc}"))


def inspect_episode(episode_dir: str | Path, config: ProjectConfig) -> EpisodeInspection:
    episode_dir = Path(episode_dir).resolve()
    try:
        metadata = _read_json(episode_dir / "metadata.json")
        stats = _read_json(episode_dir / "episode_stats.json")
    except Exception as exc:
        return EpisodeInspection(
            episode=episode_dir.name,
            path=str(episode_dir),
            valid=False,
            duration_s=0.0,
            task="",
            data_format="",
            contract_version="",
            active_arm_sides=[],
            synthetic_black_cameras=[],
            issues=[Issue("error", str(exc))],
        )

    extra = metadata.get("extra", {})
    report = EpisodeInspection(
        episode=episode_dir.name,
        path=str(episode_dir),
        valid=False,
        duration_s=float(stats.get("duration_s", 0.0)),
        task=str(metadata.get("task_description") or metadata.get("task_name") or ""),
        data_format=str(metadata.get("data_format", "")),
        contract_version=str(extra.get("contract_version") or stats.get("contract_version") or ""),
        active_arm_sides=list(extra.get("active_arm_sides", [])),
        synthetic_black_cameras=list(extra.get("synthetic_black_cameras", [])),
    )
    if report.data_format != "NEDF3":
        report.issues.append(Issue("error", f"Expected NEDF3, got {report.data_format!r}"))
    if report.contract_version != "i2rt-nedf3-v2":
        report.issues.append(Issue("error", f"Unsupported contract {report.contract_version!r}"))
    if report.duration_s < config.min_episode_seconds:
        report.issues.append(Issue("error", f"Episode is only {report.duration_s:.2f}s long"))
    missing_arms = sorted(set(config.required_arm_sides) - set(report.active_arm_sides))
    if missing_arms:
        report.issues.append(Issue("error", f"Missing active arm sides: {missing_arms}"))
    synthetic_required = sorted(set(config.required_cameras) & set(report.synthetic_black_cameras))
    if synthetic_required:
        report.issues.append(Issue("error", f"Required cameras are synthetic black streams: {synthetic_required}"))

    started_at_ns = int(stats.get("started_at_ns", 0))
    ended_at_ns = int(stats.get("ended_at_ns", 0))
    for side in config.required_arm_sides:
        for role in ("leader", "follower"):
            _check_numeric_stream(
                report,
                f"joints.{role}_{side}",
                episode_dir / "exports" / "joints" / f"{role}_{side}.npz",
                "position_rad",
                90.0,
                6,
                config,
                started_at_ns,
                ended_at_ns,
            )
            _check_numeric_stream(
                report,
                f"grippers.{role}_{side}",
                episode_dir / "exports" / "grippers" / f"{role}_{side}.npz",
                "opening_normalized",
                30.0,
                1,
                config,
                started_at_ns,
                ended_at_ns,
            )
    for camera in config.required_cameras:
        _check_camera(report, episode_dir, metadata, camera, config, started_at_ns, ended_at_ns)

    report.valid = not any(issue.severity == "error" for issue in report.issues)
    return report


def discover_episodes(raw_root: str | Path) -> list[Path]:
    raw_root = Path(raw_root).resolve()
    if raw_root.name.startswith("episode_") and (raw_root / "metadata.json").is_file():
        return [raw_root]
    return sorted(
        path
        for path in raw_root.glob("episode_*")
        if path.is_dir() and not path.name.endswith("_inprogress")
    )
