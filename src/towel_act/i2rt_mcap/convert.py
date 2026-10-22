"""Offline I2RT NEDF3 -> LeRobot v3 ACT conversion; no WebUI/hardware imports.

Observations are follower feedback. Actions are leader positions mapped into
the follower joint convention using the explicitly supplied teleop config.
Joint units are radians; grippers are normalized openings, never metres.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .contract import (
    NEDF3_CAMERA_TOPICS,
    NEDF3_I2RT_CONTRACT_VERSION,
    NEDF3_I2RT_HAND_TOPICS,
    NEDF3_I2RT_JOINT_TOPICS,
    NEDF3_METADATA_TOPIC,
    validate_topics,
)
from .reader import iter_i2rt_records

EPISODE_NAME = re.compile(r"episode_\d+$")
CAMERA_NAMES = {"left": "left_wrist", "right": "right_wrist", "head": "head"}


@dataclass(frozen=True)
class Options:
    fps: int = 30
    arms: tuple[str, ...] = ("left", "right")
    cameras: tuple[str, ...] = ("left", "right")
    gap_policy: str = "reject"
    max_gap_ms: float = 200.0
    joint_skew_ms: float = 25.0
    gripper_skew_ms: float = 60.0
    camera_skew_ms: float = 75.0
    min_seconds: float = 3.0
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.min_seconds <= 0:
            raise ValueError("fps and min_seconds must be positive")
        if (
            not self.arms
            or len(set(self.arms)) != len(self.arms)
            or not set(self.arms) <= {"left", "right"}
        ):
            raise ValueError("arms must contain unique left/right sides")
        if (
            not self.cameras
            or len(set(self.cameras)) != len(self.cameras)
            or not set(self.cameras) <= set(CAMERA_NAMES)
        ):
            raise ValueError("cameras must contain unique left/right/head roles")
        if self.gap_policy not in {"reject", "split"}:
            raise ValueError("gap_policy must be reject or split")
        if any(
            not np.isfinite(v) or v <= 0
            for v in (
                self.max_gap_ms,
                self.joint_skew_ms,
                self.gripper_skew_ms,
                self.camera_skew_ms,
                self.min_seconds,
            )
        ):
            raise ValueError("timing limits must be finite and positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")


@dataclass(frozen=True)
class Source:
    path: Path
    collector: str
    task: str | None = None
    start_s: float = 0.0
    end_s: float | None = None

    def __post_init__(self) -> None:
        if self.collector not in ("A", "B", "C", "D"):
            raise ValueError("collector must be A, B, C or D")
        if not np.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError("start_s must be finite and nonnegative")
        if self.end_s is not None and (
            not np.isfinite(self.end_s) or self.end_s <= self.start_s
        ):
            raise ValueError("end_s must be greater than start_s")


@dataclass
class Series:
    times: np.ndarray
    values: np.ndarray


@dataclass
class Episode:
    source: Source
    task: str
    streams: dict[str, Series]
    camera_files: dict[str, Path]
    recorded_start_ns: int


@dataclass
class Segment:
    times: np.ndarray
    state: np.ndarray
    action: np.ndarray
    camera_indices: dict[str, np.ndarray]


def axis_names(arms: tuple[str, ...]) -> list[str]:
    return [
        name
        for side in arms
        for name in (
            *(f"{side}_joint_{i}" for i in range(6)),
            f"{side}_gripper",
        )
    ]


def action_directions(path: Path, arms: tuple[str, ...]) -> tuple[np.ndarray, str]:
    """Read signs from a reviewed recording-time configuration, never hardware."""
    content = path.read_bytes()
    config = yaml.safe_load(content)
    parts = []
    for side in arms:
        signs = np.asarray(
            config["arms"][side]["follower"].get("joint_directions", [1] * 7),
            dtype=np.float32,
        )
        if signs.shape != (7,) or not np.isin(signs, [-1, 1]).all() or signs[6] != 1:
            raise ValueError(
                f"{side}: expected six +/-1 joint signs and +1 normalized gripper sign"
            )
        parts.append(signs)
    return np.concatenate(parts), hashlib.sha256(content).hexdigest()


def discover_sources(
    manifest_path: Path, collectors: set[str] | None = None
) -> tuple[list[Source], list[str]]:
    """Expand source roots; paths in a manifest are relative to that manifest."""
    manifest_path = manifest_path.resolve()
    config = yaml.safe_load(manifest_path.read_text())
    if (
        not isinstance(config, dict)
        or set(config) - {"version", "sources"}
        or config.get("version") != 1
    ):
        raise ValueError("manifest must contain version: 1 and sources")
    found: dict[Path, Source] = {}
    excluded = []
    for entry in config["sources"]:
        if set(entry) - {"path", "collector", "task", "exclude", "start_s", "end_s"}:
            raise ValueError(f"unsupported source fields: {entry}")
        root = (manifest_path.parent / entry["path"]).resolve()
        source = Source(
            root,
            entry["collector"],
            entry.get("task"),
            float(entry.get("start_s", 0)),
            entry.get("end_s"),
        )
        if collectors is not None and source.collector not in collectors:
            continue
        if not root.is_dir():
            raise FileNotFoundError(root)
        candidates = (
            [root]
            if EPISODE_NAME.fullmatch(root.name)
            else sorted(root.rglob("episode_*"))
        )
        blocked = set(entry.get("exclude", []))
        if any(
            not isinstance(name, str) or not EPISODE_NAME.fullmatch(name)
            for name in blocked
        ):
            raise ValueError("exclude entries must be exact episode names")
        for path in candidates:
            if (
                not path.is_dir()
                or not EPISODE_NAME.fullmatch(path.name)
                or "quarantine" in path.parts
            ):
                continue
            if path.name in blocked:
                excluded.append(str(path))
                continue
            path = path.resolve()
            if path in found:
                raise ValueError(f"duplicate source episode: {path}")
            found[path] = Source(
                path, source.collector, source.task, source.start_s, source.end_s
            )
    if not found:
        raise ValueError("no completed, selected episodes remain")
    return list(found.values()), excluded


def required_streams(options: Options) -> list[str]:
    return [
        topic
        for role in ("follower", "leader")
        for side in options.arms
        for topic in (
            NEDF3_I2RT_JOINT_TOPICS[role][side],
            NEDF3_I2RT_HAND_TOPICS[role][side],
        )
    ] + [NEDF3_CAMERA_TOPICS[role] for role in options.cameras]


def read_episode(source: Source, cache: Path, options: Options) -> Episode:
    """Spool compressed JPEGs to disk; RAM holds only timestamps and lowdim data."""
    root = source.path
    if not EPISODE_NAME.fullmatch(root.name) or "quarantine" in root.parts:
        raise ValueError(f"not a finalized episode: {root}")
    metadata = json.loads((root / "metadata.json").read_text())
    stats = json.loads((root / "episode_stats.json").read_text())
    if (
        metadata.get("data_format") != "NEDF3"
        or metadata.get("extra", {}).get("contract_version")
        != NEDF3_I2RT_CONTRACT_VERSION
    ):
        raise ValueError(f"{root}: requires I2RT NEDF3 v2 (RGB JPEG contract)")
    validate_topics(metadata["record_topic_list"])
    if stats.get("writer_error") or stats.get("dropped_batches") != 0:
        raise ValueError(f"{root}: writer errors or dropped batches")
    synthetic = set(metadata.get("extra", {}).get("synthetic_black_cameras", []))
    if synthetic.intersection(options.cameras):
        raise ValueError(f"{root}: requested camera is a synthetic black placeholder")
    task = source.task or metadata.get("task_description") or metadata.get("task_name")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"{root}: missing task description")
    needed = set(required_streams(options))
    values: dict[str, list] = {key: [] for key in needed}
    times: dict[str, list] = {key: [] for key in needed}
    camera_files = {role: cache / f"{role}.jpeg.bin" for role in options.cameras}
    counts: Counter[str] = Counter()
    with ExitStack() as stack:
        writers = {
            role: stack.enter_context(path.open("wb"))
            for role, path in camera_files.items()
        }
        for record in iter_i2rt_records(root, validate_crcs=True):
            counts[record.topic] += 1
            if record.topic not in needed:
                continue
            times[record.topic].append(record.timestamp_ns)
            if record.kind == "camera":
                if (record.image_width, record.image_height) != (
                    options.width,
                    options.height,
                ):
                    raise ValueError(
                        f"{root}/{record.role}: unexpected image dimensions"
                    )
                writer = writers[record.role]
                values[record.topic].append((writer.tell(), len(record.image)))
                writer.write(record.image)
            else:
                data = np.asarray(record.values)
                dimension = 6 if record.kind == "joints" else 1
                if data.shape != (dimension,) or not np.isfinite(data).all():
                    raise ValueError(f"{root}/{record.topic}: invalid vector")
                if (
                    record.kind == "gripper"
                    and not np.logical_and(data >= 0, data <= 1).all()
                ):
                    raise ValueError(
                        f"{root}/{record.topic}: gripper is not normalized [0,1]"
                    )
                values[record.topic].append(record.values)
    expected = {
        key: value
        for key, value in stats["topic_counts"].items()
        if key != NEDF3_METADATA_TOPIC
    }
    if dict(counts) != expected:
        raise ValueError(f"{root}: MCAP messages do not match episode_stats.json")
    streams = {}
    for key in needed:
        ts = np.asarray(times[key], dtype=np.int64)
        if len(ts) < 2 or np.any(np.diff(ts) <= 0):
            raise ValueError(f"{root}/{key}: missing or nonincreasing timestamps")
        streams[key] = Series(ts, np.asarray(values[key]))
    return Episode(
        source, task.strip(), streams, camera_files, int(stats["started_at_ns"])
    )


def nearest_indices(times: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(times, query), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(
        np.abs(times[right] - query) < np.abs(times[left] - query), right, left
    )


def align_episode(
    episode: Episode, options: Options, directions: np.ndarray
) -> tuple[list[Segment], list[dict[str, Any]]]:
    """Uniform 30 Hz timeline with bounded interpolation and explicit gap cuts."""
    streams = episode.streams
    start = max(
        max(s.times[0] for s in streams.values()),
        episode.recorded_start_ns + round(episode.source.start_s * 1e9),
    )
    end = min(s.times[-1] for s in streams.values())
    if episode.source.end_s is not None:
        end = min(end, episode.recorded_start_ns + round(episode.source.end_s * 1e9))
    if end <= start:
        raise ValueError(f"{episode.source.path}: streams have no shared time range")
    count = int((end - start) * options.fps // 1_000_000_000) + 1
    query = start + np.rint(np.arange(count) * (1e9 / options.fps)).astype(np.int64)
    valid = np.ones(count, dtype=bool)
    aligned = {}
    camera_indices = {}
    gaps = []
    camera_topics = {NEDF3_CAMERA_TOPICS[role]: role for role in options.cameras}
    for topic, series in streams.items():
        ts = series.times
        for i in np.flatnonzero(np.diff(ts) > options.max_gap_ms * 1e6):
            if ts[i + 1] > start and ts[i] < end:
                gaps.append(
                    {"topic": topic, "start_ns": int(ts[i]), "end_ns": int(ts[i + 1])}
                )
                valid &= ~((query > ts[i]) & (query < ts[i + 1]))
        nearest = nearest_indices(ts, query)
        if topic in camera_topics:
            camera_indices[camera_topics[topic]] = nearest
            tolerance = options.camera_skew_ms
        else:
            tolerance = (
                options.gripper_skew_ms
                if topic.endswith("hand_state")
                else options.joint_skew_ms
            )
            # Relative timestamps avoid losing precision in epoch nanoseconds.
            x, q = (ts - ts[0]).astype(np.float64), (query - ts[0]).astype(np.float64)
            aligned[topic] = np.column_stack(
                [
                    np.interp(q, x, series.values[:, j])
                    for j in range(series.values.shape[1])
                ]
            )
        valid &= np.abs(ts[nearest] - query) <= tolerance * 1e6
    if options.gap_policy == "reject" and (gaps or not valid.all()):
        raise ValueError(
            f"{episode.source.path}: discontinuity/alignment failure ({len(gaps)} stream gaps, {int((~valid).sum())} invalid frames); review the episode or explicitly use --gap-policy split"
        )

    def vector(role):
        return np.concatenate(
            [
                aligned[topic]
                for side in options.arms
                for topic in (
                    NEDF3_I2RT_JOINT_TOPICS[role][side],
                    NEDF3_I2RT_HAND_TOPICS[role][side],
                )
            ],
            axis=1,
        ).astype(np.float32)

    state, action = vector("follower"), vector("leader") * directions
    indices = np.flatnonzero(valid)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    minimum = max(2, int(np.ceil(options.min_seconds * options.fps)))
    segments = [
        Segment(
            query[group],
            state[group],
            action[group],
            {role: indexes[group] for role, indexes in camera_indices.items()},
        )
        for group in groups
        if len(group) >= minimum
    ]
    if not segments:
        raise ValueError(
            f"{episode.source.path}: no continuous segment meets minimum duration"
        )
    return segments, gaps


def features(options: Options) -> dict[str, dict[str, Any]]:
    axes = axis_names(options.arms)
    return {
        "observation.state": {"dtype": "float32", "shape": (len(axes),), "names": axes},
        "action": {"dtype": "float32", "shape": (len(axes),), "names": axes},
        **{
            f"observation.images.{CAMERA_NAMES[role]}": {
                "dtype": "video",
                "shape": (options.height, options.width, 3),
                "names": ["height", "width", "channels"],
            }
            for role in options.cameras
        },
    }


def write_segment(
    dataset: Any, episode: Episode, segment: Segment, options: Options
) -> None:
    import cv2

    with ExitStack() as stack:
        readers = {
            role: stack.enter_context(path.open("rb"))
            for role, path in episode.camera_files.items()
        }
        cached = {}
        for i in range(len(segment.times)):
            frame = {
                "observation.state": segment.state[i],
                "action": segment.action[i],
                "task": episode.task,
            }
            for role, indexes in segment.camera_indices.items():
                index = int(indexes[i])
                if role not in cached or cached[role][0] != index:
                    offset, size = episode.streams[NEDF3_CAMERA_TOPICS[role]].values[
                        index
                    ]
                    reader = readers[role]
                    reader.seek(int(offset))
                    image = cv2.imdecode(
                        np.frombuffer(reader.read(int(size)), dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if image is None or image.shape != (
                        options.height,
                        options.width,
                        3,
                    ):
                        raise ValueError(
                            f"{episode.source.path}/{role}: invalid JPEG payload"
                        )
                    cached[role] = (index, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                frame[f"observation.images.{CAMERA_NAMES[role]}"] = cached[role][1]
            dataset.add_frame(frame)
        dataset.save_episode()


def convert(
    sources: list[Source],
    output: Path,
    repo_id: str,
    options: Options,
    teleop_config: Path,
    excluded: list[str] | None = None,
) -> dict[str, Any]:
    from importlib.metadata import version

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if version("lerobot") != "0.4.4":
        raise ValueError(
            "This converter is validated with lerobot==0.4.4; use its independent converter profile"
        )
    if not sources:
        raise ValueError("no source episodes")
    output = output.resolve()
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError(f"output or unfinished build already exists: {output}")
    if any(
        output.is_relative_to(s.path.resolve())
        or staging.is_relative_to(s.path.resolve())
        for s in sources
    ):
        raise ValueError("output must be outside the raw episode directories")
    directions, config_sha = action_directions(teleop_config, options.arms)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=staging,
        fps=options.fps,
        features=features(options),
        robot_type="i2rt_yam",
        use_videos=True,
        vcodec="h264",
        image_writer_threads=2,
        encoder_threads=2,
    )
    provenance = []
    try:
        for source in sources:
            with tempfile.TemporaryDirectory(prefix="i2rt-jpeg-") as tmp:
                episode = read_episode(source, Path(tmp), options)
                segments, gaps = align_episode(episode, options, directions)
                for segment_index, segment in enumerate(segments):
                    index = len(provenance)
                    write_segment(dataset, episode, segment, options)
                    timings = (
                        staging
                        / "meta"
                        / "source_timestamps"
                        / f"episode_{index:06d}.npz"
                    )
                    timings.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(timings, timestamps_ns=segment.times)
                    provenance.append(
                        {
                            "episode_index": index,
                            "collector": source.collector,
                            "source_episode": str(source.path.resolve()),
                            "source_segment": segment_index,
                            "frames": len(segment.times),
                            "task": episode.task,
                            "source_start_ns": int(segment.times[0]),
                            "source_end_ns": int(segment.times[-1]),
                            "source_gaps": gaps,
                            "source_metadata_sha256": hashlib.sha256(
                                (source.path / "metadata.json").read_bytes()
                            ).hexdigest(),
                        }
                    )
                    print(
                        f"{source.collector}/{source.path.name} segment {segment_index} -> episode {index}: {len(segment.times)} frames",
                        flush=True,
                    )
        dataset.finalize()
        dataset.stop_image_writer()
        report = {
            "schema_version": 1,
            "lerobot_version": "0.4.4",
            "dataset_format": "v3.0",
            "repo_id": repo_id,
            "fps": options.fps,
            "state": "follower joint radians + normalized gripper, in selected arm order",
            "action": "leader positions multiplied by recording-time follower joint_directions; absolute follower targets",
            "axes": axis_names(options.arms),
            "action_directions": directions.tolist(),
            "teleop_config_sha256": config_sha,
            "gap_policy": options.gap_policy,
            "options": asdict(options),
            "excluded_sources": excluded or [],
            "episodes": provenance,
            "total_frames": sum(row["frames"] for row in provenance),
        }
        (staging / "meta" / "i2rt_sources.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        staging.rename(output)
        return report
    except BaseException:
        # An incomplete .building directory is intentionally never a dataset.
        try:
            dataset.finalize()
        except Exception:
            pass
        try:
            dataset.stop_image_writer()
        except Exception:
            pass
        raise
