"""Read MCAP using embedded protobuf schemas without nmx-msg or dm3-core."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .contract import (
    NEDF3_CAMERA_TOPICS,
    NEDF3_I2RT_HAND_TOPICS,
    NEDF3_I2RT_JOINT_TOPICS,
    NEDF3_METADATA_TOPIC,
)

JOINT_NAMES = tuple(f"joint{index}" for index in range(6))
CAMERA_TOPIC_INFO = {topic: role for role, topic in NEDF3_CAMERA_TOPICS.items()}
JOINT_TOPIC_INFO = {
    topic: (role, side)
    for role, side_topics in NEDF3_I2RT_JOINT_TOPICS.items()
    for side, topic in side_topics.items()
}
HAND_TOPIC_INFO = {
    topic: (role, side)
    for role, side_topics in NEDF3_I2RT_HAND_TOPICS.items()
    for side, topic in side_topics.items()
}


@dataclass(frozen=True)
class I2RTRecord:
    timestamp_ns: int
    topic: str
    kind: str
    role: str
    side: str | None = None
    values: tuple[float, ...] = ()
    image: bytes = b""
    image_width: int = 0
    image_height: int = 0


class _EmbeddedProtobufDecoder:
    def __init__(self) -> None:
        self._message_classes: dict[tuple[str, bytes], Any] = {}

    def decode(self, schema: Any, payload: bytes) -> Any:
        if schema.encoding != "protobuf":
            raise ValueError(
                f"Unsupported schema encoding {schema.encoding!r} for {schema.name!r}"
            )
        key = (str(schema.name), bytes(schema.data))
        message_class = self._message_classes.get(key)
        if message_class is None:
            message_class = self._build_message_class(schema.name, schema.data)
            self._message_classes[key] = message_class
        return message_class.FromString(payload)

    @staticmethod
    def _build_message_class(message_name: str, descriptor_data: bytes) -> Any:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

        descriptor_set = descriptor_pb2.FileDescriptorSet.FromString(descriptor_data)
        pool = descriptor_pool.DescriptorPool()
        pending = list(descriptor_set.file)
        while pending:
            remaining = []
            for descriptor in pending:
                try:
                    pool.Add(descriptor)
                except Exception:
                    remaining.append(descriptor)
            if len(remaining) == len(pending):
                names = ", ".join(descriptor.name for descriptor in remaining)
                raise ValueError(f"Could not resolve protobuf descriptors: {names}")
            pending = remaining

        descriptor = pool.FindMessageTypeByName(str(message_name))
        factory = message_factory.MessageFactory(pool)
        if hasattr(factory, "GetPrototype"):
            return factory.GetPrototype(descriptor)
        return message_factory.GetMessageClass(descriptor)


def episode_mcap_paths(episode_dir: Path) -> tuple[Path, ...]:
    """Resolve ordered MCAP shards from a finalized NEDF3 episode."""
    episode_dir = episode_dir.expanduser().resolve()
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"NEDF3 metadata not found: {metadata_path}")

    data_dir = episode_dir / "data"
    index_path = data_dir / "mcap_index.json"
    paths: list[Path] = []
    if index_path.is_file():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid MCAP index: {index_path}") from exc
        if not isinstance(entries, list):
            raise ValueError(f"MCAP index must contain a list: {index_path}")
        for entry in entries:
            filename = entry.get("filename") if isinstance(entry, dict) else None
            if not isinstance(filename, str) or not filename:
                raise ValueError(
                    f"MCAP index contains an invalid filename: {index_path}"
                )
            path = (data_dir / filename).resolve()
            if path.parent != data_dir.resolve():
                raise ValueError(
                    f"MCAP index filename escapes the data directory: {filename}"
                )
            paths.append(path)
    else:
        paths = sorted(data_dir.glob("*.mcap"))

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MCAP shard not found: {missing[0]}")
    if not paths:
        raise FileNotFoundError(f"No MCAP shards found under: {data_dir}")
    return tuple(paths)


def _iter_messages(
    shards: Sequence[Path], *, validate_crcs: bool = False
) -> Iterator[tuple[Any, Any, Any]]:
    from mcap.reader import make_reader

    for shard in shards:
        with shard.open("rb") as stream:
            yield from make_reader(stream, validate_crcs=validate_crcs).iter_messages()


def _joint_values(decoded: Any, topic: str) -> tuple[float, ...]:
    values = {str(joint.name): float(joint.pos) for joint in decoded.joints}
    missing = [name for name in JOINT_NAMES if name not in values]
    if missing:
        raise ValueError(f"{topic} is missing joints: {missing}")
    result = tuple(values[name] for name in JOINT_NAMES)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{topic} contains a non-finite joint value")
    return result


def _gripper_value(decoded: Any, topic: str) -> float:
    values = {
        str(joint.name): float(joint.pos)
        for joint in decoded.joints
        if str(joint.name) == "gripper"
    }
    if "gripper" not in values or not math.isfinite(values["gripper"]):
        raise ValueError(f"{topic} does not contain a finite gripper value")
    return values["gripper"]


def iter_i2rt_records(
    episode_dir: Path, *, validate_crcs: bool = False
) -> Iterator[I2RTRecord]:
    """Yield decoded camera, joint, and gripper records in MCAP order."""
    decoder = _EmbeddedProtobufDecoder()
    for schema, channel, message in _iter_messages(
        episode_mcap_paths(episode_dir), validate_crcs=validate_crcs
    ):
        topic = str(channel.topic)
        if topic == NEDF3_METADATA_TOPIC:
            continue
        if (
            topic not in CAMERA_TOPIC_INFO
            and topic not in JOINT_TOPIC_INFO
            and topic not in HAND_TOPIC_INFO
        ):
            continue
        decoded = decoder.decode(schema, message.data)
        timestamp_ns = int(message.log_time)
        if topic in CAMERA_TOPIC_INFO:
            rgb = decoded.rgb
            image = bytes(rgb.data)
            if not image:
                continue
            yield I2RTRecord(
                timestamp_ns=timestamp_ns,
                topic=topic,
                kind="camera",
                role=CAMERA_TOPIC_INFO[topic],
                image=image,
                image_width=int(rgb.cols),
                image_height=int(rgb.rows),
            )
        elif topic in JOINT_TOPIC_INFO:
            role, side = JOINT_TOPIC_INFO[topic]
            yield I2RTRecord(
                timestamp_ns=timestamp_ns,
                topic=topic,
                kind="joints",
                role=role,
                side=side,
                values=_joint_values(decoded, topic),
            )
        else:
            role, side = HAND_TOPIC_INFO[topic]
            yield I2RTRecord(
                timestamp_ns=timestamp_ns,
                topic=topic,
                kind="gripper",
                role=role,
                side=side,
                values=(_gripper_value(decoded, topic),),
            )
