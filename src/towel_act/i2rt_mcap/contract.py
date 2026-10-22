"""Canonical I2RT NEDF3 topics; no recorder or vendor SDK dependency."""

from __future__ import annotations

from typing import Mapping, Sequence

NEDF3_I2RT_LEGACY_BGR_JPEG_VERSION = "i2rt-nedf3-v1"
NEDF3_I2RT_CONTRACT_VERSION = "i2rt-nedf3-v2"
NEDF3_METADATA_TOPIC = "nmx/nedf/metadata"

NEDF3_CAMERA_TOPICS: Mapping[str, str] = {
    "right": "nmx/hal/camera/camera_0/rgbd",
    "left": "nmx/hal/camera/camera_1/rgbd",
    "head": "nmx/hal/camera/camera_2/rgbd",
}
NEDF3_I2RT_JOINT_TOPICS: Mapping[str, Mapping[str, str]] = {
    "leader": {
        "right": "nmx/hal/i2rt/leader/right/joint",
        "left": "nmx/hal/i2rt/leader/left/joint",
    },
    "follower": {
        "right": "nmx/hal/i2rt/follower/right/joint",
        "left": "nmx/hal/i2rt/follower/left/joint",
    },
}
NEDF3_I2RT_HAND_TOPICS: Mapping[str, Mapping[str, str]] = {
    "leader": {
        "right": "nmx/hal/i2rt/leader/right/hand_state",
        "left": "nmx/hal/i2rt/leader/left/hand_state",
    },
    "follower": {
        "right": "nmx/hal/i2rt/follower/right/hand_state",
        "left": "nmx/hal/i2rt/follower/left/hand_state",
    },
}

NEDF3_RECORD_TOPICS: tuple[str, ...] = (
    NEDF3_METADATA_TOPIC,
    NEDF3_CAMERA_TOPICS["right"],
    NEDF3_I2RT_JOINT_TOPICS["leader"]["right"],
    NEDF3_I2RT_HAND_TOPICS["leader"]["right"],
    NEDF3_I2RT_JOINT_TOPICS["follower"]["right"],
    NEDF3_I2RT_HAND_TOPICS["follower"]["right"],
    NEDF3_CAMERA_TOPICS["left"],
    NEDF3_I2RT_JOINT_TOPICS["leader"]["left"],
    NEDF3_I2RT_HAND_TOPICS["leader"]["left"],
    NEDF3_I2RT_JOINT_TOPICS["follower"]["left"],
    NEDF3_I2RT_HAND_TOPICS["follower"]["left"],
    NEDF3_CAMERA_TOPICS["head"],
)
NEDF3_RECORD_TOPIC_LIST = NEDF3_RECORD_TOPICS
HAL_CAMERA_TOPICS = tuple(NEDF3_CAMERA_TOPICS.values())


def validate_topics(topics: Sequence[str]) -> tuple[str, ...]:
    """Require the exact canonical topic set while preserving declared order."""
    resolved = tuple(str(topic) for topic in topics)
    duplicates = sorted({topic for topic in resolved if resolved.count(topic) > 1})
    if duplicates:
        raise ValueError(f"NEDF3 topic list contains duplicates: {duplicates}")
    missing = [topic for topic in NEDF3_RECORD_TOPICS if topic not in resolved]
    unexpected = [topic for topic in resolved if topic not in NEDF3_RECORD_TOPICS]
    if missing or unexpected:
        raise ValueError(
            "NEDF3 I2RT topic list differs from the canonical contract: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return resolved
