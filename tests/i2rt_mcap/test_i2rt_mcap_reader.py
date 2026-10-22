"""Exercise real MCAP/embedded protobuf input without robot or private SDKs."""

import json
from collections import Counter

import cv2
import numpy as np
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.writer import Writer
from towel_act.i2rt_mcap.contract import (
    NEDF3_CAMERA_TOPICS,
    NEDF3_I2RT_HAND_TOPICS,
    NEDF3_I2RT_JOINT_TOPICS,
    NEDF3_METADATA_TOPIC,
    NEDF3_RECORD_TOPICS,
)
from towel_act.i2rt_mcap.convert import Options, Source, align_episode, read_episode
from towel_act.i2rt_mcap.reader import episode_mcap_paths, iter_i2rt_records


def schemas():
    file = descriptor_pb2.FileDescriptorProto(
        name="i2rt_fixture.proto", package="fixture"
    )
    specs = {
        "Joint": [("name", 9, None, 1), ("pos", 1, None, 1)],
        "Arm": [("joints", 11, ".fixture.Joint", 3)],
        "Image": [("rows", 13, None, 1), ("cols", 13, None, 1), ("data", 12, None, 1)],
        "RGBD": [("rgb", 11, ".fixture.Image", 1)],
    }
    for name, fields in specs.items():
        message = file.message_type.add(name=name)
        for number, (field_name, kind, type_name, label) in enumerate(fields, 1):
            field = message.field.add(
                name=field_name, number=number, type=kind, label=label
            )
            if type_name:
                field.type_name = type_name
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    classes = {
        name: message_factory.GetMessageClass(
            pool.FindMessageTypeByName("fixture." + name)
        )
        for name in ("Arm", "RGBD")
    }
    return descriptor_pb2.FileDescriptorSet(file=[file]).SerializeToString(), classes


def write_episode(root):
    episode = root / "episode_0000"
    data = episode / "data"
    data.mkdir(parents=True)
    descriptor, classes = schemas()
    counts = Counter()
    start = 1_792_678_000_000_000_000
    with (data / "test.mcap").open("wb") as stream:
        writer = Writer(stream, enable_data_crcs=True)
        writer.start()
        arm_schema = writer.register_schema("fixture.Arm", "protobuf", descriptor)
        image_schema = writer.register_schema("fixture.RGBD", "protobuf", descriptor)
        channels = {
            topic: writer.register_channel(
                topic,
                "protobuf",
                image_schema if topic in NEDF3_CAMERA_TOPICS.values() else arm_schema,
            )
            for topic in NEDF3_RECORD_TOPICS
        }
        for index in range(11):
            stamp = start + index * 33_333_333
            messages = {}
            for role in ("leader", "follower"):
                for side in ("left", "right"):
                    arm = classes["Arm"]()
                    for joint in range(6):
                        arm.joints.add(
                            name=f"joint{joint}", pos=1 if role == "leader" else 0
                        )
                    hand = classes["Arm"]()
                    hand.joints.add(name="gripper", pos=0.5)
                    messages[NEDF3_I2RT_JOINT_TOPICS[role][side]] = arm
                    messages[NEDF3_I2RT_HAND_TOPICS[role][side]] = hand
            for topic in NEDF3_CAMERA_TOPICS.values():
                ok, jpeg = cv2.imencode(
                    ".jpg", np.full((24, 32, 3), 120, dtype=np.uint8)
                )
                assert ok
                image = classes["RGBD"]()
                image.rgb.rows, image.rgb.cols = 24, 32
                image.rgb.data = jpeg.tobytes()
                messages[topic] = image
            if index == 0:
                messages[NEDF3_METADATA_TOPIC] = classes["Arm"]()
            for topic, message in messages.items():
                writer.add_message(
                    channels[topic], stamp, message.SerializeToString(), stamp
                )
                counts[topic] += 1
        writer.finish()
    (data / "mcap_index.json").write_text(json.dumps([{"filename": "test.mcap"}]))
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "data_format": "NEDF3",
                "task_name": "fixture",
                "record_topic_list": list(NEDF3_RECORD_TOPICS),
                "extra": {
                    "contract_version": "i2rt-nedf3-v2",
                    "synthetic_black_cameras": ["head"],
                },
            }
        )
    )
    (episode / "episode_stats.json").write_text(
        json.dumps(
            {
                "started_at_ns": start,
                "topic_counts": dict(counts),
                "dropped_batches": 0,
                "writer_error": None,
            }
        )
    )
    return episode


def test_embedded_protobuf_crc_reader_and_bimanual_alignment(tmp_path):
    path = write_episode(tmp_path)
    records = list(iter_i2rt_records(path, validate_crcs=True))
    assert len(records) == 121
    assert {r.kind for r in records} == {"camera", "joints", "gripper"}
    cache = tmp_path / "cache"
    cache.mkdir()
    options = Options(width=32, height=24, min_seconds=0.1)
    episode = read_episode(Source(path, "A"), cache, options)
    segments, gaps = align_episode(episode, options, np.ones(14))
    assert not gaps and len(segments) == 1
    segment = segments[0]
    assert segment.state.shape[1] == segment.action.shape[1] == 14
    np.testing.assert_allclose(segment.action[:, :6], 1)
    np.testing.assert_allclose(segment.action[:, 7:13], 1)
    np.testing.assert_allclose(segment.state[:, :6], 0)
    np.testing.assert_allclose(segment.state[:, 7:13], 0)
    np.testing.assert_allclose(segment.state[:, [6, 13]], 0.5)
    assert set(segment.camera_indices) == {"left", "right"}


def test_mcap_index_cannot_escape_episode(tmp_path):
    episode = tmp_path / "episode_0000"
    (episode / "data").mkdir(parents=True)
    (episode / "metadata.json").write_text("{}")
    (episode / "data/mcap_index.json").write_text('[{"filename":"../../outside.mcap"}]')
    with pytest.raises(ValueError, match="escapes"):
        episode_mcap_paths(episode)
