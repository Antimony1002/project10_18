"""Hardware-free tests of ACT labels, joint conventions and gap-safe alignment."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from towel_act.i2rt_mcap.contract import NEDF3_CAMERA_TOPICS
from towel_act.i2rt_mcap.convert import (
    Episode,
    Options,
    Series,
    Source,
    action_directions,
    align_episode,
    discover_sources,
    features,
    required_streams,
    write_segment,
)


def episode(tmp_path, times, options):
    streams = {}
    for topic in required_streams(options):
        if "/camera/" in topic:
            values = np.zeros((len(times), 2), dtype=np.int64)
        else:
            size = 1 if topic.endswith("hand_state") else 6
            value = 0.25 if size == 1 else (2 if "/leader/" in topic else 1)
            values = np.full((len(times), size), value, dtype=np.float32)
        streams[topic] = Series(times, values)
    return Episode(
        Source(tmp_path / "episode_0002", "A"), "test", streams, {}, int(times[0])
    )


@pytest.mark.parametrize("arms", [("left",), ("left", "right")])
def test_state_is_feedback_and_action_is_mapped_leader_target(tmp_path, arms):
    options = Options(arms=arms, cameras=arms, fps=10, min_seconds=0.2)
    times = 1_792_674_000_000_000_000 + np.arange(10, dtype=np.int64) * 100_000_000
    raw = episode(tmp_path, times, options)
    signs = np.ones(7 * len(arms), dtype=np.float32)
    if "right" in arms:
        signs[12] = -1
    segments, gaps = align_episode(raw, options, signs)
    assert len(segments) == 1
    assert gaps == []
    assert segments[0].state.shape == (10, 7 * len(arms))
    np.testing.assert_array_equal(segments[0].state[:, :6], 1)
    np.testing.assert_array_equal(segments[0].action[:, :6], 2)
    np.testing.assert_array_equal(segments[0].action[:, 6], 0.25)
    if "right" in arms:
        np.testing.assert_array_equal(segments[0].action[:, 12], -2)
        np.testing.assert_array_equal(segments[0].state[:, 12], 1)
    assert features(options)["observation.state"]["shape"] == (7 * len(arms),)


def test_gap_is_rejected_or_split_without_interpolation_across_it(tmp_path):
    options = Options(arms=("left",), cameras=("left",), fps=10, min_seconds=0.2)
    offsets = np.r_[np.arange(5), np.arange(15, 20)] * 100_000_000
    raw = episode(tmp_path, 1_792_674_000_000_000_000 + offsets, options)
    with pytest.raises(ValueError, match="discontinuity"):
        align_episode(raw, options, np.ones(7))
    split = Options(
        arms=("left",), cameras=("left",), fps=10, min_seconds=0.2, gap_policy="split"
    )
    segments, gaps = align_episode(raw, split, np.ones(7))
    assert [len(s.times) for s in segments] == [5, 5]
    assert gaps
    assert segments[0].times[-1] < segments[1].times[0]
    assert all(np.max(np.diff(s.times)) == 100_000_000 for s in segments)


def test_short_alignment_dropout_is_not_silently_compressed(tmp_path):
    options = Options(arms=("left",), cameras=("left",), fps=30, min_seconds=0.1)
    times = 1_792_674_000_000_000_000 + np.arange(30, dtype=np.int64) * 33_333_333
    raw = episode(tmp_path, times, options)
    camera = raw.streams[NEDF3_CAMERA_TOPICS["left"]]
    camera.times = camera.times[::6]
    camera.values = camera.values[::6]
    with pytest.raises(ValueError, match="alignment failure"):
        align_episode(raw, options, np.ones(7))


def test_collector_manifest_excludes_exact_episodes_and_incomplete_paths(tmp_path):
    for name in (
        "episode_0000",
        "episode_0001",
        "episode_0002",
        "episode_0003_inprogress",
        "quarantine/episode_0004",
    ):
        (tmp_path / "A" / name).mkdir(parents=True)
    (tmp_path / "B" / "episode_0002").mkdir(parents=True)
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "sources": [
                    {
                        "path": "A",
                        "collector": "A",
                        "exclude": ["episode_0000", "episode_0001"],
                    },
                    {"path": "B", "collector": "B"},
                ],
            }
        )
    )
    sources, excluded = discover_sources(manifest)
    assert [(s.path.parent.name, s.path.name, s.collector) for s in sources] == [
        ("A", "episode_0002", "A"),
        ("B", "episode_0002", "B"),
    ]
    assert len(excluded) == 2
    assert len(discover_sources(manifest, {"A"})[0]) == 1


def test_direction_config_is_explicit_and_grippers_cannot_be_sign_flipped(tmp_path):
    config = tmp_path / "teleop.yaml"
    config.write_text(
        "arms:\n  right:\n    follower:\n      joint_directions: [1, 1, 1, 1, 1, -1, 1]\n"
    )
    signs, fingerprint = action_directions(config, ("right",))
    np.testing.assert_array_equal(signs, [1, 1, 1, 1, 1, -1, 1])
    assert len(fingerprint) == 64
    config.write_text(config.read_text().replace("-1, 1]", "-1, -1]"))
    with pytest.raises(ValueError, match="gripper sign"):
        action_directions(config, ("right",))


def test_jpeg_writer_preserves_rgb_and_does_not_add_a_synthetic_head(tmp_path):
    options = Options(
        arms=("left",), cameras=("left",), fps=10, min_seconds=0.2, width=8, height=8
    )
    times = np.arange(3, dtype=np.int64) * 100_000_000 + 1_792_674_000_000_000_000
    raw = episode(tmp_path, times, options)
    bgr = np.full((8, 8, 3), (10, 30, 240), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    cache = tmp_path / "left.bin"
    payload = encoded.tobytes()
    cache.write_bytes(payload)
    raw.camera_files = {"left": cache}
    raw.streams[NEDF3_CAMERA_TOPICS["left"]].values[:] = [0, len(payload)]
    segments, _ = align_episode(raw, options, np.ones(7))

    class Writer:
        def __init__(self):
            self.frames = []
            self.saved = 0

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved += 1

    writer = Writer()
    write_segment(writer, raw, segments[0], options)
    assert writer.saved == 1
    assert len(writer.frames) == 3
    first = writer.frames[0]
    assert set(first) == {
        "observation.state",
        "action",
        "task",
        "observation.images.left_wrist",
    }
    image = first["observation.images.left_wrist"]
    assert image.shape == (8, 8, 3)
    assert image[0, 0, 0] > 220 and image[0, 0, 2] < 30


@pytest.mark.parametrize("collector", ["E", "", "alice"])
def test_unknown_collectors_are_rejected(tmp_path, collector):
    with pytest.raises(ValueError, match="collector"):
        Source(Path(tmp_path), collector)
