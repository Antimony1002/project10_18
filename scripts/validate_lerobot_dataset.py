#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and sanity-check a converted LeRobot dataset")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--repo-id", default="local/towel_folding")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    required = {
        "observation.state",
        "action",
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    missing = sorted(required - set(dataset.features))
    if missing:
        raise ValueError(f"Converted dataset is missing features: {missing}")
    if dataset.num_episodes < 1 or len(dataset) < 1:
        raise ValueError("Converted dataset is empty")

    sample_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    samples = []
    for index in sample_indices:
        item = dataset[index]
        state = np.asarray(item["observation.state"])
        action = np.asarray(item["action"])
        if state.shape != (14,) or action.shape[-1] != 14:
            raise ValueError(f"Frame {index}: unexpected state/action shape {state.shape}/{action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"Frame {index}: state/action contains NaN or Inf")
        camera_shapes = {
            key: list(np.asarray(item[key]).shape)
            for key in required
            if key.startswith("observation.images.")
        }
        samples.append(
            {
                "index": index,
                "state_shape": list(state.shape),
                "action_shape": list(action.shape),
                "cameras": camera_shapes,
            }
        )

    steps_per_epoch = math.ceil(len(dataset) / 8)
    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "repo_id": args.repo_id,
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "fps": dataset.fps,
        "camera_keys": list(dataset.meta.camera_keys),
        "steps_per_epoch_batch_8": steps_per_epoch,
        "recommended_5_epochs_steps": steps_per_epoch * 5,
        "recommended_10_epochs_steps": steps_per_epoch * 10,
        "samples": samples,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
