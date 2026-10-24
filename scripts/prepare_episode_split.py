#!/usr/bin/env python3
"""Create a lightweight episode split view for an existing LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads


DEFAULT_VALIDATION_INDICES = "4,9,14,19,24,29,34,39,44,49"


def parse_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not indices or len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("indices must be unique non-negative integers")
    return sorted(indices)


def feature_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"invalid numeric feature array: shape={values.shape}")
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(len(values))],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing 50-episode LeRobot dataset")
    parser.add_argument("output", type=Path, help="new lightweight split-view directory")
    parser.add_argument(
        "--validation-indices",
        type=parse_indices,
        default=parse_indices(DEFAULT_VALIDATION_INDICES),
        help=f"comma-separated held-out episodes (default: {DEFAULT_VALIDATION_INDICES})",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    for required in (source / "meta" / "info.json", source / "meta" / "stats.json", source / "data"):
        if not required.exists():
            raise FileNotFoundError(required)

    info = json.loads((source / "meta" / "info.json").read_text())
    total_episodes = int(info["total_episodes"])
    validation_indices = args.validation_indices
    if validation_indices[-1] >= total_episodes:
        raise ValueError(f"validation index exceeds dataset range 0:{total_episodes}")
    validation_set = set(validation_indices)
    train_indices = [index for index in range(total_episodes) if index not in validation_set]
    if not train_indices:
        raise ValueError("split leaves no training episodes")

    dataset = pads.dataset(source / "data", format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "observation.state", "action"],
        filter=pads.field("episode_index").isin(train_indices),
    )
    observed_train_indices = set(table["episode_index"].to_pylist())
    if observed_train_indices != set(train_indices):
        raise ValueError(
            f"data rows do not cover requested train episodes: {sorted(observed_train_indices)}"
        )

    stats = json.loads((source / "meta" / "stats.json").read_text())
    for key in ("observation.state", "action"):
        stats[key] = feature_stats(np.asarray(table[key].to_pylist(), dtype=np.float64))

    building = output.with_name(f"{output.name}.building")
    if building.exists() or building.is_symlink():
        raise FileExistsError(f"stale build path exists: {building}")
    building.mkdir(parents=True)
    shutil.copytree(source / "meta", building / "meta")
    for directory in ("data", "videos"):
        src = source / directory
        if src.exists():
            os.symlink(src, building / directory, target_is_directory=True)
    (building / "meta" / "stats.json").write_text(
        json.dumps(stats, indent=4, ensure_ascii=False) + "\n"
    )
    split = {
        "source_dataset": str(source),
        "total_episodes": total_episodes,
        "train_episode_indices": train_indices,
        "validation_episode_indices": validation_indices,
        "train_frames": len(table),
        "note": "data/videos are shared; state/action stats use train episodes only",
    }
    (building / "meta" / "episode_split.json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False) + "\n"
    )
    building.rename(output)

    print(f"Created split view: {output}")
    print(f"Train: {len(train_indices)} episodes / {len(table)} frames")
    print(f"Validation: {len(validation_indices)} episodes")
    print("EPISODES=" + json.dumps(train_indices, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
