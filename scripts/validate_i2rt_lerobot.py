#!/usr/bin/env python3
"""Read a converted dataset with official LeRobot; optionally check ACT loss/backward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(root: Path, repo_id: str, act_smoke: bool = False) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = root.resolve()
    provenance = json.loads((root / "meta/i2rt_sources.json").read_text())
    dimension = len(provenance["axes"])
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    if (
        dataset.num_episodes != len(provenance["episodes"])
        or len(dataset) != provenance["total_frames"]
    ):
        raise ValueError("dataset totals differ from source provenance")
    samples = []
    cursor = 0
    for row in provenance["episodes"]:
        for index in sorted(
            {cursor, cursor + row["frames"] // 2, cursor + row["frames"] - 1}
        ):
            item = dataset[index]
            for key in ("observation.state", "action"):
                if (
                    tuple(item[key].shape) != (dimension,)
                    or not torch.isfinite(item[key]).all()
                ):
                    raise ValueError(f"invalid {key} at frame {index}")
            for key in dataset.meta.camera_keys:
                image = item[key]
                if (
                    image.ndim != 3
                    or image.shape[0] != 3
                    or not torch.isfinite(image).all()
                ):
                    raise ValueError(f"invalid video frame: {key}/{index}")
            samples.append(index)
        cursor += row["frames"]
    # Check every lowdim row, not just the video sample points.
    rows = 0
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        for key in ("observation.state", "action"):
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if values.shape != (len(table), dimension) or not np.isfinite(values).all():
                raise ValueError(f"invalid numeric values in {path}/{key}")
            if not np.logical_and(values[:, 6::7] >= 0, values[:, 6::7] <= 1).all():
                raise ValueError(f"gripper values are not normalized: {path}/{key}")
        rows += len(table)
    if rows != len(dataset):
        raise ValueError("parquet rows do not match dataset length")
    report = {
        "root": str(root),
        "format": dataset.meta.info["codebase_version"],
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "fps": dataset.fps,
        "state_action_dim": dimension,
        "cameras": list(dataset.meta.camera_keys),
        "collectors": sorted({row["collector"] for row in provenance["episodes"]}),
        "decoded_video_sample_indices": samples,
    }
    if act_smoke:
        from lerobot.configs.types import FeatureType
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.processor_act import make_act_pre_post_processors

        torch.set_num_threads(4)
        policy_features = dataset_to_policy_features(dataset.features)
        config = ACTConfig(
            device="cpu",
            input_features={
                k: v for k, v in policy_features.items() if v.type != FeatureType.ACTION
            },
            output_features={
                k: v for k, v in policy_features.items() if v.type == FeatureType.ACTION
            },
            chunk_size=60,
            n_action_steps=60,
            pretrained_backbone_weights=None,
            dim_model=64,
            dim_feedforward=128,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            n_vae_encoder_layers=1,
        )
        chunked = LeRobotDataset(
            repo_id,
            root=root,
            video_backend="pyav",
            delta_timestamps={"action": [i / dataset.fps for i in range(60)]},
        )
        batch = torch.utils.data.default_collate([chunked[0]])
        preprocessor, _ = make_act_pre_post_processors(config, dataset.meta.stats)
        batch = preprocessor(batch)
        policy = ACTPolicy(config)
        policy.train()
        loss, details = policy(batch)
        if not torch.isfinite(loss):
            raise ValueError("ACT produced a nonfinite loss")
        loss.backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError("ACT backward produced missing/nonfinite gradients")
        report["act_smoke"] = {
            "chunk_size": 60,
            "loss": float(loss.detach()),
            "details": details,
            "finite_backward": True,
            "note": "random weights, reduced transformer, CPU compatibility check; no training run/checkpoint",
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--repo-id", default="local/i2rt_act")
    parser.add_argument("--act-smoke", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.dataset, args.repo_id, args.act_smoke),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
