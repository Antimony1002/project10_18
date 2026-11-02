#!/usr/bin/env python3
"""Compare PI0 checkpoints on fixed held-out LeRobot frames, without robot control."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from train_pi0_cup_place_then_carry import install_alias_patch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoints", default="000500,001000,001500,002000,002500,003000")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    split = json.loads(args.split.read_text())
    repo_id = split["source_dataset"]
    episodes = split["val_episodes"]
    metadata = LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    first = args.checkpoints.split(",")[0]
    with safe_open(
        str(args.run_root / "checkpoints" / first / "pretrained_model" / "model.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        keys = set(handle.keys())
        aliases = {key: value for key, value in (handle.metadata() or {}).items() if value in keys}
    install_alias_patch(PI0Policy, aliases)
    rows = []
    for checkpoint in args.checkpoints.split(","):
        model_dir = args.run_root / "checkpoints" / checkpoint / "pretrained_model"
        print(f"Loading {checkpoint}", flush=True)
        policy = PI0Policy.from_pretrained(model_dir, local_files_only=True)
        policy.to(device=args.device, dtype=torch.bfloat16).eval()
        dataset = LeRobotDataset(
            repo_id,
            root=args.dataset_root,
            episodes=episodes,
            delta_timestamps=resolve_delta_timestamps(policy.config, metadata),
            video_backend="pyav",
        )
        indices = np.linspace(0, len(dataset) - 1, min(args.frames, len(dataset)), dtype=int).tolist()
        loader = DataLoader(Subset(dataset, indices), batch_size=2, shuffle=False, num_workers=0)
        preprocessor, _ = make_pre_post_processors(
            policy.config,
            pretrained_path=str(model_dir),
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        total = 0.0
        n = 0
        seen_episodes = set()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                torch.manual_seed(20261102 + i)
                torch.cuda.manual_seed_all(20261102 + i)
                seen_episodes.update(int(x) for x in batch["episode_index"].tolist())
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, _ = policy(preprocessor(batch))
                count = len(batch["episode_index"])
                total += float(loss.item()) * count
                n += count
        if seen_episodes != set(episodes):
            raise RuntimeError(f"Validation episodes missed: {set(episodes) - seen_episodes}")
        row = {"step": int(checkpoint), "val_loss": total / n, "frames": n, "episodes": len(seen_episodes)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        output = args.run_root / "validation_pi0.json"
        output.write_text(json.dumps({"method": "fixed-seed PI0 flow-matching loss", "split": str(args.split), "results": rows}, indent=2) + "\n")
        with (args.run_root / "validation_pi0.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        del policy, dataset, loader, preprocessor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
