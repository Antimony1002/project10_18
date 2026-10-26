#!/usr/bin/env python3
"""Offline ACT checkpoint evaluation on held-out LeRobot episodes.

This script never creates a robot or environment.  It predicts ACT action
chunks from recorded observations, unnormalizes them with the processor saved
inside each checkpoint, and compares them with the recorded actions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


GROUPS = {
    "left_arm": (0, 6),
    "left_gripper": (6, 7),
    "right_arm": (7, 13),
    "right_gripper": (13, 14),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episodes", default="4,9,14,19,24,29,34,39,44,49")
    parser.add_argument("--checkpoints", default="005000,006000,007000,008000,010000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Evaluate every Nth validation frame. Use 1 for every frame.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def comma_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def masked_mae_sum(error: torch.Tensor, valid: torch.Tensor, start: int, end: int) -> tuple[float, int]:
    values = error[..., start:end] * valid
    count = int(valid.sum().item()) * (end - start)
    return float(values.sum().item()), count


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: str, episodes: list[int]) -> dict:
    model_dir = args.run_root / "checkpoints" / checkpoint / "pretrained_model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {model_dir}")

    policy = ACTPolicy.from_pretrained(model_dir, local_files_only=True)
    policy.to(args.device).eval()
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(policy.config, metadata),
        video_backend="pyav",
    )
    indices = list(range(0, len(dataset), args.frame_stride))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    totals = defaultdict(float)
    counts = defaultdict(int)
    episode_sums = defaultdict(float)
    episode_counts = defaultdict(int)

    with torch.inference_mode():
        for batch in loader:
            target = batch["action"].clone()
            padding = batch["action_is_pad"].clone()
            episode_index = batch["episode_index"].clone()
            prediction = policy.predict_action_chunk(preprocessor(batch))
            prediction = postprocessor(prediction).cpu()

            error = (prediction - target).abs()
            valid = (~padding).unsqueeze(-1)
            value, count = masked_mae_sum(error, valid, 0, error.shape[-1])
            totals["chunk"] += value
            counts["chunk"] += count

            first_error = error[:, 0]
            totals["first"] += float(first_error.sum().item())
            counts["first"] += first_error.numel()

            for name, (start, end) in GROUPS.items():
                value, count = masked_mae_sum(error, valid, start, end)
                totals[name] += value
                counts[name] += count

            per_frame_sum = (error * valid).sum(dim=(1, 2))
            per_frame_count = valid.sum(dim=(1, 2)) * error.shape[-1]
            for ep, value, count in zip(episode_index.tolist(), per_frame_sum.tolist(), per_frame_count.tolist()):
                episode_sums[int(ep)] += float(value)
                episode_counts[int(ep)] += int(count)

    episode_maes = {
        str(ep): episode_sums[ep] / episode_counts[ep]
        for ep in sorted(episode_counts)
    }
    result = {
        "checkpoint": checkpoint,
        "step": int(checkpoint),
        "frames_evaluated": len(indices),
        "episodes_evaluated": len(episode_maes),
        "frame_stride": args.frame_stride,
        "chunk_mae": totals["chunk"] / counts["chunk"],
        "episode_mean_chunk_mae": sum(episode_maes.values()) / len(episode_maes),
        "first_action_mae": totals["first"] / counts["first"],
        **{f"{name}_mae": totals[name] / counts[name] for name in GROUPS},
        "per_episode_chunk_mae": episode_maes,
    }
    del policy, dataset, loader, preprocessor, postprocessor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def save_results(args: argparse.Namespace, results: list[dict]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda row: row["episode_mean_chunk_mae"])
    payload = {
        "dataset_root": str(args.dataset_root),
        "run_root": str(args.run_root),
        "episodes": [int(x) for x in comma_list(args.episodes)],
        "ranking_metric": "episode_mean_chunk_mae",
        "results": ranked,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    scalar_keys = [key for key in ranked[0] if key != "per_episode_chunk_mae"]
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_keys} for row in ranked)


def main() -> None:
    args = parse_args()
    episodes = [int(x) for x in comma_list(args.episodes)]
    results = []
    for checkpoint in comma_list(args.checkpoints):
        print(f"Evaluating checkpoint {checkpoint} ...", flush=True)
        result = evaluate_checkpoint(args, checkpoint, episodes)
        results.append(result)
        save_results(args, results)
        print(
            f"  episode_mean_chunk_mae={result['episode_mean_chunk_mae']:.6f} "
            f"first_action_mae={result['first_action_mae']:.6f}",
            flush=True,
        )
    print(f"Results written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
