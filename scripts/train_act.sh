#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_DIR}/data/lerobot/towel_folding}"
REPO_ID="${REPO_ID:-local/towel_folding}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/act_towel_folding}"
STEPS="${STEPS:-50000}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

cd "${LEROBOT_DIR}"
if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run lerobot-train)
else
  RUNNER=(lerobot-train)
fi

"${RUNNER[@]}" \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.eval_split=0.1 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=60 \
  --policy.n_action_steps=60 \
  --accelerator.mixed_precision=bf16 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_towel_folding \
  --batch_size=16 \
  --num_workers=8 \
  --steps="${STEPS}" \
  --eval_steps=2000 \
  --max_eval_samples=1024 \
  --save_freq=5000 \
  --wandb.enable="${WANDB_ENABLE}"
