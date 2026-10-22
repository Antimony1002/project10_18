#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_DIR}/data/lerobot/towel_folding}"
REPO_ID="${REPO_ID:-local/towel_folding}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/act_overfit}"
EPISODES="${EPISODES:-[0,1,2,3,4]}"
STEPS="${STEPS:-10000}"

cd "${LEROBOT_DIR}"
if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run lerobot-train)
else
  RUNNER=(lerobot-train)
fi

"${RUNNER[@]}" \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.episodes="${EPISODES}" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=60 \
  --policy.n_action_steps=60 \
  --accelerator.mixed_precision=bf16 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_towel_overfit \
  --batch_size=8 \
  --num_workers=8 \
  --steps="${STEPS}" \
  --save_freq=2000 \
  --eval_steps=0 \
  --wandb.enable=false
