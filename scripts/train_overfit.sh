#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/output}"
DATASET_NAME="${DATASET_NAME:-i2rt_a_dual_act_20261022_all_0000_0015}"
DATASET_ROOT="${DATASET_ROOT:-${OUTPUT_ROOT}/data/${DATASET_NAME}}"
REPO_ID="${REPO_ID:-Hydrogen1002/${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/runs/act_overfit_$(date +%Y%m%d_%H%M%S)}"
EPISODES="${EPISODES:-[0,1,2,3,4]}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-${OUTPUT_ROOT}/envs/act/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "Dataset not found: ${DATASET_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.episodes="${EPISODES}" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=60 \
  --policy.n_action_steps=60 \
  --policy.pretrained_backbone_weights=null \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_i2rt_overfit \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --save_freq=2000 \
  --eval_freq=0 \
  --log_freq=50 \
  --wandb.enable=false
