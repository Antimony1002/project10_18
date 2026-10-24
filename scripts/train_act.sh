#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/output}"
TASK="${TASK:-empty_tray}"
case "${TASK}" in
  empty_tray)
    DEFAULT_DATASET_NAME="i2rt_a_dual_act_20261024_empty_tray_50ep"
    ;;
  cup_tray)
    DEFAULT_DATASET_NAME="i2rt_a_dual_act_20261024_cup_tray_50ep"
    ;;
  *)
    echo "TASK must be empty_tray or cup_tray, got: ${TASK}" >&2
    exit 2
    ;;
esac
DATASET_NAME="${DATASET_NAME:-${DEFAULT_DATASET_NAME}}"
DATASET_ROOT="${DATASET_ROOT:-${OUTPUT_ROOT}/data/${DATASET_NAME}}"
REPO_ID="${REPO_ID:-Hydrogen1002/${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/runs/act_${TASK}_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
PYTHON_BIN="${PYTHON_BIN:-${OUTPUT_ROOT}/envs/act/bin/python}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-aloha-act-tray}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
EPISODES="${EPISODES:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "Dataset not found: ${DATASET_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

TRAIN_ARGS=(
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=60 \
  --policy.n_action_steps=60 \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="act_i2rt_${TASK}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --eval_freq=0 \
  --save_freq="${SAVE_FREQ}" \
  --log_freq=100 \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.mode="${WANDB_MODE}" \
  --wandb.disable_artifact="${WANDB_DISABLE_ARTIFACT}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  TRAIN_ARGS+=(--wandb.entity="${WANDB_ENTITY}")
fi
if [[ -n "${EPISODES}" ]]; then
  TRAIN_ARGS+=(--dataset.episodes="${EPISODES}")
fi

"${PYTHON_BIN}" -m lerobot.scripts.lerobot_train "${TRAIN_ARGS[@]}"
