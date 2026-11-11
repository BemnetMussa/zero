#!/usr/bin/env bash
set -euo pipefail

: "${SYZYGY_PATH:?Set SYZYGY_PATH to the mounted Syzygy 3-5 piece directory}"

if [[ ! -d "${SYZYGY_PATH}" ]]; then
  echo "SYZYGY_PATH is not a directory: ${SYZYGY_PATH}" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-src}"
export ZERO_OUTPUT_DIR="${ZERO_OUTPUT_DIR:-/kaggle/working/checkpoints}"

python -m tabula.pipeline \
  --no-tui \
  --output-dir "${ZERO_OUTPUT_DIR}" \
  --time-limit-hours "${ZERO_TIME_LIMIT_HOURS:-7.5}" \
  --games "${ZERO_GAMES:-5}" \
  --arena-games "${ZERO_ARENA_GAMES:-0}" \
  --mate-exam-every "${ZERO_MATE_EXAM_EVERY:-50}" \
  --max-game-error-rate "${ZERO_MAX_GAME_ERROR_RATE:-0.20}"
