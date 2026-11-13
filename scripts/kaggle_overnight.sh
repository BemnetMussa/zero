#!/usr/bin/env bash
# Tabula / Zero overnight Kaggle launcher.
# Run #2: zero oracles in training, repetition penalty, 128 sims, seeded from Run #1.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
export ZERO_OUTPUT_DIR="${ZERO_OUTPUT_DIR:-/kaggle/working/checkpoints}"
export TABULA_REPETITION_PENALTY="${TABULA_REPETITION_PENALTY:--0.20}"
ZERO_SIMULATIONS="${ZERO_SIMULATIONS:-128}"
ZERO_TIME_LIMIT_HOURS="${ZERO_TIME_LIMIT_HOURS:-7.5}"
SEED_NAME="weights_iter_0969.pt"
KAGGLE_INPUT="${KAGGLE_INPUT:-/kaggle/input}"
ZERO_RESUME_FROM="${ZERO_RESUME_FROM:-${KAGGLE_INPUT}/zero-model/checkpoints/${SEED_NAME}}"

# Kaggle mounts notebook outputs under different layouts; find the exact seed file.
if [[ ! -e "${ZERO_RESUME_FROM}" && -d "${KAGGLE_INPUT}" ]]; then
  mapfile -t seeds < <(find "${KAGGLE_INPUT}" -name "${SEED_NAME}" -type f 2>/dev/null)
  if [[ ${#seeds[@]} -eq 1 ]]; then
    ZERO_RESUME_FROM="${seeds[0]}"
  elif [[ ${#seeds[@]} -gt 1 ]]; then
    printf 'Multiple %s found; set ZERO_RESUME_FROM explicitly:\n' "${SEED_NAME}" >&2
    printf '  %s\n' "${seeds[@]}" >&2
    exit 2
  fi
fi

echo "=== Tabula Run #2 launcher ==="
echo "ZERO_OUTPUT_DIR=${ZERO_OUTPUT_DIR}"
echo "ZERO_SIMULATIONS=${ZERO_SIMULATIONS}"
echo "TABULA_REPETITION_PENALTY=${TABULA_REPETITION_PENALTY}"
echo "ZERO_RESUME_FROM=${ZERO_RESUME_FROM}"

# Checkpoints already in ZERO_OUTPUT_DIR take priority (session restart);
# the seed checkpoint is only required for the first launch.
if ! compgen -G "${ZERO_OUTPUT_DIR}/weights_iter_*.pt" > /dev/null \
  && [[ ! -e "${ZERO_RESUME_FROM}" ]]; then
  echo "Seed checkpoint not found: ${ZERO_RESUME_FROM}" >&2
  echo "Mount the Run #1 output or set ZERO_RESUME_FROM." >&2
  exit 2
fi

python -m tabula.pipeline \
  --no-tui \
  --output-dir "${ZERO_OUTPUT_DIR}" \
  --resume-from "${ZERO_RESUME_FROM}" \
  --time-limit-hours "${ZERO_TIME_LIMIT_HOURS}" \
  --games "${ZERO_GAMES:-5}" \
  --simulations "${ZERO_SIMULATIONS}" \
  --arena-games "${ZERO_ARENA_GAMES:-0}" \
  --mate-exam-every "${ZERO_MATE_EXAM_EVERY:-50}" \
  --max-game-error-rate "${ZERO_MAX_GAME_ERROR_RATE:-0.20}"
