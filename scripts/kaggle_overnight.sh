#!/usr/bin/env bash
# Tabula / Zero Kaggle launcher — Run #4 gated curriculum.
# Warm-start from Run #2 weights. Bidirectional mate-exam gate.
# Default: ~4.5 h canary. Overnight only if the exam holds.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
export ZERO_OUTPUT_DIR="${ZERO_OUTPUT_DIR:-/kaggle/working/checkpoints}"
export TABULA_REPETITION_PENALTY="${TABULA_REPETITION_PENALTY:--0.20}"
ZERO_SIMULATIONS="${ZERO_SIMULATIONS:-128}"
ZERO_TIME_LIMIT_HOURS="${ZERO_TIME_LIMIT_HOURS:-4.5}"
ZERO_GAMES="${ZERO_GAMES:-8}"
KAGGLE_INPUT="${KAGGLE_INPUT:-/kaggle/input}"
# Explicit overrides (either name works).
ZERO_RESUME_FROM="${ZERO_RESUME_FROM:-${ZERO_SEED_WEIGHTS:-}}"
# Gate thresholds (mates out of 20).
ZERO_CURRICULUM_UNLOCK_UP="${ZERO_CURRICULUM_UNLOCK_UP:-16}"
ZERO_CURRICULUM_DROP_DOWN="${ZERO_CURRICULUM_DROP_DOWN:-12}"
ZERO_CURRICULUM_UNLOCK_STREAK="${ZERO_CURRICULUM_UNLOCK_STREAK:-2}"

pick_newest_iter_checkpoint() {
  # Print the weights_iter_XXXX.pt with the highest iteration number under $1.
  local root="$1"
  local best="" best_n=-1
  local f base num
  while IFS= read -r -d '' f; do
    base="$(basename "$f")"
    [[ "$base" =~ ^weights_iter_([0-9]+)\.pt$ ]] || continue
    num=$((10#${BASH_REMATCH[1]}))
    if (( num > best_n )); then
      best_n=$num
      best="$f"
    fi
  done < <(find "$root" -type f -name 'weights_iter_*.pt' -print0 2>/dev/null)
  if [[ -n "$best" ]]; then
    echo "$best"
    return 0
  fi
  return 1
}

resolve_seed() {
  # 1) Explicit path
  if [[ -n "${ZERO_RESUME_FROM}" && -e "${ZERO_RESUME_FROM}" ]]; then
    echo "${ZERO_RESUME_FROM}"
    return 0
  fi
  if [[ ! -d "${KAGGLE_INPUT}" ]]; then
    return 1
  fi

  # 2) Prefer a unique latest_weights.pt under /kaggle/input (newest mtime if many)
  mapfile -t latests < <(find "${KAGGLE_INPUT}" -type f -name 'latest_weights.pt' 2>/dev/null | sort)
  if [[ ${#latests[@]} -eq 1 ]]; then
    echo "${latests[0]}"
    return 0
  fi
  if [[ ${#latests[@]} -gt 1 ]]; then
    local newest=""
    newest="$(ls -t "${latests[@]}" 2>/dev/null | head -n 1 || true)"
    if [[ -n "$newest" && -e "$newest" ]]; then
      echo "$newest"
      return 0
    fi
  fi

  # 3) Highest weights_iter_XXXX.pt anywhere under /kaggle/input
  local best=""
  if best="$(pick_newest_iter_checkpoint "${KAGGLE_INPUT}")"; then
    echo "$best"
    return 0
  fi
  return 1
}

RESUME_PATH=""
if RESUME_PATH="$(resolve_seed)"; then
  :
else
  RESUME_PATH=""
fi

echo "=== Tabula Run #4 launcher (gated curriculum) ==="
echo "ZERO_OUTPUT_DIR=${ZERO_OUTPUT_DIR}"
echo "ZERO_SIMULATIONS=${ZERO_SIMULATIONS}"
echo "ZERO_GAMES=${ZERO_GAMES}"
echo "ZERO_TIME_LIMIT_HOURS=${ZERO_TIME_LIMIT_HOURS}"
echo "TABULA_REPETITION_PENALTY=${TABULA_REPETITION_PENALTY}"
echo "ZERO_RESUME_FROM=${RESUME_PATH:-<none>}"
echo "GATE unlock>=${ZERO_CURRICULUM_UNLOCK_UP}/20 x${ZERO_CURRICULUM_UNLOCK_STREAK} | drop<=${ZERO_CURRICULUM_DROP_DOWN}/20"

mkdir -p "${ZERO_OUTPUT_DIR}" "${ZERO_OUTPUT_DIR}/pgn_dumps"

HAS_LOCAL_CKPT=0
if compgen -G "${ZERO_OUTPUT_DIR}/weights_iter_*.pt" > /dev/null; then
  HAS_LOCAL_CKPT=1
fi

# Wipe replay only on a true Run #4 cold start (no local checkpoints yet).
if [[ "${HAS_LOCAL_CKPT}" -eq 0 ]]; then
  echo "Cold start: wiping replay/ for a clean Run #4 buffer."
  rm -rf "${ZERO_OUTPUT_DIR}/replay"
fi
mkdir -p "${ZERO_OUTPUT_DIR}/replay"

if [[ "${HAS_LOCAL_CKPT}" -eq 0 ]]; then
  if [[ -z "${RESUME_PATH}" || ! -e "${RESUME_PATH}" ]]; then
    echo "Seed checkpoint not found under ${KAGGLE_INPUT}." >&2
    echo "Mount the Run #2 output or set ZERO_RESUME_FROM / ZERO_SEED_WEIGHTS." >&2
    echo "Do NOT seed from the Run #3 canary (mate skill already wiped)." >&2
    exit 2
  fi
  echo "Seeding from: ${RESUME_PATH}"
else
  echo "Resuming local Run #4 checkpoints in ${ZERO_OUTPUT_DIR} (seed ignored by pipeline)."
fi

RESUME_ARGS=()
if [[ "${HAS_LOCAL_CKPT}" -eq 0 && -n "${RESUME_PATH}" && -e "${RESUME_PATH}" ]]; then
  RESUME_ARGS=(--resume-from "${RESUME_PATH}")
fi

python -m tabula.pipeline \
  --no-tui \
  --output-dir "${ZERO_OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}" \
  --time-limit-hours "${ZERO_TIME_LIMIT_HOURS}" \
  --games "${ZERO_GAMES}" \
  --simulations "${ZERO_SIMULATIONS}" \
  --arena-games "${ZERO_ARENA_GAMES:-0}" \
  --mate-exam-every "${ZERO_MATE_EXAM_EVERY:-50}" \
  --max-game-error-rate "${ZERO_MAX_GAME_ERROR_RATE:-0.20}" \
  --curriculum-gate \
  --curriculum-unlock-up "${ZERO_CURRICULUM_UNLOCK_UP}" \
  --curriculum-drop-down "${ZERO_CURRICULUM_DROP_DOWN}" \
  --curriculum-unlock-streak "${ZERO_CURRICULUM_UNLOCK_STREAK}"
