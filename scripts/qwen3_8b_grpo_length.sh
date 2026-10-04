#!/usr/bin/env bash
# Length-penalized GRPO on Qwen3-8B: 25 steps of 16 DAPO-Math prompts x 8 rollouts (12,288-token
# budget), learning rate 1e-6. The reward (+1 correct, -1 otherwise) carries DAPO's soft overlong
# penalty, which grows linearly from 0 at 8,192 tokens to -1 at 12,288.
#
#   bash scripts/qwen3_8b_grpo_length.sh        # seed 42
#   bash scripts/qwen3_8b_grpo_length.sh 1      # seed 1
#
# Evaluation runs at step 0 and at the end (step 25). A finished run is skipped; an interrupted
# one starts over (its files are moved aside).
#
# Environment overrides:
#   LENGTH_PENALTY=1             0: vanilla GRPO
#   BENCHMARKS=math500,aime24,aime25   add aime26,hmmt26,olympiadbench for the extended suite
#   GPUS=0,1,2,3   TP=2   GPU_MEMORY_UTILIZATION=0.6   OUT=outputs   PYTHON=python   DRY_RUN=1
set -euo pipefail

SETTING=qwen3_8b
MODEL=Qwen/Qwen3-8B
REVISION=b968826d9c46dd6066d109eabc6255188de91218

SEED="${1:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "usage: $0 [SEED]" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(mkdir -p "${OUT:-$ROOT/outputs}" && cd "${OUT:-$ROOT/outputs}" && pwd)/$SETTING/seed$SEED"
DATA="$ROOT/data"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

NAME=grpo_length_penalty
ARGS=()
if [[ "${LENGTH_PENALTY:-1}" == 0 ]]; then
  NAME=grpo_vanilla
  ARGS+=(--no-length-penalty)
fi

run() {
  if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "$@"; echo; else "$@"; fi
}

run "$PYTHON" "$ROOT/data.py" prepare --out "$DATA"
mkdir -p "$OUT/logs"
echo "[$(date '+%F %T')] $SETTING $NAME, seed $SEED (log: $OUT/logs/$NAME.log)"
run "$PYTHON" "$ROOT/grpo.py" --model "$MODEL" --revision "$REVISION" --data "$DATA" --out "$OUT/$NAME" \
  --seed "$SEED" ${ARGS[@]+"${ARGS[@]}"} --benchmarks "${BENCHMARKS:-math500,aime24,aime25}" \
  --n-gpus "$(tr ',' '\n' <<<"$GPUS" | grep -c .)" --tp "${TP:-2}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.6}" 2>&1 | tee -a "$OUT/logs/$NAME.log"
echo "[$(date '+%F %T')] finished: $OUT/$NAME/eval.jsonl"
