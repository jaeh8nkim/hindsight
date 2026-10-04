#!/usr/bin/env bash
# Prompt-matched lockstep OPSD on Qwen3-8B: a correct-only and an incorrect-only branch train from
# the same start for 25 steps on the same 16 prompts per step. A candidate prompt is kept when, out
# of 4 rollouts per branch, the correct-only branch has a correct one and the incorrect-only branch
# an incorrect one; each branch then trains on its own rollout with the reflection from GT.
#
#   bash scripts/qwen3_8b_lockstep.sh        # seed 42
#   bash scripts/qwen3_8b_lockstep.sh 1      # seed 1
#
# Evaluation runs at step 0 (shared) and for each branch at the end (step 25). Branch snapshots
# (parameters and optimizer state) are swapped through $OUT/<run>/branches, which needs about
# 230 GB of disk. A finished run is skipped; an interrupted one starts over (its files are moved
# aside).
#
# Environment overrides:
#   CONTEXT=reflection_from_gt   LOSS=reverse_kl   (as in qwen3_8b.sh)
#   BENCHMARKS=math500,aime24,aime25   add aime26,hmmt26,olympiadbench for the extended suite
#   GPUS=0,1,2,3   TP=2   GPU_MEMORY_UTILIZATION=0.6   OUT=outputs   PYTHON=python
#   YES=1 (generate expert demonstrations without asking)   DRY_RUN=1 (print the commands only)
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

CONTEXT="${CONTEXT:-reflection_from_gt}"
LOSS="${LOSS:-reverse_kl}"
NAME="lockstep_${CONTEXT}_${LOSS}"

run() {
  if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "$@"; echo; else "$@"; fi
}

run "$PYTHON" "$ROOT/data.py" prepare --out "$DATA"
case "$CONTEXT" in
  reflection_from_demo|do_dont|four_line|resolve_from_demo)
    run "$PYTHON" "$ROOT/data.py" demos --out "$DATA" ${YES:+--yes} ;;
esac

mkdir -p "$OUT/logs"
echo "[$(date '+%F %T')] $SETTING $NAME, seed $SEED (log: $OUT/logs/$NAME.log)"
run "$PYTHON" "$ROOT/opsd.py" --lockstep --steps 25 --model "$MODEL" --revision "$REVISION" --data "$DATA" \
  --out "$OUT/$NAME" --seed "$SEED" --context "$CONTEXT" --loss "$LOSS" \
  --benchmarks "${BENCHMARKS:-math500,aime24,aime25}" \
  --n-gpus "$(tr ',' '\n' <<<"$GPUS" | grep -c .)" --tp "${TP:-2}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.6}" 2>&1 | tee -a "$OUT/logs/$NAME.log"
echo "[$(date '+%F %T')] finished: $OUT/$NAME/eval.jsonl"
