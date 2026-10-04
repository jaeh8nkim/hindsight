#!/usr/bin/env bash
# OPSD on AceReason-Nemotron-7B: 100 steps of 32 DAPO-Math prompts, one rollout each, learning rate 1e-6.
# The self-teacher is the frozen initial model; its default context is the reflection from GT.
#
#   bash scripts/acereason_nemotron_7b.sh        # seed 42
#   bash scripts/acereason_nemotron_7b.sh 1      # seed 1
#
# One seed drives the prompt order, sampling, and evaluation; for a multi-seed result, run the
# script once per seed. Evaluation runs at step 0, every 25 steps, and at the end. A finished run
# is skipped; an interrupted one starts over (its files are moved aside).
#
# Environment overrides:
#   CONTEXT=reflection_from_gt   teacher context: reflection_from_gt, conciseness, reflection_from_demo,
#                                do_dont, four_line, resolve_from_demo (the last four need expert
#                                demonstrations, which the script offers to generate)
#   GATING=all                   rollouts trained on: all, correct, incorrect
#   LOSS=reverse_kl              reverse_kl, forward_kl, jsd, split (reverse KL on correct rollouts,
#                                forward KL on the rest)
#   REINJECTION=0                1: repeat the context in the teacher's input every 2,048 thinking tokens
#   BENCHMARKS=math500,aime24,aime25   add aime26,hmmt26,olympiadbench for the extended suite
#   GPUS=0,1,2,3   TP=2   GPU_MEMORY_UTILIZATION=0.6   OUT=outputs   PYTHON=python
#   YES=1 (generate expert demonstrations without asking)   DRY_RUN=1 (print the commands only)
set -euo pipefail

SETTING=acereason_nemotron_7b
MODEL=nvidia/AceReason-Nemotron-7B
REVISION=7d385d309648ef2fc78eaab944d41dfa204da8d8

SEED="${1:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "usage: $0 [SEED]" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(mkdir -p "${OUT:-$ROOT/outputs}" && cd "${OUT:-$ROOT/outputs}" && pwd)/$SETTING/seed$SEED"
DATA="$ROOT/data"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONTEXT="${CONTEXT:-reflection_from_gt}"
GATING="${GATING:-all}"
LOSS="${LOSS:-reverse_kl}"
NAME="opsd_${CONTEXT}_${GATING}_${LOSS}"
ARGS=(--context "$CONTEXT" --gating "$GATING" --loss "$LOSS")
if [[ "${REINJECTION:-0}" == 1 ]]; then
  NAME="${NAME}_reinjection"
  ARGS+=(--reinjection)
fi

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
run "$PYTHON" "$ROOT/opsd.py" --model "$MODEL" --revision "$REVISION" --data "$DATA" --out "$OUT/$NAME" \
  --seed "$SEED" "${ARGS[@]}" --benchmarks "${BENCHMARKS:-math500,aime24,aime25}" \
  --n-gpus "$(tr ',' '\n' <<<"$GPUS" | grep -c .)" --tp "${TP:-2}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.6}" 2>&1 | tee -a "$OUT/logs/$NAME.log"
echo "[$(date '+%F %T')] finished: $OUT/$NAME/eval.jsonl"
