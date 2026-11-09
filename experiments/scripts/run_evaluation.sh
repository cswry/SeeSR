#!/usr/bin/env bash
# Evaluate one or more finished experiments.
# Usage:
#   GT_DIR=preset/datasets/RealSR/test_HR bash experiments/scripts/run_evaluation.sh \
#       experiments/results/RealSR baseline_ddpm50 original_ddpm50_s42 original_ddpm20_s42 combined_ddpm20_s42
# Leave GT_DIR unset for unpaired data (no-reference metrics only).
# Set LR_DIR to the LR inputs to add the LR-consistency diagnostic (--lrc; SCALE defaults to 4).
# Set PYTHON to choose the interpreter (default: python).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1  # upstream tracks __pycache__/*.pyc in git
root="${1:?usage: run_evaluation.sh RESULTS_ROOT EXPERIMENT [EXPERIMENT ...]}"; shift
[ "$#" -gt 0 ] || { echo "give at least one experiment name" >&2; exit 2; }
log_dir="${LOG_DIR:-experiments/logs}"   # set LOG_DIR to keep logs on persistent storage
mkdir -p "$log_dir"
status=0
for exp in "$@"; do
  args=(--sr_dir "$root/$exp/sr/sample00" --output_dir "$root/$exp/eval" --config experiments/configs/baseline.yaml --noref)
  if [ -n "${GT_DIR:-}" ]; then args+=(--paired --gt_dir "$GT_DIR"); fi
  if [ -n "${LR_DIR:-}" ]; then args+=(--lrc --lr_dir "$LR_DIR" --scale "${SCALE:-4}"); fi
  log="$log_dir/eval_${exp}_$(date +%Y%m%d_%H%M%S).log"
  "$PY" experiments/evaluation/evaluate.py "${args[@]}" ${EVAL_EXTRA_ARGS:-} 2>&1 | tee "$log" || status=$?
done
exit $status
