#!/usr/bin/env bash
# Usage:
#   bash experiments/scripts/run_inference.sh baseline              # track A: upstream 50-step baseline
#   bash experiments/scripts/run_inference.sh sweep                 # track B: 4 variants x 50/30/20 steps x 3 seeds
#   bash experiments/scripts/run_inference.sh sweep --seeds 42 --variants original   # upstream method, one seed
#   bash experiments/scripts/run_inference.sh sweep --variants all            # include lr_schedule_late
#   bash experiments/scripts/run_inference.sh sweep --only combined_ddpm20_s42       # a single run
#   SSR variants (spatial_sem, semantic_spatial, ...) only after Gate 0 = GO; see experiments/README.md 5a
#   bash experiments/scripts/run_inference.sh v2 --seeds 42                   # V2 SFR-OC ablation (README 5b)
#   bash experiments/scripts/run_inference.sh efficiency                      # separate sampler experiment (5c)
# Extra arguments are passed to experiments/run_inference.py (e.g. --results_root, --dry_run).
# Set PYTHON to choose the interpreter (default: python).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1  # keep the tree free of .pyc files
mode="${1:?usage: run_inference.sh baseline|sweep|v2|efficiency [extra args]}"; shift
log_dir="${LOG_DIR:-experiments/logs}"   # set LOG_DIR to keep logs on persistent storage
mkdir -p "$log_dir"
log="$log_dir/inference_${mode}_$(date +%Y%m%d_%H%M%S).log"
case "$mode" in
  baseline) "$PY" experiments/run_inference.py --config experiments/configs/baseline.yaml "$@" 2>&1 | tee "$log" ;;
  sweep)    "$PY" experiments/run_inference.py --sweep experiments/configs/sampling_sweep.yaml "$@" 2>&1 | tee "$log" ;;
  v2)       "$PY" experiments/run_inference.py --sweep experiments/configs/v2_ablation.yaml "$@" 2>&1 | tee "$log" ;;
  efficiency) "$PY" experiments/run_inference.py --sweep experiments/configs/efficiency_sweep.yaml "$@" 2>&1 | tee "$log" ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
