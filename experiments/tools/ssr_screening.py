"""SSR screening decision: does spatial semantic reliability earn the full evaluation?

Reads the small screening runs (20 RealSR + 20 DRealSR images, seed 42, 20 and 50 steps) and checks,
per dataset and step count, with images paired by name:

  where-test   semantic_spatial vs semantic_conf   (same mean strength; only the spatial arrangement differs)
  alone-test   spatial_sem      vs original
  control      spatial_shuffled vs semantic_conf   (same map values, wrong locations)

for LPIPS and DISTS (lower is better): mean paired difference and fraction of images that improved.

CONTINUE to the full evaluation only if, in EVERY (dataset, steps) cell,
  * the where-test and the alone-test improve both metrics on average, with at least `min_fraction` of
    images improving on each metric, and
  * the shuffled control does NOT reproduce it: it is not consistently better than semantic_conf by the same
    rule, and its mean improvement is below half of semantic_spatial's on each metric.
Otherwise STOP. All runs must be completed, evaluated on the same images, and use the same base seed.

  python experiments/tools/ssr_screening.py --results_root RealSR=<R>/screening/RealSR \\
      --results_root DRealSR=<R>/screening/DRealSR --steps 20 50 --seed 42 --output <R>/screening/ssr_screening.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.tools.compare_results import load_experiment, per_image_values  # noqa: E402

METRICS = ("lpips", "dists")
COMPARISONS = {"where": ("semantic_spatial", "semantic_conf"),
               "alone": ("spatial_sem", "original"),
               "shuffled": ("spatial_shuffled", "semantic_conf")}


def paired(exp, ref, metric):
    a, b = per_image_values(exp, metric), per_image_values(ref, metric)
    if set(a) != set(b):
        raise ValueError(f"{exp['name']} and {ref['name']} were evaluated on different images for {metric}")
    if not a:
        raise ValueError(f"{exp['name']}: no {metric} values (run the paired evaluation first)")
    d = [a[k] - b[k] for k in sorted(a)]
    return {"n_images": len(d), "mean_diff": statistics.fmean(d), "median_diff": statistics.median(d),
            "frac_improved": sum(x < 0 for x in d) / len(d)}


def consistent(res, min_fraction):
    return all(res[m]["mean_diff"] < 0 and res[m]["frac_improved"] >= min_fraction for m in METRICS)


def evaluate_cell(runs, min_fraction):
    """runs: variant -> loaded experiment for one (dataset, steps)."""
    out = {}
    for label, (v, ref) in COMPARISONS.items():
        missing = [x for x in (v, ref) if x not in runs]
        if missing:
            raise ValueError(f"missing runs: {missing}")
        out[label] = {m: paired(runs[v], runs[ref], m) for m in METRICS}
    where_ok = consistent(out["where"], min_fraction)
    alone_ok = consistent(out["alone"], min_fraction)
    shuffled_reproduces = consistent(out["shuffled"], min_fraction) or any(
        out["shuffled"][m]["mean_diff"] <= 0.5 * out["where"][m]["mean_diff"] < 0 for m in METRICS)
    out["checks"] = {"where_consistent": where_ok, "alone_consistent": alone_ok,
                     "shuffled_does_not_reproduce": not shuffled_reproduces}
    return out


def load_runs(root, steps, seed):
    runs = {}
    variants = {v for pair in COMPARISONS.values() for v in pair}
    for v in variants:
        path = Path(root) / f"{v}_ddpm{steps}_s{seed}"
        exp = load_experiment(path)
        status = exp["meta"].get("status")
        if status != "completed":
            raise ValueError(f"{path.name}: status {status!r}")
        runs[v] = exp
    return runs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", action="append", required=True, metavar="DATASET=DIR")
    p.add_argument("--steps", type=int, nargs="+", default=[20, 50])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_fraction", type=float, default=0.6)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    cells, problems = {}, []
    for item in a.results_root:
        ds, _, root = item.partition("=")
        for st in a.steps:
            key = f"{ds}@{st}"
            try:
                cells[key] = evaluate_cell(load_runs(root, st, a.seed), a.min_fraction)
            except (ValueError, FileNotFoundError) as e:
                problems.append(f"{key}: {e}")
    ok = not problems and cells and all(all(c["checks"].values()) for c in cells.values())
    summary = {"decision": "CONTINUE" if ok else "STOP", "problems": problems, "min_fraction": a.min_fraction,
               "cells": cells, "rule": "see tools/ssr_screening.py docstring"}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(summary, indent=2))
    for key, c in cells.items():
        print(key, c["checks"])
    for pr in problems:
        print("PROBLEM:", pr)
    print(f"SSR SCREENING DECISION: {summary['decision']}")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
