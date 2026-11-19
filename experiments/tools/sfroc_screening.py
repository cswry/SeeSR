"""SFR-OC strength selection on the SCREENING set (pre-registered rule; run before the full V2 evaluation).

Input: the runs of configs/v2_strength_screening.yaml (original, combined_spatial = V1, v2_heur_sem_g025/g050/g100)
for each dataset at 20 and 50 steps, seed 42, all evaluated with the paired metrics. Images are paired by name;
every run of a cell must be completed and evaluated on the same images.

Per candidate strength g and cell (dataset, steps), paired differences against V1 (combined_spatial):
    dPSNR, dSSIM   (higher is better)      dLPIPS, dDISTS   (lower is better)
with mean, median, fraction of images improved and a 95 % percentile-bootstrap CI of the mean (2000 resamples,
fixed seed). Also recorded: SFR-OC map statistics (sfroc_W_mean, sfroc_O_mean, sfroc_S_mean) and runtime
(sfroc_analysis_ms, sfroc_correction_ms, diffusion time) from per_image.csv.

RULE (fixed before any V2 result was seen; do not change it after looking at the numbers):
  1. A strength is ADMISSIBLE if, in EVERY cell,
       mean dLPIPS <= +LPIPS_BUDGET (0.005) and mean dDISTS <= +DISTS_BUDGET (0.005)   (perceptual cost bounded)
       and mean dPSNR > 0                                                              (fidelity improves)
  2. Among admissible strengths choose the largest average (over cells) mean dPSNR; if another admissible
     strength is within PSNR_TIE (0.05 dB) of it, take the SMALLEST such strength (least intervention).
  3. No admissible strength -> decision NONE: SFR-OC is not supported at the screened strengths; the full
     V2 ablation is not run with a tuned strength (report the screening result instead).
SSIM, the bootstrap CIs, the map statistics and the runtime are reported, not used by the rule.
The final evaluation images must not be in the screening set (experiments/README.md §5a).

  python experiments/tools/sfroc_screening.py --results_root RealSR=<R>/v2_screening/RealSR \\
      --results_root DRealSR=<R>/v2_screening/DRealSR --steps 20 50 --seed 42 \\
      --output <R>/v2_screening/sfroc_screening.json
"""
import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.tools.compare_results import load_experiment, per_image_values  # noqa: E402

REFERENCE = "combined_spatial"
CANDIDATES = {"v2_heur_sem_g025": 0.25, "v2_heur_sem_g050": 0.5, "v2_heur_sem_g100": 1.0}
METRICS = {"psnr": +1, "ssim": +1, "lpips": -1, "dists": -1}       # +1: higher is better
LPIPS_BUDGET = 0.005
DISTS_BUDGET = 0.005
PSNR_TIE = 0.05
RUNTIME_FIELDS = ("sfroc_W_mean", "sfroc_W_min", "sfroc_W_max", "sfroc_O_mean", "sfroc_S_mean",
                  "sfroc_analysis_ms", "sfroc_correction_ms", "sfroc_encode_ms", "diffusion_s")
N_BOOT = 2000


def bootstrap_ci(values, n_boot=N_BOOT, seed=0, level=0.95):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(n_boot))
    lo = means[int((1 - level) / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
    return [lo, hi]


def paired(exp, ref, metric):
    a, b = per_image_values(exp, metric), per_image_values(ref, metric)
    if set(a) != set(b):
        raise ValueError(f"{exp['name']} and {ref['name']} were evaluated on different images for {metric}")
    if not a:
        raise ValueError(f"{exp['name']}: no {metric} values (run the paired evaluation first)")
    d = [a[k] - b[k] for k in sorted(a)]
    better = METRICS[metric]
    return {"n_images": len(d), "mean_diff": statistics.fmean(d), "median_diff": statistics.median(d),
            "frac_improved": sum((x * better) > 0 for x in d) / len(d), "ci95_mean_diff": bootstrap_ci(d)}


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def run_stats(path):
    """Means of the SFR-OC map / runtime columns of per_image.csv (absent columns -> None)."""
    p = Path(path) / "per_image.csv"
    if not p.exists():
        return {}
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in RUNTIME_FIELDS:
        vals = [x for x in (_num(r.get(k)) for r in rows) if x is not None]
        out[k] = statistics.fmean(vals) if vals else None
    return out


def evaluate_cell(runs):
    out = {}
    for name in CANDIDATES:
        out[name] = {m: paired(runs[name], runs[REFERENCE], m) for m in METRICS}
        out[name]["run_stats"] = runs[name].get("run_stats", {})
    out[REFERENCE] = {"run_stats": runs[REFERENCE].get("run_stats", {})}
    return out


def admissible(cell_result):
    return (cell_result["lpips"]["mean_diff"] <= LPIPS_BUDGET and cell_result["dists"]["mean_diff"] <= DISTS_BUDGET
            and cell_result["psnr"]["mean_diff"] > 0)


def select(cells):
    """Apply the rule. cells: key -> evaluate_cell(...)."""
    table = {}
    for name, g in CANDIDATES.items():
        ok = all(admissible(c[name]) for c in cells.values())
        table[name] = {"strength": g, "admissible": ok,
                       "mean_dpsnr": statistics.fmean(c[name]["psnr"]["mean_diff"] for c in cells.values())}
    adm = [v for v in table.values() if v["admissible"]]
    if not adm:
        return None, table
    best = max(v["mean_dpsnr"] for v in adm)
    close = [v["strength"] for v in adm if v["mean_dpsnr"] >= best - PSNR_TIE]
    return min(close), table


def load_runs(root, steps, seed):
    runs = {}
    for v in (REFERENCE, *CANDIDATES):
        path = Path(root) / f"{v}_ddpm{steps}_s{seed}"
        exp = load_experiment(path)
        status = exp["meta"].get("status")
        if status != "completed":
            raise ValueError(f"{path.name}: status {status!r}")
        exp["run_stats"] = run_stats(path)
        runs[v] = exp
    return runs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", action="append", required=True, metavar="DATASET=DIR")
    p.add_argument("--steps", type=int, nargs="+", default=[20, 50])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    cells, problems = {}, []
    for item in a.results_root:
        ds, _, root = item.partition("=")
        for st in a.steps:
            key = f"{ds}@{st}"
            try:
                cells[key] = evaluate_cell(load_runs(root, st, a.seed))
            except (ValueError, FileNotFoundError, KeyError) as e:
                problems.append(f"{key}: {e}")
    strength, table = select(cells) if cells and not problems else (None, {})
    decision = "INCOMPLETE" if problems or not cells else ("SELECTED" if strength is not None else "NONE")
    summary = {"decision": decision, "selected_strength": strength, "candidates": table, "problems": problems,
               "rule": {"lpips_budget": LPIPS_BUDGET, "dists_budget": DISTS_BUDGET, "psnr_tie_db": PSNR_TIE,
                        "reference": REFERENCE, "text": "see tools/sfroc_screening.py docstring"},
               "cells": cells}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(summary, indent=2))
    for name, row in table.items():
        print(name, row)
    for pr in problems:
        print("PROBLEM:", pr)
    print(f"SFR-OC STRENGTH SCREENING: {decision}" + (f" (strength {strength})" if strength is not None else ""))
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
