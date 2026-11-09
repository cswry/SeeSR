"""Compare finished experiments of ONE dataset. Two separate tracks; they are never mixed.

reproduction (track A)
    One upstream-faithful run (baseline.yaml, upstream RNG) against the published SeeSR numbers.
    A single run: no variance or significance claims are made.

sampling (track B)
    Runs from sampling_sweep.yaml: method variants x 50 / 30 / 20 steps x seeds, all with per-image
    seeding. Runs must be identical in every relevant setting except the variant's `method` section,
    num_inference_steps and seed; within one variant the method section must be identical. Otherwise the
    comparison is REJECTED (exit 2) and every difference is listed. Results are paired by (image, seed).
    Reported per (variant, steps) and metric:
      * mean over (image, seed) pairs and the std across seeds of the dataset-level mean;
      * within-image seed std: per image, std across seeds, averaged over images (needs >= 2 seeds);
      * paired differences against two references:
          - original SeeSR at the reference step count (default 50): combined effect of steps and method;
          - original SeeSR at the SAME step count (method variants only): effect of the method alone;
        per image, Δ averaged over the paired seeds, then mean, std and standard error over images, the
        share of images that improved, and the per-seed mean Δ with its std across seeds.
    No composite score and no significance test; missing values stay empty.

  python experiments/tools/compare_results.py sampling --results_root <RESULTS>/RealSR --dataset RealSR
  python experiments/tools/compare_results.py reproduction --results_root <RESULTS>/RealSR --dataset RealSR
"""
import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import load_yaml, now_iso, upstream_defaults, write_csv, write_json  # noqa: E402

METRIC_ORDER = ["psnr", "ssim", "lpips", "dists", "musiq", "clipiqa", "lr_psnr", "lr_mae"]
EVAL_KINDS = ("paired", "noref", "lrc")
REFERENCE_VARIANT = "original"
# The only upstream arguments allowed to differ between track-B runs.
ALLOWED_ARG_DIFFERENCES = {"num_inference_steps", "seed", "output_dir", "image_path"}
# Track A: settings the baseline intentionally sets differently from bare upstream defaults.
BASELINE_EXPERIMENT_SPECIFIC = {"seed", "save_prompts"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else None


def _load_csv(path):
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_experiment(path):
    path = Path(path)
    meta = _load_json(path / "run_metadata.json")
    if meta is None:
        raise FileNotFoundError(f"{path}/run_metadata.json not found")
    exp = {"name": path.name, "meta": meta, "summaries": {}, "per_image": {}}
    for kind in EVAL_KINDS:
        s = _load_json(path / "eval" / f"{kind}_summary.json")
        if s is not None:
            exp["summaries"][kind] = s
            exp["per_image"][kind] = {r["image"]: r for r in _load_csv(path / "eval" / f"{kind}_per_image.csv")
                                      if r.get("status") == "ok"}
    return exp


def steps_of(exp):
    return exp["meta"]["upstream_args"]["num_inference_steps"]


def seed_of(exp):
    return exp["meta"]["upstream_args"]["seed"]


def variant_of(exp):
    return exp["meta"]["config"].get("experiment", {}).get("variant", "default")


def method_of(exp):
    return json.dumps(exp["meta"]["config"].get("method") or {}, sort_keys=True)


def metric_info(exp):
    """{metric: lower_better}"""
    out = {}
    for s in exp["summaries"].values():
        for name, desc in s["metrics"].items():
            out[name] = desc["lower_better"]
    return out


def per_image_values(exp, metric):
    for kind in EVAL_KINDS:
        rows = exp["per_image"].get(kind, {})
        vals = {}
        for img, r in rows.items():
            v = r.get(metric)
            if v not in (None, "") and math.isfinite(float(v)):
                vals[img] = float(v)
        if vals:
            return vals
    return {}


# --------------------------------------------------------------------------- #
# Compatibility (track B)
# --------------------------------------------------------------------------- #
def signature(exp):
    """Every setting that must be identical across track-B runs, flattened to {field: value}."""
    m = exp["meta"]
    sig = {
        "status": m.get("status"),
        "dataset.n_images": m["dataset"].get("n_images"),
        "dataset.filenames_sha256": m["dataset"].get("filenames_sha256"),
        "dataset.lr_dir_fingerprint": m["dataset"].get("lr_dir_fingerprint"),
        "attention_backend": m.get("attention_backend"),
        "git.commit": m.get("git", {}).get("commit"),
        "git.uncommitted_files": tuple(m.get("git", {}).get("uncommitted_files") or ()),
        "environment.gpu_name": m["environment"].get("gpu_name"),
    }
    for k, v in m["upstream_args"].items():  # CFG, precision, preprocessing, tiling, colour correction, prompts...
        if k not in ALLOWED_ARG_DIFFERENCES:
            sig[f"upstream_args.{k}"] = v
    for k, v in (m["config"].get("runtime") or {}).items():  # scheduler, xformers, reseed policy
        sig[f"runtime.{k}"] = v
    for k, v in (m.get("checkpoints") or {}).items():
        sig[f"checkpoint.{k}"] = v.get("fingerprint") or (v.get("path"), v.get("size_bytes"))
    sched = m.get("scheduler") or {}
    sig["scheduler.class"] = sched.get("class")
    sig["scheduler.config"] = json.dumps(sched.get("config"), sort_keys=True)
    for k, v in (m.get("runtime_verified") or {}).items():  # actual dtypes and attention processors
        sig[f"runtime_verified.{k}"] = json.dumps(v, sort_keys=True)
    for name in ("torch", "diffusers", "transformers", "xformers", "pyiqa"):
        sig[f"package.{name}"] = m["environment"].get("packages", {}).get(name)
    for kind, s in exp["summaries"].items():
        sig[f"eval.{kind}.metrics"] = json.dumps(s["metrics"], sort_keys=True)
        sig[f"eval.{kind}.n_evaluated_ok"] = s.get("n_evaluated_ok")
        sig[f"eval.{kind}.n_errors"] = s.get("n_errors")
        sig[f"eval.{kind}.gt_dir"] = s.get("gt_dir")
        if kind == "lrc":
            sig["eval.lrc.lr_dir"] = s.get("lr_dir")
            sig["eval.lrc.scale"] = s.get("scale")
    sig["eval.kinds"] = tuple(sorted(exp["summaries"]))
    return sig


def compatibility(exps):
    """Return every difference that makes the runs incomparable (empty = compatible)."""
    problems = []
    for e in exps:
        if e["meta"].get("status") != "completed":
            problems.append(f"{e['name']}: status is {e['meta'].get('status')!r}")
        if not e["meta"]["config"].get("runtime", {}).get("reseed_per_image"):
            problems.append(f"{e['name']}: reseed_per_image is not true (track-A runs cannot be paired)")
        if not e["summaries"]:
            problems.append(f"{e['name']}: not evaluated")
    sigs = {e["name"]: signature(e) for e in exps}
    for field in sorted(set().union(*sigs.values())):
        values = {n: s.get(field) for n, s in sigs.items()}
        if len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) > 1:
            groups = defaultdict(list)
            for n, v in values.items():
                groups[json.dumps(v, sort_keys=True, default=str)].append(n)
            problems.append(f"{field} differs: " + "; ".join(f"{v} <- {sorted(ns)}" for v, ns in groups.items()))
    methods = defaultdict(set)
    for e in exps:
        methods[variant_of(e)].add(method_of(e))
    for variant, ms in methods.items():
        if len(ms) > 1:
            problems.append(f"variant {variant!r} has {len(ms)} different method configurations")
    seen = defaultdict(list)
    for e in exps:
        seen[(variant_of(e), steps_of(e), seed_of(e))].append(e["name"])
    for key, names in seen.items():
        if len(names) > 1:
            problems.append(f"several runs for variant={key[0]}, steps={key[1]}, seed={key[2]}: {names}")
    return problems


# --------------------------------------------------------------------------- #
# Statistics (track B)
# --------------------------------------------------------------------------- #
def _sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else None


def _tmean(runs, key):
    vals = [r["meta"].get("timing", {}).get(key) for r in runs.values()]
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def paired_difference(runs, ref, metric, lower_better):
    """runs, ref: {seed: exp}. Per image, Δ averaged over common seeds; summary over images."""
    common_seeds = sorted(set(runs) & set(ref))
    per_img, seed_level = defaultdict(list), []
    for sd in common_seeds:
        a, b = per_image_values(runs[sd], metric), per_image_values(ref[sd], metric)
        pair = [a[i] - b[i] for i in sorted(set(a) & set(b))]
        if pair:
            seed_level.append(statistics.fmean(pair))
        for i in set(a) & set(b):
            per_img[i].append(a[i] - b[i])
    d = [statistics.fmean(v) for v in per_img.values()]
    if not d:
        return None
    sd_d = _sd(d)
    better = sum((x < 0) if lower_better else (x > 0) for x in d)
    return {"paired_on_seeds": " ".join(map(str, common_seeds)), "n_images": len(d),
            "mean_diff": statistics.fmean(d), "sd_diff": sd_d,
            "se_diff": sd_d / math.sqrt(len(d)) if sd_d is not None else None,
            "frac_images_better": better / len(d),
            "per_seed_mean_diff": " ".join(f"{x:+.4f}" for x in seed_level),
            "sd_of_per_seed_mean_diff": _sd(seed_level)}


def sampling_tables(exps, reference_steps):
    groups = defaultdict(dict)  # (variant, steps) -> seed -> exp
    for e in exps:
        groups[(variant_of(e), steps_of(e))][seed_of(e)] = e
    variants = sorted({v for v, _ in groups}, key=lambda v: (v != REFERENCE_VARIANT, v))
    ref_variant = REFERENCE_VARIANT if REFERENCE_VARIANT in variants else variants[0]
    if (ref_variant, reference_steps) not in groups:
        raise ValueError(f"no {ref_variant} runs with {reference_steps} steps")
    lower = metric_info(exps[0])
    metrics = [m for m in METRIC_ORDER if m in lower]
    ref50 = groups[(ref_variant, reference_steps)]
    summary, diffs = [], []
    for variant in variants:
        for steps in sorted({s for v, s in groups if v == variant}, reverse=True):
            runs = groups[(variant, steps)]
            seeds = sorted(runs)
            any_run = runs[seeds[0]]
            row = {"variant": variant, "steps": steps, "seeds": " ".join(map(str, seeds)), "n_seeds": len(seeds),
                   "n_images": any_run["meta"]["dataset"]["n_images"],
                   "executed_steps": ",".join(sorted({",".join(map(str, r["meta"].get("scheduler", {}).get(
                       "denoising_steps_per_call", []))) for r in runs.values()})),
                   "lr_mean_factor": any_run["meta"].get("method", {}).get("lr_mean_factor_scheduled")}
            row.update(diffusion_s_per_image=_tmean(runs, "per_image_diffusion_s_mean_excl_first"),
                       total_s_per_image=_tmean(runs, "per_image_total_s_mean_excl_first"),
                       total_inference_s_per_run=_tmean(runs, "sum_per_image_total_s"),
                       peak_vram_allocated_mb=max((r["meta"].get("timing", {}).get("peak_vram_allocated_mb_max") or 0
                                                   for r in runs.values()), default=0) or None,
                       peak_vram_reserved_mb=max((r["meta"].get("timing", {}).get("peak_vram_reserved_mb_max") or 0
                                                  for r in runs.values()), default=0) or None,
                       step_reduction_pct=100 * (1 - steps / reference_steps))
            ref_t = _tmean(ref50, "per_image_diffusion_s_mean_excl_first")
            row["diffusion_latency_change_pct_vs_ref"] = (100 * (row["diffusion_s_per_image"] / ref_t - 1)
                                                          if row["diffusion_s_per_image"] and ref_t else None)
            for m in metrics:
                per_seed = {sd: per_image_values(runs[sd], m) for sd in seeds}
                pooled = [v for vals in per_seed.values() for v in vals.values()]
                seed_means = [statistics.fmean(v.values()) for v in per_seed.values() if v]
                row[f"{m}_mean"] = statistics.fmean(pooled) if pooled else None
                row[f"{m}_sd_between_seed_means"] = _sd(seed_means)
                images = sorted(set.intersection(*(set(v) for v in per_seed.values()))) if per_seed else []
                within = [_sd([per_seed[sd][i] for sd in seeds]) for i in images] if len(seeds) > 1 else []
                row[f"{m}_within_image_seed_sd"] = statistics.fmean(within) if within else None
                comparisons = []
                if (variant, steps) != (ref_variant, reference_steps):
                    comparisons.append((f"{ref_variant}@{reference_steps}", ref50))
                if variant != ref_variant and (ref_variant, steps) in groups and steps != reference_steps:
                    comparisons.append((f"{ref_variant}@{steps}", groups[(ref_variant, steps)]))
                for ref_label, ref_runs in comparisons:
                    d = paired_difference(runs, ref_runs, m, lower[m])
                    if d:
                        diffs.append({"variant": variant, "steps": steps, "reference": ref_label, "metric": m,
                                      "lower_better": lower[m], **d})
            summary.append(row)
    return summary, diffs, metrics, ref_variant


def _f(v, nd=4):
    if v is None or v == "":
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def sampling_markdown(dataset, summary, diffs, metrics, reference_steps, ref_variant):
    L = [f"# {dataset}: controlled sampling comparison and method ablation (track B)", "",
         f"Runs paired by (image, seed). Reference: `{ref_variant}` at {reference_steps} steps. Within a variant "
         "only steps and seed differ; between variants only the `method` section.", "",
         "## Metrics per variant and step count", "",
         "mean = over all (image, seed) pairs; sd seeds = std across seeds of the dataset mean; "
         "within-image seed sd = per-image std across seeds, averaged over images.", ""]
    head = ["variant", "steps (executed)", "seeds", "n img"] + [m.upper() for m in metrics]
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in summary:
        cells = [r["variant"], f"{r['steps']} ({r['executed_steps'] or '–'})", r["seeds"], str(r["n_images"])]
        for m in metrics:
            cells.append(f"{_f(r[f'{m}_mean'])} (sd seeds {_f(r[f'{m}_sd_between_seed_means'])}; "
                         f"within-img {_f(r[f'{m}_within_image_seed_sd'])})")
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## Runtime (first image excluded as warm-up; mean over seeds)", "",
          "| variant | steps | diffusion s/img | total s/img | total inference s/run | peak VRAM alloc / reserved MB | "
          "step −% | diffusion latency change % | mean LR factor |", "|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        L.append(f"| {r['variant']} | {r['steps']} | {_f(r['diffusion_s_per_image'], 3)} | {_f(r['total_s_per_image'], 3)} | "
                 f"{_f(r['total_inference_s_per_run'], 1)} | {_f(r['peak_vram_allocated_mb'])} / {_f(r['peak_vram_reserved_mb'])} | "
                 f"{_f(r['step_reduction_pct'], 1)} | {_f(r['diffusion_latency_change_pct_vs_ref'], 1)} | {_f(r['lr_mean_factor'], 3)} |")
    if diffs:
        L += ["", "## Paired differences", "",
              "Δ = (row) − (reference) per image, averaged over the paired seeds; then mean / sd / standard error over "
              "images. ↑ / ↓ = direction of improvement.", "",
              "| variant | steps | vs | metric | mean Δ | sd Δ | SE | images better | n img | per-seed mean Δ | sd of per-seed Δ |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in diffs:
            L.append(f"| {d['variant']} | {d['steps']} | {d['reference']} | {d['metric'].upper()} "
                     f"{'↓' if d['lower_better'] else '↑'} | {d['mean_diff']:+.4f} | {_f(d['sd_diff'])} | {_f(d['se_diff'])} | "
                     f"{100 * d['frac_images_better']:.0f}% | {d['n_images']} | {d['per_seed_mean_diff']} | "
                     f"{_f(d['sd_of_per_seed_mean_diff'])} |")
    L += ["", "With one seed, seed variation is not estimated (sd columns empty).",
          "No-reference metrics (MUSIQ, CLIPIQA) do not measure fidelity or semantic correctness; LR-consistency "
          "(LR_PSNR, LR_MAE) is a diagnostic that rewards blur and assumes bicubic degradation.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Track A
# --------------------------------------------------------------------------- #
def reproduction(exp, published):
    m = exp["meta"]
    checks = []
    if m.get("status") != "completed":
        checks.append(f"status is {m.get('status')!r}")
    if m["config"].get("runtime", {}).get("reseed_per_image"):
        checks.append("reseed_per_image is true: not upstream RNG behaviour")
    if m.get("method", {}).get("is_baseline") is False:
        checks.append("method components are enabled: not original SeeSR")
    defaults = upstream_defaults()
    for k, v in m["upstream_args"].items():
        if k in defaults and k not in BASELINE_EXPERIMENT_SPECIFIC and k not in {
                "seesr_model_path", "ram_ft_path", "pretrained_model_path", "image_path", "output_dir"} and v != defaults[k]:
            checks.append(f"upstream_args.{k}={v!r} differs from upstream default {defaults[k]!r}")
    if (m.get("runtime_verified") or {}).get("unet_dtype") not in (None, "torch.float16"):
        checks.append(f"unet dtype {m['runtime_verified']['unet_dtype']} (upstream fp16)")
    if m.get("scheduler", {}).get("class") not in (None, "DDPMScheduler"):
        checks.append(f"scheduler {m['scheduler']['class']}")
    if m.get("scheduler", {}).get("denoising_steps_per_call") not in (None, [50]):
        checks.append(f"executed steps {m['scheduler']['denoising_steps_per_call']}")
    rows = []
    for name in METRIC_ORDER:
        vals = list(per_image_values(exp, name).values())
        if not vals and (published is None or name not in published["values"]):
            continue
        rep = statistics.fmean(vals) if vals else None
        pub = published["values"].get(name) if published else None
        rows.append({"metric": name, "reproduced": rep, "published": pub,
                     "difference": rep - pub if rep is not None and pub is not None else None, "n_images": len(vals)})
    return rows, checks


def reproduction_markdown(dataset, exp_name, rows, checks, published):
    L = [f"# {dataset}: reproduction of original SeeSR (track A)", "",
         f"Run `{exp_name}` (upstream RNG and defaults) vs {published['source'] if published else 'no published values'}.",
         "A single run: differences mix implementation, library and seed effects and carry no error bar.", "",
         "| metric | reproduced | published | difference | n |", "|---|---|---|---|---|"]
    for r in rows:
        diff = "–" if r["difference"] is None else f"{r['difference']:+.4f}"
        L.append(f"| {r['metric'].upper()} | {_f(r['reproduced'])} | {_f(r['published'])} | {diff} | {r['n_images']} |")
    L += ["", "## Configuration checks", ""] + ([f"* {c}" for c in checks] or ["* matches upstream defaults (seed and save_prompts are recorded experiment settings)"])
    return "\n".join(L) + "\n"



# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def track_b_runs(root):
    """Run folders with per-image seeding (track B). Returns (names, {excluded folder: reason})."""
    names, excluded = [], {}
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or d.name == "comparison":
            continue
        if "__replaced_" in d.name:
            excluded[d.name] = "replaced by a later run"
            continue
        meta = _load_json(d / "run_metadata.json")
        if meta is None:
            excluded[d.name] = "no run_metadata.json"
        elif not meta["config"].get("runtime", {}).get("reseed_per_image"):
            excluded[d.name] = "upstream RNG (track A)"
        else:
            names.append(d.name)
    return names, excluded


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="track", required=True)
    s = sub.add_parser("sampling", help="track B: controlled step-count comparison and method ablation")
    s.add_argument("--results_root", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--experiments", nargs="*", help="default: every run folder with per-image seeding")
    s.add_argument("--reference_steps", type=int, default=50)
    s.add_argument("--output_dir")
    r = sub.add_parser("reproduction", help="track A: upstream run vs published numbers")
    r.add_argument("--results_root", required=True)
    r.add_argument("--dataset", required=True)
    r.add_argument("--experiment", default="baseline_ddpm50")
    r.add_argument("--published", default=str(Path(__file__).resolve().parents[1] / "configs" / "published_seesr.yaml"))
    r.add_argument("--output_dir")
    a = p.parse_args(argv)
    root = Path(a.results_root)
    out = Path(a.output_dir) if a.output_dir else root / "comparison"

    if a.track == "reproduction":
        exp = load_experiment(root / a.experiment)
        pub = load_yaml(a.published)
        published = {"source": pub.get("source"), "values": pub["datasets"][a.dataset]} if a.dataset in pub.get("datasets", {}) else None
        rows, checks = reproduction(exp, published)
        write_json(out / "reproduction.json", {"created_at": now_iso(), "dataset": a.dataset, "experiment": a.experiment,
                                                "published": published, "rows": rows, "checks": checks})
        md = reproduction_markdown(a.dataset, a.experiment, rows, checks, published)
        (out / "reproduction.md").write_text(md, encoding="utf-8")
        print(md)
        return 2 if checks else 0

    if a.experiments:
        names, excluded = a.experiments, {}
    else:
        names, excluded = track_b_runs(root)
        for n, why in excluded.items():
            print(f"not compared: {n} ({why})", file=sys.stderr)
    if not names:
        print(f"ERROR: no track-B runs under {root}", file=sys.stderr)
        return 2
    exps = [load_experiment(root / n) for n in names]
    problems = compatibility(exps)
    report = {"created_at": now_iso(), "dataset": a.dataset, "experiments": names, "excluded": excluded,
              "compatible": not problems, "differences": problems}
    if problems:
        write_json(out / "sampling_comparison.json", report)
        print("REJECTED: runs are not comparable\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    summary, diffs, metrics, ref_variant = sampling_tables(exps, a.reference_steps)
    write_csv(out / "sampling_summary.csv", summary)
    write_csv(out / "sampling_paired_differences.csv", diffs)
    write_json(out / "sampling_comparison.json", {
        **report, "reference": {"variant": ref_variant, "steps": a.reference_steps},
        "methods": {variant_of(e): json.loads(method_of(e)) for e in exps},
        "summary": summary, "paired_differences": diffs})
    md = sampling_markdown(a.dataset, summary, diffs, metrics, a.reference_steps, ref_variant)
    (out / "sampling_comparison.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
