"""Evaluate super-resolved images with paired and/or no-reference metrics.

Paired (needs ground truth):   PSNR, SSIM, LPIPS, DISTS
No-reference (SR images only): MUSIQ, CLIPIQA
LR consistency (diagnostic):   PSNR / MAE of bicubic-downsampled SR vs the LR input (--lrc)

Outputs (in --output_dir):
  paired_per_image.csv, paired_summary.json   (with --paired)
  noref_per_image.csv,  noref_summary.json    (with --noref)

Exit status: 0 = all requested images evaluated; 2 = validation problems
(unmatched files without --allow_missing, duplicate keys, size mismatch,
unreadable image, per-image metric failure); 3 = a metric backend (pyiqa or
its weights) could not be loaded. Results that could be computed are still
written, and every problem is listed in the summary; nothing is skipped silently.

Examples:
  python experiments/evaluation/evaluate.py --paired --noref \
      --sr_dir experiments/results/baseline_ddpm50/sr/sample00 \
      --gt_dir preset/datasets/RealSR/test_HR \
      --output_dir experiments/results/baseline_ddpm50/eval
  python experiments/evaluation/evaluate.py --noref \
      --sr_dir experiments/results/reallr200_ddpm50/sr/sample00 \
      --output_dir experiments/results/reallr200_ddpm50/eval
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import git_info, list_images, load_yaml, now_iso, package_versions, write_csv, write_json  # noqa: E402
from experiments.evaluation.metrics import (  # noqa: E402
    NOREF_DISCLAIMER, NOREF_METRICS, PAIRED_METRICS, MetricUnavailable, build_metrics, describe, pyiqa_factory,
)


# --------------------------------------------------------------------------- #
# File matching and image loading (no torch needed)
# --------------------------------------------------------------------------- #
def match_key(path, strip_suffixes=()):
    stem = Path(path).stem
    for suffix in sorted(strip_suffixes, key=len, reverse=True):
        if suffix and stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def index_by_key(files, strip_suffixes):
    index, duplicates = {}, {}
    for f in files:
        key = match_key(f, strip_suffixes)
        if key in index:
            duplicates.setdefault(key, [index[key].name]).append(f.name)
        else:
            index[key] = f
    return index, duplicates


def match_pairs(sr_files, gt_files, strip_suffixes=()):
    sr_idx, sr_dup = index_by_key(sr_files, strip_suffixes)
    gt_idx, gt_dup = index_by_key(gt_files, strip_suffixes)
    keys = sorted(set(sr_idx) & set(gt_idx))
    return {
        "pairs": [(k, sr_idx[k], gt_idx[k]) for k in keys],
        "missing_gt": sorted(sr_idx[k].name for k in set(sr_idx) - set(gt_idx)),
        "missing_sr": sorted(gt_idx[k].name for k in set(gt_idx) - set(sr_idx)),
        "duplicates": {"sr": sr_dup, "gt": gt_dup},
    }


def load_rgb(path):
    """Load an image as uint8 RGB numpy array. Returns (array, note)."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        mode = im.mode
        if mode == "RGB":
            note = ""
        elif mode in ("L", "RGBA", "P", "LA", "CMYK", "I;16", "I"):
            note = f"converted {mode}->RGB"
        else:
            raise ValueError(f"unsupported image mode {mode}")
        if mode in ("I;16", "I"):
            raise ValueError(f"high bit-depth mode {mode} is not supported; convert to 8-bit first")
        arr = np.asarray(im.convert("RGB"))
    return arr, note


def to_tensor(arr, device):
    import torch
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(rows, metric_names):
    out = {}
    for name in metric_names:
        vals = [r[name] for r in rows if isinstance(r.get(name), float) and math.isfinite(r[name])]
        if not vals:
            out[name] = {"n": 0, "mean": None, "std": None, "min": None, "max": None}
            continue
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))  # population std (ddof=0)
        out[name] = {"n": len(vals), "mean": mean, "std": std, "min": min(vals), "max": max(vals)}
    return out


def evaluate_items(items, metrics, device):
    """items: list of (key, sr_path, gt_path_or_None). Returns (rows, problems)."""
    rows, problems = [], []
    for key, sr_path, gt_path in items:
        row = {"image": key, "sr_file": sr_path.name}
        if gt_path is not None:
            row["gt_file"] = gt_path.name
        try:
            sr, note_sr = load_rgb(sr_path)
            notes = [f"sr {note_sr}"] if note_sr else []
            row["height"], row["width"] = sr.shape[0], sr.shape[1]
            gt_t = None
            if gt_path is not None:
                gt, note_gt = load_rgb(gt_path)
                if note_gt:
                    notes.append(f"gt {note_gt}")
                if gt.shape != sr.shape:
                    raise ValueError(f"size mismatch: SR {sr.shape[1]}x{sr.shape[0]} vs GT {gt.shape[1]}x{gt.shape[0]}")
                gt_t = to_tensor(gt, device)
            sr_t = to_tensor(sr, device)
            for name, fn in metrics.items():
                row[name] = fn(sr_t, gt_t) if gt_path is not None else fn(sr_t)
            row["status"] = "ok"
            row["notes"] = "; ".join(notes)
        except Exception as e:  # recorded per image, reported, and reflected in the exit code
            row["status"] = "error"
            row["notes"] = f"{type(e).__name__}: {e}"
            problems.append(f"{key}: {row['notes']}")
        rows.append(row)
    return rows, problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sr_dir", required=True, help="directory of super-resolved images")
    p.add_argument("--gt_dir", help="directory of HR ground-truth images (required for --paired)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--paired", action="store_true", help="compute paired metrics")
    p.add_argument("--noref", action="store_true", help="compute no-reference metrics")
    p.add_argument("--lrc", action="store_true",
                   help="LR-consistency diagnostic (needs --lr_dir); see evaluation/lr_consistency.py")
    p.add_argument("--lr_dir", help="directory of the LR inputs (for --lrc)")
    p.add_argument("--scale", type=int, default=4, help="SR scale factor for --lrc (default 4)")
    p.add_argument("--paired_metrics", nargs="+", default=None, help=f"default: config or {list(PAIRED_METRICS)}")
    p.add_argument("--noref_metrics", nargs="+", default=None, help=f"default: config or {list(NOREF_METRICS)}")
    p.add_argument("--config", help="experiment YAML; reads its `evaluation` and `dataset.strip_suffixes` sections")
    p.add_argument("--strip_suffix", action="append", default=None,
                   help="suffix removed from filename stems before matching, e.g. _HR (repeatable)")
    y = p.add_mutually_exclusive_group()
    y.add_argument("--y_channel", dest="y_channel", action="store_true", default=None, help="PSNR/SSIM on Y (default)")
    y.add_argument("--rgb", dest="y_channel", action="store_false", help="PSNR/SSIM on RGB")
    p.add_argument("--crop_border", type=int, default=None, help="pixels cropped from each border for PSNR/SSIM (default 0)")
    p.add_argument("--allow_missing", action="store_true",
                   help="evaluate the matched subset even if some files have no counterpart (still reported)")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--overwrite", action="store_true", help="replace existing summaries in --output_dir")
    return p.parse_args(argv)


def resolve_settings(opts):
    cfg_eval, cfg_strip = {}, []
    if opts.config:
        cfg = load_yaml(opts.config)
        cfg_eval = cfg.get("evaluation", {}) or {}
        cfg_strip = (cfg.get("dataset", {}) or {}).get("strip_suffixes", []) or []
    return {
        "paired_metrics": opts.paired_metrics or cfg_eval.get("paired_metrics") or list(PAIRED_METRICS),
        "noref_metrics": opts.noref_metrics or cfg_eval.get("noref_metrics") or list(NOREF_METRICS),
        "strip_suffixes": opts.strip_suffix if opts.strip_suffix is not None else cfg_strip,
        "y_channel": opts.y_channel if opts.y_channel is not None else bool(cfg_eval.get("y_channel", True)),
        "crop_border": opts.crop_border if opts.crop_border is not None else int(cfg_eval.get("crop_border", 0)),
    }


def run(opts, factory=pyiqa_factory):
    if not (opts.paired or opts.noref or opts.lrc):
        raise SystemExit("ERROR: choose --paired, --noref and/or --lrc")
    if opts.paired and not opts.gt_dir:
        raise SystemExit("ERROR: --paired requires --gt_dir")
    if opts.lrc and not opts.lr_dir:
        raise SystemExit("ERROR: --lrc requires --lr_dir")
    if opts.lr_dir and not Path(opts.lr_dir).is_dir():
        raise SystemExit(f"ERROR: --lr_dir not found: {opts.lr_dir}")
    settings = resolve_settings(opts)
    sr_dir = Path(opts.sr_dir)
    out_dir = Path(opts.output_dir)
    if not sr_dir.is_dir():
        raise SystemExit(f"ERROR: --sr_dir not found: {sr_dir}")
    if opts.gt_dir and not Path(opts.gt_dir).is_dir():
        raise SystemExit(f"ERROR: --gt_dir not found: {opts.gt_dir}")

    targets = (["paired"] if opts.paired else []) + (["noref"] if opts.noref else []) + (["lrc"] if opts.lrc else [])
    if not opts.overwrite:
        existing = [t for t in targets if (out_dir / f"{t}_summary.json").exists()]
        if existing:
            raise SystemExit(f"ERROR: {existing} summaries already exist in {out_dir}; use --overwrite")

    device = opts.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    sr_files = list_images(sr_dir)
    common_meta = {
        "created_at": now_iso(),
        "git": git_info(),
        "packages": package_versions(),
        "device": device,
        "sr_dir": str(sr_dir.resolve()),
        "n_sr_files": len(sr_files),
        "strip_suffixes": settings["strip_suffixes"],
        "std_definition": "population standard deviation (ddof=0) over successfully evaluated images",
    }
    exit_code = 0

    if opts.paired:
        gt_files = list_images(opts.gt_dir)
        m = match_pairs(sr_files, gt_files, settings["strip_suffixes"])
        problems, warnings = [], []
        if m["duplicates"]["sr"] or m["duplicates"]["gt"]:
            problems.append(f"duplicate match keys: {m['duplicates']}")
        if not m["pairs"]:
            problems.append("no matching SR/GT pairs")
        unmatched = []
        if m["missing_gt"]:
            unmatched.append(f"{len(m['missing_gt'])} SR file(s) without GT")
        if m["missing_sr"]:
            unmatched.append(f"{len(m['missing_sr'])} GT file(s) without SR")
        # Unmatched files block evaluation unless --allow_missing; either way they are listed.
        (warnings if opts.allow_missing else problems).extend(unmatched)
        blocking = bool(problems)
        rows, img_problems = [], []
        if not blocking:
            metrics = build_metrics(settings["paired_metrics"], PAIRED_METRICS, device,
                                    settings["y_channel"], settings["crop_border"], factory)
            rows, img_problems = evaluate_items(m["pairs"], metrics, device)
        problems += img_problems
        summary = {
            **common_meta,
            "mode": "paired",
            "gt_dir": str(Path(opts.gt_dir).resolve()),
            "n_gt_files": len(gt_files),
            "n_pairs_matched": len(m["pairs"]),
            "n_evaluated_ok": sum(r["status"] == "ok" for r in rows),
            "n_errors": sum(r["status"] == "error" for r in rows),
            "evaluated": not blocking,
            "missing_gt": m["missing_gt"],
            "missing_sr": m["missing_sr"],
            "duplicates": m["duplicates"],
            "problems": problems,
            "warnings": warnings,
            "metrics": describe(settings["paired_metrics"], PAIRED_METRICS, settings["y_channel"], settings["crop_border"]),
            "aggregate": aggregate([r for r in rows if r["status"] == "ok"], settings["paired_metrics"]),
        }
        write_csv(out_dir / "paired_per_image.csv", rows,
                  ["image", "sr_file", "gt_file", "height", "width", *settings["paired_metrics"], "status", "notes"])
        write_json(out_dir / "paired_summary.json", summary)
        _report("paired", summary, problems + [f"(warning) {w}" for w in warnings])
        exit_code = 2 if problems else exit_code

    if opts.noref:
        items = [(f.stem, f, None) for f in sr_files]
        problems = [] if items else ["no SR images found"]
        rows = []
        if items:
            metrics = build_metrics(settings["noref_metrics"], NOREF_METRICS, device, True, 0, factory)
            rows, img_problems = evaluate_items(items, metrics, device)
            problems += img_problems
        summary = {
            **common_meta,
            "mode": "noref",
            "disclaimer": NOREF_DISCLAIMER,
            "n_evaluated_ok": sum(r["status"] == "ok" for r in rows),
            "n_errors": sum(r["status"] == "error" for r in rows),
            "problems": problems,
            "metrics": describe(settings["noref_metrics"], NOREF_METRICS, True, 0),
            "aggregate": aggregate([r for r in rows if r["status"] == "ok"], settings["noref_metrics"]),
        }
        write_csv(out_dir / "noref_per_image.csv", rows,
                  ["image", "sr_file", "height", "width", *settings["noref_metrics"], "status", "notes"])
        write_json(out_dir / "noref_summary.json", summary)
        _report("noref", summary, problems)
        exit_code = 2 if problems else exit_code

    if opts.lrc:
        code = run_lr_consistency(opts, settings, sr_files, out_dir, common_meta)
        exit_code = code or exit_code
    return exit_code


def run_lr_consistency(opts, settings, sr_files, out_dir, common_meta):
    """LR-consistency diagnostic; separate outputs lrc_per_image.csv / lrc_summary.json."""
    from experiments.evaluation.lr_consistency import LRC_DISCLAIMER, LRC_METRICS, lr_consistency
    lr_files = list_images(opts.lr_dir)
    m = match_pairs(sr_files, lr_files, settings["strip_suffixes"])
    problems = []
    if m["missing_gt"]:
        problems.append(f"{len(m['missing_gt'])} SR file(s) without LR input")
    if m["missing_sr"]:
        problems.append(f"{len(m['missing_sr'])} LR input(s) without SR output")
    if m["duplicates"]["sr"] or m["duplicates"]["gt"]:
        problems.append(f"duplicate match keys: {m['duplicates']}")
    rows = []
    for key, sr_path, lr_path in m["pairs"]:
        row = {"image": key, "sr_file": sr_path.name, "lr_file": lr_path.name}
        try:
            sr, _ = load_rgb(sr_path)
            lr, _ = load_rgb(lr_path)
            row.update(lr_consistency(sr, lr, opts.scale), status="ok", notes="")
        except Exception as e:
            row.update(status="error", notes=f"{type(e).__name__}: {e}")
            problems.append(f"{key}: {row['notes']}")
        rows.append(row)
    names = list(LRC_METRICS)
    summary = {
        **common_meta, "mode": "lrc", "disclaimer": LRC_DISCLAIMER,
        "lr_dir": str(Path(opts.lr_dir).resolve()), "scale": opts.scale,
        "n_evaluated_ok": sum(r["status"] == "ok" for r in rows),
        "n_errors": sum(r["status"] == "error" for r in rows),
        "problems": problems,
        "metrics": {n: {"lower_better": d["lower_better"], "definition": d["definition"]} for n, d in LRC_METRICS.items()},
        "aggregate": aggregate([r for r in rows if r["status"] == "ok"], names),
    }
    write_csv(out_dir / "lrc_per_image.csv", rows, ["image", "sr_file", "lr_file", *names, "status", "notes"])
    write_json(out_dir / "lrc_summary.json", summary)
    _report("lrc", summary, problems)
    return 2 if problems else 0


def _report(mode, summary, problems):
    print(f"[{mode}] evaluated {summary['n_evaluated_ok']} image(s), errors: {summary['n_errors']}")
    for name, agg in summary["aggregate"].items():
        if agg["n"]:
            print(f"    {name:8s} mean={agg['mean']:.4f} std={agg['std']:.4f} (n={agg['n']})")
    for p in problems:
        print(f"    PROBLEM: {p}")


def main(argv=None):
    try:
        return run(parse_args(argv))
    except MetricUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
