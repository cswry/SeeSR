"""Validate checkpoints and datasets before spending GPU time. Reads files only; never modifies them.

  python experiments/tools/validate_inputs.py checkpoints --config experiments/configs/baseline.yaml
  python experiments/tools/validate_inputs.py dataset --name RealSR --lr_dir <LR> --gt_dir <HR> --scale 4 \
      --output experiments/results/RealSR/dataset_report.json
  python experiments/tools/validate_inputs.py dataset --name RealLR200 --lr_dir <LR>   # unpaired

Exit status: 0 = no problems, 2 = problems found (listed in the output and report).
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import IMAGE_EXTENSIONS, list_images, load_yaml, now_iso, resolve_path, write_json  # noqa: E402
from experiments.evaluation.evaluate import match_pairs  # noqa: E402

WEIGHT_NAMES = ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin",
                "model.safetensors", "pytorch_model.bin")
LFS_POINTER = b"version https://git-lfs"


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def _check_file(path, problems, min_bytes=1):
    p = Path(path)
    if not p.is_file():
        problems.append(f"missing: {p}")
        return None
    size = p.stat().st_size
    with open(p, "rb") as f:
        head = f.read(len(LFS_POINTER))
    if head == LFS_POINTER:
        problems.append(f"git-lfs pointer, not the real file (run `git lfs pull`): {p}")
    elif size < min_bytes:
        problems.append(f"suspiciously small ({size} bytes): {p}")
    return size


def _check_component(folder, problems, need_weights=True):
    folder = Path(folder)
    info = {"path": str(folder)}
    cfg_path = folder / "config.json"
    if _check_file(cfg_path, problems) is not None:
        info["config"] = json.loads(cfg_path.read_text(encoding="utf-8"))
    if need_weights:
        found = [n for n in WEIGHT_NAMES if (folder / n).is_file()]
        if not found:
            problems.append(f"no weight file ({' / '.join(WEIGHT_NAMES)}) in {folder}")
        else:
            info["weights"] = {n: _check_file(folder / n, problems, min_bytes=1_000_000) for n in found}
    return info


def check_checkpoints(models):
    problems, report = [], {}
    sd = resolve_path(models["pretrained_model_path"])
    sched_path = sd / "scheduler" / "scheduler_config.json"
    if _check_file(sched_path, problems) is not None:
        sched = json.loads(sched_path.read_text(encoding="utf-8"))
        report["scheduler_config"] = sched
        if sched.get("prediction_type", "epsilon") != "epsilon":
            problems.append(f"SD-2-base scheduler prediction_type={sched.get('prediction_type')} (expected epsilon)")
        report["ddpm_timesteps"] = ddpm_timesteps(sched)
    for name in ("vocab.json", "merges.txt"):
        _check_file(sd / "tokenizer" / name, problems)
    _check_file(sd / "feature_extractor" / "preprocessor_config.json", problems)
    report["text_encoder"] = _check_component(sd / "text_encoder", problems)
    report["vae"] = _check_component(sd / "vae", problems)

    seesr = resolve_path(models["seesr_model_path"])
    for part in ("unet", "controlnet"):
        info = _check_component(seesr / part, problems)
        flag = info.get("config", {}).get("use_image_cross_attention")
        info["use_image_cross_attention"] = flag
        if "config" in info and flag is not True:
            problems.append(f"seesr/{part}/config.json: use_image_cross_attention={flag!r} (SeeSR soft prompts need true)")
        info.pop("config", None)
        report[f"seesr_{part}"] = info

    for key in ("ram_ft_path", "ram_pretrained_path"):
        path = resolve_path(models[key])
        report[key] = {"path": str(path), "size_bytes": _check_file(path, problems, min_bytes=1_000_000)}
    for key in ("text_encoder", "vae"):
        report[key].pop("config", None)
    return report, problems


def ddpm_timesteps(sched_config, steps=(50, 30, 20)):
    """Timesteps DDPMScheduler would use with this config (as test_seesr.py loads it)."""
    try:
        from diffusers import DDPMScheduler
    except ImportError:
        return "diffusers not importable; skipped"
    s = DDPMScheduler.from_config(sched_config)
    out = {}
    for n in steps:
        s.set_timesteps(n)
        t = [int(x) for x in s.timesteps]
        out[str(n)] = {"count": len(t), "first": t[0], "last": t[-1], "executed_with_start_steps_999": sum(x <= 999 for x in t)}
    return out


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def _image_info(path):
    from PIL import Image
    with Image.open(path) as im:
        return {"width": im.width, "height": im.height, "mode": im.mode}


def _content_duplicates(files):
    by_hash = {}
    for f in files:
        by_hash.setdefault(hashlib.sha256(Path(f).read_bytes()).hexdigest(), []).append(Path(f).name)
    return [names for names in by_hash.values() if len(names) > 1]


def check_dataset(lr_dir, gt_dir=None, scale=4, strip_suffixes=(), process_size=512):
    problems, warnings = [], []
    lr_dir = Path(lr_dir)
    report = {"lr_dir": str(lr_dir.resolve()), "gt_dir": str(Path(gt_dir).resolve()) if gt_dir else None,
              "paired": gt_dir is not None, "scale": scale, "strip_suffixes": list(strip_suffixes)}

    def scan(folder, label):
        if not folder.is_dir():
            problems.append(f"{label} directory not found: {folder}")
            return []
        files = list_images(folder)
        others = sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() not in IMAGE_EXTENSIONS)
        subdirs = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if others:
            msg = f"{label}: non-image files {others[:5]}{' ...' if len(others) > 5 else ''}"
            # upstream globs '<lr_dir>/*.*' and opens every match as an image
            (problems if label == "LR" and any("." in o for o in others) else warnings).append(msg)
        if subdirs:
            warnings.append(f"{label}: subdirectories are ignored: {subdirs[:5]}")
        if not files:
            problems.append(f"{label}: no images in {folder}")
        dup = _content_duplicates(files)
        if dup:
            problems.append(f"{label}: byte-identical images under different names: {dup[:5]}")
        return files

    lr_files = scan(lr_dir, "LR")
    infos = {}
    for f in lr_files:
        try:
            infos[f.name] = _image_info(f)
        except Exception as e:
            problems.append(f"LR unreadable {f.name}: {type(e).__name__}: {e}")
    report["n_lr"] = len(lr_files)
    report["lr_sizes"] = dict(Counter(f"{i['width']}x{i['height']}" for i in infos.values()).most_common())
    report["lr_modes"] = dict(Counter(i["mode"] for i in infos.values()))
    min_side = process_size // scale
    small = sorted(n for n, i in infos.items() if min(i["width"], i["height"]) < min_side)
    report["n_lr_upsampled_by_upstream"] = len(small)
    if small:
        warnings.append(f"{len(small)} LR image(s) have a side < {min_side}px; upstream test_seesr.main upsamples "
                        "them before SR and resizes the output back to input size x scale")

    if gt_dir is not None:
        gt_files = scan(Path(gt_dir), "GT")
        m = match_pairs(lr_files, gt_files, strip_suffixes)
        report.update(n_gt=len(gt_files), n_pairs=len(m["pairs"]), missing_gt=m["missing_gt"],
                      missing_lr=m["missing_sr"], duplicate_keys=m["duplicates"])
        if m["missing_gt"]:
            problems.append(f"{len(m['missing_gt'])} LR image(s) without GT, e.g. {m['missing_gt'][:3]}")
        if m["missing_sr"]:
            problems.append(f"{len(m['missing_sr'])} GT image(s) without LR, e.g. {m['missing_sr'][:3]}")
        if m["duplicates"]["sr"] or m["duplicates"]["gt"]:
            problems.append(f"duplicate filename keys after suffix stripping: {m['duplicates']}")
        bad, gt_sizes = [], Counter()
        for key, lr, gt in m["pairs"]:
            if lr.name not in infos:
                continue
            try:
                g = _image_info(gt)
            except Exception as e:
                problems.append(f"GT unreadable {gt.name}: {type(e).__name__}: {e}")
                continue
            gt_sizes[f"{g['width']}x{g['height']}"] += 1
            l = infos[lr.name]
            if (g["width"], g["height"]) != (l["width"] * scale, l["height"] * scale):
                bad.append(f"{key}: LR {l['width']}x{l['height']} x{scale} != GT {g['width']}x{g['height']}")
        report["gt_sizes"] = dict(gt_sizes.most_common())
        report["size_mismatches"] = bad
        if bad:
            problems.append(f"{len(bad)} pair(s) where GT size != LR size x {scale}, e.g. {bad[:3]}")
    report["problems"], report["warnings"] = problems, warnings
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checkpoints", help="check model weights listed in a config")
    c.add_argument("--config", default="experiments/configs/baseline.yaml")
    c.add_argument("--output", help="write the report as JSON")
    d = sub.add_parser("dataset", help="check an LR folder and optional HR folder")
    d.add_argument("--name", required=True)
    d.add_argument("--lr_dir", required=True)
    d.add_argument("--gt_dir")
    d.add_argument("--scale", type=int, default=4)
    d.add_argument("--process_size", type=int, default=512)
    d.add_argument("--strip_suffix", action="append", default=[])
    d.add_argument("--output", help="write the report as JSON")
    a = p.parse_args(argv)

    if a.cmd == "checkpoints":
        report, problems = check_checkpoints(load_yaml(a.config)["models"])
        report = {"checked_at": now_iso(), "config": a.config, **report, "problems": problems}
        print(json.dumps({k: report[k] for k in ("ddpm_timesteps",) if k in report}, indent=2))
    else:
        report = {"checked_at": now_iso(), "name": a.name,
                  **check_dataset(a.lr_dir, a.gt_dir, a.scale, a.strip_suffix, a.process_size)}
        problems = report["problems"]
        keys = ("n_lr", "n_gt", "n_pairs", "lr_sizes", "gt_sizes", "lr_modes", "n_lr_upsampled_by_upstream")
        print(json.dumps({k: report[k] for k in keys if k in report}, indent=2))
        for w in report["warnings"]:
            print(f"WARNING: {w}")
    for prob in problems:
        print(f"PROBLEM: {prob}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    if a.output:
        write_json(a.output, report)
    return 2 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
