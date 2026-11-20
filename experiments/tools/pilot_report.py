"""Check, sanitise and plot the single-image V2.1 pilot (Canon_001_LR4) from its saved artifacts.

Expected layout (copied from the Colab results on Drive; see assets/pilots/canon_001/README.md):

    assets/pilots/canon_001/<variant>/paired_per_image.csv     evaluate.py --paired output (one row)
    assets/pilots/canon_001/<variant>/run_metadata.json         optional: run_inference.py metadata
    assets/pilots/canon_001/<variant>/sr.png                     optional: the SR output
    variants: original, heuristic, learned

  python experiments/tools/pilot_report.py check     # every CSV present, one image, finite metrics; the tables in
                                                     # README.md and docs/results.md match the CSVs to 4 decimals;
                                                     # metadata has no Drive paths; prints each run's settings
  python experiments/tools/pilot_report.py sanitize  # replace absolute Colab/Drive paths in run_metadata.json
  python experiments/tools/pilot_report.py figures [--lr LR.png --gt GT.png]
                                                     # assets/figures/pilot_metrics.svg (from the CSVs) and, from
                                                     # the SR images (+ optional LR/GT), assets/comparisons/canon_001.png

Nothing here computes metrics: values come from the CSVs written by experiments/evaluation/evaluate.py.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
PILOT = REPO / "assets" / "pilots" / "canon_001"
VARIANTS = (("original", "Original SeeSR"), ("heuristic", "Heuristic SFR-OC"), ("learned", "Learned LORNet"))
METRICS = (("psnr", "PSNR", True), ("ssim", "SSIM", True), ("lpips", "LPIPS", False), ("dists", "DISTS", False))
PRIVATE_PATH = re.compile(r"/content/drive/[^\"']*|/content/[^\"']*|[A-Za-z]:\\\\[^\"']*|/home/[^\"']*|/Users/[^\"']*")


def _rel(p):
    try:
        return Path(p).resolve().relative_to(REPO)
    except ValueError:
        return Path(p)


def read_metrics(pilot=PILOT):
    out, problems = {}, []
    for key, _ in VARIANTS:
        f = pilot / key / "paired_per_image.csv"
        if not f.exists():
            problems.append(f"missing {_rel(f)}")
            continue
        rows = [r for r in csv.DictReader(f.open(newline="", encoding="utf-8")) if r.get("status", "ok") in ("ok", "")]
        if len(rows) != 1:
            problems.append(f"{_rel(f)}: expected one evaluated image, found {len(rows)}")
            continue
        vals = {}
        for m, _, _ in METRICS:
            try:
                vals[m] = float(rows[0][m])
            except (KeyError, ValueError):
                problems.append(f"{_rel(f)}: no numeric '{m}'")
        if not all(math.isfinite(v) for v in vals.values()):
            problems.append(f"{_rel(f)}: non-finite metric")
        out[key] = {"image": rows[0].get("image"), **vals}
    images = {v["image"] for v in out.values()}
    if len(images) > 1:
        problems.append(f"variants evaluated different images: {sorted(images)}")
    return out, problems


def readme_table(text):
    """{variant: {metric: value}} from the README pilot table (rows 'PSNR ↑ | a | b | c')."""
    table = {}
    for m, label, _ in METRICS:
        num = r"\s*\**([\d.]+)\**\s*\|"              # a cell, optionally **bold**
        hit = re.search(rf"^\|\s*{label}[^|]*\|{num}{num}{num}", text, re.M)
        if hit:
            for (key, _), v in zip(VARIANTS, hit.groups()):
                table.setdefault(key, {})[m] = float(v)
    return table


def cmd_check(pilot=PILOT, docs=(REPO / "README.md", REPO / "docs" / "results.md")):
    vals, problems = read_metrics(pilot)
    for doc in docs:
        if not (vals and doc.exists()):
            continue
        table = readme_table(doc.read_text(encoding="utf-8"))
        for key in vals:
            for m, label, _ in METRICS:
                if m in vals[key] and round(vals[key][m], 4) != table.get(key, {}).get(m):
                    problems.append(f"{_rel(doc)} {label} for {key}: {table.get(key, {}).get(m)} != CSV "
                                    f"{vals[key][m]:.4f}")
    for f in sorted(pilot.glob("*/*.json")):
        hits = sorted(set(PRIVATE_PATH.findall(f.read_text(encoding="utf-8"))))
        if hits:
            problems.append(f"{_rel(f)}: absolute private paths (run 'sanitize'): {hits[:3]}")
    for meta in pilot.glob("*/run_metadata.json"):
        data = json.loads(meta.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        lr_dir = str(cfg.get("dataset", {}).get("lr_dir", ""))
        inf, sfr = cfg.get("inference", {}), data.get("method", {}).get("sfr_oc") or {}
        v1 = [k for k in ("semantic_confidence", "lr_guidance", "spatial_semantic")
              if (data.get("method", {}).get(k) or {}).get("enabled")]
        print(f"{meta.parent.name}: version {data.get('method', {}).get('version')} "
              f"signature {data.get('method', {}).get('signature')} | steps {inf.get('num_inference_steps')} "
              f"seed {inf.get('seed')} cfg {inf.get('guidance_scale')} | V1 {v1 or 'off'} | sfr_oc enabled {sfr.get('enabled')} "
              f"strength {sfr.get('strength')} reliability {sfr.get('reliability')} map "
              f"{sfr.get('fusion', {}).get('map')} reference {sfr.get('reference')} | lr_dir {lr_dir}")
        if re.search(r"(^|/)Train(/|$)", lr_dir):
            problems.append(f"{_rel(meta)}: input from a training split ({lr_dir}); LORNet was trained "
                            "on RealSR V3 Train: the learned pilot would not be held out")
    for p in problems:
        print("PROBLEM:", p)
    print("pilot check:", "OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


def cmd_sanitize(pilot=PILOT):
    for meta in sorted(pilot.glob("*/*.json")):
        text = meta.read_text(encoding="utf-8")
        new = re.sub(r"/content/drive/MyDrive/seesr", "<DRIVE_ROOT>", text)
        new = re.sub(r"/content/drive/[^\"']*?/seesr", "<DRIVE_ROOT>", new)
        new = new.replace("/content/SeeSR", "<REPO>")
        if new != text:
            meta.write_text(new, encoding="utf-8")
            print("sanitised", _rel(meta))
    return 0


def metrics_svg(vals):
    """Small-multiple bar chart (one panel per metric), plain SVG, no plotting dependency."""
    colors = {"original": "#8a8f98", "heuristic": "#d9822b", "learned": "#2e8b57"}
    W, H, pw, ph, top = 880, 300, 200, 190, 60
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             'font-family="Helvetica, Arial, sans-serif">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="26" font-size="16" font-weight="bold">Single-image pilot, Canon_001_LR4 (512 × 512)'
             '</text>',
             '<text x="20" y="44" font-size="12" fill="#555">One image: not a benchmark result. '
             'Arrow: direction of better.</text>']
    for i, (m, label, higher) in enumerate(METRICS):
        x0 = 20 + i * (pw + 15)
        series = [vals[k][m] for k, _ in VARIANTS if k in vals]
        lo, hi = min(series), max(series)
        pad = (hi - lo) * 0.6 or abs(hi) * 0.05 or 1.0
        lo, hi = lo - pad, hi + pad * 0.4
        parts.append(f'<text x="{x0 + pw / 2}" y="{top + 8}" text-anchor="middle" font-size="13" '
                     f'font-weight="bold">{label} {"↑" if higher else "↓"}</text>')
        base = top + ph
        bw = 44
        for j, (k, name) in enumerate(VARIANTS):
            if k not in vals:
                continue
            v = vals[k][m]
            h = (v - lo) / (hi - lo) * (ph - 40)
            x = x0 + 20 + j * (bw + 14)
            parts.append(f'<rect x="{x}" y="{base - h:.1f}" width="{bw}" height="{h:.1f}" fill="{colors[k]}"/>')
            parts.append(f'<text x="{x + bw / 2}" y="{base - h - 4:.1f}" text-anchor="middle" font-size="10.5">'
                         f'{v:.4f}</text>')
        parts.append(f'<line x1="{x0 + 10}" y1="{base}" x2="{x0 + pw - 10}" y2="{base}" stroke="#333"/>')
        parts.append(f'<text x="{x0 + pw / 2}" y="{base + 14}" text-anchor="middle" font-size="9.5" fill="#777">'
                     f'axis from {lo:.3f} (not zero)</text>')
    for j, (k, name) in enumerate(VARIANTS):
        x = 20 + j * 190
        parts.append(f'<rect x="{x}" y="{H - 26}" width="12" height="12" fill="{colors[k]}"/>')
        parts.append(f'<text x="{x + 18}" y="{H - 16}" font-size="12">{name}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def comparison_png(pilot, lr_path, gt_path, out, crop=192, origin=None):
    """[LR (bicubic x4) |] Original | Heuristic | Learned [| GT]: full images (top) and a centre crop (bottom).
    LR and GT are optional (they are dataset images and not committed)."""
    from PIL import Image, ImageDraw
    srs = {k: pilot / k / "sr.png" for k, _ in VARIANTS}
    missing = [str(_rel(p)) for p in srs.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"SR images missing: {missing}")
    ref = Image.open(srs["original"]).convert("RGB")
    W, H = ref.size
    tiles = []
    if lr_path:
        tiles.append(("LR (bicubic x4)", Image.open(lr_path).convert("RGB").resize((W, H), Image.BICUBIC)))
    tiles += [(name, Image.open(srs[k]).convert("RGB").resize((W, H))) for k, name in VARIANTS]
    if gt_path:
        tiles.append(("GT", Image.open(gt_path).convert("RGB").resize((W, H))))
    s = 256
    cx, cy = origin if origin else ((W - crop) // 2, (H - crop) // 2)
    canvas = Image.new("RGB", (len(tiles) * (s + 8) + 8, 2 * s + 60), "white")
    d = ImageDraw.Draw(canvas)
    for i, (name, im) in enumerate(tiles):
        x = 8 + i * (s + 8)
        canvas.paste(im.resize((s, s), Image.BICUBIC), (x, 24))
        canvas.paste(im.crop((cx, cy, cx + crop, cy + crop)).resize((s, s), Image.NEAREST), (x, 32 + s))
        d.text((x, 6), name, fill="black")
    d.text((8, 2 * s + 42), f"top: full {W}x{H} output (downscaled); bottom: {crop}x{crop} crop at ({cx}, {cy}), "
           "enlarged",
           fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def cmd_figures(pilot=PILOT, lr=None, gt=None, crop=None):
    vals, problems = read_metrics(pilot)
    if problems:
        for p in problems:
            print("PROBLEM:", p)
        return 1
    svg = REPO / "assets" / "figures" / "pilot_metrics.svg"
    svg.parent.mkdir(parents=True, exist_ok=True)
    svg.write_text(metrics_svg(vals), encoding="utf-8")
    print("wrote", _rel(svg))
    if all((pilot / k / "sr.png").exists() for k, _ in VARIANTS):
        print("wrote", _rel(comparison_png(pilot, Path(lr) if lr else None, Path(gt) if gt else None,
                                           REPO / "assets" / "comparisons" / "canon_001.png",
                                           crop=crop[2] if crop else 192, origin=crop[:2] if crop else None)))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("check", "sanitize", "figures"))
    p.add_argument("--pilot_dir", default=str(PILOT))
    p.add_argument("--lr", help="figures: the LR input image (not committed)")
    p.add_argument("--gt", help="figures: the GT image (not committed)")
    p.add_argument("--crop", type=int, nargs=3, metavar=("X", "Y", "SIZE"), help="figures: crop region (default: centre 192)")
    a = p.parse_args(argv)
    pilot = Path(a.pilot_dir).resolve()
    if a.command == "check":
        return cmd_check(pilot)
    if a.command == "sanitize":
        return cmd_sanitize(pilot)
    return cmd_figures(pilot, a.lr, a.gt, a.crop)


if __name__ == "__main__":
    sys.exit(main())
