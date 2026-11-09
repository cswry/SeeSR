"""Gate 0: does DAPE's tagging-head attention localise the tags it predicts?

Spatial semantic reliability (cfa_seesr/spatial_reliability.py) assumes that the cross-attention of a tag
query over DAPE's 12x12 image tokens points at the image region that supports the tag. This diagnostic tests
that assumption with an occlusion experiment, using only DAPE (no diffusion model):

  for every LR image and every upstream-selected tag i (S0):
    1. capture tag i's attention map A_i over the grid (pooled token excluded; mean over heads; last layer)
    2. occlude the top `top_frac` of cells by A_i            -> logit z_i(top)
    3. occlude `n_random` control regions of the same area     -> mean logit z_i(rand)
       control 'shifted' (default): the same cell pattern, circularly shifted by a random non-zero offset
         (same size AND the same clustering, different location);
       control 'scattered': random cells (drawn once per image, shared by its tags)
    4. drop_top = z_i - z_i(top), drop_rand = z_i - z_i(rand), d_i = drop_top - drop_rand

  d_i > 0 means hiding the attended cells removes more evidence for the tag than hiding random cells.

Statistics. Tags of one image share the image, the occlusions and the random draws, so they are NOT
independent: the unit of analysis is the image, d_img = mean of d_i over its tags. Reported: tag-level
descriptives; image-level mean/median, 95% bootstrap CI over images, a one-sided paired sign-flip
randomisation test over images, the Wilcoxon signed-rank test (if scipy is installed) and the fraction of
images with d_img > 0.

Decision (fixed before running; the thresholds are in the summary):
  GO     if n_images >= min_images AND bootstrap CI lower bound > 0 AND sign-flip p < alpha
         AND fraction of images with d_img > 0 >= min_positive_fraction
  NO-GO  otherwise (including too few images with tags).

Masking and its limitations.
  * Occlusion happens in DAPE's input space (the normalised 384x384 tensor), one grid cell = 384/g pixels,
    so masks align with the token grid. Fill 'local_mean' (default) replaces a cell by the image's
    area-downsampled-to-the-grid and bilinearly upsampled version (keeps local colour, removes detail);
    'channel_mean' uses the per-channel image mean (stronger, more out-of-distribution).
  * Swin-L has a large receptive field (shifted windows, patch merging): a token is not a function of its
    cell alone, so occlusion probes, not isolates, spatial evidence.
  * Occlusion is itself a distribution shift; both conditions share it, which is why the comparison is
    against random occlusions of equal area rather than against the unoccluded image alone.
  * Scene-level tags ("outdoor", "image") have no localised support and pull the effect towards zero.
  * Attended cells are often clustered, and clustered occlusion can be more damaging than scattered
    occlusion regardless of attention. The default 'shifted' control keeps the pattern's shape, removing
    that bias; with 'scattered' the test is biased towards GO. Circular shifts wrap at the borders, which
    splits clusters that touch an edge (a small bias in the same direction).
  * The LR inputs are degraded; attention quality on clean images says nothing about this setting, which
    is why the test runs on the LR benchmark images themselves.

Outputs (in --output_dir): gate0_per_tag.csv, gate0_per_image.csv, gate0_summary.json (with "decision"),
overlays/<dataset>__<image>.png (LR image, attention of up to four tags with their attended cells outlined,
and the SSR grid at contrast 1).

  python experiments/tools/gate0_attention.py --images RealSR=<LR dir> --images DRealSR=<LR dir> \\
      --n_images 50 --output_dir <RESULTS>/gate0
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_CRITERIA = {"min_images": 20, "alpha": 0.05, "min_positive_fraction": 0.6}


# --------------------------------------------------------------------------- #
# Masks and occlusion
# --------------------------------------------------------------------------- #
def top_cells(attention_map, frac):
    """Indices (flattened, row-major) of the ceil(frac * N) cells with the highest attention."""
    import torch
    flat = attention_map.flatten()
    k = max(1, int(math.ceil(frac * flat.numel())))
    return torch.argsort(flat, descending=True, stable=True)[:k].tolist()


def random_cell_sets(n_cells, k, n_sets, seed):
    rng = random.Random(seed)
    return [sorted(rng.sample(range(n_cells), k)) for _ in range(n_sets)]


def shifted_cell_sets(cells, g, n_sets, seed):
    """The same cell pattern, circularly shifted by n_sets distinct random non-zero offsets (dy, dx)."""
    rng = random.Random(seed)
    offsets = [(dy, dx) for dy in range(g) for dx in range(g) if (dy, dx) != (0, 0)]
    out = []
    for dy, dx in rng.sample(offsets, min(n_sets, len(offsets))):
        out.append(sorted(((c // g + dy) % g) * g + (c % g + dx) % g for c in cells))
    return out


def cell_mask(cells, g, size):
    """[size, size] float mask (1 = occluded) for the given flattened grid cells."""
    import torch
    import torch.nn.functional as F
    m = torch.zeros(g * g)
    m[list(cells)] = 1.0
    return F.interpolate(m.view(1, 1, g, g), size=(size, size), mode="nearest")[0, 0]


def fill_image(x, g, how):
    """Replacement content for occluded pixels. x: [1, 3, S, S] (DAPE input space)."""
    import torch.nn.functional as F
    if how == "local_mean":
        small = F.adaptive_avg_pool2d(x, (g, g))
        return F.interpolate(small, size=x.shape[-2:], mode="bilinear", align_corners=False)
    if how == "channel_mean":
        return x.mean(dim=(2, 3), keepdim=True).expand_as(x)
    raise ValueError(f"unknown fill {how!r}")


def occlude(x, mask, fill):
    return x * (1 - mask) + fill * mask


# --------------------------------------------------------------------------- #
# Statistics (image is the unit of analysis)
# --------------------------------------------------------------------------- #
def image_level(per_tag):
    by = {}
    for r in per_tag:
        by.setdefault((r["dataset"], r["image"]), []).append(r)
    out = []
    for (ds, img), rows in sorted(by.items()):
        diffs = [r["diff"] for r in rows]
        out.append({"dataset": ds, "image": img, "n_tags": len(rows),
                    "mean_drop_top": statistics.fmean(r["drop_top"] for r in rows),
                    "mean_drop_rand": statistics.fmean(r["drop_rand"] for r in rows),
                    "mean_diff": statistics.fmean(diffs),
                    "frac_tags_positive": sum(d > 0 for d in diffs) / len(diffs)})
    return out


def bootstrap_ci(values, n_boot=10000, level=0.95, seed=0):
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(n_boot))
    lo = means[int(math.floor((1 - level) / 2 * n_boot))]
    hi = means[min(n_boot - 1, int(math.ceil((1 + level) / 2 * n_boot)) - 1)]
    return lo, hi


def sign_flip_p(values, n_perm=20000, seed=0):
    """One-sided p-value for mean(values) > 0 under random sign flips (exact if n <= 16)."""
    n = len(values)
    if n == 0:
        return None
    obs = statistics.fmean(values)
    if n <= 16:
        count = total = 0
        for bits in range(2 ** n):
            s = sum(v if (bits >> i) & 1 else -v for i, v in enumerate(values)) / n
            count += s >= obs - 1e-15
            total += 1
        return count / total
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        s = sum(v if rng.random() < 0.5 else -v for v in values) / n
        count += s >= obs - 1e-15
    return (count + 1) / (n_perm + 1)


def wilcoxon_p(values):
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    nz = [v for v in values if v != 0]
    if len(nz) < 2:
        return None
    return float(wilcoxon(nz, alternative="greater").pvalue)


def summarize(per_tag, n_images_total, criteria=None, settings=None):
    criteria = {**DEFAULT_CRITERIA, **(criteria or {})}
    imgs = image_level(per_tag)
    d_img = [r["mean_diff"] for r in imgs]

    def desc(vals):
        if not vals:
            return {"n": 0, "mean": None, "median": None, "sd": None}
        return {"n": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals),
                "sd": statistics.stdev(vals) if len(vals) > 1 else None}

    lo, hi = bootstrap_ci(d_img)
    p_flip = sign_flip_p(d_img)
    frac_pos = (sum(d > 0 for d in d_img) / len(d_img)) if d_img else None
    checks = {
        "enough_images": len(imgs) >= criteria["min_images"],
        "ci_lower_above_zero": lo is not None and lo > 0,
        "sign_flip_significant": p_flip is not None and p_flip < criteria["alpha"],
        "consistent_across_images": frac_pos is not None and frac_pos >= criteria["min_positive_fraction"],
    }
    decision = "GO" if all(checks.values()) else "NO-GO"
    return {
        "decision": decision,
        "checks": checks,
        "criteria": criteria,
        "settings": settings or {},
        "n_images_evaluated": n_images_total,
        "n_images_with_tags": len(imgs),
        "n_tags": len(per_tag),
        "tag_level": {"drop_top": desc([r["drop_top"] for r in per_tag]),
                      "drop_rand": desc([r["drop_rand"] for r in per_tag]),
                      "paired_diff": desc([r["diff"] for r in per_tag]),
                      "attention_entropy_norm": desc([r["entropy_norm"] for r in per_tag]),
                      "note": "descriptive only: tags of one image are not independent"},
        "image_level": {"mean_diff": desc(d_img),
                        "bootstrap_95ci_mean_diff": [lo, hi],
                        "sign_flip_p_one_sided": p_flip,
                        "wilcoxon_p_one_sided": wilcoxon_p(d_img),
                        "fraction_images_positive": frac_pos,
                        "mean_drop_top": desc([r["mean_drop_top"] for r in imgs]),
                        "mean_drop_rand": desc([r["mean_drop_rand"] for r in imgs])},
        "interpretation": ("GO: hiding the attended cells reduces the tag logit more than hiding random cells "
                           "of equal area, consistently across images; SSR variants may be screened."
                           if decision == "GO" else
                           "NO-GO: DAPE's tagging attention is not shown to localise tag evidence on these "
                           "images; do not run SSR variants in the main experiments."),
    }


# --------------------------------------------------------------------------- #
# Running DAPE
# --------------------------------------------------------------------------- #
def select_images(sources, n_images):
    """sources: [(dataset, dir)]. Deterministic, order-independent choice: sha256 of 'dataset/name'.

    Images are split as evenly as possible between the sources."""
    per = []
    for ds, d in sources:
        files = sorted((p for p in Path(d).iterdir() if p.suffix.lower() in IMAGE_EXT),
                       key=lambda p: hashlib.sha256(f"{ds}/{p.name}".encode()).hexdigest())
        per.append([(ds, p) for p in files])
    out, i = [], 0
    while len(out) < n_images and any(per):
        lst = per[i % len(per)]
        if lst:
            out.append(lst.pop(0))
        i += 1
    return out


def entropy_norm(a):
    import torch
    p = a.flatten().double().clamp_min(1e-12)
    return float(-(p * p.log()).sum() / math.log(p.numel()))


def analyse_image(model, x, dataset, name, args, overlays_dir=None, pil_image=None):
    """Per-tag rows for one preprocessed image x [1, 3, S, S] (DAPE input space)."""
    import torch

    from cfa_seesr.semantic_confidence import tag_confidence
    from cfa_seesr.spatial_reliability import (SpatialSemanticConfig, capture_dape, grid_side, reliability_grid,
                                               tag_attention_maps, upstream_tag_indices)

    with torch.no_grad(), capture_dape(model, args.layer) as cap:
        _, logits, _ = model.condition_forward(x, only_feature=False)
    att = cap["attention"][0]                                  # [heads, K, T]
    z0 = logits[0].float()
    idx = upstream_tag_indices(z0, model.class_threshold, model.delete_tag_index)
    g = grid_side(att.shape[-1])
    if not idx:
        return [], None
    maps = tag_attention_maps(att, idx)                        # [n, g, g]
    conf, _ = tag_confidence(z0, model.class_threshold, 1.0)
    size = x.shape[-1]
    fill = fill_image(x, g, args.fill)
    k = max(1, int(math.ceil(args.top_frac * g * g)))
    seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{dataset}/{name}".encode()).digest()[:4], "little")
    scattered = random_cell_sets(g * g, k, args.n_random, seed) if args.control == "scattered" else None

    def logits_for(cell_sets):
        out = []
        for b in range(0, len(cell_sets), args.batch):
            chunk = cell_sets[b:b + args.batch]
            xs = torch.cat([occlude(x, cell_mask(c, g, size).to(x), fill) for c in chunk])
            with torch.no_grad():
                _, lg, _ = model.condition_forward(xs, only_feature=False)
            out.append(lg.float())
        return torch.cat(out)

    tops = [top_cells(maps[j], args.top_frac) for j in range(len(idx))]
    z_top = logits_for(tops)                                   # [n, K]
    if scattered is not None:
        z_shared = logits_for(scattered).mean(dim=0)           # [K], shared by all tags of the image
    rows = []
    for j, i in enumerate(idx):
        if scattered is not None:
            z_rand_i = float(z_shared[i])
        else:
            z_rand_i = float(logits_for(shifted_cell_sets(tops[j], g, args.n_random, seed + j))[:, i].mean())
        drop_top = float(z0[i] - z_top[j, i])
        drop_rand = float(z0[i] - z_rand_i)
        a = maps[j]
        rows.append({"dataset": dataset, "image": name, "tag": str(model.tag_list[i]), "class_index": i,
                     "p": float(torch.sigmoid(z0[i])), "tau": float(model.class_threshold[i]),
                     "c": float(conf[i]), "logit": float(z0[i]), "logit_top": float(z_top[j, i]),
                     "logit_rand_mean": z_rand_i, "drop_top": drop_top, "drop_rand": drop_rand,
                     "diff": drop_top - drop_rand, "entropy_norm": entropy_norm(a),
                     "top_mass": float(a.flatten()[tops[j]].sum())})
    grid = None
    if overlays_dir is not None and pil_image is not None:
        conf_hat = [min(1.0, max(0.0, 2 * float(conf[i]) - 1)) for i in idx]
        C = max(0.0, min(1.0, (statistics.fmean(float(conf[i]) for i in idx) - 0.5) / 0.5))
        grid, _ = reliability_grid(maps, conf_hat, C, C, SpatialSemanticConfig(enabled=True, contrast=1.0))
        save_overlay(pil_image, maps, tops, [r["tag"] for r in rows], grid, g,
                     Path(overlays_dir) / f"{dataset}__{Path(name).stem}.png")
    return rows, grid


def save_overlay(image, maps, tops, tags, grid, g, path, panel=256, max_tags=4):
    """LR image | attention of up to four tags (red; attended cells outlined) | SSR grid (blue = low)."""
    import numpy as np
    from PIL import Image, ImageDraw
    base = np.asarray(image.convert("RGB").resize((panel, panel), Image.BICUBIC)).astype(np.float32)
    order = sorted(range(len(tags)), key=lambda j: -float(maps[j].max()))[:max_tags]
    panels = [Image.fromarray(base.astype(np.uint8))]
    cell = panel / g
    for j in order:
        a = maps[j].numpy()
        a = (a - a.min()) / (a.max() - a.min() + 1e-12)
        heat = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((panel, panel), Image.BILINEAR),
                          dtype=np.float32) / 255.0
        img = base * (1 - 0.5 * heat[..., None]) + np.array([255, 0, 0], np.float32) * 0.5 * heat[..., None]
        im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        for c in tops[j]:
            r, q = divmod(c, g)
            d.rectangle([q * cell, r * cell, (q + 1) * cell - 1, (r + 1) * cell - 1], outline=(255, 255, 0))
        d.text((4, 4), tags[j][:30], fill=(255, 255, 255))
        panels.append(im)
    if grid is not None:
        v = grid.numpy()
        rgb = np.stack([v, v, np.ones_like(v)], -1) * 255   # 1 -> white, 0 -> blue
        im = Image.fromarray(rgb.astype(np.uint8)).resize((panel, panel), Image.NEAREST)
        ImageDraw.Draw(im).text((4, 4), f"SSR grid (lambda=1) {v.min():.2f}-{v.max():.2f}", fill=(0, 0, 0))
        panels.append(im)
    out = Image.new("RGB", (panel * len(panels), panel), (0, 0, 0))
    for n, p in enumerate(panels):
        out.paste(p, (n * panel, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", action="append", required=True, metavar="DATASET=LR_DIR",
                   help="LR image folder (repeatable)")
    p.add_argument("--n_images", type=int, default=50)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ram_ft_path", default="preset/models/DAPE.pth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer", type=int, default=-1, help="tagging-head layer (-1 = last, as used by SSR)")
    p.add_argument("--top_frac", type=float, default=0.2)
    p.add_argument("--n_random", type=int, default=8)
    p.add_argument("--fill", choices=("local_mean", "channel_mean"), default="local_mean")
    p.add_argument("--control", choices=("shifted", "scattered"), default="shifted")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_overlays", action="store_true")
    a = p.parse_args(argv)

    import torch
    from PIL import Image
    os.chdir(REPO)   # upstream load_tag_model uses the relative path preset/models/ram_swin_large_14m.pth
    import test_seesr  # upstream module, unmodified

    sources = []
    for item in a.images:
        ds, _, d = item.partition("=")
        if not d or not Path(d).is_dir():
            p.error(f"--images expects DATASET=DIR with an existing directory, got {item!r}")
        sources.append((ds, d))
    chosen = select_images(sources, a.n_images)
    if not chosen:
        p.error("no images found")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = test_seesr.load_tag_model(argparse.Namespace(ram_ft_path=a.ram_ft_path), device)
    model.eval()

    per_tag = []
    for n, (ds, path) in enumerate(chosen, 1):
        image = Image.open(path).convert("RGB")
        x = test_seesr.ram_transforms(test_seesr.tensor_transforms(image).unsqueeze(0).to(device))
        rows, _ = analyse_image(model, x, ds, path.name, a,
                                overlays_dir=None if a.no_overlays else out / "overlays", pil_image=image)
        per_tag.extend(rows)
        print(f"[{n}/{len(chosen)}] {ds}/{path.name}: {len(rows)} tag(s)")

    settings = {"top_frac": a.top_frac, "n_random": a.n_random, "fill": a.fill, "control": a.control, "layer": a.layer, "seed": a.seed,
                "images": [f"{ds}/{p.name}" for ds, p in chosen], "ram_ft_path": a.ram_ft_path, "device": device}
    summary = summarize(per_tag, len(chosen), settings=settings)
    write_csv(out / "gate0_per_tag.csv", per_tag)
    write_csv(out / "gate0_per_image.csv", image_level(per_tag))
    (out / "gate0_summary.json").write_text(json.dumps(summary, indent=2))
    il = summary["image_level"]
    print(json.dumps({"decision": summary["decision"], "checks": summary["checks"],
                      "n_images_with_tags": summary["n_images_with_tags"], "n_tags": summary["n_tags"],
                      "mean_diff": il["mean_diff"]["mean"], "ci": il["bootstrap_95ci_mean_diff"],
                      "p_sign_flip": il["sign_flip_p_one_sided"]}, indent=1))
    print(f"GATE 0 DECISION: {summary['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
