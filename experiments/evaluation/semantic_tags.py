"""Optional semantic-consistency check: RAM tag agreement between SR and HR images.

Protocol
  1. Tag every HR (ground-truth) image and every SR image with the ORIGINAL RAM
     model (ram.models.ram, Swin-L, upstream per-class thresholds) using the
     same preprocessing as upstream utils_data/make_tags.py.
  2. Per image, treat the HR tag set as a *reference* (not ground truth) and
     report precision, recall, F1 and Jaccard of the SR tag set, plus the tags
     only present in SR ("extra") and only present in HR ("lost").

Caveats (also written into the summary):
  * RAM tags of the HR image are model predictions, not human labels.
  * SeeSR is conditioned on DAPE, which is distilled from RAM, so agreement
    with RAM is biased toward SeeSR's own semantic prior. Use this only to
    compare SeeSR variants with each other, never as an absolute semantic score.
  * RAM thresholds were tuned on clean images; results are not calibrated.

Requires preset/models/ram_swin_large_14m.pth and preferably a GPU.

Example:
  python experiments/evaluation/semantic_tags.py \
      --sr_dir experiments/results/baseline_ddpm50/sr/sample00 \
      --gt_dir preset/datasets/RealSR/test_HR \
      --output_dir experiments/results/baseline_ddpm50/eval
"""
import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # upstream tracks __pycache__/*.pyc; do not rewrite them
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.common import git_info, list_images, now_iso, write_csv, write_json  # noqa: E402
from experiments.evaluation.evaluate import aggregate, match_pairs  # noqa: E402

CAVEATS = [
    "HR tags are RAM predictions used as a reference, not ground truth.",
    "SeeSR's DAPE is distilled from RAM, so this agreement is biased toward SeeSR's own semantic prior; "
    "compare variants relative to each other only.",
    "RAM per-class thresholds were tuned on clean images and are not calibrated for this use.",
]


def parse_tags(text):
    return {t.strip() for t in text.split(",") if t.strip()}


def tag_agreement(sr_tags, ref_tags):
    inter = sr_tags & ref_tags
    union = sr_tags | ref_tags
    precision = len(inter) / len(sr_tags) if sr_tags else (1.0 if not ref_tags else 0.0)
    recall = len(inter) / len(ref_tags) if ref_tags else (1.0 if not sr_tags else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    jaccard = len(inter) / len(union) if union else 1.0
    return {
        "tag_precision": precision, "tag_recall": recall, "tag_f1": f1, "tag_jaccard": jaccard,
        "n_sr_tags": len(sr_tags), "n_ref_tags": len(ref_tags),
        "extra_tags": ", ".join(sorted(sr_tags - ref_tags)),
        "lost_tags": ", ".join(sorted(ref_tags - sr_tags)),
        "sr_tags": ", ".join(sorted(sr_tags)),
        "ref_tags": ", ".join(sorted(ref_tags)),
    }


def load_ram(device):
    os.chdir(REPO_ROOT)  # upstream RAM resolves config/tag-list paths relative to the package
    from ram.models.ram import ram
    model = ram(pretrained="preset/models/ram_swin_large_14m.pth", image_size=384, vit="swin_l")
    return model.eval().to(device)


def make_tagger(device):
    import torch
    from PIL import Image
    from torchvision import transforms
    from ram import inference_ram

    model = load_ram(device)
    tf = transforms.Compose([  # identical to utils_data/make_tags.py
        transforms.ToTensor(),
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def tag(path):
        with Image.open(path) as im:
            x = tf(im.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            return parse_tags(inference_ram(x, model)[0])

    return tag


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sr_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--strip_suffix", action="append", default=[])
    p.add_argument("--device", default="cuda")
    opts = p.parse_args(argv)

    # Resolve before load_ram() changes the working directory.
    sr_dir, gt_dir, out_dir = (Path(x).resolve() for x in (opts.sr_dir, opts.gt_dir, opts.output_dir))
    m = match_pairs(list_images(sr_dir), list_images(gt_dir), opts.strip_suffix)
    if m["missing_gt"] or m["missing_sr"] or not m["pairs"]:
        print(f"ERROR: unmatched files: missing_gt={m['missing_gt'][:5]} missing_sr={m['missing_sr'][:5]}", file=sys.stderr)
        return 2
    tag = make_tagger(opts.device)
    rows = []
    for key, sr_path, gt_path in m["pairs"]:
        rows.append({"image": key, **tag_agreement(tag(sr_path), tag(gt_path))})
    keys = ["tag_precision", "tag_recall", "tag_f1", "tag_jaccard"]
    write_csv(out_dir / "semantic_per_image.csv", rows)
    write_json(out_dir / "semantic_summary.json", {
        "created_at": now_iso(), "git": git_info(), "sr_dir": str(sr_dir), "gt_dir": str(gt_dir),
        "tagger": "ram.models.ram (RAM Swin-L, ram_swin_large_14m.pth, upstream per-class thresholds)",
        "caveats": CAVEATS, "n_images": len(rows), "aggregate": aggregate(rows, keys),
    })
    print(f"semantic tag agreement written for {len(rows)} images -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
