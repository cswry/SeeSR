"""Hold-out evidence for a pilot input: is the pilot LR image the StableSR test file, and is it absent from the
LORNet training data?

Three checks, all recorded in one JSON file:

1. Identity. SHA-256 of every pilot LR (and, optionally, GT) file, compared with the file of the same name in the
   StableSR test folder, plus a search of that folder by hash (a renamed copy is still found). The run's recorded
   `dataset.lr_dir_fingerprint` (experiments/common.file_record) is recomputed from the pilot folder and compared
   with `--expect-lr-fingerprint`, which ties the files checked here to the files the pilot runs actually read.
2. Byte duplicates. SHA-256 of every image under the training folders; any file identical to a pilot file is listed.
3. Content overlap. A 128 px test crop is never byte-identical to a full training image, so each pilot LR image is
   also searched inside every training LR image by normalised cross-correlation (grayscale, all positions).
   Training LR images registered at HR size (RealSR V3 'LR4') are first bicubic-downsampled by `--scale`, exactly
   as experiments/training/data.py does, so both are at the LR resolution SeeSR sees. The best score per training
   image is recorded; the top matches are listed. A crop cut from a training image scores close to 1.0.
   Training LR images are selected by stem suffix (`--train-lr-regex`, case-insensitive). The search counts as
   complete only if every selected LR image was searched and there are at least as many LR as HR images.

Run where the data is (e.g. Colab with Drive mounted; <DATA> = your datasets folder).
Read-only: nothing is written except `--out`.

  python experiments/tools/check_pilot_holdout.py \
      --pilot-lr-dir  <DATA>/pilot_realsr_lr \
      --pilot-gt-dir  <DATA>/pilot_realsr_gt \
      --test-lr-dir   <DATA>/StableSR_testsets/RealSRVal_crop128/test_LR \
      --test-hr-dir   <DATA>/StableSR_testsets/RealSRVal_crop128/test_HR \
      --train-dir     <DATA>/RealSR_V3/Canon/Train/4 \
      --train-dir     <DATA>/RealSR_V3/Nikon/Train/4 \
      --expect-lr-fingerprint aecdd6dd33825d3692b4700bb42c39ad715bc470d37d66e68b1f068a1f775e08 \
      --out pilot_holdout_evidence.json

Exit code 0 = identity confirmed, complete search, no overlap above `--threshold`; 1 otherwise.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import file_record, now_iso  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
PRIVATE_PREFIXES = ("/content/drive/MyDrive/seesr", "/content/drive/MyDrive", "/content/SeeSR", "/content")


def images(d, pattern="*"):
    return sorted(p for p in Path(d).rglob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def by_stem(files, regex):
    """Files whose stem matches `regex` (case-insensitive), e.g. LR images by their _LR4 / _lr4 / -LR suffix."""
    r = re.compile(regex, re.IGNORECASE)
    return [p for p in files if r.search(p.stem)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gray(img):
    return np.asarray(img.convert("L"), dtype=np.float64)


def as_lr(path, scale, lr_size=None):
    """Training LR image at LR resolution (data.load_pair geometry): HR-size images are bicubic-downsampled."""
    im = Image.open(path)
    if scale > 1 and (lr_size is None or max(im.size) > 2 * max(lr_size)):
        W, H = im.size[0] // scale * scale, im.size[1] // scale * scale
        im = im.crop((0, 0, W, H)).resize((W // scale, H // scale), Image.BICUBIC)
    return gray(im)


def ncc_max(image, template):
    """Max zero-mean normalised cross-correlation of `template` over all valid positions of `image` (FFT)."""
    th, tw = template.shape
    ih, iw = image.shape
    if th > ih or tw > iw:
        return float("nan"), None
    t = template - template.mean()
    tn = np.sqrt((t ** 2).sum())
    if tn == 0:
        return float("nan"), None
    fh, fw = ih + th - 1, iw + tw - 1
    corr = np.fft.irfft2(np.fft.rfft2(image, (fh, fw)) * np.fft.rfft2(t[::-1, ::-1], (fh, fw)), (fh, fw))
    corr = corr[th - 1:ih, tw - 1:iw]                   # valid positions; sum(I * t) since t has zero mean
    ii = np.pad(image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ii2 = np.pad(image ** 2, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def box(a):
        return a[th:, tw:] - a[:-th, tw:] - a[th:, :-tw] + a[:-th, :-tw]

    n = th * tw
    var = box(ii2) - box(ii) ** 2 / n
    score = corr / (np.sqrt(np.clip(var, 1e-12, None)) * tn)
    score[var < 1e-6] = -1.0
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(score[y, x]), [int(x), int(y)]


def public(path):
    s = str(path)
    for p in PRIVATE_PREFIXES:
        if s.startswith(p):
            return "<DRIVE_ROOT>" + s[len(p):] if p.startswith("/content/drive") else "<REPO>" + s[len(p):]
    return s


def identity(pilot_dir, test_dir):
    test = {p.name: p for p in images(test_dir)} if test_dir else {}
    by_hash = {}
    for p in test.values():
        by_hash.setdefault(sha256(p), []).append(p.name)
    rows = []
    for p in images(pilot_dir):
        h = sha256(p)
        same = test.get(p.name)
        rows.append({"file": p.name, "sha256": h, "size_bytes": p.stat().st_size,
                     "test_file_sha256": sha256(same) if same else None,
                     "identical_to_test_file": bool(same) and sha256(same) == h,
                     "test_files_with_same_hash": by_hash.get(h, [])})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot-lr-dir", required=True)
    ap.add_argument("--pilot-gt-dir")
    ap.add_argument("--test-lr-dir", required=True)
    ap.add_argument("--test-hr-dir")
    ap.add_argument("--train-dir", action="append", required=True)
    ap.add_argument("--train-lr-regex", default=r"[_-]lr\d*$", help="training LR stems (case-insensitive)")
    ap.add_argument("--train-hr-regex", default=r"[_-]hr$", help="training HR stems, for the coverage count")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--expect-lr-fingerprint")
    ap.add_argument("--threshold", type=float, default=0.95, help="content-overlap NCC flag level")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    ev = {"created_at": now_iso(), "tool": "experiments/tools/check_pilot_holdout.py",
          "folders": {"pilot_lr": public(a.pilot_lr_dir),
                      "pilot_gt": public(a.pilot_gt_dir) if a.pilot_gt_dir else None,
                      "test_lr": public(a.test_lr_dir), "test_hr": public(a.test_hr_dir) if a.test_hr_dir else None,
                      "train": [public(d) for d in a.train_dir]}}

    ev["lr_identity"] = identity(a.pilot_lr_dir, a.test_lr_dir)
    if a.pilot_gt_dir:
        ev["gt_identity"] = identity(a.pilot_gt_dir, a.test_hr_dir)
    fp = file_record(a.pilot_lr_dir).get("fingerprint")
    ev["lr_dir_fingerprint"] = {"recomputed": fp, "expected": a.expect_lr_fingerprint,
                                "match": (fp == a.expect_lr_fingerprint) if a.expect_lr_fingerprint else None}

    pilot = {p.name: p for p in images(a.pilot_lr_dir)}
    if a.pilot_gt_dir:
        pilot.update({"GT:" + p.name: p for p in images(a.pilot_gt_dir)})
    pilot_hash = {sha256(p): k for k, p in pilot.items()}
    train_all = [p for d in a.train_dir for p in images(d)]
    ev["byte_duplicates"] = {"n_training_files_hashed": len(train_all),  # every image, HR and LR
                             "duplicates": [{"pilot": pilot_hash[h], "training_file": public(p)}
                                            for p in train_all if (h := sha256(p)) in pilot_hash]}

    templates = {p.name: gray(Image.open(p)) for p in images(a.pilot_lr_dir)}
    train_lr = by_stem(train_all, a.train_lr_regex)
    n_hr = len(by_stem(train_all, a.train_hr_regex))
    content = {}
    for name, t in templates.items():
        scores = []
        for p in train_lr:
            s, xy = ncc_max(as_lr(p, a.scale, t.shape[::-1]), t)
            scores.append({"training_file": public(p), "ncc": round(s, 6), "xy_lr": xy})
        scores.sort(key=lambda r: -np.nan_to_num(r["ncc"], nan=-2))
        valid = [r["ncc"] for r in scores if r["ncc"] == r["ncc"]]
        content[name] = {"n_training_lr_images": len(train_lr), "n_training_hr_images": n_hr,
                         "n_searched": len(valid), "not_searched": [r["training_file"] for r in scores
                                                                    if r["ncc"] != r["ncc"]],
                         "max_ncc": max(valid) if valid else None,
                         "median_best_ncc": float(np.median(valid)) if valid else None,
                         "top": scores[:a.top],
                         "flagged": [r for r in scores if r["ncc"] == r["ncc"] and r["ncc"] >= a.threshold]}
    ev["content_overlap"] = {"method": f"max zero-mean NCC, grayscale, training LR downsampled 1/{a.scale} "
                                       "(bicubic) when registered at HR size", "threshold": a.threshold,
                             "per_pilot_image": content}

    ident_ok = bool(ev["lr_identity"]) and all(r["identical_to_test_file"] for r in ev["lr_identity"])
    if a.pilot_gt_dir:
        ident_ok = ident_ok and all(r["identical_to_test_file"] for r in ev.get("gt_identity", []))
    fp_ok = ev["lr_dir_fingerprint"]["match"] is not False
    overlap = bool(ev["byte_duplicates"]["duplicates"]) or any(c["flagged"] for c in content.values())
    # complete = every training LR image searched, and at least one LR image per HR image (no LR files missed)
    searched = all(c["n_searched"] > 0 and c["n_searched"] == c["n_training_lr_images"]
                   and c["n_training_lr_images"] >= c["n_training_hr_images"] for c in content.values())
    ev["verdict"] = {"pilot_is_stablesr_test_file": ident_ok,
                     "fingerprint_matches_runs": ev["lr_dir_fingerprint"]["match"],
                     "training_overlap_found": overlap, "training_search_complete": searched,
                     "held_out": ident_ok and fp_ok and searched and not overlap}
    Path(a.out).write_text(json.dumps(ev, indent=2) + "\n")
    print(json.dumps(ev["verdict"], indent=2))
    for name, c in content.items():
        print(f"{name}: searched {c['n_searched']} of {c['n_training_lr_images']} training LR images "
              f"({c['n_training_hr_images']} HR), max NCC {c['max_ncc']}, median {c['median_best_ncc']}")
    return 0 if ev["verdict"]["held_out"] else 1


if __name__ == "__main__":
    sys.exit(main())
