"""Write the V2 SCREENING set: LR/GT image pairs from LORNet's VALIDATION split (never the test sets).

The SFR-OC strength screening (configs/v2_strength_screening.yaml) and the map-validity test
(tools/sfroc_map_validity.py) must not use the final evaluation images. This tool takes n images of the
hash-selected validation split of experiments/training (same discovery, exclusions and split as `train_lornet.py
cache`, so they are held out from LORNet training too) and writes, per image, one centre crop:
    <output>/<group>/lr/<stem>.png    LR crop (crop_hr / scale px; LR registered at HR size is downscaled first,
                                      exactly as in the training cache)
    <output>/<group>/gt/<stem>.png    matching HR crop (crop_hr px)
(<group> = the source's `group` in the config, e.g. RealSR for both RealSR cameras; default: the source name)
plus manifest.json (ids, split digest, crop boxes). Images are selected by sha256 of their id (deterministic).

  python experiments/training/make_screening_set.py --config experiments/training/configs/lornet.yaml \\
      --n 20 --crop_hr 512 --output <RESULTS>/v2_screening_data
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.training import data as D  # noqa: E402
from experiments.training.train_lornet import load_config  # noqa: E402


def build(cfg, n, crop_hr, output):
    d = cfg["data"]
    scale = int(d["scale"])
    if crop_hr % (8 * scale):
        raise ValueError(f"crop_hr must be a multiple of {8 * scale}")
    pairs = D.discover_pairs(d["sources"])
    _, val = D.split_pairs(pairs, float(d["val_fraction"]), exclude=D.excluded_stems(d.get("exclude_dirs")))
    groups = {s["name"]: s.get("group", s["name"]) for s in d["sources"]}
    chosen = sorted(val, key=lambda p: hashlib.sha256(p["id"].encode()).hexdigest())
    out = Path(output)
    written = {}
    for p in chosen:
        src = groups[p["source"]]
        if len(written.get(src, [])) >= n:
            continue
        lr, hr, mode = D.load_pair(p, scale)
        W, H = hr.size
        c = min(crop_hr, W // (8 * scale) * 8 * scale, H // (8 * scale) * 8 * scale)
        if c < 8 * scale:
            continue
        x, y = (W - c) // 2 // scale * scale, (H - c) // 2 // scale * scale
        for sub in ("lr", "gt"):
            (out / src / sub).mkdir(parents=True, exist_ok=True)
        hr.crop((x, y, x + c, y + c)).save(out / src / "gt" / f"{p['stem']}.png")
        lr.crop((x // scale, y // scale, (x + c) // scale, (y + c) // scale)).save(out / src / "lr" / f"{p['stem']}.png")
        written.setdefault(src, []).append({"id": p["id"], "stem": p["stem"], "box": [x, y, c], "lr_mode": mode})
    manifest = {"source_split": "lornet validation", "val_ids_sha256": D.ids_digest(val), "n_per_group": n,
                "crop_hr": crop_hr, "scale": scale, "images": written,
                "note": "held out from LORNet training and disjoint from the evaluation sets (exclude_dirs)"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--n", type=int, default=20, help="images per group")
    p.add_argument("--crop_hr", type=int, default=512)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    m = build(load_config(a.config, a.set), a.n, a.crop_hr, a.output)
    print({k: len(v) for k, v in m["images"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
