"""Paired LR/HR discovery, leakage-safe splits, crops and the frozen-VAE latent cache for LORNet.

Sources (configs/*.yaml, `data.sources`): each names an HR folder and an LR folder (they may be the same folder
with different filename patterns, as in RealSR V3) of a TRAINING split. Pairs are matched by a normalised stem
(suffixes such as _HR, _LR4, _x4 removed). Validation images are a deterministic hash-selected fraction of the
training pairs (by stem, so all crops of an image stay on one side). Test/evaluation images never enter:
  * a source path containing a path component named like a test split (test, Test, test_LR, ...) is refused;
  * every stem in `data.exclude_dirs` (e.g. the LR folders used for evaluation) is collected and any overlap with
    a training or validation stem raises an error.

LR geometry: if an LR image is 1/scale of its HR (DRealSR x4), it is used as is. If it has the HR size (RealSR V3
'LR4' images are registered at the HR resolution), it is bicubic-downsampled by `scale` to give the LR input
SeeSR sees at inference (as the 128 px RealSR test crops are). Recorded in the cache manifest.

Crops: HR crops of `crop_hr` px (multiple of 8 * scale) on a regular grid (top-left aligned, at most
`crops_per_image`, evenly spread); the LR crop is the corresponding region. The conditioning image is then built
with the upstream resizing (cfa_seesr.latent_targets.upscale_like_upstream) and encoded with the frozen VAE,
as is the HR crop. Cached per crop: LR crop (uint8), E = LocalAverage(mean_c |LPF(z_lr - z_hr)|) (float16).
"""
import hashlib
import json
import re
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
TEST_DIR_RE = re.compile(r"^(test|testing|testset|eval|evaluation)([_-]?(x\d+|lr\d*|hr|gt|data|set))*$", re.IGNORECASE)
STEM_SUFFIX_RE = re.compile(r"([_-](hr|lr\d*|x\d+|gt))+$", re.IGNORECASE)


def normalise_stem(name):
    return STEM_SUFFIX_RE.sub("", Path(name).stem).lower()


def _list(dirpath, pattern):
    d = Path(dirpath)
    if not d.is_dir():
        raise FileNotFoundError(f"not a directory: {d}")
    return sorted(p for p in d.glob(pattern or "*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def _check_not_test(path):
    for part in Path(path).parts:
        if TEST_DIR_RE.match(part):
            raise ValueError(f"training source {path} looks like a test split (component '{part}'); "
                             "LORNet is trained on training splits only")


def discover_pairs(sources):
    """sources: [{name, hr_dir, lr_dir, hr_pattern?, lr_pattern?}] -> [{id, source, hr, lr}] (sorted)."""
    pairs = []
    for src in sources:
        name = src["name"]
        for key in ("hr_dir", "lr_dir"):
            _check_not_test(src[key])
        hr = {normalise_stem(p.name): p for p in _list(src["hr_dir"], src.get("hr_pattern"))}
        lr = {normalise_stem(p.name): p for p in _list(src["lr_dir"], src.get("lr_pattern"))}
        if src["hr_dir"] == src["lr_dir"] and src.get("hr_pattern") == src.get("lr_pattern"):
            raise ValueError(f"{name}: HR and LR come from the same folder with the same pattern")
        common = sorted(set(hr) & set(lr))
        if not common:
            raise ValueError(f"{name}: no HR/LR pairs matched in {src['hr_dir']} / {src['lr_dir']}")
        pairs += [{"id": f"{name}/{s}", "stem": s, "source": name, "hr": str(hr[s]), "lr": str(lr[s])}
                  for s in common]
    ids = [p["id"] for p in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pair ids")
    return pairs


def excluded_stems(dirs):
    out = set()
    for d in dirs or []:
        out |= {normalise_stem(p.name) for p in _list(d, "*")}
    return out


def split_pairs(pairs, val_fraction, salt="lornet-val", exclude=()):
    """Deterministic train/val split by sha256(salt:id); raises on any overlap with `exclude` stems."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    leak = sorted({p["stem"] for p in pairs} & set(exclude))
    if leak:
        raise ValueError(f"{len(leak)} training stems also appear in the evaluation folders, e.g. {leak[:5]}")
    train, val = [], []
    for p in pairs:
        h = int.from_bytes(hashlib.sha256(f"{salt}:{p['id']}".encode()).digest()[:8], "big") / 2 ** 64
        (val if h < val_fraction else train).append(p)
    if not train or not val:
        raise ValueError(f"split produced {len(train)} train / {len(val)} val images; need both")
    return train, val


def ids_digest(pairs):
    return hashlib.sha256("\n".join(sorted(p["id"] for p in pairs)).encode()).hexdigest()


def load_pair(pair, scale):
    """-> (LR PIL at 1/scale, HR PIL, lr_mode) with HR cropped to a multiple of scale."""
    from PIL import Image
    hr = Image.open(pair["hr"]).convert("RGB")
    lr = Image.open(pair["lr"]).convert("RGB")
    W, H = hr.size
    W, H = W // scale * scale, H // scale * scale
    hr = hr.crop((0, 0, W, H))
    if lr.size[0] >= W and lr.size[1] >= H:                       # registered at HR size
        lr = lr.crop((0, 0, W, H)).resize((W // scale, H // scale), Image.BICUBIC)
        mode = "downscaled_from_hr_size"
    else:
        if abs(lr.size[0] * scale - hr.size[0]) > scale or abs(lr.size[1] * scale - hr.size[1]) > scale:
            raise ValueError(f"{pair['id']}: LR {lr.size} is neither HR size nor 1/{scale} of HR {hr.size}")
        lr = lr.crop((0, 0, W // scale, H // scale))
        mode = "native"
    return lr, hr, mode


def crop_boxes(W, H, crop, n):
    """Up to n top-left corners of crop x crop boxes on an even grid inside W x H."""
    if W < crop or H < crop:
        return []
    nx, ny = max(1, (W - crop) // crop + 1), max(1, (H - crop) // crop + 1)
    xs = [round(i * (W - crop) / max(nx - 1, 1)) for i in range(nx)] if nx > 1 else [0]
    ys = [round(j * (H - crop) / max(ny - 1, 1)) for j in range(ny)] if ny > 1 else [0]
    boxes = [(x, y) for y in ys for x in xs]
    if len(boxes) > n:
        step = len(boxes) / n
        boxes = [boxes[int(i * step)] for i in range(n)]
    return boxes


def build_cache(pairs, split, out_dir, vae, cfg, log=print):
    """Encode crops of `pairs` and write <out_dir>/<split>.pt; returns the manifest entry."""
    import numpy as np
    import torch

    from cfa_seesr.latent_targets import (crop_common, discrepancy, encode_latent, pil_to_tensor01,
                                          upscale_like_upstream)
    d = cfg["data"]
    scale, crop = int(d["scale"]), int(d["crop_hr"])
    if crop % (8 * scale):
        raise ValueError(f"crop_hr must be a multiple of 8 * scale = {8 * scale}")
    tg = cfg["target"]
    items, modes = [], {}
    for i, pair in enumerate(pairs):
        lr, hr, mode = load_pair(pair, scale)
        modes[mode] = modes.get(mode, 0) + 1
        for (x, y) in crop_boxes(hr.size[0], hr.size[1], crop, int(d["crops_per_image"])):
            hr_c = hr.crop((x, y, x + crop, y + crop))
            lr_c = lr.crop((x // scale, y // scale, (x + crop) // scale, (y + crop) // scale))
            cond = upscale_like_upstream(lr_c, scale, int(d.get("process_size", 512)))
            z_lr = encode_latent(vae, pil_to_tensor01(cond))
            z_hr = encode_latent(vae, pil_to_tensor01(hr_c))
            z_lr, z_hr = crop_common(z_lr, z_hr)
            E = discrepancy(z_lr, z_hr, tg["lowpass"], int(tg["window"]))[0, 0]
            items.append({"id": pair["id"], "box": [x, y], "lr": torch.from_numpy(np.array(lr_c, dtype=np.uint8)),
                          "E": E.half().cpu()})
        if (i + 1) % 20 == 0:
            log(f"[cache {split}] {i + 1}/{len(pairs)} images, {len(items)} crops")
    if not items:
        raise ValueError(f"{split}: no crops (images smaller than crop_hr={crop}?)")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"items": items, "split": split}, out / f"{split}.pt")
    return {"split": split, "n_images": len(pairs), "n_crops": len(items), "ids_sha256": ids_digest(pairs),
            "lr_modes": modes}


class CropDataset:
    """Torch dataset over a cached split: (conditioning image [3, H, W] in [0, 1], E [1, h, w]).

    augment: random flips / 90-degree rotations applied identically to the image and the latent-grid map (both
    grids are aligned: H = 8 h)."""

    def __init__(self, cache_file, cfg, augment=False):
        import torch
        blob = torch.load(cache_file, map_location="cpu")
        self.items = blob["items"]
        self.scale = int(cfg["data"]["scale"])
        self.process_size = int(cfg["data"].get("process_size", 512))
        self.augment = augment     # random draws from torch's global RNG: seeded by seed_everything in the main
                                   # process and by the DataLoader generator in worker processes (reproducible)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch
        from PIL import Image

        from cfa_seesr.latent_targets import pil_to_tensor01, upscale_like_upstream
        it = self.items[idx]
        cond = upscale_like_upstream(Image.fromarray(it["lr"].numpy()), self.scale, self.process_size)
        img = pil_to_tensor01(cond)[0]
        E = it["E"].float()[None]
        h, w = E.shape[-2:]
        img = img[:, : h * 8, : w * 8]
        if self.augment:
            r = int(torch.randint(0, 8, (1,)))
            if r & 1:
                img, E = img.flip(-1), E.flip(-1)
            if r & 2:
                img, E = img.flip(-2), E.flip(-2)
            if r & 4:
                img, E = img.transpose(-1, -2), E.transpose(-1, -2)
        return img.contiguous(), E.contiguous()


def write_manifest(out_dir, manifest):
    Path(out_dir, "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_manifest(out_dir):
    p = Path(out_dir, "manifest.json")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: build the cache first (train_lornet.py cache ...)")
    return json.loads(p.read_text())
