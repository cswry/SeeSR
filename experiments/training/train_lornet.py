"""LORNet training CLI (CFA-SeeSR V2): build the frozen-VAE latent cache, train, evaluate.

  python experiments/training/train_lornet.py cache    --config experiments/training/configs/lornet.yaml
  python experiments/training/train_lornet.py train    --config experiments/training/configs/lornet.yaml
  python experiments/training/train_lornet.py evaluate --config experiments/training/configs/lornet.yaml \\
         [--checkpoint <output_dir>/checkpoints/best.pt]

Overrides: --set key.sub=value (YAML value; list items by index, e.g. data.sources.0.hr_dir=/path).

Independent of the inference notebook and of SeeSR's UNet/ControlNet/DAPE: only the SD2 VAE is loaded (frozen,
no_grad) by `cache`; `train` and `evaluate` read the cache only. Nothing is back-propagated through SeeSR.

Outputs in output_dir:
  cache/manifest.json, cache/train.pt, cache/val.pt      crops, discrepancy maps E, tau, split digests
  checkpoints/best.pt, checkpoints/last.pt                 cfa-seesr-lornet checkpoints (model + tau + provenance)
  train_log.jsonl                                          one line per epoch
  evaluation.json                                          validation metrics of LORNet and of the heuristic map
"""
import argparse
import contextlib
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from experiments.training import data as D  # noqa: E402
from experiments.training import metrics as M  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def set_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[int(k)] if isinstance(node, list) else node[k]
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        if last not in node:
            raise KeyError(f"unknown config key {dotted}")
        node[last] = value


def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides:
        k, _, v = item.partition("=")
        set_path(cfg, k.strip(), yaml.safe_load(v))
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    from cfa_seesr.lornet import validate_model_config
    for sec in ("data", "target", "model", "train", "vae"):
        if sec not in cfg:
            raise ValueError(f"config section '{sec}' missing")
    validate_model_config(cfg["model"])
    t = cfg["train"]
    if t["loss"] not in M.LOSSES:
        raise ValueError(f"train.loss must be one of {M.LOSSES}")
    if t["loss"] == "gaussian_nll" and not cfg["model"].get("uncertainty"):
        raise ValueError("train.loss gaussian_nll needs model.uncertainty: true")
    if t["tv_weight"] < 0 or t["epochs"] < 1 or t["batch_size"] < 1:
        raise ValueError("train: tv_weight >= 0, epochs >= 1, batch_size >= 1")
    if t.get("selection_metric", "val_loss") != "val_loss":
        raise ValueError("train.selection_metric: only val_loss is supported")


def out_dir(cfg):
    p = Path(cfg["output_dir"])
    return p if p.is_absolute() else REPO / p


def seed_everything(seed, deterministic=True):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass


def device_of():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def load_vae(cfg, device):
    import torch
    from diffusers import AutoencoderKL
    path = Path(cfg["vae"]["pretrained_model_path"])
    path = path if path.is_absolute() else REPO / path
    vae = AutoencoderKL.from_pretrained(str(path), subfolder="vae")
    dtype = torch.float16 if (device.type == "cuda" and cfg["vae"].get("dtype") == "float16") else torch.float32
    vae = vae.to(device=device, dtype=dtype).eval()
    vae.requires_grad_(False)
    return vae


def cmd_cache(cfg, vae=None, log=print):
    import torch

    from cfa_seesr.latent_targets import estimate_tau
    d = cfg["data"]
    pairs = D.discover_pairs(d["sources"])
    train, val = D.split_pairs(pairs, float(d["val_fraction"]), exclude=D.excluded_stems(d.get("exclude_dirs")))
    vae = vae if vae is not None else load_vae(cfg, device_of())
    cache = out_dir(cfg) / "cache"
    entries = [D.build_cache(train, "train", cache, vae, cfg, log), D.build_cache(val, "val", cache, vae, cfg, log)]
    E_train = [it["E"] for it in torch.load(cache / "train.pt", map_location="cpu")["items"]]
    tau = estimate_tau(E_train)
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "tau": tau,
                "tau_rule": "median(E over training crops) / ln 2", "target": cfg["target"], "data": cfg["data"],
                "splits": {e["split"]: e for e in entries},
                "vae": {"pretrained_model_path": cfg["vae"]["pretrained_model_path"],
                        "dtype": str(next(vae.parameters()).dtype),
                        "scaling_factor": float(vae.config.scaling_factor),
                        "encoding": "posterior mean * scaling_factor of (image * 2 - 1)"}}
    D.write_manifest(cache, manifest)
    log(f"cache written to {cache}: {entries[0]['n_crops']} train / {entries[1]['n_crops']} val crops, tau={tau:.5f}")
    return manifest


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def _seed_worker(base, worker_id):
    import numpy as np
    random.seed(base + worker_id)
    np.random.seed((base + worker_id) % 2 ** 32)


def _loader(ds, cfg, shuffle, seed):
    import functools

    import torch
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=shuffle,
                                       num_workers=int(cfg["train"].get("num_workers", 0)), generator=g,
                                       worker_init_fn=functools.partial(_seed_worker, seed), drop_last=False)


def run_epoch(model, loader, cfg, tau, device, optimizer=None, scaler=None):
    import torch

    from cfa_seesr.latent_targets import teacher_target
    from cfa_seesr.lornet import compute_features
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    use_amp = bool(cfg["train"].get("amp")) and device.type == "cuda"
    for img, E in loader:
        img, E = img.to(device), E.to(device)
        target = teacher_target(E, tau)
        amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
        with torch.set_grad_enabled(train), amp_ctx:
            logit, logvar = model.forward_features(compute_features(img, model.config["features"]))
        loss, _ = M.reliability_loss(cfg["train"]["loss"], logit.float(),
                                     None if logvar is None else logvar.float(), target, float(cfg["train"]["tv_weight"]))
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total += float(loss) * img.shape[0]
        n += img.shape[0]
    return total / max(n, 1)


def cmd_train(cfg, log=print):
    import torch

    from cfa_seesr.lornet import LORNet, conv_macs, make_checkpoint, parameter_count
    seed = int(cfg["seed"])
    seed_everything(seed, bool(cfg["train"].get("deterministic", True)))
    root = out_dir(cfg)
    manifest = D.read_manifest(root / "cache")
    if manifest["target"] != cfg["target"]:
        raise ValueError("config target differs from the cache's target: rebuild the cache")
    tau = float(manifest["tau"])
    device = device_of()
    tr = D.CropDataset(root / "cache" / "train.pt", cfg, augment=bool(cfg["train"].get("augment", True)))
    va = D.CropDataset(root / "cache" / "val.pt", cfg, augment=False)
    tl, vl = _loader(tr, cfg, True, seed), _loader(va, cfg, False, seed)
    model = LORNet(**cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and cfg["train"].get("amp")) else None
    ck_dir = root / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / "train_log.jsonl"
    log_path.write_text("")
    target = {"tau": tau, **copy.deepcopy(cfg["target"])}
    provenance = {"seed": seed, "config": cfg, "splits": manifest["splits"], "cache_created": manifest["created"],
                  "n_parameters": parameter_count(model)}
    best, best_epoch, patience = math.inf, -1, int(cfg["train"]["early_stopping_patience"])
    for epoch in range(epochs):
        t0 = time.perf_counter()
        tr_loss = run_epoch(model, tl, cfg, tau, device, opt, scaler)
        with torch.no_grad():
            va_loss = run_epoch(model, vl, cfg, tau, device)
        sched.step()
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "lr": opt.param_groups[0]["lr"],
               "seconds": time.perf_counter() - t0}
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        log(json.dumps(rec))
        if va_loss < best:
            best, best_epoch = va_loss, epoch
            torch.save(make_checkpoint(model, target, {**provenance, "epoch": epoch, "val_loss": va_loss,
                                                       "selection": "lowest val_loss"}), ck_dir / "best.pt")
        torch.save(make_checkpoint(model, target, {**provenance, "epoch": epoch, "val_loss": va_loss}),
                   ck_dir / "last.pt")
        if epoch - best_epoch >= patience:
            log(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break
    model.eval()
    summary = {"best_epoch": best_epoch, "best_val_loss": best, "n_parameters": parameter_count(model),
               "macs_512": conv_macs(model, 512, 512), "device": str(device), "tau": tau}
    (root / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def inference_time_ms(model, device, size=512, reps=20):
    import torch
    x = torch.rand(1, 3, size, size, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        if device.type == "cuda":
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                model(x)
            e.record()
            e.synchronize()
            return s.elapsed_time(e) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        return (time.perf_counter() - t0) * 1000 / reps


def cmd_evaluate(cfg, checkpoint=None, log=print):
    import torch

    from cfa_seesr.latent_targets import pil_to_tensor01, teacher_target, upscale_like_upstream
    from cfa_seesr.lornet import conv_macs, load_lornet, parameter_count
    from cfa_seesr.observation_reliability import analyse, resize_map
    from PIL import Image
    root = out_dir(cfg)
    manifest = D.read_manifest(root / "cache")
    ck = Path(checkpoint) if checkpoint else root / "checkpoints" / "best.pt"
    device = device_of()
    model, meta = load_lornet(ck, device)
    tau = float(meta["target"]["tau"])
    if abs(tau - float(manifest["tau"])) > 1e-9 * max(1.0, tau):
        raise ValueError("checkpoint tau differs from the cache tau (different training cache?)")
    items = torch.load(root / "cache" / "val.pt", map_location="cpu")["items"]
    preds, heur, targets, errors = [], [], [], []
    scale, ps = int(cfg["data"]["scale"]), int(cfg["data"].get("process_size", 512))
    with torch.no_grad():
        for it in items:
            lr = Image.fromarray(it["lr"].numpy())
            img = pil_to_tensor01(upscale_like_upstream(lr, scale, ps)).to(device)
            E = it["E"].float()
            h, w = E.shape
            img = img[..., : h * 8, : w * 8]
            preds.append(model(img)[0, 0].cpu())
            O = analyse(img, lr=lr)[0]
            heur.append(resize_map(O, (h, w)).cpu())
            targets.append(teacher_target(E, tau))
            errors.append(E)
    res = {"checkpoint": str(ck), "split": "val", "n_crops": len(items),
           "lornet": M.summarise_maps(preds, targets, errors, calibrated=True),
           "heuristic_O": M.summarise_maps(heur, targets, errors, calibrated=False),
           "n_parameters": parameter_count(model), "macs_512": conv_macs(model, 512, 512),
           "inference_ms_512": inference_time_ms(model, device), "device": str(device),
           "note": ("validation crops of the training datasets (held out by image). Agreement with the teacher "
                    "is a proxy; whether correcting toward the LR latent helps is tested separately by "
                    "experiments/tools/sfroc_map_validity.py on V1 outputs.")}
    (root / "evaluation.json").write_text(json.dumps(res, indent=2))
    log(json.dumps({k: res[k] for k in ("lornet", "heuristic_O", "n_parameters", "inference_ms_512")}, indent=2))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("cache", "train", "evaluate"))
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--checkpoint", default=None, help="evaluate: checkpoint (default <output_dir>/checkpoints/best.pt)")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.command == "cache":
        cmd_cache(cfg)
    elif a.command == "train":
        cmd_train(cfg)
    else:
        cmd_evaluate(cfg, a.checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
