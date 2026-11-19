"""Does a reliability map predict WHERE the low-frequency LR correction helps? (counterfactual screening test)

Uses finished V1 outputs (no diffusion run): for each image, the V1 SR image, its GT and its LR are encoded with
the frozen SeeSR VAE in the SAME way (cfa_seesr/latent_targets.py: upstream resizing of the LR, image * 2 - 1,
posterior mean * scaling_factor, top-left common crop). Then, per latent location,
    benefit = mean_c |LPF(z_sr - z_gt)| - mean_c |LPF(z_sr - z_gt) + gamma LPF(z_lr - z_sr)|
            (> 0 where the predefined low-frequency correction toward the LR latent reduces the error)
    g       = mean_c |LPF(z_sr - z_gt)| - mean_c |LPF(z_sr - z_lr)|        (the audit's screening indicator)
and each candidate map M is compared with them:
    Spearman(M, benefit) per image; top-minus-bottom quartile of M: mean benefit difference;
    aggregate: mean over images with a 95 % percentile-bootstrap CI (images are the unit; 2000 resamples).
Maps: heuristic O (observation_reliability.analyse on the conditioning image, noise estimated on the original LR),
learned O (LORNet, if --lornet_checkpoint), semantic 1 - S (SSR grid of the V1 run, else component A's
reliability, else absent), fused 0.5 O + 0.5 (1 - S) for each O.

Decision per map (fixed in advance): SUPPORTED if the CI lower bounds of BOTH the mean Spearman and the mean
quartile benefit difference are > 0 (at least 10 images); otherwise UNSUPPORTED (or INSUFFICIENT). If no fused map is supported, spatial fusion is not
supported by this test and the uniform map (v2_uniform) stays the candidate.

The benefit is a latent-space PROXY measured on V1 outputs (one correction applied to the final latent, not inside
the diffusion trajectory); it does not replace the paired image-metric evaluation.
Run it on SCREENING / validation images; --split_label test is refused unless --allow_final_test is given.

  python experiments/tools/sfroc_map_validity.py --run_dir <R>/screening/RealSR/combined_spatial_ddpm50_s42 \\
      --lr_dir <screening LR> --gt_dir <screening GT> --pretrained_model_path preset/models/stable-diffusion-2-base \\
      [--lornet_checkpoint <ckpt>] --split_label screening --output <R>/screening/map_validity_RealSR.json
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.training.metrics import bootstrap_ci, quartile_contrast, spearman  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _find(d, stem):
    for p in sorted(Path(d).iterdir()):
        if p.suffix.lower() in IMAGE_EXTS and p.stem == stem:
            return p
    return None


def semantic_maps(run_dir):
    """{image stem: S grid (numpy) or float} from the V1 run's spatial_maps.npz / semantic_confidence.jsonl."""
    import numpy as np
    out = {}
    jl = Path(run_dir) / "semantic_confidence.jsonl"
    if jl.exists():
        for line in jl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("reliability") is not None:
                    out[Path(r["image"]).stem] = float(r["reliability"])
    npz = Path(run_dir) / "spatial_maps.npz"
    if npz.exists():
        with np.load(npz) as z:
            for k in z.files:
                out[Path(k).stem] = z[k]
    return out


def candidate_maps(cond01, lr_pil, latent_hw, S, lornet=None):
    """{name: [h, w] tensor} for the conditioning image [1, 3, H, W]: built on the conditioning image's full
    latent grid (H/8, W/8), as at inference, then cropped top-left to latent_hw (the common SR/GT/LR crop)."""
    full_hw = (cond01.shape[-2] // 8, cond01.shape[-1] // 8)
    maps = _maps_full(cond01, lr_pil, full_hw, S, lornet)
    return {k: v[: latent_hw[0], : latent_hw[1]] for k, v in maps.items()}


def _maps_full(cond01, lr_pil, latent_hw, S, lornet):
    import torch
    import torch.nn.functional as F

    from cfa_seesr.observation_reliability import analyse, resize_map
    maps = {"heuristic_O": resize_map(analyse(cond01, lr=lr_pil)[0], latent_hw).float().cpu()}
    if lornet is not None:
        with torch.no_grad():
            m = lornet(cond01.to(next(lornet.parameters()).device))[0, 0].float().cpu()
        maps["learned_O"] = m[: latent_hw[0], : latent_hw[1]]
    if S is not None:
        if isinstance(S, float):
            s = torch.full(latent_hw, S)
        else:
            s = F.interpolate(torch.as_tensor(S).float()[None, None], size=latent_hw, mode="bilinear",
                              align_corners=False)[0, 0]
        maps["semantic_1mS"] = 1 - s
        for k in [k for k in maps if k.endswith("_O")]:
            maps[f"fused_{k[:-2]}"] = (0.5 * maps[k] + 0.5 * (1 - s)).clamp(0, 1)
    return maps


def evaluate_image(vae, sr_pil, gt_pil, lr_pil, S=None, lornet=None, gamma=1.0, lowpass=None, upscale=4,
                   process_size=512):
    from cfa_seesr.latent_targets import counterfactual, encode_latent, pil_to_tensor01, upscale_like_upstream
    cond = pil_to_tensor01(upscale_like_upstream(lr_pil, upscale, process_size))

    def enc(img):
        t = pil_to_tensor01(img)
        return encode_latent(vae, t[..., : t.shape[-2] // 8 * 8, : t.shape[-1] // 8 * 8])

    cf = counterfactual(enc(sr_pil), enc(gt_pil), encode_latent(vae, cond), gamma, lowpass)
    benefit, g = cf["benefit"][0, 0].cpu(), cf["g"][0, 0].cpu()
    hw = tuple(benefit.shape)
    maps = candidate_maps(cond, lr_pil, hw, S, lornet)
    row = {"latent_h": hw[0], "latent_w": hw[1], "benefit_mean": float(benefit.mean()),
           "frac_benefit_positive": float((benefit > 0).float().mean()), "g_mean": float(g.mean())}
    for name, m in maps.items():
        m = m[: hw[0], : hw[1]]
        row[f"{name}__spearman_benefit"] = spearman(m, benefit)
        row[f"{name}__quartile_benefit"] = quartile_contrast(m, benefit)
        row[f"{name}__spearman_g"] = spearman(m, g)
    return row, list(maps)


MIN_IMAGES = 10


def aggregate(rows, names, min_images=MIN_IMAGES):
    out = {}
    for n in names:
        res = {}
        for key in ("spearman_benefit", "quartile_benefit", "spearman_g"):
            vals = [r[f"{n}__{key}"] for r in rows if r.get(f"{n}__{key}") == r.get(f"{n}__{key}")]
            res[key] = {"n_images": len(vals), "mean": statistics.fmean(vals) if vals else float("nan"),
                        "ci95": bootstrap_ci(vals)}
        lo_s, lo_q = res["spearman_benefit"]["ci95"][0], res["quartile_benefit"]["ci95"][0]
        if res["spearman_benefit"]["n_images"] < min_images:
            res["decision"] = f"INSUFFICIENT (< {min_images} images)"
        else:
            res["decision"] = "SUPPORTED" if (lo_s > 0 and lo_q > 0) else "UNSUPPORTED"
        out[n] = res
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="finished V1 run (sr/sample00, spatial_maps.npz, ...)")
    p.add_argument("--lr_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--pretrained_model_path", default="preset/models/stable-diffusion-2-base")
    p.add_argument("--lornet_checkpoint", default=None)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian low-pass std (latent px), as SFR-OC")
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--process_size", type=int, default=512)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--split_label", required=True, help="e.g. screening / validation (recorded)")
    p.add_argument("--allow_final_test", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    if a.split_label.lower() in ("test", "final", "final_test") and not a.allow_final_test:
        p.error("refusing to screen maps on the final test set (pass --allow_final_test to override knowingly)")
    import torch
    from diffusers import AutoencoderKL
    from PIL import Image
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vae = AutoencoderKL.from_pretrained(a.pretrained_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    lornet = None
    if a.lornet_checkpoint:
        from cfa_seesr.lornet import load_lornet
        lornet, _ = load_lornet(a.lornet_checkpoint, device)
    sr_dir = Path(a.run_dir) / "sr" / "sample00"
    sems = semantic_maps(a.run_dir)
    srs = sorted(q for q in sr_dir.iterdir() if q.suffix.lower() in IMAGE_EXTS)[: a.max_images]
    rows, names = [], []
    for sr_path in srs:
        stem = sr_path.stem
        lr_p, gt_p = _find(a.lr_dir, stem), _find(a.gt_dir, stem)
        if lr_p is None or gt_p is None:
            raise FileNotFoundError(f"{stem}: LR or GT image not found")
        row, names = evaluate_image(vae, Image.open(sr_path).convert("RGB"), Image.open(gt_p).convert("RGB"),
                                    Image.open(lr_p).convert("RGB"), sems.get(stem), lornet, a.gamma,
                                    {"method": "gaussian", "sigma": a.sigma}, a.upscale, a.process_size)
        rows.append({"image": stem, **row})
    if not rows:
        raise SystemExit("no SR images found")
    agg = aggregate(rows, names)
    fused = {k: v["decision"] for k, v in agg.items() if k.startswith("fused_")}
    summary = {"run_dir": str(a.run_dir), "split_label": a.split_label, "n_images": len(rows), "gamma": a.gamma,
               "sigma": a.sigma, "lornet_checkpoint": a.lornet_checkpoint, "maps": agg,
               "spatial_fusion": ("SUPPORTED" if any(v == "SUPPORTED" for v in fused.values()) else
                                  "UNSUPPORTED (keep the uniform map, v2_uniform, as the candidate)") if fused else
                                 "NOT TESTED (no semantic map in the run)",
               "rule": "see tools/sfroc_map_validity.py docstring"}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    with out.with_suffix(".per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for n, v in agg.items():
        print(f"{n}: spearman {v['spearman_benefit']['mean']:+.3f} {v['spearman_benefit']['ci95']}  -> {v['decision']}")
    print("spatial fusion:", summary["spatial_fusion"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
