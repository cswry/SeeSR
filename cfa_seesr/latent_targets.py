"""Latent-space quantities shared by LORNet training (targets) and the map-validity tool (counterfactual test).

All images go through ONE path, the one SeeSR inference uses:
  * LR -> conditioning image: upstream test_seesr.py resizing (enlarge to process_size // upscale if smaller,
    x upscale, then floor to a multiple of 8), bicubic (PIL's default filter for RGB images, made explicit);
  * image in [0, 1] -> latent: vae.encode(image * 2 - 1).latent_dist.MEAN * vae.config.scaling_factor
    (the pipeline samples the posterior for its LR start; the mean is used here so targets are deterministic).
HR and SR images are encoded the same way; pairs are cropped to their common latent size (top-left anchored, as
the datasets are registered at the top-left corner).

Teacher target for LORNet (a PROXY for observation fidelity, not proof that correcting toward z_lr helps):
    E        = LocalAverage_k( mean_c | LPF(z_lr - z_hr) | )          [B, 1, h, w]
    R_teacher = exp(-E / tau)                                         in (0, 1]
tau = median(E over all training pixels) / ln 2  (estimated on the TRAINING split only, saved with the model), so
a pixel with the median training discrepancy has target 0.5.

Counterfactual benefit (map-validity test): with the predefined low-frequency correction toward the LR latent
    d  = LPF(z_lr - z_sr)                       (what SFR-OC would add, uniform W, gamma)
    e  = LPF(z_sr - z_gt)                       (low-frequency error of the uncorrected output)
    benefit = mean_c |e| - mean_c |e + gamma d|  (> 0: the correction reduces the error here)
and the audit's screening indicator g = mean_c |e| - mean_c |LPF(z_sr - z_lr)|.
"""
import math

DEFAULT_LOWPASS = {"method": "gaussian", "sigma": 3.0, "cutoff": 0.25}


def upscale_like_upstream(lr, upscale=4, process_size=512):
    """PIL LR image -> PIL conditioning image exactly as test_seesr.py resizes it (bicubic)."""
    from PIL import Image
    img = lr.convert("RGB")
    w, h = img.size
    if w < process_size // upscale or h < process_size // upscale:
        scale = (process_size // upscale) / min(w, h)
        img = img.resize((int(scale * w), int(scale * h)), Image.BICUBIC)
    img = img.resize((img.size[0] * upscale, img.size[1] * upscale), Image.BICUBIC)
    return img.resize((img.size[0] // 8 * 8, img.size[1] // 8 * 8), Image.BICUBIC)


def pil_to_tensor01(img):
    import numpy as np
    import torch
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None].contiguous()


def encode_latent(vae, img01):
    """[B, 3, H, W] in [0, 1] -> posterior mean * scaling_factor, float32. H, W multiples of 8."""
    import torch
    if img01.shape[-1] % 8 or img01.shape[-2] % 8:
        raise ValueError(f"image size {tuple(img01.shape[-2:])} is not a multiple of 8")
    p = next(vae.parameters())
    with torch.no_grad():
        x = (img01 * 2 - 1).to(device=p.device, dtype=p.dtype)
        z = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
    return z.float()


def crop_common(*zs):
    h = min(z.shape[-2] for z in zs)
    w = min(z.shape[-1] for z in zs)
    return [z[..., :h, :w] for z in zs]


def _lp(x, lowpass):
    from .frequency import lowpass as lp
    cfg = {**DEFAULT_LOWPASS, **(lowpass or {})}
    return lp(x, cfg["method"], cfg["sigma"], cfg["cutoff"])


def local_average(x, window):
    """Box average over window x window (reflect padding), same size."""
    import torch.nn.functional as F
    if window <= 1:
        return x
    r = window // 2
    pad = F.pad(x, (r, window - 1 - r, r, window - 1 - r), mode="replicate")
    return F.avg_pool2d(pad, window, stride=1)


def discrepancy(z_lr, z_hr, lowpass=None, window=3):
    """E [B, 1, h, w] = LocalAverage(mean_c |LPF(z_lr - z_hr)|)."""
    z_lr, z_hr = crop_common(z_lr.float(), z_hr.float())
    e = _lp(z_lr - z_hr, lowpass).abs().mean(dim=1, keepdim=True)
    return local_average(e, window)


def estimate_tau(E_values):
    """tau from TRAINING discrepancies (iterable of tensors): median / ln 2 (median pixel -> target 0.5)."""
    import torch
    flat = torch.cat([e.flatten().float().cpu() for e in E_values])
    if flat.numel() == 0:
        raise ValueError("no training discrepancies to estimate tau from")
    med = float(flat.median())
    if not med > 0:
        raise ValueError(f"median training discrepancy is {med}; cannot set tau")
    return med / math.log(2.0)


def teacher_target(E, tau):
    import torch
    if not tau > 0:
        raise ValueError("tau must be > 0")
    return torch.exp(-E / tau)


def counterfactual(z_sr, z_gt, z_lr, gamma=1.0, lowpass=None):
    """Per-location benefit of the predefined low-frequency LR correction and the audit indicator g.
    Returns dict of [B, 1, h, w] tensors: benefit, g, err_before, err_after."""
    z_sr, z_gt, z_lr = crop_common(z_sr.float(), z_gt.float(), z_lr.float())
    e = _lp(z_sr - z_gt, lowpass)
    d = _lp(z_lr - z_sr, lowpass)
    before = e.abs().mean(dim=1, keepdim=True)
    after = (e + gamma * d).abs().mean(dim=1, keepdim=True)
    g = before - d.abs().mean(dim=1, keepdim=True)
    return {"benefit": before - after, "g": g, "err_before": before, "err_after": after}
