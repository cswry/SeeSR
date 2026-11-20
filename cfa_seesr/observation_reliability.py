"""SFR-OC part A: local observation-reliability analyzer (training-free, interpretable heuristics).

Analysis domain. When the runner provides the ORIGINAL LR image (the image DAPE receives; before any upscaling),
the analysis runs on it, so the noise estimate and every local statistic refer to the same pixels and the same
noise. Otherwise (direct pipeline use, tests) the conditioning image (upscaled LR, [B,3,H,W] in [0,1]) is
area-downsampled by `analysis_scale` and used instead; `noise_source` records which domain was used. The CFG batch
duplicates one image: all batch items are checked to be identical (a deferred device check, see `checks`) and
item 0 is analysed. Y = BT.601 luminance; local windows are Gaussian with sigma_w = window / 4 pixels.

Noise model and estimate. With the original LR ("white" model): sigma_n = 1.4826 * median(|L|) / 6 over the
low-gradient pixels, L = Immerkaer's operator [[1,-2,1],[-2,4,-2],[1,-2,1]] (response to white noise: std
6 sigma). Low-gradient pixels: gradient magnitude after Gaussian smoothing with sigma 2 (so the mask follows
structure rather than the noise itself, which limits selection bias) at or below `low_gradient_quantile`.
Assumptions: additive, white, signal-independent, roughly Gaussian noise; real LR inputs (demosaicking, JPEG,
resampling) violate them and the estimate is then a heuristic. Each statistic's expected noise contribution
("floor" f_k) follows from sigma_n and the statistic's filter (sum of squared filter taps).
Without the LR ("empirical" model, the resampled conditioning image): the upscaling makes the noise spatially
correlated, so the white model would underestimate it; each floor f_k is instead measured directly as the median
of statistic k over the low-gradient pixels. Limitation: an image with no flat region (texture everywhere) then
gets floors that contain signal, and its indicators are suppressed toward 0 (conservative).

Indicators ("high" = more trustworthy observed structure), from noise-compensated statistics
x_s = max(x - kappa_k f_k, 0). kappa_k = 1 + z * (relative std of statistic k under the noise model, computed
from its filter and the window), z = `significance` (default 2): a margin for the positive bias of max(., 0).
Significance-type normalisation SNR(x_s) = x_s / (x_s + kappa_k f_k + a_k), a_k = an absolute reference energy
(`reference_level`^2, or (level/2)^2): for a clean image a saturating function of the signal energy, and for
noise only a ratio of noise-scaled quantities, i.e. bounded and independent of the noise level. No per-image
rescaling, so values are comparable between images and adding noise cannot raise an indicator by renormalisation.

  edge_coherence   SNR(l1') * coherence'; l1', l2' = structure-tensor eigenvalues (Sobel/8 gradients, window
                   sigma_w = window / 4) minus kappa f_struct; coherence' = ((l1'-l2')/(l1'+l2'+f+floor))^2.
  frequency_energy SNR of the window energy of the difference of Gaussians G1 Y - G2 Y.
  contrast         SNR of the local variance.
  noise            (var_s + floor^2) / (var_s + sigma_n^2 + floor^2): share of local variance that is signal;
                   `noise_floor` (one 8-bit level) keeps clean flat areas from being called unreliable.
  sharpness        on Y_s = G_pre Y (`presmooth_sigma`): with compensated gradient energies G (Y_s) and G_b
                   (Y_s re-blurred, sigma 1.5), sharpness = (sqrt G - sqrt G_b) / (sqrt G + sqrt(kappa f_G) + floor):
                   the share of the gradient a mild re-blur removes; low for blurred structure and for noise.

O = sum_k w_k I_k / sum_k w_k (weights >= 0, default 1), Gaussian-smoothed (smoothing_sigma), in [0, 1].
Behaviour checked by tests (experiments/tests/test_sfroc_remediation.py): the mean of O does not increase with
added white noise (averaged over noise draws) for flat, textured, edge and mixed scenes with the original LR,
and for scenes with flat regions without it. O is a heuristic score, NOT a calibrated probability.

Limitations: the indicators are correlated (most measure local signal activity); compensation assumes the noise
model above; windows are fixed; agreement with ground truth is tested separately (tools/sfroc_map_validity.py,
and against the LORNet teacher in experiments/training).

No host synchronisation: every statistic stays a device tensor; `stats` holds tensors that the controller
converts once after inference.
"""
import math
from dataclasses import dataclass, field

INDICATORS = ("edge_coherence", "frequency_energy", "contrast", "noise", "sharpness")
IMMERKAER = ((1.0, -2.0, 1.0), (-2.0, 4.0, -2.0), (1.0, -2.0, 1.0))


def _default_weights():
    return {k: 1.0 for k in INDICATORS}


@dataclass(frozen=True)
class ObservationReliabilityConfig:
    analysis_scale: int = 4                  # fallback only: area-downsampling of the conditioning image
    window: int = 7                          # local window (pixels at analysis resolution)
    weights: dict = field(default_factory=_default_weights)
    reference_level: float = 0.02            # absolute normalisation N(x) = x / (x + c_k), c_k ~ this (luminance)
    significance: float = 2.0                # z: noise variance subtracted with margin kappa = 1 + z sqrt(2/n_eff)
    smoothing_sigma: float = 1.0             # Gaussian smoothing of O (analysis pixels); 0 = none
    low_gradient_quantile: float = 0.5       # pixels used by the noise estimate
    noise_floor: float = 1.0 / 255.0         # stabilising floor (std units, ~one 8-bit level)
    presmooth_sigma: float = 1.0             # pre-smoothing before the sharpness indicator

    def __post_init__(self):
        if self.analysis_scale < 1 or self.window < 3:
            raise ValueError("observation: analysis_scale >= 1 and window >= 3 required")
        unknown = set(self.weights) - set(INDICATORS)
        if unknown:
            raise ValueError(f"observation.weights: unknown indicators {sorted(unknown)}; expected {INDICATORS}")
        if any(v < 0 or not math.isfinite(v) for v in self.weights.values()):
            raise ValueError("observation.weights must be finite and >= 0")
        if sum(self.weights.get(k, 0.0) for k in INDICATORS) <= 0:
            raise ValueError("observation.weights: at least one weight must be > 0")
        if not self.reference_level > 0 or self.significance < 0:
            raise ValueError("observation: reference_level > 0 and significance >= 0 required")
        if self.smoothing_sigma < 0 or self.presmooth_sigma < 0:
            raise ValueError("observation: smoothing_sigma and presmooth_sigma must be >= 0")
        if not 0.0 < self.low_gradient_quantile <= 1.0:
            raise ValueError("observation.low_gradient_quantile must be in (0, 1]")
        if self.noise_floor <= 0:
            raise ValueError("observation.noise_floor must be > 0")

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        if "weights" in d:   # unspecified indicators keep their default weight; set one to 0 to ablate it
            d["weights"] = {**_default_weights(), **dict(d["weights"])}
        return cls(**d)


def _blur(x, sigma):
    from .frequency import gaussian_lowpass
    return gaussian_lowpass(x, sigma)


def _sobel_kernels(device):
    import torch
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device) / 8.0
    return kx, kx.t().contiguous()


def _sobel(y):
    import torch.nn.functional as F
    kx, ky = _sobel_kernels(y.device)
    p = F.pad(y, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(p, kx.view(1, 1, 3, 3)), F.conv2d(p, ky.view(1, 1, 3, 3))


def _impulse_response(fn, size=33, device="cpu"):
    import torch
    d = torch.zeros(1, 1, size, size, device=device)
    d[..., size // 2, size // 2] = 1.0
    return fn(d)


def _autocorr(h):
    import torch.nn.functional as F
    n = h.shape[-1]
    return F.conv2d(F.pad(h, (n // 2, n // 2, n // 2, n // 2)), h)       # [1,1,n,n], lag 0 at the centre


def relative_std(filter_fn, ws, device="cpu"):
    """Relative std of the window average E_w[(h * n)^2] for white noise n: sqrt(2 sum_lag A_w(lag) rho_h(lag)^2),
    A_w = autocorrelation of the window weights, rho_h = normalised autocorrelation of the filter h. The
    significance margin of each statistic uses its own value (correlated filters fluctuate more)."""
    h = _impulse_response(filter_fn, device=device)
    w = _impulse_response(lambda t: _blur(t, ws), device=device)
    rho = _autocorr(h)
    rho = rho / rho[..., rho.shape[-2] // 2, rho.shape[-1] // 2]
    return (2.0 * (_autocorr(w) * rho * rho).sum()).sqrt()


def _snr(x_s, floor_noise, a):
    """Significance-type indicator x_s / (x_s + noise floor + a): for a clean image a saturating function of the
    signal energy (a = absolute reference energy); for noise only, bounded and independent of the noise level."""
    return x_s / (x_s + floor_noise + a)


def _saturate(x, c):
    """Absolute normalisation x / (x + c) in [0, 1): monotone, the same for every image (no per-image rescaling,
    so a noise-compensated statistic that decreases with noise gives an indicator that decreases with noise)."""
    return x / (x + c)


def _kernel_energy(fn, size=33, device="cpu"):
    """sum of squares of the impulse response of a linear filter fn ([1,1,H,W] -> same)."""
    import torch
    d = torch.zeros(1, 1, size, size, device=device)
    d[..., size // 2, size // 2] = 1.0
    return (fn(d) ** 2).sum()


def low_gradient_mask(y, q=0.5):
    """[1,1,H,W] bool: pixels whose gradient magnitude after Gaussian smoothing (sigma 2: the mask follows
    structure, not the noise itself, which limits selection bias) is at or below its q-quantile."""
    import torch
    gx, gy = _sobel(_blur(y, 2.0))
    g = (gx * gx + gy * gy).sqrt()
    g = g.float()          # torch.quantile accepts only float32/float64 (fp16 under CUDA autocast failed)
    return g <= torch.quantile(g.flatten(), q)


def _masked_median(x, mask):
    import torch
    return torch.nanmedian(torch.where(mask, x, torch.full_like(x, float("nan"))))


def estimate_noise_sigma(y, low_gradient_quantile=0.5):
    """Robust white-noise std of a [1, 1, H, W] image (device tensor, 0-dim). 0 for images smaller than 3x3."""
    import torch
    import torch.nn.functional as F
    H, W = y.shape[-2:]
    if H < 3 or W < 3:
        return torch.zeros((), device=y.device)
    k = torch.tensor(IMMERKAER, device=y.device).view(1, 1, 3, 3)
    L = F.conv2d(y, k)                                             # valid region, (H-2) x (W-2)
    mask = low_gradient_mask(y, low_gradient_quantile)[..., 1:-1, 1:-1]
    return 1.4826 * _masked_median(L.abs(), mask) / 6.0


def immerkaer_sigma(y):
    """Immerkaer (1996) global estimate (mean |L|; not robust to edges). Kept for comparison/tests."""
    import torch
    import torch.nn.functional as F
    H, W = y.shape[-2:]
    if H < 3 or W < 3:
        return 0.0
    k = torch.tensor(IMMERKAER, device=y.device).view(1, 1, 3, 3)
    r = F.conv2d(y, k).abs().sum()
    return float(r * math.sqrt(math.pi / 2.0) / (6.0 * (W - 2) * (H - 2)))


def _to_tensor_image(img, device):
    """PIL / numpy HxWx3 uint8 / tensor [3,H,W] or [1,3,H,W] in [0,1] -> [1,3,H,W] float32 on device."""
    import numpy as np
    import torch
    if isinstance(img, torch.Tensor):
        t = img.float()
        if t.ndim == 3:
            t = t[None]
        return t.to(device)
    a = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(device)


def analyse(image, cfg=None, lr=None):
    """Observation-reliability map.

    image: conditioning image [B, 3, H, W] in [0, 1]; lr: optional original LR image (PIL, array or tensor).
    Returns (O, components, stats, checks):
      O           [h, w] float32 in [0, 1] at the analysis resolution
      components  {indicator: [h, w] map in [0, 1]}
      stats       dict of 0-dim device tensors / python values (no host sync); see tensor_stats_to_python()
      checks      {"cfg_batch_identical": 0-dim bool tensor}
    """
    import torch
    import torch.nn.functional as F
    cfg = cfg or ObservationReliabilityConfig()
    checks = {"cfg_batch_identical": (image == image[0:1]).all()}
    if lr is not None:
        x = _to_tensor_image(lr, image.device).clamp(0.0, 1.0)
        source = "original_lr"
        noise_model = "white"
    else:
        x = image[0:1].float().clamp(0.0, 1.0)
        if cfg.analysis_scale > 1:
            x = F.adaptive_avg_pool2d(x, (max(1, x.shape[-2] // cfg.analysis_scale),
                                          max(1, x.shape[-1] // cfg.analysis_scale)))
        source = "conditioning_downsampled"
        noise_model = "empirical"
    y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    ws = cfg.window / 4.0
    c = cfg.reference_level
    floor2 = cfg.noise_floor ** 2
    eps = 1e-12
    # raw local statistics (signal + noise)
    sobel_energy = 12.0 / 64.0                                      # sum of squares of Sobel/8
    gx, gy = _sobel(y)
    jxx, jyy, jxy = _blur(gx * gx, ws), _blur(gy * gy, ws), _blur(gx * gy, ws)
    tr = jxx + jyy
    disc = ((jxx - jyy) ** 2 + 4 * jxy ** 2).clamp_min(0).sqrt()
    dog = lambda t: _blur(t, 1.0) - _blur(t, 2.0)  # noqa: E731
    band = dog(y)
    e_raw = _blur(band * band, ws)
    mean = _blur(y, ws)
    var = (_blur(y * y, ws) - mean * mean).clamp_min(0)
    ys = _blur(y, cfg.presmooth_sigma)
    sx, sy = _sobel(ys)
    bx, by = _sobel(_blur(ys, 1.5))
    G_raw = _blur(sx * sx + sy * sy, ws)
    Gb_raw = _blur(bx * bx + by * by, ws)

    # expected noise contribution ("floor") of each statistic
    if noise_model == "white":       # original LR: white-noise model, sigma from Immerkaer on low-gradient pixels
        sigma_n = estimate_noise_sigma(y, cfg.low_gradient_quantile)
        noise_var = sigma_n ** 2
        f_struct = noise_var * sobel_energy                         # per eigenvalue (isotropic noise tensor)
        f_band = noise_var * _kernel_energy(dog, device=y.device)
        f_var = noise_var
        f_G = 2.0 * noise_var * _kernel_energy(lambda t: _sobel(_blur(t, cfg.presmooth_sigma))[0], device=y.device)
        f_Gb = 2.0 * noise_var * _kernel_energy(lambda t: _sobel(_blur(_blur(t, cfg.presmooth_sigma), 1.5))[0],
                                                device=y.device)
    else:                            # resampled image: noise is correlated, so measure each floor directly as the
        mask = low_gradient_mask(y, cfg.low_gradient_quantile)      # median of the statistic over flat pixels
        f_var = _masked_median(var, mask)
        f_struct = _masked_median(tr / 2, mask)
        f_band = _masked_median(e_raw, mask)
        f_G = _masked_median(G_raw, mask)
        f_Gb = _masked_median(Gb_raw, mask)
        noise_var = f_var
        sigma_n = f_var.sqrt()

    # significance margins kappa_k = 1 + z * relative std of statistic k under the noise model
    z = cfg.significance
    dev = y.device
    k_var = 1.0 + z * relative_std(lambda t: t, ws, dev)
    k_struct = 1.0 + z * relative_std(lambda t: _sobel(t)[0], ws, dev)
    k_band = 1.0 + z * relative_std(dog, ws, dev)
    k_G = 1.0 + z * relative_std(lambda t: _sobel(_blur(t, cfg.presmooth_sigma))[0], ws, dev)
    k_Gb = 1.0 + z * relative_std(lambda t: _sobel(_blur(_blur(t, cfg.presmooth_sigma), 1.5))[0], ws, dev)

    # structure tensor, noise-compensated; strength as a significance ratio of the dominant eigenvalue
    l1 = ((tr + disc) / 2 - k_struct * f_struct).clamp_min(0)
    l2 = ((tr - disc) / 2 - k_struct * f_struct).clamp_min(0)
    coherence = ((l1 - l2) / (l1 + l2 + f_struct + floor2 * sobel_energy + eps)) ** 2
    edge = (_snr(l1, k_struct * f_struct, (0.5 * c) ** 2) * coherence).clamp(0, 1)

    # band energy (difference of Gaussians), noise-compensated
    e_s = (e_raw - k_band * f_band).clamp_min(0)
    energy = _snr(e_s, k_band * f_band, (0.5 * c) ** 2)

    # local variance, noise-compensated
    var_s = (var - k_var * f_var).clamp_min(0)
    contrast = _snr(var_s, k_var * f_var, c * c)
    noise = ((var_s + floor2) / (var_s + noise_var + floor2)).clamp(0, 1)

    # sharpness from noise-compensated gradient ENERGIES of the pre-smoothed image, before and after a mild
    # re-blur: the share of the gradient that the re-blur removes (low for blurred structure; for noise only a
    # ratio of two noise-scaled quantities, i.e. independent of the noise level)
    g = (G_raw - k_G * f_G).clamp_min(0).sqrt()
    gb = (Gb_raw - k_Gb * f_Gb).clamp_min(0).sqrt()
    sharpness = ((g - gb) / (g + (k_G * f_G).sqrt() + cfg.noise_floor * math.sqrt(sobel_energy) + eps)).clamp(0, 1)

    comps = {"edge_coherence": edge, "frequency_energy": energy, "contrast": contrast, "noise": noise,
             "sharpness": sharpness}
    total = sum(cfg.weights.get(k, 0.0) for k in INDICATORS)
    O = sum(cfg.weights.get(k, 0.0) * comps[k] for k in INDICATORS) / total
    if cfg.smoothing_sigma > 0:
        O = _blur(O, cfg.smoothing_sigma)
    O = O.clamp(0.0, 1.0)
    checks["O_finite"] = torch.isfinite(O).all()
    comps = {k: v[0, 0] for k, v in comps.items()}
    stats = {"analysis_size": [int(O.shape[-2]), int(O.shape[-1])], "noise_source": source,
             "noise_sigma_estimate": sigma_n.detach(), "noise_model": noise_model, "O_mean": O.mean(), "O_min": O.min(), "O_max": O.max(),
             **{f"{k}_mean": v.mean() for k, v in comps.items()},
             "weights": {k: cfg.weights.get(k, 0.0) for k in INDICATORS}}
    return O[0, 0], comps, stats, checks


def tensor_stats_to_python(stats):
    """Convert 0-dim tensors in a stats dict to floats (one host transfer per value; call after inference)."""
    import torch
    out = {}
    for k, v in stats.items():
        out[k] = float(v) if isinstance(v, torch.Tensor) and v.numel() == 1 else v
    return out


def resize_map(m, size):
    """[h, w] map -> [H, W] (area when shrinking, bilinear when enlarging); stays within [min, max] of m."""
    import torch.nn.functional as F
    t = m.float()[None, None]
    H, W = size
    if H <= t.shape[-2] and W <= t.shape[-1]:
        out = F.adaptive_avg_pool2d(t, (H, W))
    else:
        out = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return out[0, 0].clamp(m.min(), m.max())
