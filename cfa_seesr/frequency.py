"""Frequency decomposition used by SFR-OC (latent space, per channel).

    x = LPF(x) + HPF(x),   HPF(x) := x - LPF(x)      (exact by construction)

Two low-pass filters (neither is described as perfectly band-limited unless stated):
  gaussian  separable Gaussian, sigma in latent pixels (1 latent pixel = 8 output pixels), reflect padding
            (replicate when the map is smaller than the kernel radius). Transfer function
            H(f) = exp(-2 pi^2 sigma^2 f^2), f in cycles per latent pixel. It is NOT band-limited: it attenuates
            smoothly. Half amplitude at f_1/2 = sqrt(ln 2 / (2 pi^2)) / sigma ~= 0.187 / sigma; for sigma = 3
            that is ~0.0625 cycles/latent px, i.e. a period of ~16 latent px = ~128 output px. (sigma = 3 is NOT a
            "24-output-pixel bandwidth": 24 px is sigma itself, not the passband.)
  fft       ideal radial low-pass in the 2-D DFT domain, cutoff as a fraction of the Nyquist frequency
            (0 < cutoff <= 1; 1 keeps every frequency). The input is reflect-padded by half its size on every side
            before the transform (and cropped after), so opposite image borders no longer wrap into each other; the
            ideal cutoff can still ring (Gibbs) near strong edges. On the padded grid the filter is idempotent; on
            the cropped result it is idempotent up to the padding approximation.

`split_lowpass(x, ...)` applies the "half" operator used twice by the SFR-OC correction
Delta = LPF_h(W * LPF_h(r)) so that, for a spatially uniform W, LPF_h(LPF_h(.)) equals the documented single
filter: Gaussian with sigma / sqrt(2) per pass (Gaussian variances add: two passes = sigma), ideal FFT with the
same cutoff per pass (idempotent). Both are linear, per channel, computed in float32, returned in the input dtype.
"""
import math

METHODS = ("gaussian", "fft")


def _gaussian_kernel(sigma, device):
    import torch
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), radius


def gaussian_lowpass(x, sigma):
    """x: [B, C, H, W]. sigma <= 0 returns x unchanged (no filtering)."""
    import torch.nn.functional as F
    if sigma <= 0:
        return x
    dtype = x.dtype
    t = x.float()
    k, r = _gaussian_kernel(sigma, t.device)
    C = t.shape[1]
    mode = "reflect" if r < min(t.shape[-2:]) else "replicate"
    t = F.pad(t, (r, r, 0, 0), mode=mode)
    t = F.conv2d(t, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    t = F.pad(t, (0, 0, r, r), mode=mode)
    t = F.conv2d(t, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return t.to(dtype)


def _ideal_mask(H, W, cutoff, device):
    import torch
    fy = torch.fft.fftfreq(H, device=device).abs() / 0.5          # 0 .. 1 (Nyquist)
    fx = torch.fft.rfftfreq(W, device=device) / 0.5
    return ((fy[:, None] ** 2 + fx[None, :] ** 2).sqrt() <= cutoff).float()


def fft_lowpass(x, cutoff, pad=True):
    """Ideal radial low-pass; cutoff in (0, 1] as a fraction of Nyquist (1 = identity).

    pad=True (default): reflect-pad by floor(dim/2) (at most dim-1) on each side, filter, crop back.
    """
    import torch
    import torch.nn.functional as F
    if cutoff >= 1.0:
        return x
    dtype = x.dtype
    t = x.float()
    H, W = t.shape[-2:]
    ph, pw = (min(H // 2, H - 1), min(W // 2, W - 1)) if pad else (0, 0)
    if ph or pw:
        t = F.pad(t, (pw, pw, ph, ph), mode="reflect")
    Hp, Wp = t.shape[-2:]
    out = torch.fft.irfft2(torch.fft.rfft2(t) * _ideal_mask(Hp, Wp, cutoff, t.device), s=(Hp, Wp))
    out = out[..., ph:ph + H, pw:pw + W]
    return out.to(dtype)


def lowpass(x, method="gaussian", sigma=3.0, cutoff=0.25):
    if method == "gaussian":
        return gaussian_lowpass(x, sigma)
    if method == "fft":
        return fft_lowpass(x, cutoff)
    raise ValueError(f"unknown low-pass method {method!r}; expected one of {METHODS}")


def split_lowpass(x, method="gaussian", sigma=3.0, cutoff=0.25):
    """One pass of the operator applied twice in Delta = LPF_h(W * LPF_h(r)) (see module docstring)."""
    if method == "gaussian":
        return gaussian_lowpass(x, sigma / math.sqrt(2.0))
    return lowpass(x, method, sigma, cutoff)


def decompose(x, method="gaussian", sigma=3.0, cutoff=0.25):
    """(low, high) with low + high == x (up to float rounding of the subtraction)."""
    low = lowpass(x, method, sigma, cutoff)
    return low, x - low


def gaussian_half_amplitude_period(sigma):
    """Period (in the same pixel unit as sigma) at which the Gaussian low-pass transmits half the amplitude."""
    return sigma / math.sqrt(math.log(2.0) / (2.0 * math.pi ** 2))


def high_frequency_energy_fraction(x, threshold):
    """Fraction of the energy of x (per sample, all channels) above `threshold` x Nyquist (radial, DFT without
    padding). Diagnostic used by the leakage tests; float64."""
    import torch
    t = x.double()
    H, W = t.shape[-2:]
    spec = torch.fft.rfft2(t).abs() ** 2
    fy = torch.fft.fftfreq(H, dtype=torch.float64).abs() / 0.5
    fx = torch.fft.rfftfreq(W, dtype=torch.float64) / 0.5
    hi = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt() > threshold
    # rfft keeps half of the spectrum: weight the non-redundant columns twice for an exact Parseval split
    w = torch.full((W // 2 + 1,), 2.0, dtype=torch.float64)
    w[0] = 1.0
    if W % 2 == 0:
        w[-1] = 1.0
    total = (spec * w).sum()
    return float((spec * w * hi).sum() / total) if float(total) > 0 else 0.0
