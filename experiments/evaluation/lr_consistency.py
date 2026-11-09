"""LR-consistency diagnostic: does the SR output, degraded back to LR size, match the LR input?

    lr_psnr = PSNR( D(SR), LR )     on RGB, data range 255     (higher = more consistent)
    lr_mae  = mean |D(SR) - LR|     on RGB, 0..255 scale       (lower = more consistent)

D is bicubic downsampling by the SR scale factor (PIL Image.resize with BICUBIC, which applies an
antialiasing kernel when shrinking). Assumptions and limits:
  * Real LR images (RealSR, DRealSR, RealLR200) were NOT produced by bicubic downsampling of a clean
    image; they contain blur, noise and compression. A perfect SR result therefore does not reach zero
    error, and an output that reproduces the LR degradations can score better than a faithful one.
  * Upstream applies AdaIN (or wavelet) colour correction towards the LR image, which already pulls
    low-frequency colour statistics towards the LR input.
  * It measures consistency with the input, not quality: blurry outputs score well. Use it only next to
    the paired (PSNR/SSIM/LPIPS/DISTS) and no-reference metrics, e.g. to see whether a guidance change
    moves outputs towards or away from the observation; it is not a substitute for GT evaluation.
"""
import math

import numpy as np
from PIL import Image

LRC_METRICS = {
    "lr_psnr": {"lower_better": False, "definition": "PSNR(bicubic_down(SR), LR), RGB, range 255"},
    "lr_mae": {"lower_better": True, "definition": "mean |bicubic_down(SR) - LR|, RGB, 0..255"},
}
LRC_DISCLAIMER = (
    "Diagnostic only: compares the bicubic-downsampled SR with the real LR input. Real degradations are "
    "not bicubic, colour correction pulls outputs towards LR, and blurry outputs score well. Not a measure "
    "of reconstruction quality and not a substitute for ground-truth evaluation."
)


def lr_consistency(sr_rgb, lr_rgb, scale):
    """sr_rgb, lr_rgb: uint8 arrays [H, W, 3]. Requires SR size == LR size x scale exactly."""
    lh, lw = lr_rgb.shape[:2]
    if sr_rgb.shape[:2] != (lh * scale, lw * scale):
        raise ValueError(f"SR {sr_rgb.shape[1]}x{sr_rgb.shape[0]} is not LR {lw}x{lh} x {scale}")
    down = np.asarray(Image.fromarray(sr_rgb).resize((lw, lh), Image.BICUBIC), dtype=np.float64)
    diff = down - lr_rgb.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    return {"lr_psnr": math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse),
            "lr_mae": float(np.mean(np.abs(diff)))}
