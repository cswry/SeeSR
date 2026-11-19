"""Losses and map-agreement metrics for LORNet (CPU/GPU, torch only)."""
import math
import random
import statistics


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def charbonnier(pred, target, eps=1e-3):
    """Robust regression (smooth L1-like): mean sqrt((p - t)^2 + eps^2) - eps."""
    return (((pred - target) ** 2 + eps ** 2).sqrt() - eps).mean()


def soft_bce(logit, target):
    """Binary cross-entropy with soft labels in [0, 1] (calibrated-probability view of the target)."""
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(logit, target)


def gaussian_nll(pred, logvar, target):
    """Heteroscedastic regression: the uncertainty head predicts log sigma^2 of the target per location."""
    logvar = logvar.clamp(-10, 6)
    return (0.5 * ((pred - target) ** 2 * (-logvar).exp() + logvar)).mean()


def total_variation(m):
    return (m[..., 1:, :] - m[..., :-1, :]).abs().mean() + (m[..., :, 1:] - m[..., :, :-1]).abs().mean()


LOSSES = ("charbonnier", "bce", "gaussian_nll")


def reliability_loss(kind, logit, logvar, target, tv_weight):
    import torch
    pred = torch.sigmoid(logit)
    if kind == "charbonnier":
        main = charbonnier(pred, target)
    elif kind == "bce":
        main = soft_bce(logit.float(), target.float())
    elif kind == "gaussian_nll":
        if logvar is None:
            raise ValueError("loss gaussian_nll needs model.uncertainty: true")
        main = gaussian_nll(pred, logvar, target)
    else:
        raise ValueError(f"loss must be one of {LOSSES}")
    tv = total_variation(pred)
    return main + tv_weight * tv, {"main": main.detach(), "tv": tv.detach()}


# --------------------------------------------------------------------------- #
# Agreement metrics
# --------------------------------------------------------------------------- #
def _rank(x):
    import torch
    r = torch.empty_like(x, dtype=torch.float64)
    r[x.argsort()] = torch.arange(x.numel(), dtype=torch.float64)
    return r


def pearson(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = (a.norm() * b.norm()).item()
    return float((a * b).sum() / den) if den > 0 else float("nan")


def spearman(a, b):
    return pearson(_rank(a.flatten()), _rank(b.flatten()))


def quartile_contrast(score, value):
    """mean(value | score in top quartile) - mean(value | score in bottom quartile)."""
    s, v = score.flatten().double(), value.flatten().double()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    top, bot = v[s >= q3], v[s <= q1]
    if top.numel() == 0 or bot.numel() == 0:
        return float("nan")
    return float(top.mean() - bot.mean())


def ece(pred, target, bins=10):
    """Expected calibration error between predicted and teacher reliability (equal-width bins of pred)."""
    p, t = pred.flatten().double(), target.flatten().double()
    total, n = 0.0, p.numel()
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if m.any():
            total += m.sum().item() / n * abs(p[m].mean().item() - t[m].mean().item())
    return total


def bootstrap_ci(values, n_boot=2000, seed=0, level=0.95):
    vals = [v for v in values if v == v]
    if len(vals) < 2:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(vals, k=len(vals))) for _ in range(n_boot))
    return [means[int((1 - level) / 2 * n_boot)], means[min(n_boot - 1, int((1 + level) / 2 * n_boot))]]


def summarise_maps(preds, targets, errors, calibrated=True):
    """preds/targets/errors: lists of [h, w] tensors per crop (pred reliability, teacher R, discrepancy E).
    calibrated=False (heuristic maps): only rank-based metrics are meaningful."""
    import torch
    P = torch.cat([p.flatten() for p in preds]).double()
    T = torch.cat([t.flatten() for t in targets]).double()
    E = torch.cat([e.flatten() for e in errors]).double()
    per = [spearman(p, t) for p, t in zip(preds, targets)]
    per = [v for v in per if not math.isnan(v)]
    out = {"n_maps": len(preds), "spearman_pooled": spearman(P, T),
           "spearman_per_map_mean": statistics.fmean(per) if per else float("nan"),
           "spearman_per_map_ci95": bootstrap_ci(per),
           "E_bottom_minus_top_quartile": -quartile_contrast(P, E)}       # > 0: high reliability = low E
    if calibrated:
        out.update({"mae": float((P - T).abs().mean()), "rmse": float(((P - T) ** 2).mean().sqrt()),
                    "pearson": pearson(P, T), "ece": ece(P, T)})
    return out
