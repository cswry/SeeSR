"""Metric registry backed by pyiqa (https://github.com/chaofengc/IQA-PyTorch).

Paired (full-reference) and no-reference metrics are kept in separate tables so
one can never be computed as the other by accident. All metric objects take
float tensors shaped [1, 3, H, W] in [0, 1].
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricSpec:
    pyiqa_name: str
    lower_better: bool
    options: dict = field(default_factory=dict)
    uses_color_options: bool = False  # receives y_channel / crop_border settings


PAIRED_METRICS = {
    "psnr": MetricSpec("psnr", lower_better=False, uses_color_options=True),
    "ssim": MetricSpec("ssim", lower_better=False, uses_color_options=True),
    "lpips": MetricSpec("lpips", lower_better=True),   # pyiqa default: AlexNet, LPIPS v0.1
    "dists": MetricSpec("dists", lower_better=True),
}

NOREF_METRICS = {
    "musiq": MetricSpec("musiq", lower_better=False),  # pyiqa default: KonIQ-trained MUSIQ
    "clipiqa": MetricSpec("clipiqa", lower_better=False),
}

NOREF_DISCLAIMER = (
    "No-reference scores estimate perceived quality only. They do not measure fidelity to the "
    "ground truth or semantic correctness, and can reward hallucinated detail."
)


class MetricUnavailable(RuntimeError):
    """Raised when a metric backend (package or weights) cannot be loaded."""


def pyiqa_factory(spec, device, y_channel, crop_border):
    try:
        import pyiqa
    except ImportError as e:
        raise MetricUnavailable("pyiqa is not installed (pip install -r experiments/requirements-metrics.txt)") from e
    kwargs = dict(spec.options)
    if spec.uses_color_options:
        kwargs.update(test_y_channel=bool(y_channel), crop_border=int(crop_border))
    try:
        metric = pyiqa.create_metric(spec.pyiqa_name, device=device, as_loss=False, **kwargs)
    except Exception as e:  # missing weights / network errors surface here
        raise MetricUnavailable(f"pyiqa could not create '{spec.pyiqa_name}': {type(e).__name__}: {e}") from e

    def compute(sr, gt=None):
        import torch
        with torch.no_grad():
            value = metric(sr, gt) if gt is not None else metric(sr)
        return float(value.reshape(-1)[0].item())

    return compute


def build_metrics(names, table, device="cpu", y_channel=True, crop_border=0, factory=pyiqa_factory):
    """Return ``{name: callable}``. Raises MetricUnavailable / KeyError early."""
    unknown = [n for n in names if n not in table]
    if unknown:
        raise KeyError(f"Unknown metric(s) {unknown}; available: {sorted(table)}")
    return {n: factory(table[n], device, y_channel, crop_border) for n in names}


def describe(names, table, y_channel, crop_border):
    out = {}
    for n in names:
        spec = table[n]
        desc = {"pyiqa_name": spec.pyiqa_name, "lower_better": spec.lower_better, **spec.options}
        if spec.uses_color_options:
            desc.update(test_y_channel=bool(y_channel), crop_border=int(crop_border))
        out[n] = desc
    return out
