"""LORNet: Learned Observation Reliability Network (CFA-SeeSR V2, the only trainable part).

Input: the conditioning image the pipeline passes to the ControlNet (upscaled LR, [B, 3, H, W] in [0, 1], H and W
multiples of 8), turned into a COMPACT set of per-pixel features (FEATURES; default y, grad, contrast, hf).
Output: a reliability map in (0, 1) at latent resolution [B, 1, H/8, W/8], trained to predict the teacher
R = exp(-E / tau) of latent_targets.py (low-frequency agreement between the LR and HR latents of the frozen VAE).

Network (fully convolutional; any H, W that are multiples of 8; internally padded to a multiple of 16):
    features --conv3x3/2--> 16 @ /2 --DS/2 + ResDS--> 24 @ /4 --DS/2 + ResDS--> 32 @ /8 (skip)
             --DS/2 + ResDS--> 48 @ /16 --multi-dilation DS context (d = 1, 2, 4)--> 48 @ /16
             --bilinear x2, concat skip, 1x1--> 32 @ /8 --ResDS--> head: 3x3 (16) -> 1x1 -> sigmoid   [/8 = latent]
    DS = depthwise 3x3 + pointwise 1x1; ResDS = x + DS(SiLU(GN(DS(x)))); GN = GroupNorm(4).
    Optional uncertainty head (uncertainty: true): a second 1x1 output, the log-variance for a Gaussian NLL loss.

Checkpoints are dicts with format "cfa-seesr-lornet"; load_lornet() validates format, version, configuration,
target definition and parameter values and raises LORNetCheckpointError otherwise: a learned map is never
produced by randomly initialised weights.
"""
import math

FEATURES = ("y", "grad", "contrast", "hf", "rgb")
DEFAULT_FEATURES = ("y", "grad", "contrast", "hf")
FEATURE_CHANNELS = {"y": 1, "grad": 1, "contrast": 1, "hf": 1, "rgb": 3}
FEATURE_SCALE = {"grad": 0.25, "contrast": 0.1, "hf": 0.05}     # fixed scalings to ~[0, 1] on natural images
CHECKPOINT_FORMAT = "cfa-seesr-lornet"
CHECKPOINT_VERSION = 1


class LORNetCheckpointError(RuntimeError):
    pass


def validate_model_config(cfg):
    cfg = dict(cfg or {})
    feats = tuple(cfg.get("features", DEFAULT_FEATURES))
    bad = [f for f in feats if f not in FEATURES]
    if not feats or bad or len(set(feats)) != len(feats):
        raise ValueError(f"lornet.features must be a non-empty list of distinct {FEATURES}; got {feats}")
    widths = tuple(int(w) for w in cfg.get("widths", (16, 24, 32, 48)))
    if len(widths) != 4 or any(w < 4 or w % 4 for w in widths):
        raise ValueError("lornet.widths: four positive multiples of 4")
    return {"features": list(feats), "widths": list(widths), "uncertainty": bool(cfg.get("uncertainty", False))}


def in_channels(features):
    return sum(FEATURE_CHANNELS[f] for f in features)


def _gauss(x, sigma):
    from .frequency import gaussian_lowpass
    return gaussian_lowpass(x, sigma)


def compute_features(image, features=DEFAULT_FEATURES):
    """[B, 3, H, W] in [0, 1] -> [B, C, H, W] float32 (C = in_channels(features))."""
    import torch
    import torch.nn.functional as F
    img = image.float().clamp(0, 1)
    y = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3])
    out = []
    for f in features:
        if f == "y":
            out.append(y)
        elif f == "rgb":
            out.append(img)
        elif f == "grad":
            ys = _gauss(y, 1.0)
            p = F.pad(ys, (1, 1, 1, 1), mode="replicate")
            gx = (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) * 0.5
            gy = (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) * 0.5
            out.append(((gx ** 2 + gy ** 2 + 1e-12).sqrt() / FEATURE_SCALE["grad"]).clamp(0, 4))
        elif f == "contrast":
            m = _gauss(y, 2.0)
            var = (_gauss(y * y, 2.0) - m * m).clamp_min(0)
            out.append(((var + 1e-12).sqrt() / FEATURE_SCALE["contrast"]).clamp(0, 4))
        elif f == "hf":
            out.append((_gauss((y - _gauss(y, 2.0)).abs(), 2.0) / FEATURE_SCALE["hf"]).clamp(0, 4))
    return torch.cat(out, dim=1)


def _build_module_classes():
    import torch
    from torch import nn

    class DS(nn.Module):
        def __init__(self, cin, cout, stride=1, dilation=1):
            super().__init__()
            self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=dilation, dilation=dilation, groups=cin,
                                padding_mode="replicate")
            self.pw = nn.Conv2d(cin, cout, 1)

        def forward(self, x):
            return self.pw(self.dw(x))

    class ResDS(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.a, self.n, self.b = DS(c, c), nn.GroupNorm(4, c), DS(c, c)

        def forward(self, x):
            return x + self.b(nn.functional.silu(self.n(self.a(x))))

    class Context(nn.Module):
        """Multi-scale context: parallel dilated DS convolutions (d = 1, 2, 4), summed, residual."""

        def __init__(self, c):
            super().__init__()
            self.branches = nn.ModuleList([DS(c, c, dilation=d) for d in (1, 2, 4)])
            self.n = nn.GroupNorm(4, c)

        def forward(self, x):
            return x + nn.functional.silu(self.n(sum(b(x) for b in self.branches)))

    class LORNet(nn.Module):
        def __init__(self, features=DEFAULT_FEATURES, widths=(16, 24, 32, 48), uncertainty=False):
            super().__init__()
            cfg = validate_model_config({"features": features, "widths": widths, "uncertainty": uncertainty})
            self.config = cfg
            w0, w1, w2, w3 = cfg["widths"]
            self.stem = nn.Sequential(nn.Conv2d(in_channels(cfg["features"]), w0, 3, stride=2, padding=1,
                                                padding_mode="replicate"), nn.SiLU())
            self.down1 = nn.Sequential(DS(w0, w1, stride=2), nn.SiLU(), ResDS(w1))
            self.down2 = nn.Sequential(DS(w1, w2, stride=2), nn.SiLU(), ResDS(w2))
            self.down3 = nn.Sequential(DS(w2, w3, stride=2), nn.SiLU(), ResDS(w3), Context(w3))
            self.fuse = nn.Conv2d(w3 + w2, w2, 1)
            self.refine = ResDS(w2)
            self.head = nn.Sequential(nn.Conv2d(w2, 16, 3, padding=1, padding_mode="replicate"), nn.SiLU())
            self.out = nn.Conv2d(16, 2 if cfg["uncertainty"] else 1, 1)

        def forward_features(self, feats):
            """feats [B, C, H, W] (H, W multiples of 8) -> (logit [B,1,H/8,W/8], logvar or None)."""
            H, W = feats.shape[-2:]
            if H % 8 or W % 8:
                raise ValueError(f"LORNet input {H}x{W}: height and width must be multiples of 8")
            ph, pw = (-H) % 16, (-W) % 16
            if ph or pw:
                feats = nn.functional.pad(feats, (0, pw, 0, ph), mode="replicate")
            s8 = self.down2(self.down1(self.stem(feats)))
            s16 = self.down3(s8)
            up = nn.functional.interpolate(s16, size=s8.shape[-2:], mode="bilinear", align_corners=False)
            x = self.refine(self.fuse(torch.cat([up, s8], dim=1)))
            o = self.out(self.head(x))[..., : H // 8, : W // 8]
            return o[:, 0:1], (o[:, 1:2] if self.config["uncertainty"] else None)

        def forward(self, image):
            """image [B, 3, H, W] in [0, 1] -> reliability [B, 1, H/8, W/8] in (0, 1)."""
            logit, _ = self.forward_features(compute_features(image, self.config["features"]))
            return torch.sigmoid(logit)

    return LORNet


_LORNET = None


def LORNet(**kw):  # noqa: N802 (class factory: torch is imported lazily)
    global _LORNET
    if _LORNET is None:
        _LORNET = _build_module_classes()
    return _LORNET(**kw)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def conv_macs(model, height, width):
    """Multiply-accumulates of all convolutions for one image of height x width (counted with forward hooks)."""
    import torch
    from torch import nn
    total = {"macs": 0}
    hooks = []

    def hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        total["macs"] += out.numel() // out.shape[0] * (m.in_channels // m.groups) * k

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, height, width, device=next(model.parameters()).device))
    finally:
        for h in hooks:
            h.remove()
    return total["macs"]


def make_checkpoint(model, target, training=None):
    """target: {"tau", "lowpass", "window"} from latent_targets; training: provenance (split, seed, metrics)."""
    from . import METHOD_VERSION
    tau = float(target["tau"])
    if not (tau > 0 and math.isfinite(tau)):
        raise ValueError("checkpoint target.tau must be a positive finite number")
    return {"format": CHECKPOINT_FORMAT, "format_version": CHECKPOINT_VERSION, "method_version": METHOD_VERSION,
            "model_config": dict(model.config), "target": dict(target), "training": dict(training or {}),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}


def resolve_checkpoint(path):
    """Absolute path; a relative path is taken relative to the working directory, else to the repository root."""
    from pathlib import Path
    p = Path(str(path)).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return Path(__file__).resolve().parents[1] / p


def checkpoint_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(resolve_checkpoint(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lornet(path, device="cpu"):
    """Load and validate a LORNet checkpoint. Returns (model in eval mode, checkpoint metadata without weights)."""
    import torch
    from pathlib import Path
    if path is None or str(path).strip() in ("", "None", "null"):
        raise LORNetCheckpointError("no LORNet checkpoint given (sfr_oc.learned.checkpoint)")
    p = resolve_checkpoint(path)
    if not p.is_file():
        raise LORNetCheckpointError(f"LORNet checkpoint not found: {path}")
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:                                    # torch < 1.13 has no weights_only
        ck = torch.load(p, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        raise LORNetCheckpointError(f"cannot read LORNet checkpoint {p}: {exc}") from exc
    if not isinstance(ck, dict) or ck.get("format") != CHECKPOINT_FORMAT:
        raise LORNetCheckpointError(f"{p} is not a {CHECKPOINT_FORMAT} checkpoint")
    if ck.get("format_version") != CHECKPOINT_VERSION:
        raise LORNetCheckpointError(f"{p}: checkpoint format_version {ck.get('format_version')} != {CHECKPOINT_VERSION}")
    for key in ("model_config", "state_dict", "target"):
        if key not in ck:
            raise LORNetCheckpointError(f"{p}: missing '{key}'")
    tau = ck["target"].get("tau")
    if not (isinstance(tau, (int, float)) and tau > 0 and math.isfinite(tau)):
        raise LORNetCheckpointError(f"{p}: invalid target.tau {tau!r}")
    try:
        cfg = validate_model_config(ck["model_config"])
        model = LORNet(**cfg)
        model.load_state_dict(ck["state_dict"], strict=True)
    except (ValueError, RuntimeError, KeyError) as exc:
        raise LORNetCheckpointError(f"{p}: weights do not match the model configuration: {exc}") from exc
    if not all(torch.isfinite(t).all() for t in ck["state_dict"].values() if t.is_floating_point()):
        raise LORNetCheckpointError(f"{p}: non-finite weights")
    meta = {k: v for k, v in ck.items() if k != "state_dict"}
    return model.to(device).eval(), meta
