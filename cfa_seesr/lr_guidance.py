"""Component B: timestep-dependent LR (ControlNet) conditioning strength.

Where the LR image enters SeeSR (verified in code): pipelines/pipeline_seesr.py calls
    self.controlnet(latent, t, encoder_hidden_states=..., image_encoder_hidden_states=...,
                    controlnet_cond=LR, conditioning_scale=conditioning_scale, ...)
once per denoising step (and per latent tile in the tiled path), with `t` the actual scheduler timestep.
models/controlnet.py multiplies every down-block residual and the mid-block residual by
`conditioning_scale`; the pipeline adds them to the UNet skip connections. Upstream uses a constant 1.0.

Schedule. With t the scheduler timestep (DDPM, t in {T, ..., 1}, T = num_train_timesteps - 1 = 999;
large t = early, noisy steps), the effective scale is

    s_lr(t) = conditioning_scale * f(t),     f(t) = s_end + (s_start - s_end) * (t / T)^gamma

  s_start  factor at t = T (start of sampling)
  s_end    factor at t = 0 (end of sampling)
  gamma    shape: 1 = linear in t; > 1 keeps f near s_end for longer; < 1 moves towards s_start sooner

f depends on the actual timestep, not on the loop index, so the same t gets the same factor at 50, 30 or
20 steps. For gamma = 1 and executed timesteps spread evenly over [1, T], the mean factor is about
(s_start + s_end) / 2, so a schedule like 1.25 -> 0.75 redistributes guidance over time while keeping
its average near the trained strength of 1.0; the exact mean over the executed timesteps is recorded.

No extra network evaluations: only the scalar passed to the existing ControlNet call changes.
"""
from dataclasses import dataclass

SCHEDULES = ("constant", "power")


@dataclass(frozen=True)
class LRGuidanceConfig:
    enabled: bool = False
    schedule: str = "constant"
    scale_start: float = 1.0     # s_start, factor at t = T
    scale_end: float = 1.0       # s_end, factor at t = 0
    gamma: float = 1.0

    def __post_init__(self):
        if self.schedule not in SCHEDULES:
            raise ValueError(f"lr_guidance.schedule must be one of {SCHEDULES}")
        if self.gamma <= 0:
            raise ValueError("lr_guidance.gamma must be > 0")
        if self.scale_start < 0 or self.scale_end < 0:
            raise ValueError("lr_guidance scales must be >= 0")

    @classmethod
    def from_dict(cls, d):
        return cls(**(d or {}))

    def is_identity(self):
        return (not self.enabled or self.schedule == "constant"
                or (self.scale_start == 1.0 and self.scale_end == 1.0))


def schedule_factor(t, cfg, max_timestep):
    """f(t) for an actual scheduler timestep t (int, float or 0-d tensor)."""
    if cfg.is_identity():
        return 1.0
    t = float(t)
    u = min(1.0, max(0.0, t / float(max_timestep)))
    return cfg.scale_end + (cfg.scale_start - cfg.scale_end) * u ** cfg.gamma


def mean_factor(timesteps, cfg, max_timestep):
    ts = [float(t) for t in timesteps]
    return sum(schedule_factor(t, cfg, max_timestep) for t in ts) / len(ts) if ts else None
