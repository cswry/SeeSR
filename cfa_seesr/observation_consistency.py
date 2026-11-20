"""SFR-OC: Semantic- and Fidelity-Reliability guided, Frequency-selective Observation Consistency.

Parts B (frequency-selective observation consistency) and C (reliability fusion); part A is
observation_reliability.py. Applied around the scheduler step: NO UNet or ControlNet evaluation is added.

Reference latent. z_lr = mean of the VAE posterior of the conditioning image (the upscaled LR, batch item 0)
times vae.config.scaling_factor: the VAE and scaling convention of the pipeline's LR-embedded start
(`vae.encode(image*2-1).latent_dist.sample() * scaling_factor`, pipeline_seesr.py:989-990), but the MEAN, so the
reference is deterministic and no RNG draw is added.
  reference='extra_encode' (default since 2.1.1): one extra VAE encoder pass of item 0 per pipeline call (the CFG
    copies of the conditioning image are checked identical when the maps are built). The V2.1 heuristic pilot ran
    this: with V1 enabled on the pinned torch 2.0.1 stack, the pipeline's own posterior failed the deferred
    CFG-identical check twice (its two rows were not bit-identical; cause not isolated). The learned pilot, run
    earlier without V1 in a different environment, used the pipeline posterior and passed the check.
  reference='pipeline' (opt-in): reuse the posterior of the pipeline's own LR encode (start_point 'lr'; no extra
    pass), with the deferred check that its CFG rows are identical.

Correction at a reverse step scheduler.step(model_output, t, x_t, ...), a = alphas_cumprod[t]:

    x0     = predicted clean latent (scheduler.config.prediction_type)
               epsilon:       x0 = (x_t - sqrt(1-a) eps) / sqrt(a)
               v_prediction:  x0 = sqrt(a) x_t - sqrt(1-a) v
               sample:        x0 = model_output
    r      = z_lr - x0
    Delta  = gamma(t) * LPF_h(W * LPF_h(r))        band='low'  (default)
    Delta  = gamma(t) * W * r                      band='full' (ablation: all frequencies)
    x0'    = x0 + Delta

LPF_h is ONE pass of the documented low-pass applied twice: Gaussian with sigma/sqrt(2) per pass (variances add,
so LPF_h(LPF_h(.)) is the documented Gaussian of std sigma), ideal FFT with the same cutoff per pass (idempotent).
For a spatially uniform W this equals gamma * W * LPF(r); for a spatially varying W the outer pass removes the
high frequencies that the multiplication by W creates (W * LPF(r) is not band-limited). A Gaussian is not
band-limited either: it attenuates smoothly (half amplitude at a period of ~5.3 sigma latent px).

Applying x0' to the scheduler (precision='output', default). DDPM and DDIM (with clipping and thresholding off,
which check_scheduler enforces) are LINEAR in x0 for fixed x_t and noise:
    DDPM:  x_{t-1} = c0 x0 + c_t x_t + sigma_t z,     c0 = sqrt(a_prev) beta_t / (1 - a)
    DDIM:  x_{t-1} = sqrt(a_prev) x0 + sqrt(1 - a_prev - s^2) eps(x0, x_t) + s z,
           eps(x0, x_t) = (x_t - sqrt(a) x0)/sqrt(1-a)  =>  c0 = sqrt(a_prev) - sqrt(1 - a_prev - s^2) sqrt(a)/sqrt(1-a)
(beta_t = 1 - a/a_prev; s = eta * sqrt(variance(t, prev)); prev / a_prev exactly as the scheduler computes them.)
So step(x0') = step(x0) + c0 * Delta: the ORIGINAL step runs on the ORIGINAL model output (same dtype, same RNG
draw as V1) and c0 * Delta is added to its output in float32 before casting to the output dtype. This avoids the
fp16 loss of the alternative below, where the scheduler re-derives x0 from a rounded eps' and amplifies the
rounding by sqrt(1-a)/sqrt(a) (~13x at t = 981; see tests/test_sfroc.py::test_fp16_*).
With fp16 latents (upstream runs mixed_precision fp16) the corrected output is still rounded to fp16: a displacement
c0*Delta below half an fp16 ulp of x_{t-1} is lost at that step (the representational floor that V1's own update
also has). precision='output_carry' (option, not the default) keeps that rounding remainder in float32 and adds it at
the next corrected step (error feedback; the carry is <= half an ulp per element and dropped at the first
uncorrected step), so the accumulated displacement is preserved to within one rounding.
precision='model_output' keeps the earlier formulation (rewrite the model output in its own parameterisation):
    epsilon: eps' = eps - sqrt(a)/sqrt(1-a) Delta;  v_prediction: v' = v - Delta/sqrt(1-a);  sample: x0'.

gamma(t) = strength * f(t/T), T = num_train_timesteps - 1, f from SCHEDULES, f = 0 outside [t_min, t_max].
Where gamma(t) == 0 (always, when strength == 0) the original step is called with the original arguments and its
result is returned untouched: an exact no-op.

Fusion (part C), at latent resolution:
    W = clip(w_obs * O + w_sem * (1 - S) + bias, w_min, w_max)        map='fused'
    W = mean(W_fused) everywhere                                       map='mean_matched' (control)
    W = 1;  W = O;  W = 1 - S                                          map='uniform' / 'observation' / 'semantic'
  O  observation reliability: part A (heuristic), LORNet (reliability 'learned'; cfa_seesr/lornet.py, validated
     checkpoint only) or a blend; S = SSR map, else component A's C, else semantic_default.
  control='shuffled' permutes W in bands (rows, then columns; any size): same values, other places.
  Interpretation (hypothesis, tested by tools/sfroc_map_validity.py): pull toward the LR observation where it is
  trustworthy (high O) and where the semantic prior is least trusted (low S).

Supported: DDPMScheduler, DDIMScheduler; prediction_type epsilon / v_prediction / sample; variance_type not learned;
clip_sample and thresholding off. Anything else is rejected at installation.

Synchronisation. During denoising no tensor is converted to a Python value: the timestep is read from a host copy
of scheduler.timesteps taken once per call (and checked against the passed value on device), diagnostics stay on
device, and checks (finite maps, identical CFG copies, timestep agreement) are evaluated once in summary(), after
the pipeline call. Timing uses CUDA events on CUDA (GPU time of the SFR-OC kernels only; previously queued
UNet/ControlNet work is not included) and perf_counter on CPU; profile=true adds explicit synchronisation.
"""
import hashlib
import math
from dataclasses import dataclass, field

from .observation_reliability import ObservationReliabilityConfig

SCHEDULES = ("constant", "power", "window")
BANDS = ("low", "full")
MAP_MODES = ("fused", "mean_matched", "uniform", "observation", "semantic")
CONTROLS = ("none", "shuffled")
PRECISIONS = ("output", "output_carry", "model_output")
RELIABILITY_SOURCES = ("heuristic", "learned", "blend")
REFERENCES = ("extra_encode", "pipeline")
SUPPORTED_SCHEDULERS = ("DDPMScheduler", "DDIMScheduler")
SUPPORTED_PREDICTION_TYPES = ("epsilon", "v_prediction", "sample")
FORMULATION = "Delta = gamma(t) * LPF_h(W * LPF_h(z_lr - x0)); x_{t-1}' = step(x0) + c0(t) * Delta"


@dataclass(frozen=True)
class FusionConfig:
    map: str = "fused"
    w_observation: float = 0.5
    w_semantic: float = 0.5
    bias: float = 0.0
    w_min: float = 0.0
    w_max: float = 1.0
    semantic_default: float = 0.5       # S when neither component A nor SSR is active
    control: str = "none"
    shuffle_block: int = 8              # latent pixels per shuffled band
    shuffle_seed: int = 0

    def __post_init__(self):
        if self.map not in MAP_MODES:
            raise ValueError(f"sfr_oc.fusion.map must be one of {MAP_MODES}")
        if self.control not in CONTROLS:
            raise ValueError(f"sfr_oc.fusion.control must be one of {CONTROLS}")
        for name in ("w_observation", "w_semantic"):
            if getattr(self, name) < 0:
                raise ValueError(f"sfr_oc.fusion.{name} must be >= 0")
        if not 0.0 <= self.w_min <= self.w_max <= 1.0:
            raise ValueError("sfr_oc.fusion: need 0 <= w_min <= w_max <= 1")
        if not 0.0 <= self.semantic_default <= 1.0:
            raise ValueError("sfr_oc.fusion.semantic_default must be in [0, 1]")
        if self.shuffle_block < 1:
            raise ValueError("sfr_oc.fusion.shuffle_block must be >= 1")


@dataclass(frozen=True)
class LearnedReliabilityConfig:
    """LORNet (cfa_seesr/lornet.py). Used only when sfr_oc.reliability is 'learned' or 'blend'."""
    checkpoint: object = None           # path to a cfa-seesr-lornet checkpoint; validated when installed
    blend: float = 0.5                  # reliability 'blend': O = blend * O_learned + (1 - blend) * O_heuristic

    def __post_init__(self):
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("sfr_oc.learned.blend must be in [0, 1]")


@dataclass(frozen=True)
class SFROCConfig:
    enabled: bool = False
    strength: float = 0.0               # gamma_max in [0, 1]; 0 = exact no-op
    band: str = "low"                   # low (frequency-selective) | full (ablation: all frequencies)
    lowpass: str = "gaussian"           # gaussian | fft
    sigma: float = 3.0                  # Gaussian std in latent px; half amplitude at a period of ~16 latent px
    cutoff: float = 0.25                # fft cutoff as a fraction of Nyquist
    schedule: str = "power"             # constant | power | window
    power: float = 1.0                  # f = (t/T)^power
    t_min: int = 0                      # no correction for t < t_min
    t_max: int = 1000                   # no correction for t > t_max
    reference: str = "extra_encode"     # extra_encode: one VAE encode of conditioning item 0 (default, 2.1.1)
                                        # pipeline: reuse the pipeline's LR-start posterior (opt-in)
    precision: str = "output"           # output: add c0*Delta to the step output | output_carry: + fp32 error
                                        # feedback of the rounding remainder | model_output: legacy eps' path
    profile: bool = False               # explicit synchronisation around timed blocks (benchmarks only)
    save_maps: bool = True
    reliability: str = "heuristic"      # heuristic (observation_reliability) | learned (LORNet) | blend
    observation: ObservationReliabilityConfig = field(default_factory=ObservationReliabilityConfig)
    learned: LearnedReliabilityConfig = field(default_factory=LearnedReliabilityConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    def __post_init__(self):
        if self.reliability not in RELIABILITY_SOURCES:
            raise ValueError(f"sfr_oc.reliability must be one of {RELIABILITY_SOURCES}")
        if self.enabled and self.uses_learned and self.learned.checkpoint in (None, ""):
            raise ValueError("sfr_oc.reliability uses LORNet but sfr_oc.learned.checkpoint is not set "
                             "(a learned map is never produced without a trained checkpoint)")
        if not 0.0 <= self.strength <= 1.0 or not math.isfinite(self.strength):
            raise ValueError("sfr_oc.strength must be in [0, 1]")
        if self.band not in BANDS:
            raise ValueError(f"sfr_oc.band must be one of {BANDS}")
        if self.lowpass not in ("gaussian", "fft"):
            raise ValueError("sfr_oc.lowpass must be gaussian or fft")
        if self.sigma <= 0 or not 0.0 < self.cutoff <= 1.0:
            raise ValueError("sfr_oc: sigma > 0 and 0 < cutoff <= 1 required")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"sfr_oc.schedule must be one of {SCHEDULES}")
        if self.power <= 0:
            raise ValueError("sfr_oc.power must be > 0")
        if not 0 <= self.t_min <= self.t_max:
            raise ValueError("sfr_oc: need 0 <= t_min <= t_max")
        if self.reference == "auto":
            raise ValueError("sfr_oc.reference 'auto' was removed in 2.1.1: use 'extra_encode' (default) or "
                             "'pipeline' (reuse the pipeline's own LR encode, what 'auto' did)")
        if self.reference not in REFERENCES:
            raise ValueError(f"sfr_oc.reference must be one of {REFERENCES}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"sfr_oc.precision must be one of {PRECISIONS}")

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        if "observation" in d:
            d["observation"] = ObservationReliabilityConfig.from_dict(d["observation"])
        if "fusion" in d:
            d["fusion"] = FusionConfig(**dict(d["fusion"] or {}))
        if "learned" in d:
            d["learned"] = LearnedReliabilityConfig(**dict(d["learned"] or {}))
        return cls(**d)

    @property
    def uses_learned(self):
        return self.reliability in ("learned", "blend")

    @property
    def is_noop(self):
        return self.strength == 0.0


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def gamma_at(t, cfg, max_t):
    """gamma(t) for an actual DDPM timestep t (0 .. max_t); t must be a host number."""
    t = float(t)
    if cfg.strength == 0.0 or t < cfg.t_min or t > cfg.t_max:
        return 0.0
    if cfg.schedule in ("constant", "window"):
        f = 1.0
    else:
        f = min(1.0, max(0.0, t / max_t)) ** cfg.power
    return cfg.strength * f


# --------------------------------------------------------------------------- #
# Scheduler parameterisation
# --------------------------------------------------------------------------- #
def check_scheduler(scheduler):
    """Return the prediction type, or raise for configurations whose step is not handled exactly."""
    name = type(scheduler).__name__
    if name not in SUPPORTED_SCHEDULERS:
        raise ValueError(f"SFR-OC does not support {name}; supported: {SUPPORTED_SCHEDULERS}")
    cfg = scheduler.config
    pt = cfg.get("prediction_type", "epsilon")
    if pt not in SUPPORTED_PREDICTION_TYPES:
        raise ValueError(f"SFR-OC does not support prediction_type={pt!r}; supported: {SUPPORTED_PREDICTION_TYPES}")
    if str(cfg.get("variance_type", "fixed_small")).startswith("learned"):
        raise ValueError(f"SFR-OC does not support variance_type={cfg.get('variance_type')!r} (learned variance: the "
                         "model output carries extra channels and the step is not linear in x0)")
    if cfg.get("thresholding", False):
        raise ValueError("SFR-OC does not support thresholding=True (the step would not be linear in x0)")
    if cfg.get("clip_sample", False):
        raise ValueError("SFR-OC does not support clip_sample=True (the step would not be linear in x0)")
    return pt


def predicted_x0(model_output, sample, a, prediction_type):
    sa, s1 = math.sqrt(a), math.sqrt(1.0 - a)
    if prediction_type == "epsilon":
        return (sample - s1 * model_output) / sa
    if prediction_type == "v_prediction":
        return sa * sample - s1 * model_output
    if prediction_type == "sample":
        return model_output
    raise ValueError(f"unsupported prediction_type {prediction_type!r}")


def apply_x0_delta(model_output, delta, a, prediction_type):
    """model_output' such that the scheduler's x0 becomes x0 + delta, with x_t unchanged."""
    sa, s1 = math.sqrt(a), math.sqrt(1.0 - a)
    if prediction_type == "epsilon":
        return model_output - (sa / s1) * delta
    if prediction_type == "v_prediction":
        return model_output - delta / s1
    if prediction_type == "sample":
        return model_output + delta
    raise ValueError(f"unsupported prediction_type {prediction_type!r}")


def x0_coefficient(scheduler, t, eta=0.0):
    """c0 with step(x0 + Delta) = step(x0) + c0 * Delta (DDPM / DDIM, no clipping/thresholding). Host float."""
    name = type(scheduler).__name__
    acp = scheduler.alphas_cumprod
    a = float(acp[t])
    if name == "DDPMScheduler":
        prev = scheduler.previous_timestep(t)
        prev = int(prev)
        a_prev = float(acp[prev]) if prev >= 0 else float(scheduler.one)
        beta = 1.0 - a / a_prev
        return math.sqrt(a_prev) * beta / (1.0 - a)
    if name == "DDIMScheduler":
        prev = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        a_prev = float(acp[prev]) if prev >= 0 else float(scheduler.final_alpha_cumprod)
        var = float(scheduler._get_variance(t, prev)) if eta else 0.0
        s2 = (eta ** 2) * var
        return math.sqrt(a_prev) - math.sqrt(max(1.0 - a_prev - s2, 0.0)) * math.sqrt(a) / math.sqrt(1.0 - a)
    raise ValueError(f"no x0 coefficient for {name}")


def correction_delta(model_output, sample, a, z_ref, W, gamma, cfg, prediction_type):
    """(Delta, x0, r) in float32 outside CUDA autocast (under test_seesr's autocast the filters run in fp16).
    Delta = gamma * LPF_h(W * LPF_h(r)) (band low) or gamma * W * r (band full)."""
    import torch

    from .frequency import split_lowpass
    mo, xt = model_output.float(), sample.float()
    x0 = predicted_x0(mo, xt, a, prediction_type)
    r = z_ref.to(device=x0.device, dtype=torch.float32) - x0
    w = W.to(device=x0.device, dtype=torch.float32)
    if cfg.band == "low":
        delta = gamma * split_lowpass(w * split_lowpass(r, cfg.lowpass, cfg.sigma, cfg.cutoff),
                                      cfg.lowpass, cfg.sigma, cfg.cutoff)
    else:
        delta = gamma * w * r
    return delta, x0, r


def correct(model_output, sample, timestep, scheduler, z_ref, W, gamma, cfg, prediction_type):
    """precision='model_output' form: corrected model output (input dtype) and on-device diagnostics."""
    t = int(timestep)
    a = float(scheduler.alphas_cumprod[t])
    if not 0.0 < a < 1.0:
        raise ValueError(f"alphas_cumprod[{t}] = {a}: cannot convert the model output")
    delta, x0, r = correction_delta(model_output, sample, a, z_ref, W, gamma, cfg, prediction_type)
    out = apply_x0_delta(model_output.float(), delta, a, prediction_type).to(model_output.dtype)
    return out, step_diagnostics(t, gamma, r, delta)


def step_diagnostics(t, gamma, r, delta):
    """Per-step diagnostics as 0-dim device tensors (no host transfer)."""
    return {"t": t, "gamma": gamma, "residual_abs_mean": r.abs().mean().detach(),
            "delta_abs_mean": delta.abs().mean().detach()}


def apply_to_step_output(out, delta, c0, x0=None, carry=None, return_carry=False):
    """Add c0 * delta (+ carry) to the scheduler output (tuple or SchedulerOutput) in float32, keeping the output
    dtype. return_carry: also return the float32 rounding remainder (None for float32 outputs)."""
    remainder = {}

    def shift(prev):
        target = prev.float() + c0 * delta.to(prev.device)
        if carry is not None:
            target = target + carry
        res = target.to(prev.dtype)
        remainder["carry"] = None if prev.dtype == target.dtype else (target - res.float())
        return res
    if isinstance(out, tuple):
        out = (shift(out[0]),) + tuple(out[1:])
    else:
        out.prev_sample = shift(out.prev_sample)
        if getattr(out, "pred_original_sample", None) is not None and x0 is not None:
            po = out.pred_original_sample
            out.pred_original_sample = (po.float() + delta.to(po.device)).to(po.dtype)
    return (out, remainder["carry"]) if return_carry else out


# --------------------------------------------------------------------------- #
# Fusion (part C)
# --------------------------------------------------------------------------- #
def _bands(n, block):
    return [(i, min(i + block, n)) for i in range(0, n, block)]


def _perm(n, gen):
    import torch
    if n < 2:
        return torch.arange(n)
    p = torch.randperm(n, generator=gen)
    if torch.equal(p, torch.arange(n)):
        p = torch.roll(p, 1)
    return p


def shuffle_blocks(m, block, seed, key=""):
    """Permute a [H, W] map in bands of `block` pixels: first the row bands, then the column bands (the last band
    may be shorter, so any H, W works and every pixel moves with its band). The multiset of values, hence the mean
    and the distribution, is unchanged; local structure inside a band is kept."""
    import torch
    H, W = m.shape
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    gen = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF)
    rows, cols = _bands(H, block), _bands(W, block)
    out = m
    if len(rows) > 1:
        out = torch.cat([out[a:b] for a, b in (rows[i] for i in _perm(len(rows), gen).tolist())], dim=0)
    if len(cols) > 1:
        out = torch.cat([out[:, a:b] for a, b in (cols[i] for i in _perm(len(cols), gen).tolist())], dim=1)
    return out.clone() if out is m else out


def fuse(O_latent, S_latent, cfg, key=""):
    """W [h, w] in [w_min, w_max] from O and S at latent resolution (both [h, w] in [0, 1]). No host sync."""
    import torch
    f = cfg
    if f.map == "uniform":
        W = torch.ones_like(O_latent)
    elif f.map == "observation":
        W = O_latent.clone()
    elif f.map == "semantic":
        W = 1.0 - S_latent
    else:
        W = f.w_observation * O_latent + f.w_semantic * (1.0 - S_latent) + f.bias
    W = W.clamp(f.w_min, f.w_max)
    if f.map == "mean_matched":       # same per-image mean as the fused map, no spatial variation
        W = torch.full_like(W, 1.0) * W.mean()
    if f.control == "shuffled":
        W = shuffle_blocks(W, f.shuffle_block, f.shuffle_seed, key)
    return W


def map_stats(prefix, m):
    """0-dim device tensors (converted once, after inference)."""
    m = m.float()
    return {f"{prefix}_mean": m.mean(), f"{prefix}_min": m.min(), f"{prefix}_max": m.max(),
            f"{prefix}_std": m.std(unbiased=False)}


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
class SectionTimer:
    """Accumulates the duration of code sections.

    CUDA: a pair of CUDA events per section (asynchronous; the elapsed time is read once in total_ms(), which
    synchronises on the last event). The start event completes only after the work already queued on the stream
    (e.g. the UNet/ControlNet of the current step), so that work is excluded; the measured interval is GPU time
    from the first to the last SFR-OC kernel, including GPU idle time while the CPU launches them.
    CPU: perf_counter. strict=True (profile) additionally synchronises the device around each section.
    """

    def __init__(self, strict=False):
        self.strict = strict
        self.events = []
        self.cpu_s = 0.0
        self._open = None

    def start(self, device):
        import time

        import torch
        if device.type == "cuda" and torch.cuda.is_available():
            if self.strict:
                torch.cuda.synchronize(device)
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._open = ("cuda", e, device)
        else:
            self._open = ("cpu", time.perf_counter(), device)

    def stop(self):
        import time

        import torch
        kind, start, device = self._open
        self._open = None
        if kind == "cuda":
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            if self.strict:
                torch.cuda.synchronize(device)
            self.events.append((start, e))
        else:
            self.cpu_s += time.perf_counter() - start

    def total_ms(self):
        if self.events:
            self.events[-1][1].synchronize()
        return 1000.0 * self.cpu_s + sum(s.elapsed_time(e) for s, e in self.events)

    @property
    def method(self):
        return "cuda_events" if self.events else "perf_counter"


# --------------------------------------------------------------------------- #
# Per-call controller (state shared with the hooks)
# --------------------------------------------------------------------------- #
class SFROCController:
    """Holds the per-pipeline-call reference latent, maps, timing, diagnostics and deferred checks."""

    def __init__(self, cfg, max_t, prediction_type, lornet=None, lornet_meta=None):
        if cfg.uses_learned and lornet is None:
            raise ValueError("SFR-OC: reliability '%s' needs a loaded LORNet" % cfg.reliability)
        self.lornet = lornet
        self.lornet_meta = lornet_meta or {}
        self.cfg = cfg
        self.max_t = max_t
        self.prediction_type = prediction_type
        self.reset()

    def reset(self):
        self.image = None                # conditioning image of the current call
        self.lr_image = None             # original LR image (optional, set by the runner)
        self.encode_dist = None          # posterior captured from the pipeline's own vae.encode
        self.z_ref = None
        self.reference_source = None
        self.W = None                    # [1, 1, h, w] float32
        self.O_latent = None
        self.components = None
        self.map_info = {}
        self.step_log = []
        self.checks = {}                 # name -> 0-dim bool tensor or bool, evaluated in summary()
        self.host_timesteps = None
        self.carry = None                # float32 rounding remainder (precision='output_carry')
        self.step_index = 0
        self.analysis_timer = SectionTimer(self.cfg.profile)
        self.correction_timer = SectionTimer(self.cfg.profile)
        self.encode_timer = SectionTimer(self.cfg.profile)
        self._summary = None

    def build_maps(self, image, latent_hw, semantic_latent, key=""):
        """image: conditioning image [B, 3, H, W] in [0, 1]; semantic_latent: [h, w] S map."""
        import torch

        from .observation_reliability import analyse, resize_map
        cfg = self.cfg
        stats, comps, extra = {}, {}, {}
        if cfg.reliability in ("heuristic", "blend"):
            O_h, comps, stats, checks = analyse(image, cfg.observation, lr=self.lr_image)
            O_h = resize_map(O_h, latent_hw)
            self.checks.update(checks)
        else:
            if image.shape[0] > 1:
                self.checks["cfg_batch_identical"] = (image == image[0:1]).all()
        if cfg.uses_learned:
            dev = next(self.lornet.parameters()).device
            with torch.no_grad():
                O_l = self.lornet(image[0:1].to(device=dev, dtype=torch.float32))[0, 0]
            if tuple(O_l.shape) != tuple(latent_hw):
                raise RuntimeError(f"LORNet map {tuple(O_l.shape)} does not match the latent size {latent_hw}")
            O_l = O_l.to(image.device)
            self.checks["O_learned_finite"] = torch.isfinite(O_l).all()
            extra = map_stats("O_learned", O_l)
            comps = {**comps, "learned": O_l}
        if cfg.reliability == "heuristic":
            O_lat = O_h
        elif cfg.reliability == "learned":
            O_lat = O_l
        else:
            O_lat = cfg.learned.blend * O_l + (1.0 - cfg.learned.blend) * O_h.to(O_l.device)
        if cfg.reliability != "heuristic":
            stats = {**stats, **map_stats("O", O_lat)}
        W = fuse(O_lat, semantic_latent.to(O_lat.device), cfg.fusion, key)
        self.O_latent, self.components = O_lat, comps
        self.W = W[None, None].float()
        self.checks["W_finite"] = torch.isfinite(W).all()
        self.map_info = {**stats, **extra, **map_stats("S", semantic_latent), **map_stats("W", W),
                         "latent_size": list(latent_hw), "map_mode": cfg.fusion.map, "control": cfg.fusion.control,
                         "reliability": cfg.reliability}
        return self.W

    def next_timestep(self, scheduler, timestep):
        """Host value of the current timestep without synchronising on the device tensor."""
        import torch
        if isinstance(timestep, torch.Tensor):
            if self.host_timesteps is None:
                self.host_timesteps = [int(v) for v in scheduler.timesteps.tolist()]   # one transfer per call
            if self.step_index >= len(self.host_timesteps):
                raise RuntimeError("SFR-OC: more scheduler steps than scheduler.timesteps")
            t = self.host_timesteps[self.step_index]
            ok = (timestep == t).all()
            prev = self.checks.get("timesteps_match")
            self.checks["timesteps_match"] = ok if prev is None else (prev & ok)
        else:
            t = int(timestep)
        self.step_index += 1
        return t

    def validate(self):
        bad = [k for k, v in self.checks.items() if not bool(v)]
        if bad:
            raise RuntimeError(f"SFR-OC deferred checks failed: {bad}")

    def summary(self):
        """Host-side summary of the last pipeline call (transfers diagnostics once; raises if a check failed)."""
        if self._summary is not None:
            return self._summary
        import torch

        from .observation_reliability import tensor_stats_to_python
        self.validate()
        applied = [s for s in self.step_log if s["gamma"] > 0]
        steps = []
        if applied:
            vals = torch.stack([torch.stack([s["residual_abs_mean"], s["delta_abs_mean"]]) for s in applied]).cpu()
            for s, (ra, da) in zip(applied, vals.tolist()):
                steps.append({"t": s["t"], "gamma": s["gamma"], "c0": s.get("c0"), "residual_abs_mean": ra,
                              "delta_abs_mean": da})
        zero = [{"t": s["t"], "gamma": 0.0} for s in self.step_log if s["gamma"] == 0]
        self.step_records = sorted(steps + zero, key=lambda s: -s["t"])
        info = tensor_stats_to_python(self.map_info)
        cfg = self.cfg
        self._summary = {
            "strength": cfg.strength, "band": cfg.band, "lowpass": cfg.lowpass,
            "sigma": cfg.sigma if cfg.lowpass == "gaussian" else None,
            "cutoff": cfg.cutoff if cfg.lowpass == "fft" else None,
            "schedule": cfg.schedule, "power": cfg.power, "t_min": cfg.t_min, "t_max": cfg.t_max,
            "precision": cfg.precision, "formulation": FORMULATION, "reliability": cfg.reliability,
            "lornet_checkpoint": self.lornet_meta.get("path"), "lornet_sha256": self.lornet_meta.get("sha256"),
            "lornet_tau": (self.lornet_meta.get("target") or {}).get("tau"),
            "prediction_type": self.prediction_type, "reference_source": self.reference_source,
            "steps_seen": len(self.step_log), "steps_corrected": len(applied),
            "gamma_mean": (sum(s["gamma"] for s in applied) / len(applied)) if applied else 0.0,
            "delta_abs_mean": (sum(s["delta_abs_mean"] for s in steps) / len(steps)) if steps else 0.0,
            "analysis_ms": self.analysis_timer.total_ms(), "correction_ms": self.correction_timer.total_ms(),
            "encode_ms": self.encode_timer.total_ms(), "timing_method": self.correction_timer.method
            if applied else self.analysis_timer.method, **info,
        }
        return self._summary
