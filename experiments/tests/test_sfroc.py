"""CFA-SeeSR V2: SFR-OC (observation reliability, frequency decomposition, fusion, scheduler-step correction),
alternative-sampler compatibility and V2 configuration. CPU only; the pipeline tests use the tiny instance of the
real upstream pipeline (tests/tiny_pipeline.py)."""
import json
import math
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from cfa_seesr import MethodConfig  # noqa: E402
from cfa_seesr.frequency import decompose, fft_lowpass, gaussian_lowpass, lowpass  # noqa: E402
from cfa_seesr.observation_consistency import (FusionConfig, SFROCConfig, apply_x0_delta, check_scheduler,  # noqa: E402
                                               correct, fuse, gamma_at, predicted_x0, shuffle_blocks)
from cfa_seesr.observation_reliability import (INDICATORS, ObservationReliabilityConfig, analyse,  # noqa: E402
                                               immerkaer_sigma, resize_map)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _sched(**over):
    from diffusers import DDIMScheduler, DDPMScheduler

    from experiments.tests.tiny_pipeline import SD2_BASE_SCHEDULER
    cfg = {k: v for k, v in SD2_BASE_SCHEDULER.items() if k != "_class_name"}
    cls = over.pop("cls", DDPMScheduler)
    return cls.from_config({**cfg, **over}) if cls is not DDIMScheduler else DDIMScheduler.from_config({**cfg, **over})


def _rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _natural_image(size=64, seed=0):
    """Smooth shapes + an edge + texture + noise, values in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    base = 0.3 + 0.4 * (xx > 0.5).float() + 0.1 * torch.sin(12 * yy)
    tex = 0.1 * torch.sin(60 * xx) * (yy > 0.6).float()
    img = (base + tex)[None].repeat(3, 1, 1) + 0.03 * torch.randn(3, size, size, generator=g)
    return img.clamp(0, 1)[None]


# --------------------------------------------------------------------------- #
# Frequency decomposition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method,kw", [("gaussian", {"sigma": 2.0}), ("gaussian", {"sigma": 20.0}),
                                       ("fft", {"cutoff": 0.3})])
def test_decomposition_reconstructs(method, kw):
    x = _rand(2, 4, 16, 24)
    low, high = decompose(x, method, **kw)
    assert low.shape == x.shape and torch.allclose(low + high, x, atol=1e-6)


def test_lowpass_properties():
    const = torch.full((1, 4, 12, 12), 0.7)
    assert torch.allclose(gaussian_lowpass(const, 3.0), const, atol=1e-6)
    assert torch.allclose(fft_lowpass(const, 0.2), const, atol=1e-5)
    checker = ((torch.arange(16)[:, None] + torch.arange(16)[None]) % 2).float()[None, None] * 2 - 1
    assert gaussian_lowpass(checker, 2.0).abs().max() < 0.05          # Nyquist pattern removed
    assert fft_lowpass(checker, 0.5).abs().max() < 1e-5
    x = _rand(1, 4, 16, 16)
    assert torch.equal(fft_lowpass(x, 1.0), x) and torch.equal(gaussian_lowpass(x, 0.0), x)
    low = fft_lowpass(x, 0.3, pad=False)
    assert torch.allclose(fft_lowpass(low, 0.3, pad=False), low, atol=1e-5)   # ideal circular low-pass: idempotent
    half = x.half()
    assert gaussian_lowpass(half, 2.0).dtype == torch.float16 and fft_lowpass(half, 0.3).dtype == torch.float16
    with pytest.raises(ValueError):
        lowpass(x, "wavelet")


# --------------------------------------------------------------------------- #
# Observation reliability (part A)
# --------------------------------------------------------------------------- #
def test_observation_map_bounds_components_and_determinism():
    img = _natural_image(64).repeat(2, 1, 1, 1)                        # CFG batch: item 0 is used
    O, comps, stats, checks = analyse(img)
    assert bool(checks["cfg_batch_identical"]) and bool(checks["O_finite"])
    assert O.shape == (16, 16) and O.dtype == torch.float32
    assert torch.isfinite(O).all() and 0 <= O.min() and O.max() <= 1
    assert set(comps) == set(INDICATORS) and all(0 <= c.min() and c.max() <= 1 for c in comps.values())
    assert stats["analysis_size"] == [16, 16] and set(stats["weights"]) == set(INDICATORS)
    O2 = analyse(img)[0]
    assert torch.equal(O, O2)
    # the vertical edge at x = 0.5 is the most coherent structure
    assert comps["edge_coherence"][:, 7:9].mean() > comps["edge_coherence"][:, :4].mean()


def test_single_indicator_weight_isolates_it():
    img = _natural_image(64)
    for k in INDICATORS:
        w = {i: (1.0 if i == k else 0.0) for i in INDICATORS}
        O, comps, _, _ = analyse(img, ObservationReliabilityConfig(weights=w, smoothing_sigma=0.0))
        assert torch.allclose(O, comps[k], atol=1e-6)


def test_constant_image_and_noise_estimate():
    O, comps, stats, _ = analyse(torch.full((1, 3, 32, 32), 0.5))
    assert torch.isfinite(O).all() and float(stats["noise_sigma_estimate"]) == 0.0
    clean = torch.full((1, 1, 64, 64), 0.5)
    noisy = clean + 0.05 * _rand(1, 1, 64, 64)
    assert immerkaer_sigma(clean) == 0.0 and 0.03 < immerkaer_sigma(noisy) < 0.07


@pytest.mark.parametrize("kw", [dict(window=1), dict(analysis_scale=0), dict(weights={"edge": 1.0}),
                                dict(weights={k: 0.0 for k in INDICATORS}), dict(reference_level=0.0),
                                dict(significance=-1.0), dict(noise_floor=0.0), dict(low_gradient_quantile=0.0)])
def test_observation_config_validation(kw):
    with pytest.raises(ValueError):
        ObservationReliabilityConfig(**kw)


def test_resize_map_bounds():
    m = torch.rand(16, 16)
    for size in ((64, 64), (8, 8), (64, 48), (10, 30)):
        r = resize_map(m, size)
        assert r.shape == size and m.min() - 1e-6 <= r.min() and r.max() <= m.max() + 1e-6
    assert abs(float(resize_map(m, (8, 8)).mean()) - float(m.mean())) < 1e-5


# --------------------------------------------------------------------------- #
# Fusion (part C)
# --------------------------------------------------------------------------- #
def test_fusion_bounds_modes_and_shuffle():
    O, S = torch.rand(32, 32), torch.rand(32, 32)
    for w_o, w_s, b in ((0.5, 0.5, 0.0), (2.0, 2.0, 0.5), (1.0, 0.0, -0.3), (0.0, 1.0, 0.0)):
        W = fuse(O, S, FusionConfig(w_observation=w_o, w_semantic=w_s, bias=b))
        assert torch.isfinite(W).all() and 0 <= W.min() and W.max() <= 1
    W = fuse(O, S, FusionConfig())
    assert torch.allclose(W, (0.5 * O + 0.5 * (1 - S)).clamp(0, 1))
    assert torch.equal(fuse(O, S, FusionConfig(map="uniform")), torch.ones_like(O))
    assert torch.equal(fuse(O, S, FusionConfig(map="observation")), O)
    assert torch.allclose(fuse(O, S, FusionConfig(map="semantic")), 1 - S)
    assert fuse(O, S, FusionConfig(w_min=0.2, w_max=0.6)).min() >= 0.2
    sh = fuse(O, S, FusionConfig(control="shuffled"), key="img")
    assert torch.equal(sh.flatten().sort().values, W.flatten().sort().values) and not torch.equal(sh, W)
    assert torch.equal(sh, fuse(O, S, FusionConfig(control="shuffled"), key="img"))   # deterministic per key
    odd = torch.arange(10 * 13, dtype=torch.float32).reshape(10, 13)
    s = shuffle_blocks(odd, 4, 0)                                                     # partial last bands move too
    assert torch.equal(s.flatten().sort().values, odd.flatten()) and not torch.equal(s, odd)


# --------------------------------------------------------------------------- #
# Correction maths and scheduler handling (part B)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pt", ["epsilon", "v_prediction", "sample"])
def test_correction_moves_x0_by_exactly_delta(pt):
    sched = _sched(prediction_type=pt)
    xt, mo, z = _rand(1, 4, 8, 8, seed=1), _rand(1, 4, 8, 8, seed=2), _rand(1, 4, 8, 8, seed=3)
    t = 500
    a = float(sched.alphas_cumprod[t])
    W = torch.ones(1, 1, 8, 8)
    cfg = SFROCConfig(enabled=True, strength=1.0, band="full")
    out, diag = correct(mo, xt, t, sched, z, W, 1.0, cfg, pt)
    assert torch.allclose(predicted_x0(out, xt, a, pt), z, atol=1e-4)     # full correction lands on z_lr
    out, _ = correct(mo, xt, t, sched, z, W, 0.25, cfg, pt)
    x0 = predicted_x0(mo, xt, a, pt)
    assert torch.allclose(predicted_x0(out, xt, a, pt), x0 + 0.25 * (z - x0), atol=1e-4)
    # the scheduler itself sees the corrected x0: compare with its own step on an equivalent output
    torch.manual_seed(0)
    s1 = sched.step(out, t, xt, generator=torch.Generator().manual_seed(0)).prev_sample
    torch.manual_seed(0)
    s2 = sched.step(apply_x0_delta(mo, 0.25 * (z - x0), a, pt), t, xt, generator=torch.Generator().manual_seed(0)).prev_sample
    assert torch.allclose(s1, s2, atol=1e-5)


def test_low_band_changes_mostly_low_frequencies_of_x0():
    """The x0 change is the two-pass low-passed residual: most of its energy lies below the cutoff, unlike the
    residual itself (measured with the unpadded DFT; padding at the borders causes some leakage)."""
    from cfa_seesr.frequency import high_frequency_energy_fraction as hf
    sched = _sched()
    xt, mo, z = _rand(1, 4, 32, 32, seed=1), _rand(1, 4, 32, 32, seed=2), _rand(1, 4, 32, 32, seed=3)
    a = float(sched.alphas_cumprod[700])
    cfg = SFROCConfig(enabled=True, strength=1.0, band="low", lowpass="fft", cutoff=0.3)
    out, _ = correct(mo, xt, 700, sched, z, torch.ones(1, 1, 32, 32), 1.0, cfg, "epsilon")
    x0, x0c = predicted_x0(mo, xt, a, "epsilon"), predicted_x0(out, xt, a, "epsilon")
    assert hf(x0c - x0, 0.4) < 0.25 * hf(z - x0, 0.4)


def test_spatial_map_and_batch_broadcast_and_dtype():
    sched = _sched()
    xt, mo = _rand(2, 4, 8, 8, seed=1).half(), _rand(2, 4, 8, 8, seed=2).half()     # batch of 2, fp16
    z = _rand(1, 4, 8, 8, seed=3)
    W = torch.zeros(1, 1, 8, 8)
    W[..., :4] = 1.0
    cfg = SFROCConfig(enabled=True, strength=1.0, band="full")
    out, _ = correct(mo, xt, 300, sched, z, W, 1.0, cfg, "epsilon")
    assert out.dtype == torch.float16 and out.shape == mo.shape
    assert torch.equal(out[..., 4:], mo[..., 4:])                       # W = 0: untouched (bit-exact)
    assert not torch.equal(out[..., :4], mo[..., :4])


def test_scheduler_and_prediction_type_checks():
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
    assert check_scheduler(_sched()) == "epsilon"
    assert check_scheduler(_sched(cls=DDIMScheduler, prediction_type="v_prediction")) == "v_prediction"
    from experiments.tests.tiny_pipeline import SD2_BASE_SCHEDULER
    cfg = {k: v for k, v in SD2_BASE_SCHEDULER.items() if k != "_class_name"}
    for cls in (DPMSolverMultistepScheduler, EulerDiscreteScheduler):
        with pytest.raises(ValueError, match="does not support"):
            check_scheduler(cls.from_config(cfg))
    s = _sched()
    s.register_to_config(prediction_type="flow")
    with pytest.raises(ValueError, match="prediction_type"):
        check_scheduler(s)
    with pytest.raises(ValueError):
        predicted_x0(torch.zeros(1), torch.zeros(1), 0.5, "flow")


def test_gamma_schedules():
    cfg = SFROCConfig(enabled=True, strength=0.8, schedule="power", power=2.0)
    vals = [gamma_at(t, cfg, 999) for t in (999, 700, 300, 0)]
    assert vals == sorted(vals, reverse=True) and abs(vals[0] - 0.8) < 1e-12 and vals[-1] == 0.0
    w = SFROCConfig(enabled=True, strength=0.5, schedule="window", t_min=200, t_max=800)
    assert gamma_at(900, w, 999) == 0 and gamma_at(500, w, 999) == 0.5 and gamma_at(100, w, 999) == 0
    assert gamma_at(500, SFROCConfig(enabled=True, schedule="constant"), 999) == 0.0   # strength 0
    assert gamma_at(500, SFROCConfig(enabled=True, strength=0.3, schedule="constant"), 999) == 0.3


@pytest.mark.parametrize("kw", [dict(strength=1.5), dict(strength=float("nan")), dict(band="mid"), dict(sigma=0),
                                dict(cutoff=0), dict(schedule="cosine"), dict(t_min=10, t_max=5), dict(power=0),
                                dict(reference="cache")])
def test_sfroc_config_validation(kw):
    with pytest.raises(ValueError):
        SFROCConfig(**kw)


@pytest.mark.parametrize("kw", [dict(map="max"), dict(control="flip"), dict(w_observation=-1), dict(w_min=0.8, w_max=0.2),
                                dict(semantic_default=2), dict(shuffle_block=0)])
def test_fusion_config_validation(kw):
    with pytest.raises(ValueError):
        FusionConfig(**kw)


# --------------------------------------------------------------------------- #
# Hooks on the tiny real pipeline
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    pytest.importorskip("diffusers")
    from experiments.tests.tiny_pipeline import build_pipeline
    return build_pipeline(tmp_path_factory.mktemp("tok_sfroc"))


def _img(seed, size=32):
    from PIL import Image
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8))


def _run(pipe, img, steps=3, **over):
    from experiments.tests.tiny_pipeline import call_kwargs
    ram = torch.randn(1, 5, 512, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(7)
    kw = call_kwargs(img, steps, torch.Generator().manual_seed(7), ram)
    kw.update(over)
    return pipe("a, clean", img, **kw).images[0]


def _with_sfroc(pipe, cfg, fn, state=None):
    from cfa_seesr.hooks import MethodHooks, MethodState
    state = state or MethodState()
    before = [vars(o).get(n) for o, n in ((pipe.vae, "encode"), (pipe, "prepare_image"), (pipe.scheduler, "step"))]
    hooks = MethodHooks()
    hooks.install_observation_consistency(pipe, cfg, state)
    try:
        return fn(state)
    finally:
        hooks.remove()
        after = [vars(o).get(n) for o, n in ((pipe.vae, "encode"), (pipe, "prepare_image"), (pipe.scheduler, "step"))]
        assert after == before                           # every wrapper removed


def test_zero_strength_is_an_exact_noop(pipe):
    img = _img(1)
    ref = _run(pipe, img)
    out, state = _with_sfroc(pipe, SFROCConfig(enabled=True, strength=0.0), lambda st: (_run(pipe, img), st))
    assert np.array_equal(out, ref)
    log = state.sfroc.step_log
    assert len(log) == 3 and all(s["gamma"] == 0.0 for s in log)
    assert state.sfroc.W is not None                     # maps are still built (logged), just not applied
    assert np.array_equal(_run(pipe, img), ref)          # hooks fully removed


def test_correction_changes_output_deterministically_without_extra_passes(pipe):
    img = _img(2)
    ref = _run(pipe, img)
    calls = {"unet": 0, "controlnet": 0, "encode": 0}
    h1 = pipe.unet.register_forward_hook(lambda *a: calls.__setitem__("unet", calls["unet"] + 1))
    h2 = pipe.controlnet.register_forward_hook(lambda *a: calls.__setitem__("controlnet", calls["controlnet"] + 1))
    try:
        cfg = SFROCConfig(enabled=True, strength=0.8, schedule="constant")
        o1, st = _with_sfroc(pipe, cfg, lambda s: (_run(pipe, img), s))
        n_unet, n_cn = calls["unet"], calls["controlnet"]
        o2, _ = _with_sfroc(pipe, cfg, lambda s: (_run(pipe, img), s))
    finally:
        h1.remove(), h2.remove()
    assert n_unet == n_cn == 3                           # one CFG-batched call per step, as without SFR-OC
    assert np.array_equal(o1, o2) and not np.array_equal(o1, ref)
    summ = st.sfroc.summary()
    assert summ["reference_source"] == "extra_encode" and summ["steps_corrected"] == 3
    assert summ["prediction_type"] == "epsilon" and summ["W_min"] >= 0 and summ["W_max"] <= 1


def _encode_spy(pipe, seen):
    original = pipe.vae.encode

    def spy(*a, **k):
        out = original(*a, **k)
        seen.setdefault("means", []).append(out.latent_dist.mean.detach().clone())
        seen.setdefault("inputs", []).append(a[0].detach().clone())
        return out
    return spy


def test_reference_latent_pipeline_mode_reuses_the_pipeline_posterior(pipe):
    """reference 'pipeline' (opt-in): no extra VAE pass; z_lr = posterior mean of the pipeline's LR encode."""
    img = _img(3)
    seen = {}
    pipe.vae.encode = _encode_spy(pipe, seen)
    try:
        cfg = SFROCConfig(enabled=True, strength=0.5, schedule="constant", reference="pipeline")
        st = _with_sfroc(pipe, cfg, lambda s: (_run(pipe, img), s)[1])
    finally:
        del pipe.vae.encode
    assert len(seen["means"]) == 1                       # no extra VAE pass with start_point 'lr'
    expected = seen["means"][0][0:1].float() * pipe.vae.config.scaling_factor
    assert torch.allclose(st.sfroc.z_ref, expected) and st.sfroc.reference_source == "pipeline_encode"
    assert "reference_cfg_identical" in st.sfroc.checks  # the deferred CFG check is kept in this mode


def test_reference_latent_default_encodes_conditioning_item_0(pipe):
    """reference 'extra_encode' (default since 2.1.1, as run in the V2.1 heuristic pilot): one extra VAE encode of
    batch item 0 of the conditioning image, posterior mean x scaling_factor."""
    img = _img(3)
    seen = {}
    pipe.vae.encode = _encode_spy(pipe, seen)
    try:
        cfg = SFROCConfig(enabled=True, strength=0.5, schedule="constant")
        assert cfg.reference == "extra_encode"
        st = _with_sfroc(pipe, cfg, lambda s: (_run(pipe, img), s)[1])
    finally:
        del pipe.vae.encode
    assert len(seen["means"]) == 2                       # pipeline's LR start + one reference encode
    assert seen["inputs"][1].shape[0] == 1
    assert torch.equal(seen["inputs"][1], (st.sfroc.image[0:1] * 2 - 1).to(seen["inputs"][1].dtype))
    expected = seen["means"][1].float() * pipe.vae.config.scaling_factor
    assert torch.equal(st.sfroc.z_ref, expected) and st.sfroc.reference_source == "extra_encode"
    assert bool(st.sfroc.checks["cfg_batch_identical"])  # the image's CFG copies are still checked
    st.sfroc.summary()                                   # all deferred checks pass


def test_reference_pipeline_mode_reports_differing_cfg_rows(pipe):
    """The opt-in mode keeps the strict check: differing posterior rows fail summary()."""
    import types
    from cfa_seesr.hooks import MethodHooks, MethodState
    hooks, state = MethodHooks(), MethodState()
    ctrl = hooks.install_observation_consistency(pipe, SFROCConfig(enabled=True, strength=0.5, reference="pipeline"),
                                                 state)
    try:
        ctrl.image = torch.rand(2, 3, 64, 64)
        ctrl.W = torch.ones(1, 1, 8, 8)
        mean = torch.randn(2, 4, 8, 8)
        ctrl.encode_dist = types.SimpleNamespace(mean=mean)
        pipe.scheduler.set_timesteps(3)
        xt, mo = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        pipe.scheduler.step(mo, pipe.scheduler.timesteps[0], xt)
        with pytest.raises(RuntimeError, match="reference_cfg_identical"):
            ctrl.summary()
    finally:
        hooks.remove()


def test_correction_time_excludes_the_reference_encode(pipe):
    """The V2.1 heuristic pilot reported correction_ms ~ encode_ms because the first corrected step's timer also
    covered the reference encode. The encode is now timed only as encode_ms."""
    import time
    original = pipe.vae.encode

    def slow(*a, **k):
        time.sleep(0.2)
        return original(*a, **k)

    pipe.vae.encode = slow
    try:
        cfg = SFROCConfig(enabled=True, strength=0.5, schedule="constant")
        st = _with_sfroc(pipe, cfg, lambda s: (_run(pipe, _img(5)), s)[1])
    finally:
        del pipe.vae.encode
    summ = st.sfroc.summary()
    assert summ["encode_ms"] >= 190 and summ["correction_ms"] < 150


def test_reference_auto_is_rejected():
    with pytest.raises(ValueError, match="removed in 2.1.1"):
        SFROCConfig(enabled=True, strength=0.5, reference="auto")


def test_noise_start_uses_one_extra_encode(pipe):
    img = _img(4)
    cfg = SFROCConfig(enabled=True, strength=0.5, schedule="constant")
    st = _with_sfroc(pipe, cfg, lambda s: (_run(pipe, img, start_point="noise"), s)[1])
    assert st.sfroc.reference_source == "extra_encode" and st.sfroc.summary()["encode_ms"] > 0


def test_tiled_sampling_corrects_the_full_latent(pipe):
    img = _img(5, 48)                                    # latent 6x6, tile 4 -> tiled branch
    cfg = SFROCConfig(enabled=True, strength=0.5, schedule="constant")
    out, st = _with_sfroc(pipe, cfg, lambda s: (_run(pipe, img, steps=2, latent_tiled_size=4, latent_tiled_overlap=1), s))
    assert np.isfinite(out).all() and st.sfroc.summary()["steps_corrected"] == 2
    assert tuple(st.sfroc.W.shape[-2:]) == (6, 6)


def test_ddim_supported_and_dpm_rejected(pipe):
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler

    from cfa_seesr.hooks import MethodHooks, MethodState
    original = pipe.scheduler
    try:
        pipe.scheduler = DDIMScheduler.from_config(original.config)
        img = _img(6)
        ref = _run(pipe, img)
        out = _with_sfroc(pipe, SFROCConfig(enabled=True, strength=0.5, schedule="constant"), lambda s: _run(pipe, img))
        assert np.isfinite(out).all() and not np.array_equal(out, ref)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(original.config)
        hooks = MethodHooks()
        with pytest.raises(ValueError, match="does not support"):
            hooks.install_observation_consistency(pipe, SFROCConfig(enabled=True, strength=0.5), MethodState())
        hooks.remove()
    finally:
        pipe.scheduler = original


@pytest.mark.parametrize("name", ["DDIMScheduler", "DPMSolverMultistepScheduler"])
def test_alternative_samplers_run_the_upstream_pipeline(pipe, name):
    """Compatibility of the efficiency-experiment samplers with pipeline_seesr (step signature, generator,
    add_noise for the LR-embedded start, step count). Says nothing about quality."""
    import diffusers
    original = pipe.scheduler
    try:
        pipe.scheduler = getattr(diffusers, name).from_config(original.config)
        n = {"steps": 0}
        step = pipe.scheduler.step
        pipe.scheduler.step = lambda *a, **k: (n.__setitem__("steps", n["steps"] + 1), step(*a, **k))[1]
        a, b = _run(pipe, _img(7), steps=4), None
        del pipe.scheduler.step
        b = _run(pipe, _img(7), steps=4)
        assert np.isfinite(a).all() and n["steps"] == 4 and np.array_equal(a, b)   # deterministic
    finally:
        pipe.scheduler = original


# --------------------------------------------------------------------------- #
# Runner integration and configuration
# --------------------------------------------------------------------------- #
def _fake_upstream(pipe, upstream):
    mod = types.ModuleType("fake_test_seesr")
    mod.load_seesr_pipeline = lambda *a, **k: pipe
    mod.load_tag_model = lambda *a, **k: None
    mod.get_validation_prompt = lambda *a, **k: ("unused", None)
    mod.tensor_transforms = upstream.tensor_transforms
    mod.ram_transforms = upstream.ram_transforms
    return mod


def _runner_output(pipe, upstream, method_dict, img, scheduler="DDPMScheduler"):
    from experiments import run_inference as ri
    from experiments.tests.test_spatial import _model_with_tags
    from experiments.tests.tiny_pipeline import call_kwargs
    method = MethodConfig.from_dict(method_dict)
    mod = _fake_upstream(pipe, upstream)
    rec = ri.Recorder(["img"], method=method)
    rec.scheduler_name = scheduler
    original_sched = pipe.scheduler
    original_step = pipe.scheduler.step
    ri.install_hooks(mod, rec, {})
    try:
        timed = mod.load_seesr_pipeline()
        dape, _ = _model_with_tags(upstream, min_tags=3)
        mod.get_validation_prompt(types.SimpleNamespace(prompt=""), img, dape)
        ram = torch.randn(1, 5, 512, generator=torch.Generator().manual_seed(1))
        torch.manual_seed(7)
        out = timed("a, clean", img, **call_kwargs(img, 3, torch.Generator().manual_seed(7), ram)).images[0]
        return out, rec, type(pipe.scheduler).__name__
    finally:
        ri.remove_hooks(mod, rec)
        pipe.scheduler = original_sched
        pipe.scheduler.step = original_step
        if "step" in vars(pipe.scheduler) and pipe.scheduler.step is original_step:
            del pipe.scheduler.step


@pytest.fixture(scope="module")
def upstream_test_seesr():
    import sys
    sys.dont_write_bytecode = True
    import test_seesr
    return test_seesr


def _v2_variants():
    from experiments.common import load_sweep
    return {n.rsplit("_ddpm", 1)[0]: c["method"] for n, c in load_sweep(CONFIGS / "v2_ablation.yaml")
            if n.endswith("_ddpm20_s42")}


def test_v2_zero_equals_v1_and_full_differs_through_the_runner(pipe, upstream_test_seesr):
    pytest.importorskip("loralib")
    v = _v2_variants()
    img = _img(8)
    v1, _, _ = _runner_output(pipe, upstream_test_seesr, v["combined_spatial"], img)
    zero, rec0, _ = _runner_output(pipe, upstream_test_seesr, v["v2_heur_sem_zero"], img)
    full, rec, _ = _runner_output(pipe, upstream_test_seesr, v["v2_heur_sem"], img)
    assert np.array_equal(zero, v1)                      # no-op control reproduces V1 exactly
    assert not np.array_equal(full, v1)
    row = rec.rows[0]
    assert row["sfroc_steps_corrected"] == 3 and row["sfroc_reference"] == "extra_encode"
    assert 0 <= row["sfroc_W_min"] <= row["sfroc_W_mean"] <= row["sfroc_W_max"] <= 1
    assert rec0.rows[0]["sfroc_steps_corrected"] == 0
    r = rec.sfroc_records[0]
    assert r["semantic_source"] == "ssr_map" and r["band"] == "low" and len(r["steps"]) == 3
    assert {"W", "O"} <= set(rec.sfroc_maps["img"])


def test_runner_sampler_swap(pipe, upstream_test_seesr):
    pytest.importorskip("loralib")
    out, rec, name = _runner_output(pipe, upstream_test_seesr, {}, _img(9), scheduler="DDIMScheduler")
    assert name == "DDIMScheduler" and np.isfinite(out).all() and rec.rows[0]["denoising_steps"] == 3
    assert type(pipe.scheduler).__name__ == "DDPMScheduler"    # fixture restored


def test_disabled_by_default_and_configs():
    from experiments import run_inference as ri
    from experiments.common import load_sweep, load_yaml, sweep_default_variants
    m = MethodConfig()
    assert not m.sfroc_active and m.sfr_oc.strength == 0.0 and m.is_baseline
    base = load_yaml(CONFIGS / "baseline.yaml")
    bm = MethodConfig.from_dict(base["method"])
    assert bm.is_baseline and not bm.sfroc_active and bm.sfr_oc == SFROCConfig.from_dict(base["method"]["sfr_oc"])
    assert base["runtime"]["scheduler"] == "DDPMScheduler"
    # V1 sweep untouched: no SFR-OC anywhere
    assert not any(MethodConfig.from_dict(c["method"]).sfroc_active for _, c in load_sweep(CONFIGS / "sampling_sweep.yaml"))
    assert sweep_default_variants(CONFIGS / "sampling_sweep.yaml") == ["original", "semantic_conf", "lr_schedule", "combined"]
    v = {k: MethodConfig.from_dict(c) for k, c in _v2_variants().items()}
    assert sweep_default_variants(CONFIGS / "v2_ablation.yaml") == [
        "original", "combined_spatial", "v2_heur", "v2_learned", "v2_heur_sem", "v2_learned_sem", "v2_uniform",
        "v2_meanmatched", "v2_shuffled", "v2_zero"]
    assert v["original"].is_baseline and not v["combined_spatial"].sfroc_active
    sfroc_variants = [n for n in v if n.startswith("v2_")]
    for name in sfroc_variants:
        assert v[name].semantic_confidence == v["combined_spatial"].semantic_confidence
        assert v[name].lr_guidance == v["combined_spatial"].lr_guidance
        assert v[name].spatial_semantic == v["combined_spatial"].spatial_semantic
    expect = {"v2_heur": ("low", "observation", "none", "heuristic"),
              "v2_learned": ("low", "observation", "none", "learned"),
              "v2_heur_sem": ("low", "fused", "none", "heuristic"), "v2_learned_sem": ("low", "fused", "none", "learned"),
              "v2_uniform": ("low", "uniform", "none", "heuristic"),
              "v2_meanmatched": ("low", "mean_matched", "none", "learned"),
              "v2_shuffled": ("low", "fused", "shuffled", "learned"), "v2_zero": ("low", "fused", "none", "learned"),
              "v2_global": ("full", "uniform", "none", "heuristic"), "v2_spatial": ("full", "fused", "none", "heuristic"),
              "v2_heur_sem_meanmatched": ("low", "mean_matched", "none", "heuristic"),
              "v2_heur_sem_shuffled": ("low", "fused", "shuffled", "heuristic"),
              "v2_heur_sem_zero": ("low", "fused", "none", "heuristic"),
              "v2_blend_sem": ("low", "fused", "none", "blend"), "v2_sem_only": ("low", "semantic", "none", "heuristic")}
    assert set(expect) == set(sfroc_variants)
    for name, (band, mp, ctl, rel) in expect.items():
        s = v[name].sfr_oc
        assert (s.band, s.fusion.map, s.fusion.control, s.reliability) == (band, mp, ctl, rel)
        assert s.strength == (0.0 if name.endswith("zero") else 0.5)
    # one LORNet checkpoint for every learned variant; the other settings identical between paired variants
    ckpts = {v[n].sfr_oc.learned.checkpoint for n in sfroc_variants if v[n].sfr_oc.uses_learned}
    assert len(ckpts) == 1 and None not in ckpts
    for a_, b_ in (("v2_learned_sem", "v2_meanmatched"), ("v2_learned_sem", "v2_shuffled"),
                   ("v2_learned_sem", "v2_zero"), ("v2_heur_sem", "v2_heur_sem_meanmatched")):
        sa, sb = v[a_].sfr_oc, v[b_].sfr_oc
        assert (sa.sigma, sa.schedule, sa.power, sa.precision, sa.fusion.w_observation, sa.fusion.w_semantic) == \
               (sb.sigma, sb.schedule, sb.power, sb.precision, sb.fusion.w_observation, sb.fusion.w_semantic)
    # same-named runs are identical to sampling_sweep.yaml's
    a = dict(load_sweep(CONFIGS / "v2_ablation.yaml"))
    b = dict(load_sweep(CONFIGS / "sampling_sweep.yaml"))
    for n in ("original_ddpm20_s42", "combined_spatial_ddpm50_s123"):
        assert {k: x for k, x in a[n].items() if k != "experiment"} == {k: x for k, x in b[n].items() if k != "experiment"}
    # SFR-OC + an unsupported sampler is rejected at configuration time
    cfg = load_yaml(CONFIGS / "baseline.yaml")
    cfg["method"]["sfr_oc"]["enabled"] = True
    cfg["runtime"]["sampler_override"] = "DPMSolverMultistepScheduler"
    with pytest.raises(ri.ConfigError, match="sfr_oc"):
        ri.method_config(cfg)


def test_efficiency_sweep_and_scheduler_validation(tmp_path):
    from experiments import run_inference as ri
    from experiments.common import load_sweep, load_yaml
    runs = dict(load_sweep(CONFIGS / "efficiency_sweep.yaml"))
    assert {c["runtime"]["scheduler"] for c in runs.values()} == {"DDPMScheduler"}
    assert {c["runtime"]["sampler_override"] for c in runs.values()} == {None, "DDIMScheduler",
                                                                         "DPMSolverMultistepScheduler"}
    base = load_yaml(CONFIGS / "baseline.yaml")
    assert base["runtime"]["sampler_override"] is None
    ri.build_upstream_args(runs["original_ddim20_s42"], tmp_path)
    bad = load_yaml(CONFIGS / "baseline.yaml")
    bad["runtime"]["sampler_override"] = "EulerDiscreteScheduler"
    with pytest.raises(ri.ConfigError, match="sampler_override"):
        ri.build_upstream_args(bad, tmp_path)
    assert all(c["runtime"]["reseed_per_image"] for c in runs.values())
    assert {c["experiment"]["variant"] for c in runs.values()} == {"original", "original_ddim", "original_dpmpp"}
    assert set(ri.ALTERNATIVE_SCHEDULERS) == {"DDIMScheduler", "DPMSolverMultistepScheduler"}


def test_compare_results_sampler_differences():
    from experiments.tools import compare_results as cr

    def exp(name, variant, sched):
        meta = {"status": "completed", "dataset": {}, "environment": {},
                "upstream_args": {"num_inference_steps": 20, "seed": 42},
                "config": {"experiment": {"variant": variant},
                           "runtime": {"scheduler": "DDPMScheduler", "reseed_per_image": True,
                                       "sampler_override": None if sched == "DDPMScheduler" else sched},
                           "method": {}, "inference": {"num_inference_steps": 20, "seed": 42}},
                "scheduler": {"class": sched, "config": {"x": sched}}}
        return {"name": name, "meta": meta, "summaries": {"paired": {"metrics": {}}}, "per_image": {}}

    a, b = exp("a", "original", "DDPMScheduler"), exp("b", "original_ddim", "DDIMScheduler")
    assert any("scheduler" in p for p in cr.compatibility([a, b]))
    assert not [p for p in cr.compatibility([a, b], allow_sampler=True) if "scheduler" in p]
    c = exp("c", "original", "DDIMScheduler")
    assert any("mixes samplers" in p for p in cr.compatibility([a, c], allow_sampler=True))


def test_benchmark_tool_runs_on_cpu():
    from experiments.tools import benchmark_sfroc
    out = benchmark_sfroc.run("cpu", (64,), repeats=2)
    r = out["results"][0]
    assert r["latent_size"] == 8 and r["analysis"]["median_ms"] > 0 and r["correction_per_step"]["median_ms"] > 0
    assert math.isfinite(r["per_image_ms_at_50_steps_heuristic"]) and r["lornet"]["median_ms"] > 0
    assert out["lornet"]["weights"] == "random (timing only)" and out["lornet"]["parameters"] > 0
    json.dumps(out)
