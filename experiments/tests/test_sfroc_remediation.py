"""CFA-SeeSR V2 audit remediation: mean-matched and shuffled controls, noise-compensated observation reliability,
two-pass band-limited correction, output-space application (exact against the real diffusers step), FP16
displacement, scheduler rejections, deferred checks, timing semantics and host-sync discipline. CPU only;
CUDA-specific behaviour is marked and skipped where no GPU is present."""
import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from cfa_seesr.frequency import (fft_lowpass, gaussian_half_amplitude_period,  # noqa: E402
                                 high_frequency_energy_fraction, lowpass, split_lowpass)
from cfa_seesr.observation_consistency import (FusionConfig, SectionTimer, SFROCConfig,  # noqa: E402
                                               apply_to_step_output, apply_x0_delta, check_scheduler,
                                               correction_delta, fuse, predicted_x0, shuffle_blocks,
                                               x0_coefficient)
from cfa_seesr.observation_reliability import (ObservationReliabilityConfig, analyse,  # noqa: E402
                                               estimate_noise_sigma)


def _sched(cls_name="DDPMScheduler", **over):
    import diffusers

    from experiments.tests.tiny_pipeline import SD2_BASE_SCHEDULER
    cfg = {k: v for k, v in SD2_BASE_SCHEDULER.items() if k != "_class_name"}
    return getattr(diffusers, cls_name).from_config({**cfg, **over})


def _rand(*shape, seed=0, dtype=torch.float32):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _scene(kind, size=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    if kind == "flat":
        img = torch.full((size, size), 0.5)
    elif kind == "texture":                                  # texture everywhere, below the LR Nyquist frequency
        img = 0.5 + 0.2 * torch.sin(20 * xx) * torch.sin(17 * yy)
    elif kind == "edges":
        img = 0.2 + 0.6 * ((xx > 0.33) ^ (yy > 0.6)).float()
    else:  # mixed
        img = 0.3 + 0.4 * (xx > 0.5).float() + 0.1 * torch.sin(12 * yy) + 0.1 * torch.sin(30 * xx) * (yy > 0.6)
        img = img + 0.02 * torch.rand(size, size, generator=g)
    return img.clamp(0, 1)[None, None].repeat(1, 3, 1, 1)


def _noisy(img, sigma, seed=1):
    return (img + sigma * torch.randn(img.shape, generator=torch.Generator().manual_seed(seed))).clamp(0, 1)


# --------------------------------------------------------------------------- #
# 1. Controls: mean-matched and shuffled maps
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hw", [(32, 32), (33, 47), (8, 5), (64, 96)])
def test_mean_matched_map(hw):
    O, S = torch.rand(*hw), torch.rand(*hw)
    fused = fuse(O, S, FusionConfig(map="fused"))
    mm = fuse(O, S, FusionConfig(map="mean_matched"))
    assert torch.allclose(mm.mean(), fused.mean(), atol=1e-6)
    assert bool((mm == mm.flatten()[0]).all()) and float(fused.std(unbiased=False)) > 1e-3   # exactly constant
    clipped = fuse(O, S, FusionConfig(map="mean_matched", w_min=0.4, w_max=0.6, bias=0.3))
    ref = fuse(O, S, FusionConfig(map="fused", w_min=0.4, w_max=0.6, bias=0.3))
    assert torch.allclose(clipped, torch.full_like(ref, float(ref.mean())), atol=1e-6)   # mean of the CLIPPED map


@pytest.mark.parametrize("hw,block", [((32, 32), 8), ((33, 47), 8), ((10, 13), 4), ((7, 64), 8), ((1, 40), 8),
                                      ((5, 5), 8), ((64, 96), 3)])
def test_shuffle_preserves_distribution_for_any_size(hw, block):
    m = torch.rand(*hw, generator=torch.Generator().manual_seed(3))
    s = shuffle_blocks(m, block, seed=0, key="img")
    assert s.shape == m.shape
    assert torch.equal(s.flatten().sort().values, m.flatten().sort().values)          # same multiset
    assert torch.allclose(s.mean(), m.mean())
    n_bands = math.ceil(hw[0] / block) + math.ceil(hw[1] / block)
    if max(math.ceil(hw[0] / block), math.ceil(hw[1] / block)) > 1:
        assert not torch.equal(s, m)                                                  # never the identity
    else:
        assert n_bands == 2 and torch.equal(s, m)                                     # single band: nothing to move
    assert torch.equal(s, shuffle_blocks(m, block, seed=0, key="img"))                # deterministic
    assert not torch.equal(shuffle_blocks(m, block, seed=0, key="other"), s) or min(hw) < 2 * block


def test_shuffled_fusion_keeps_mean_and_values():
    O, S = torch.rand(33, 41), torch.rand(33, 41)
    W = fuse(O, S, FusionConfig())
    sh = fuse(O, S, FusionConfig(control="shuffled"), key="x")
    assert torch.allclose(sh.mean(), W.mean()) and torch.equal(sh.flatten().sort().values, W.flatten().sort().values)


# --------------------------------------------------------------------------- #
# 2. Observation reliability: noise compensation
# --------------------------------------------------------------------------- #
NOISE = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16)
DRAWS = 6


def _mean_O_curve(kind, with_lr):
    """Mean of O per noise level, averaged over noise draws (the property is about the expectation). The LR is the
    1/4 image the conditioning image is made from; noise is added to the LR."""
    from PIL import Image
    clean = _scene(kind)
    lr_clean = torch.nn.functional.interpolate(clean, scale_factor=0.25, mode="area")
    curve = []
    for s in NOISE:
        acc = 0.0
        for d in range(DRAWS):
            lr = _noisy(lr_clean, s, seed=100 + d)
            cond = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
            lr_img = Image.fromarray((lr[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)) if with_lr else None
            O, _, _, checks = analyse(cond, lr=lr_img)
            assert torch.isfinite(O).all() and bool(checks["O_finite"])
            acc += float(O.mean()) / DRAWS
        curve.append(acc)
    return curve


@pytest.mark.parametrize("kind", ["mixed", "flat", "texture", "edges"])
def test_mean_reliability_does_not_increase_with_noise_original_lr(kind):
    """Main path (the runner passes the original LR): white-noise model. No clipping of O to any reference."""
    c = _mean_O_curve(kind, True)
    assert all(b <= a + 1e-3 for a, b in zip(c, c[1:])), c
    assert c[-1] < c[0] - 0.05, c                                                    # noise is actually seen


@pytest.mark.parametrize("kind", ["mixed", "flat", "edges"])
def test_mean_reliability_does_not_increase_with_noise_fallback(kind):
    """Fallback (no LR): empirical floors measured on flat pixels of the resampled image."""
    c = _mean_O_curve(kind, False)
    assert all(b <= a + 1e-3 for a, b in zip(c, c[1:])), c


def test_fallback_without_flat_regions_is_conservative():
    """Documented limitation: texture everywhere + no LR -> floors contain signal -> O suppressed toward 0."""
    c = _mean_O_curve("texture", False)
    assert max(c) < 0.02, c


def test_clean_beats_noisy_on_noise_indicator_and_sharpness_not_inflated():
    clean = _scene("texture", 128)
    noisy = _noisy(clean, 0.05)
    _, c_clean, _, _ = analyse(clean)
    _, c_noisy, _, _ = analyse(noisy)
    assert float(c_noisy["noise"].mean()) < float(c_clean["noise"].mean())
    assert float(c_noisy["sharpness"].mean()) <= float(c_clean["sharpness"].mean()) + 1e-3
    flat_noisy = _noisy(_scene("flat", 128), 0.05)
    _, c_flat, _, _ = analyse(flat_noisy)
    assert float(c_flat["sharpness"].mean()) < 0.2                                    # noise is not "sharp detail"
    assert float(c_flat["edge_coherence"].mean()) < 0.2


def test_noise_estimator_tracks_true_sigma():
    base = _scene("edges", 128)[:, :1]                         # piecewise constant: no built-in noise
    prev = -1.0
    for s in (0.0, 0.01, 0.03, 0.06):
        est = float(estimate_noise_sigma(_noisy(base, s, seed=5)))
        assert est >= prev - 1e-4 and abs(est - s) < 0.35 * s + 0.006, (s, est)
        prev = est


def test_constant_image_and_determinism():
    O, comps, stats, _ = analyse(torch.full((1, 3, 64, 64), 0.3))
    assert torch.isfinite(O).all() and float(stats["noise_sigma_estimate"]) < 1e-3          # float rounding only
    O2, _, s2, _ = analyse(torch.full((1, 3, 64, 64), 0.3), lr=torch.full((1, 3, 16, 16), 0.3))
    assert torch.isfinite(O2).all() and float(s2["noise_sigma_estimate"]) < 1e-6
    assert all(torch.isfinite(c).all() for c in comps.values())
    img = _noisy(_scene("mixed", 64), 0.02)
    a, b = analyse(img), analyse(img)
    assert torch.equal(a[0], b[0]) and all(torch.equal(a[1][k], b[1][k]) for k in a[1])


def test_cfg_batch_check_is_deferred_and_detects_mismatch():
    img = _scene("mixed", 64)
    _, _, _, ok = analyse(torch.cat([img, img]))
    assert isinstance(ok["cfg_batch_identical"], torch.Tensor) and bool(ok["cfg_batch_identical"])
    _, _, _, bad = analyse(torch.cat([img, img.flip(-1)]))
    assert not bool(bad["cfg_batch_identical"])


def test_indicator_maps_are_saved_by_the_controller():
    from cfa_seesr.observation_consistency import SFROCController
    ctrl = SFROCController(SFROCConfig(enabled=True, strength=0.5), 999, "epsilon")
    ctrl.build_maps(_scene("mixed", 64), (8, 8), torch.full((8, 8), 0.5))
    assert {"edge_coherence", "frequency_energy", "contrast", "noise", "sharpness"} <= set(ctrl.components)
    s = ctrl.summary()
    assert 0 <= s["W_min"] <= s["W_mean"] <= s["W_max"] <= 1 and s["noise_source"] == "conditioning_downsampled"


# --------------------------------------------------------------------------- #
# 4. Two-pass band-limited correction, Fourier leakage, padding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method,kw", [("gaussian", {"sigma": 3.0}), ("fft", {"cutoff": 0.25})])
@pytest.mark.parametrize("hw", [(32, 32), (33, 31), (24, 40)])
def test_two_pass_reduces_leakage_of_spatial_maps(method, kw, hw):
    r = _rand(1, 4, *hw, seed=2)
    W = torch.rand(1, 1, *hw, generator=torch.Generator().manual_seed(4))
    one_pass = W * lowpass(r, method, **kw)
    two_pass = split_lowpass(W * split_lowpass(r, method, **kw), method, **kw)
    thr = 0.5
    assert high_frequency_energy_fraction(two_pass, thr) < 0.6 * high_frequency_energy_fraction(one_pass, thr)
    assert high_frequency_energy_fraction(two_pass, thr) < 0.5 * high_frequency_energy_fraction(r, thr)


@pytest.mark.parametrize("hw", [(32, 32), (33, 47)])
def test_uniform_map_two_pass_equals_documented_single_pass(hw):
    """W = w constant: LPF_h(w LPF_h r) = w LPF(r) (Gaussian: variances add; the discrete kernels agree to the
    sampling error checked here; FFT: same cutoff, idempotent up to the reflect padding at the border)."""
    r = _rand(1, 4, *hw, seed=6)
    two = split_lowpass(0.7 * split_lowpass(r, "gaussian", 3.0, 0.25), "gaussian", 3.0, 0.25)
    one = 0.7 * lowpass(r, "gaussian", 3.0, 0.25)
    assert (two - one).abs().max() < 0.02 * one.abs().max()
    two_f = split_lowpass(0.7 * split_lowpass(r, "fft", 3.0, 0.25, ), "fft", 3.0, 0.25)
    one_f = 0.7 * lowpass(r, "fft", 3.0, 0.25)
    inner = (slice(None), slice(None), slice(6, -6), slice(6, -6))
    assert (two_f - one_f)[inner].abs().mean() < 0.1 * one_f[inner].abs().mean()


def test_padded_fft_has_no_wraparound():
    x = torch.zeros(1, 1, 32, 32)
    x[..., :, :3] = 1.0                                       # bright left border only
    padded = fft_lowpass(x, 0.25, pad=True)
    circular = fft_lowpass(x, 0.25, pad=False)
    assert circular[..., :, -3:].abs().mean() > 0.05         # circular filtering leaks to the right border
    assert padded[..., :, -3:].abs().mean() < 0.2 * circular[..., :, -3:].abs().mean()


def test_gaussian_bandwidth_documentation_value():
    assert abs(gaussian_half_amplitude_period(3.0) - 16.0) < 0.05          # latent px; x8 VAE -> ~128 output px
    k = torch.zeros(1, 1, 256, 1)
    k[..., 128, 0] = 1
    resp = torch.fft.rfft(lowpass(k, "gaussian", 3.0, 0.25)[0, 0, :, 0]).abs()
    f_half = (resp - 0.5).abs().argmin().item() / 256                     # cycles per latent px
    assert abs(1 / f_half - 16.0) < 1.0


# --------------------------------------------------------------------------- #
# 7. Output-space application: exact against the real scheduler step
# --------------------------------------------------------------------------- #
def _check_output_space(sched, pt, t, eta=None):
    xt, mo = _rand(1, 4, 8, 8, seed=1, dtype=torch.float64), _rand(1, 4, 8, 8, seed=2, dtype=torch.float64)
    delta = 0.3 * _rand(1, 4, 8, 8, seed=3, dtype=torch.float64)
    a = float(sched.alphas_cumprod[t])
    kw = {} if eta is None else {"eta": eta}
    step = lambda m: sched.step(m, t, xt, generator=torch.Generator().manual_seed(11), **kw)  # noqa: E731
    ref = step(apply_x0_delta(mo, delta, a, pt)).prev_sample                     # x0' through the scheduler
    c0 = x0_coefficient(sched, t, eta or 0.0)
    out = apply_to_step_output(step(mo), delta, c0, x0=True)
    assert torch.allclose(out.prev_sample, ref, atol=1e-6, rtol=1e-6), (pt, t, eta)
    assert torch.allclose(out.pred_original_sample, predicted_x0(mo, xt, a, pt) + delta, atol=1e-6)
    tup = apply_to_step_output(sched.step(mo, t, xt, generator=torch.Generator().manual_seed(11), return_dict=False,
                                          **kw), delta, c0)
    assert isinstance(tup, tuple) and torch.allclose(tup[0], ref, atol=1e-6)


@pytest.mark.parametrize("pt", ["epsilon", "v_prediction", "sample"])
def test_output_space_equals_x0_shift_ddpm(pt):
    s = _sched("DDPMScheduler", prediction_type=pt)
    s.set_timesteps(20)
    for t in (int(s.timesteps[0]), int(s.timesteps[10]), int(s.timesteps[-1])):     # incl. the last step (prev < 0)
        _check_output_space(s, pt, t)


@pytest.mark.parametrize("pt", ["epsilon", "v_prediction"])
@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_output_space_equals_x0_shift_ddim(pt, eta):
    s = _sched("DDIMScheduler", prediction_type=pt)
    s.set_timesteps(20)
    for t in (int(s.timesteps[0]), int(s.timesteps[7]), int(s.timesteps[-1])):
        _check_output_space(s, pt, t, eta)


def test_correction_delta_formula():
    s = _sched()
    xt, mo, z = _rand(1, 4, 16, 16, seed=1), _rand(1, 4, 16, 16, seed=2), _rand(1, 4, 16, 16, seed=3)
    W = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(5))
    a = float(s.alphas_cumprod[600])
    cfg = SFROCConfig(enabled=True, strength=1.0, band="low", lowpass="gaussian", sigma=3.0)
    delta, x0, r = correction_delta(mo, xt, a, z, W, 0.4, cfg, "epsilon")
    ref = 0.4 * split_lowpass(W * split_lowpass(z - x0, "gaussian", 3.0, 0.25), "gaussian", 3.0, 0.25)
    assert torch.allclose(delta, ref, atol=1e-6) and torch.allclose(r, z - x0)
    full, _, _ = correction_delta(mo, xt, a, z, W, 0.4, SFROCConfig(enabled=True, strength=1.0, band="full"), "epsilon")
    assert torch.allclose(full, 0.4 * W * (z - x0), atol=1e-6)


def test_batch_broadcast_of_reference_and_map():
    s = _sched()
    xt, mo, z = _rand(2, 4, 8, 8, seed=1), _rand(2, 4, 8, 8, seed=2), _rand(1, 4, 8, 8, seed=3)
    delta, _, _ = correction_delta(mo, xt, float(s.alphas_cumprod[500]), z, torch.ones(1, 1, 8, 8), 1.0,
                                   SFROCConfig(enabled=True, strength=1.0, band="full"), "epsilon")
    assert delta.shape == mo.shape


# --------------------------------------------------------------------------- #
# 7b. FP16
# --------------------------------------------------------------------------- #
def test_fp16_output_path_is_within_one_rounding_and_beats_model_output_path():
    s = _sched()
    s.set_timesteps(20)
    t = int(s.timesteps[1])                                                   # high t: worst amplification
    xt, mo = _rand(1, 4, 64, 64, seed=1).half(), _rand(1, 4, 64, 64, seed=2).half()
    delta = 0.03 * _rand(1, 4, 64, 64, seed=3)
    a, c0 = float(s.alphas_cumprod[t]), x0_coefficient(s, t)
    step = lambda m: s.step(m, t, xt, generator=torch.Generator().manual_seed(1))  # noqa: E731
    base = step(mo).prev_sample
    out = apply_to_step_output(step(mo), delta, c0).prev_sample
    legacy = step(apply_x0_delta(mo.float(), delta, a, "epsilon").half()).prev_sample
    assert out.dtype == torch.float16 and legacy.dtype == torch.float16
    target = base.float() + c0 * delta
    ulp = torch.finfo(torch.float16).eps * target.abs().clamp_min(2 ** -14)
    assert ((out.float() - target).abs() <= 0.5 * ulp * 1.01 + 1e-7).all()       # only the final rounding
    ref = c0 * delta
    err_out = ((out.float() - base.float()) - ref).abs().mean()
    err_legacy = ((legacy.float() - base.float()) - ref).abs().mean()
    assert err_out < err_legacy


def test_fp16_carry_preserves_accumulated_displacement():
    """Sub-ulp displacements are lost step by step without the carry and preserved with it (error feedback)."""
    x = torch.full((1, 4, 16, 16), 1.0, dtype=torch.float16)
    small = torch.full((1, 4, 16, 16), 1e-4)                   # below half an fp16 ulp at 1.0 (~4.9e-4)
    plain, carried, carry = x.clone(), x.clone(), None
    for _ in range(20):
        plain = apply_to_step_output((plain,), small, 1.0)[0]
        out, carry = apply_to_step_output((carried,), small, 1.0, carry=carry, return_carry=True)
        carried = out[0]
    assert torch.equal(plain, x)                                                # all lost
    assert abs(float(carried.float().mean()) - (1.0 + 20 * 1e-4)) < 1e-3       # preserved to one rounding


def test_float32_carry_is_none_and_zero_delta_is_exact():
    x = _rand(1, 4, 8, 8)
    out, carry = apply_to_step_output((x,), torch.zeros_like(x), 0.7, return_carry=True)
    assert carry is None and torch.equal(out[0], x)


# --------------------------------------------------------------------------- #
# 8. Scheduler safeguards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("over,match", [({"variance_type": "learned"}, "learned"),
                                        ({"variance_type": "learned_range"}, "learned"),
                                        ({"thresholding": True}, "thresholding"),
                                        ({"clip_sample": True}, "clip_sample")])
def test_unsupported_scheduler_configurations_are_rejected(over, match):
    cls = "DDIMScheduler" if "variance_type" not in over else "DDPMScheduler"
    with pytest.raises(ValueError, match=match):
        check_scheduler(_sched(cls, **over))


def test_supported_configurations_pass():
    assert check_scheduler(_sched()) == "epsilon"
    assert check_scheduler(_sched("DDIMScheduler", prediction_type="v_prediction")) == "v_prediction"


# --------------------------------------------------------------------------- #
# 3. Timing and host synchronisation
# --------------------------------------------------------------------------- #
def test_section_timer_cpu():
    t = SectionTimer()
    t.start(torch.device("cpu"))
    sum(range(10000))
    t.stop()
    assert t.method == "perf_counter" and t.total_ms() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available: event timing unverified here")
def test_section_timer_cuda_events():
    t = SectionTimer()
    x = torch.randn(1024, 1024, device="cuda")
    t.start(x.device)
    for _ in range(10):
        x = x @ x.T / 1024
    t.stop()
    assert t.method == "cuda_events" and t.total_ms() > 0


def _hooked_step(pipe, cfg, n_steps=4, timestep_offset=0):
    """Install SFR-OC on the tiny pipeline with a stub scheduler step; return (controller, step fn, conversions)."""
    from cfa_seesr.hooks import MethodHooks, MethodState
    sched = pipe.scheduler
    sched.set_timesteps(n_steps)
    stub = lambda mo, t, x, *a, **k: (x.clone(),)  # noqa: E731
    sched.step = stub
    hooks = MethodHooks()
    state = MethodState()
    ctrl = hooks.install_observation_consistency(pipe, cfg, state)
    ctrl.W = torch.ones(1, 1, 8, 8)
    ctrl.z_ref = _rand(1, 4, 8, 8, seed=9)
    ctrl.reference_source = "test"
    return hooks, ctrl, sched


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    from experiments.tests.tiny_pipeline import build_pipeline
    return build_pipeline(tmp_path_factory.mktemp("tok_rem"))


def test_step_wrapper_converts_no_latent_tensor_to_host(pipe, monkeypatch):
    """The only tensor -> Python conversions in the corrected step are scheduler constants (alphas_cumprod
    entries, 'one'); latents, model outputs, maps and diagnostics stay tensors (no device sync on CUDA)."""
    cfg = SFROCConfig(enabled=True, strength=0.8, schedule="constant")
    hooks, ctrl, sched = _hooked_step(pipe, cfg)
    try:
        allowed = set(sched.alphas_cumprod.tolist()) | {1.0}
        seen = []
        orig = {n: getattr(torch.Tensor, n) for n in ("__float__", "item", "__bool__", "__int__")}

        def spy(name):
            def f(self, *a, **k):
                v = orig[name](self, *a, **k)
                seen.append((name, float(v) if not isinstance(v, bool) else v, self.numel()))
                return v
            return f

        xt, mo = _rand(1, 4, 8, 8, seed=1), _rand(1, 4, 8, 8, seed=2)
        timesteps = list(sched.timesteps)
        for n in orig:
            monkeypatch.setattr(torch.Tensor, n, spy(n))
        for t in timesteps:
            pipe.scheduler.step(mo, t, xt)
        monkeypatch.undo()
        bad = [s for s in seen if not (s[0] in ("__float__", "item") and s[2] == 1 and
                                       any(abs(s[1] - v) <= 1e-12 * max(1, abs(v)) for v in allowed))]
        assert not bad, bad
        assert len(ctrl.step_log) == len(timesteps) and all(torch.is_tensor(d["delta_abs_mean"]) for d in ctrl.step_log)
        summ = ctrl.summary()                                               # one transfer, after the call
        assert summ["steps_corrected"] == len(timesteps)
    finally:
        hooks.remove()
        del pipe.scheduler.step


def test_deferred_timestep_mismatch_is_reported(pipe):
    cfg = SFROCConfig(enabled=True, strength=0.8, schedule="constant")
    hooks, ctrl, sched = _hooked_step(pipe, cfg)
    try:
        xt, mo = _rand(1, 4, 8, 8, seed=1), _rand(1, 4, 8, 8, seed=2)
        ts = list(sched.timesteps)
        pipe.scheduler.step(mo, ts[0], xt)
        pipe.scheduler.step(mo, ts[0], xt)                                 # wrong: should be ts[1]
        with pytest.raises(RuntimeError, match="timesteps_match"):
            ctrl.summary()
    finally:
        hooks.remove()
        del pipe.scheduler.step


def test_deferred_cfg_check_fails_summary():
    from cfa_seesr.observation_consistency import SFROCController
    ctrl = SFROCController(SFROCConfig(enabled=True, strength=0.5), 999, "epsilon")
    img = _scene("mixed", 64)
    ctrl.build_maps(torch.cat([img, img.flip(-1)]), (8, 8), torch.full((8, 8), 0.5))
    with pytest.raises(RuntimeError, match="cfg_batch_identical"):
        ctrl.summary()


# --------------------------------------------------------------------------- #
# Through the pipeline: precision modes, mean-matched, metadata
# --------------------------------------------------------------------------- #
def _run(pipe, img, steps=3):
    from experiments.tests.tiny_pipeline import call_kwargs
    ram = torch.randn(1, 5, 512, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(7)
    return pipe("a, clean", img, **call_kwargs(img, steps, torch.Generator().manual_seed(7), ram)).images[0]


def _img(seed, size=32):
    from PIL import Image
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8))


def _with(pipe, cfg, img):
    from cfa_seesr.hooks import MethodHooks, MethodState
    hooks, state = MethodHooks(), MethodState()
    hooks.install_observation_consistency(pipe, cfg, state)
    try:
        return _run(pipe, img), state.sfroc.summary(), state.sfroc
    finally:
        hooks.remove()


@pytest.mark.parametrize("sched_name", ["DDPMScheduler", "DDIMScheduler"])
def test_output_and_model_output_precision_agree_in_fp32(pipe, sched_name):
    import diffusers
    original = pipe.scheduler
    try:
        pipe.scheduler = getattr(diffusers, sched_name).from_config(original.config)
        img = _img(3)
        base = dict(enabled=True, strength=0.7, schedule="constant")
        a, sa, ca = _with(pipe, SFROCConfig(**base, precision="output"), img)
        b, sb, _ = _with(pipe, SFROCConfig(**base, precision="model_output"), img)
        c, sc, _ = _with(pipe, SFROCConfig(**base, precision="output_carry"), img)
        assert sa["steps_corrected"] == sb["steps_corrected"] == 3
        assert np.abs(a.astype(int) - b.astype(int)).max() <= 2                 # same maths, different rounding
        assert np.array_equal(a, c)                                             # fp32 latents: no carry
        corrected = [r for r in ca.step_records if r["gamma"] > 0]
        assert sa["precision"] == "output" and len(corrected) == 3 and all(r["c0"] > 0 for r in corrected)
    finally:
        pipe.scheduler = original


def test_mean_matched_map_through_pipeline(pipe):
    from PIL import Image
    img = Image.fromarray((_scene("mixed", 64)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    _, s, ctrl = _with(pipe, SFROCConfig(enabled=True, strength=0.5, fusion=FusionConfig(map="mean_matched")), img)
    assert ctrl.W.unique().numel() == 1 and 0 < s["W_mean"] < 1
    _, s2, _ = _with(pipe, SFROCConfig(enabled=True, strength=0.5), img)
    assert abs(s2["W_mean"] - s["W_mean"]) < 1e-6 and s2["W_std"] > 1e-3


def test_method_signature_and_version():
    import cfa_seesr
    from cfa_seesr import MethodConfig
    assert cfa_seesr.METHOD_VERSION.startswith("2.")
    assert cfa_seesr.method_signature(MethodConfig()) == "upstream"
    v1 = MethodConfig.from_dict({"semantic_confidence": {"enabled": True}})
    assert cfa_seesr.method_signature(v1) == "cfa-seesr-v1"
    v2 = MethodConfig.from_dict({"sfr_oc": {"enabled": True, "strength": 0.0}})
    assert cfa_seesr.method_signature(v2) == "cfa-seesr-v2"
