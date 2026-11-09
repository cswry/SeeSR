"""Tests for the Confidence- and Fidelity-Aware extension (cfa_seesr).

* Formulas: confidence, tag selection, reliability, soft-prompt scale, LR schedule.
* Upstream equivalence of the DAPE path: upstream's own RAMLora.generate_tag / condition_forward /
  generate_image_embeds and test_seesr.get_validation_prompt, bound to small random modules (the real
  RAMLora needs bert-base-uncased and 5.6 GB of weights).
* Hooks on a tiny instance of the real upstream pipeline: disabled = identical outputs, s = 0 equals
  removing the image-attention blocks, a constant LR schedule equals upstream conditioning_scale, the
  schedule uses actual timesteps, and hooks are removed cleanly.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from cfa_seesr import LRGuidanceConfig, MethodConfig, SemanticConfidenceConfig, mean_factor, schedule_factor  # noqa: E402
from cfa_seesr.semantic_confidence import (aggregate_reliability, select_tags, soft_prompt_scale,  # noqa: E402
                                           tag_confidence)

TAGS = np.array([f"tag{i}" for i in range(8)])
THRESH = torch.tensor([0.5, 0.65, 0.8, 0.9, 0.65, 0.95, 0.7, 0.6])


def logits_for(probs):
    p = torch.tensor(probs, dtype=torch.float32)
    return torch.log(p) - torch.log1p(-p)


# --------------------------------------------------------------------------- #
# Component A: formulas
# --------------------------------------------------------------------------- #
def test_confidence_is_threshold_centred_and_monotone():
    z = torch.linspace(-6, 6, 101)
    tau = torch.full_like(z, 0.8)
    c, margin = tag_confidence(z, tau, 1.0)
    assert torch.all(c[1:] >= c[:-1])
    p = torch.sigmoid(z)
    assert torch.equal(c > 0.5, p > tau)                      # c_i > 1/2  <=>  p_i > tau_i
    at_threshold, _ = tag_confidence(logits_for([0.8]), torch.tensor([0.8]), 1.0)
    assert at_threshold.item() == pytest.approx(0.5, abs=1e-5)
    # T = 1: c = odds_p / (odds_p + odds_tau)
    p0, t0 = 0.9, 0.8
    c1, _ = tag_confidence(logits_for([p0]), torch.tensor([t0]), 1.0)
    assert c1.item() == pytest.approx((p0 / (1 - p0)) / (p0 / (1 - p0) + t0 / (1 - t0)), rel=1e-5)
    # larger T flattens towards 1/2
    c2, _ = tag_confidence(logits_for([p0]), torch.tensor([t0]), 4.0)
    assert 0.5 < c2.item() < c1.item()


def test_default_selection_equals_upstream_rule():
    probs = [0.7, 0.6, 0.85, 0.91, 0.99, 0.5, 0.71, 0.2]
    r = select_tags(logits_for(probs), THRESH, TAGS, [], SemanticConfidenceConfig(enabled=True))
    upstream = [i for i, (p, t) in enumerate(zip(probs, THRESH.tolist())) if p > t]
    assert r.prompt_tags == ", ".join(TAGS[upstream]) == r.upstream_tags
    assert r.n_selected == r.n_upstream == len(upstream)


def test_stricter_selection_is_a_log_odds_subset():
    probs = [0.7, 0.66, 0.85, 0.91, 0.99, 0.96, 0.71, 0.61]
    cfg = SemanticConfidenceConfig(enabled=True, min_confidence=1 / (1 + np.exp(-1.0)))   # margin > 1 nat
    r = select_tags(logits_for(probs), THRESH, TAGS, [], cfg)
    z = logits_for(probs)
    margin = z - torch.log(THRESH / (1 - THRESH))
    expected = [i for i in range(8) if margin[i] > 1.0]
    assert r.prompt_tags == ", ".join(TAGS[expected])
    assert set(r.prompt_tags.split(", ")) <= set(r.upstream_tags.split(", "))
    assert r.n_selected < r.n_upstream


def test_delete_index_is_respected():
    probs = [0.99] * 8
    r = select_tags(logits_for(probs), THRESH, TAGS, [2, 5], SemanticConfidenceConfig(enabled=True))
    assert "tag2" not in r.prompt_tags and "tag5" not in r.prompt_tags and r.n_upstream == 6


def test_empty_and_uncertain_predictions_fall_back():
    cfg = SemanticConfidenceConfig(enabled=True, min_confidence=0.9, soft_scale_min=0.5, soft_scale_max=1.0)
    none = select_tags(logits_for([0.1] * 8), THRESH, TAGS, [], cfg)
    assert none.prompt_tags == "" and none.reliability == 0.0 and none.soft_scale == 0.5
    # tags pass upstream thresholds, but none is confident enough: empty hard prompt, low but non-zero reliability
    barely = select_tags(logits_for((THRESH + 0.01).tolist()), THRESH, TAGS, [], cfg)
    assert barely.prompt_tags == "" and barely.n_upstream == 8 and 0.0 < barely.reliability < 0.2
    assert 0.5 < barely.soft_scale < 0.6


def test_reliability_and_scale_bounds():
    rng = np.random.default_rng(0)
    cfg = SemanticConfidenceConfig(enabled=True, soft_scale_min=0.3, soft_scale_max=0.9)
    for _ in range(200):
        r = select_tags(torch.tensor(rng.normal(0, 4, 8), dtype=torch.float32), THRESH, TAGS, [], cfg)
        assert 0.0 <= r.reliability <= 1.0
        assert 0.3 <= r.soft_scale <= 0.9
    assert aggregate_reliability([]) == 0.0
    assert aggregate_reliability([0.5, 0.5]) == 0.0 and aggregate_reliability([1.0]) == 1.0
    assert aggregate_reliability([0.6, 1.0], "min") <= aggregate_reliability([0.6, 1.0], "mean")
    assert soft_prompt_scale(2.0, cfg) == 0.9 and soft_prompt_scale(-1.0, cfg) == 0.3


@pytest.mark.parametrize("kwargs", [dict(temperature=0), dict(min_confidence=1.0), dict(min_confidence=0.0),
                                    dict(soft_scale_min=0.8, soft_scale_max=0.5), dict(aggregate="max")])
def test_semantic_config_validation(kwargs):
    with pytest.raises(ValueError):
        SemanticConfidenceConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Component B: schedule
# --------------------------------------------------------------------------- #
def test_schedule_endpoints_shape_and_identity():
    cfg = LRGuidanceConfig(enabled=True, schedule="power", scale_start=1.25, scale_end=0.75, gamma=1.0)
    assert schedule_factor(999, cfg, 999) == pytest.approx(1.25)
    assert schedule_factor(0, cfg, 999) == pytest.approx(0.75)
    assert schedule_factor(torch.tensor(500), cfg, 999) == pytest.approx(0.75 + 0.5 * 500 / 999)
    ts = list(range(999, -1, -37))
    vals = [schedule_factor(t, cfg, 999) for t in ts]
    assert all(a >= b for a, b in zip(vals, vals[1:]))                      # early strong, late weaker
    convex = LRGuidanceConfig(enabled=True, schedule="power", scale_start=1.25, scale_end=0.75, gamma=2.0)
    assert schedule_factor(500, convex, 999) < schedule_factor(500, cfg, 999)
    for identity in (LRGuidanceConfig(), LRGuidanceConfig(enabled=True),
                     LRGuidanceConfig(enabled=False, schedule="power", scale_start=2.0),
                     LRGuidanceConfig(enabled=True, schedule="power", scale_start=1.0, scale_end=1.0)):
        assert identity.is_identity() and schedule_factor(700, identity, 999) == 1.0


def test_schedule_uses_actual_timesteps_and_is_near_mean_preserving():
    from diffusers import DDPMScheduler
    from experiments.tests.tiny_pipeline import SD2_BASE_SCHEDULER
    cfg = LRGuidanceConfig(enabled=True, schedule="power", scale_start=1.25, scale_end=0.75)
    sched = DDPMScheduler.from_config(SD2_BASE_SCHEDULER)
    factors = {}
    for n in (50, 30, 20):
        sched.set_timesteps(n)
        ts = [int(t) for t in sched.timesteps]
        factors[n] = dict(zip(ts, (schedule_factor(t, cfg, 999) for t in ts)))
        assert abs(mean_factor(ts, cfg, 999) - 1.0) < 0.02
    # the same timestep gets the same factor regardless of the step count
    common = set(factors[50]) & set(factors[20])
    assert common and all(factors[50][t] == factors[20][t] for t in common)


@pytest.mark.parametrize("kwargs", [dict(schedule="cosine"), dict(gamma=0), dict(scale_start=-1)])
def test_lr_config_validation(kwargs):
    with pytest.raises(ValueError):
        LRGuidanceConfig(**kwargs)


def test_method_config_switches():
    assert MethodConfig.from_dict(None).is_baseline
    a = MethodConfig.from_dict({"semantic_confidence": {"enabled": True}})
    b = MethodConfig.from_dict({"lr_guidance": {"enabled": True, "schedule": "power", "scale_start": 1.25, "scale_end": 0.75}})
    ab = MethodConfig.from_dict({"semantic_confidence": {"enabled": True},
                                 "lr_guidance": {"enabled": True, "schedule": "power", "scale_start": 1.2}})
    assert (a.semantic_active, a.lr_active, a.is_baseline) == (True, False, False)
    assert (b.semantic_active, b.lr_active, b.is_baseline) == (False, True, False)
    assert (ab.semantic_active, ab.lr_active) == (True, True)
    # an enabled but identity schedule is not "active": no hook is installed
    assert MethodConfig.from_dict({"lr_guidance": {"enabled": True}}).is_baseline
    with pytest.raises(ValueError):
        MethodConfig.from_dict({"typo_section": {}})


# --------------------------------------------------------------------------- #
# Upstream DAPE code bound to small random modules
# --------------------------------------------------------------------------- #
class _TaggingHead(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, encoder_embeds, encoder_hidden_states, encoder_attention_mask, return_dict, mode):
        ctx = encoder_hidden_states.mean(dim=1, keepdim=True)
        return (torch.tanh(self.lin(encoder_embeds + ctx)),)


class _PatchEncoder(torch.nn.Module):
    """[B, 3, H, W] -> [B, tokens, 24], standing in for the Swin-L visual encoder."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 8, 8)
        self.proj = torch.nn.Linear(8, 24)

    def forward(self, x):
        return self.proj(self.conv(x).flatten(2).transpose(1, 2))


def tiny_dape(n_class=40, dim=16, seed=0):
    """Object exposing the attributes RAMLora's methods use, with RAMLora's own methods attached."""
    pytest.importorskip("loralib")
    from ram.models.ram_lora import RAMLora
    torch.manual_seed(seed)

    class TinyDAPE(torch.nn.Module):
        generate_tag = RAMLora.generate_tag
        condition_forward = RAMLora.condition_forward
        generate_image_embeds = RAMLora.generate_image_embeds

        def __init__(self):
            super().__init__()
            self.visual_encoder = _PatchEncoder()
            self.image_proj = torch.nn.Linear(24, dim)
            self.wordvec_proj = torch.nn.Identity()
            self.label_embed = torch.nn.Parameter(torch.randn(n_class, dim))
            self.tagging_head = _TaggingHead(dim)
            self.fc = torch.nn.Linear(dim, 1)
            torch.nn.init.normal_(self.fc.weight, std=3.0)
            self.num_class = n_class
            self.class_threshold = torch.rand(n_class) * 0.4 + 0.5
            self.tag_list = np.array([f"t{i}" for i in range(n_class)])
            self.tag_list_chinese = self.tag_list
            self.delete_tag_index = [3]

    return TinyDAPE().eval()


@pytest.fixture(scope="module")
def upstream_test_seesr():
    sys.dont_write_bytecode = True
    import test_seesr
    return test_seesr


def _image(seed):
    from PIL import Image
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8))


def test_extension_prompt_matches_upstream_when_selection_is_default(upstream_test_seesr):
    import cfa_seesr
    args = types.SimpleNamespace(prompt="")
    found_tags = False
    for seed in range(6):
        model = tiny_dape(seed=seed)
        img = _image(seed)
        up_prompt, up_embeds = upstream_test_seesr.get_validation_prompt(args, img, model, device="cpu")
        prompt, embeds, result = cfa_seesr.semantic_validation_prompt(
            upstream_test_seesr, args, img, model, SemanticConfidenceConfig(enabled=True))
        assert prompt == up_prompt
        assert torch.equal(embeds, up_embeds)
        assert result.upstream_tags == upstream_test_seesr.inference(
            upstream_test_seesr.ram_transforms(upstream_test_seesr.tensor_transforms(img).unsqueeze(0)), model)[0]
        found_tags |= result.n_upstream > 0
    assert found_tags   # the comparison is not vacuous


# --------------------------------------------------------------------------- #
# Hooks on the tiny real pipeline
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    pytest.importorskip("diffusers")
    from experiments.tests.tiny_pipeline import build_pipeline
    return build_pipeline(tmp_path_factory.mktemp("tok_method"))


@pytest.fixture(scope="module")
def inputs():
    img = _image(11).resize((32, 32))
    ram = torch.randn(1, 5, 512, generator=torch.Generator().manual_seed(1))
    return img, ram


def _run(pipe, inputs, steps=4, seed=42, conditioning_scale=1.0):
    from experiments.tests.tiny_pipeline import call_kwargs
    img, ram = inputs
    torch.manual_seed(seed)
    kw = call_kwargs(img, steps, torch.Generator().manual_seed(seed), ram)
    kw["conditioning_scale"] = conditioning_scale
    return pipe("a, clean", img, **kw).images[0]


def test_soft_scale_one_is_exact_and_zero_bypasses_image_attention(pipe, inputs):
    from cfa_seesr.hooks import MethodHooks, MethodState, image_attention_blocks
    ref = _run(pipe, inputs)
    state, hooks = MethodState(), MethodHooks()
    n = hooks.install_soft_prompt_scaling(pipe, state)
    try:
        assert n == len(image_attention_blocks(pipe.unet)) + len(image_attention_blocks(pipe.controlnet)) > 0
        state.soft_scale = 1.0
        assert np.array_equal(_run(pipe, inputs), ref)
        state.soft_scale = 0.5
        half = _run(pipe, inputs)
        assert not np.array_equal(half, ref)
        state.soft_scale = 0.0
        zero = _run(pipe, inputs)
    finally:
        hooks.remove()
    assert np.array_equal(_run(pipe, inputs), ref)   # removed cleanly

    # s = 0 must equal a pipeline whose image-attention blocks are skipped entirely
    class Skip(torch.nn.Module):
        def forward(self, hidden_states, *a, **k):
            return (hidden_states,)

    saved = []
    for model in (pipe.unet, pipe.controlnet):
        for name, mod in model.named_modules():
            if name.endswith("image_attentions"):
                saved.append((mod, list(mod)))
                for i in range(len(mod)):
                    mod[i] = Skip()
    try:
        skipped = _run(pipe, inputs)
    finally:
        for mod, original in saved:
            for i, m in enumerate(original):
                mod[i] = m
    assert np.allclose(zero, skipped, atol=1e-6)
    assert np.array_equal(_run(pipe, inputs), ref)


def test_constant_lr_factor_equals_upstream_conditioning_scale(pipe, inputs):
    from cfa_seesr.hooks import MethodHooks, MethodState
    cfg = LRGuidanceConfig(enabled=True, schedule="power", scale_start=0.6, scale_end=0.6)
    expected = _run(pipe, inputs, conditioning_scale=0.6)
    state, hooks = MethodState(), MethodHooks()
    hooks.install_lr_schedule(pipe, cfg, state)
    try:
        got = _run(pipe, inputs)
    finally:
        hooks.remove()
    assert np.array_equal(got, expected)
    assert "forward" not in vars(pipe.controlnet)


def test_schedule_is_applied_per_actual_timestep(pipe, inputs):
    from cfa_seesr.hooks import MethodHooks, MethodState
    cfg = LRGuidanceConfig(enabled=True, schedule="power", scale_start=1.25, scale_end=0.75)
    ref = _run(pipe, inputs, steps=5)
    state, hooks = MethodState(), MethodHooks()
    hooks.install_lr_schedule(pipe, cfg, state)
    try:
        out = _run(pipe, inputs, steps=5)
    finally:
        hooks.remove()
    ts = [float(t) for t in pipe.scheduler.timesteps]
    assert [t for t, _ in state.lr_log] == ts                    # one ControlNet call per step (CFG batched)
    assert [f for _, f in state.lr_log] == [schedule_factor(t, cfg, 999) for t in ts]
    assert not np.array_equal(out, ref)
    assert np.array_equal(_run(pipe, inputs, steps=5), ref)


def test_disabled_method_installs_nothing_and_preserves_outputs(pipe, inputs):
    """Original-baseline regression through the experiment runner's hooks."""
    from experiments import run_inference as ri
    ref = _run(pipe, inputs)
    unet_hooks = sum(len(m._forward_hooks) for m in pipe.unet.modules())
    for method in (MethodConfig(),
                   MethodConfig.from_dict({"semantic_confidence": {"enabled": False, "soft_scale_min": 0.2},
                                           "lr_guidance": {"enabled": False, "schedule": "power", "scale_start": 2.0}}),
                   MethodConfig.from_dict({"lr_guidance": {"enabled": True, "schedule": "constant"}})):
        rec = ri.Recorder(["x"], method=method)
        ri.install_method(pipe, rec)
        assert rec.method_hooks is None
        assert sum(len(m._forward_hooks) for m in pipe.unet.modules()) == unet_hooks
        assert "forward" not in vars(pipe.controlnet)
        assert np.array_equal(_run(pipe, inputs), ref)


def test_runner_hooks_with_both_components(pipe, inputs, upstream_test_seesr):
    """install_hooks + get_prompt + TimedPipeline with the combined method, then clean removal."""
    from experiments import run_inference as ri
    from experiments.tests.tiny_pipeline import call_kwargs
    method = MethodConfig.from_dict({
        "semantic_confidence": {"enabled": True, "min_confidence": 0.7310585786300049,
                                "soft_scale_min": 0.5, "soft_scale_max": 1.0},
        "lr_guidance": {"enabled": True, "schedule": "power", "scale_start": 1.25, "scale_end": 0.75}})
    img, ram = inputs
    ref = _run(pipe, inputs)
    mod = types.ModuleType("fake_test_seesr")
    mod.load_seesr_pipeline = lambda *a, **k: pipe
    mod.load_tag_model = lambda *a, **k: None
    mod.get_validation_prompt = lambda *a, **k: ("unused", None)
    mod.tensor_transforms = upstream_test_seesr.tensor_transforms
    mod.ram_transforms = upstream_test_seesr.ram_transforms
    rec = ri.Recorder(["img"], method=method)
    original_step = pipe.scheduler.step
    ri.install_hooks(mod, rec, {})
    try:
        timed = mod.load_seesr_pipeline()
        dape = tiny_dape(seed=2)
        prompt, embeds = mod.get_validation_prompt(types.SimpleNamespace(prompt=""), img, dape)
        assert prompt.endswith(", ,")
        row = rec.rows[0]
        assert 0.5 <= row["soft_scale"] <= 1.0 and row["n_tags_selected"] <= row["n_tags_upstream"]
        assert rec.method_state.soft_scale == row["soft_scale"]
        torch.manual_seed(42)
        timed("a, clean", img, **call_kwargs(img, 4, torch.Generator().manual_seed(42), ram))
        assert row["controlnet_calls"] == 4 and row["lr_factor_first"] > row["lr_factor_last"]
        assert rec.semantic_records[0]["image"] == "img" and "tag_scores" in rec.semantic_records[0]
    finally:
        ri.remove_hooks(mod, rec)
        pipe.scheduler.step = original_step
    assert "forward" not in vars(pipe.controlnet)
    assert np.array_equal(_run(pipe, inputs), ref)
