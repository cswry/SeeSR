"""Spatial semantic reliability (cfa_seesr/spatial_reliability.py), Gate 0 and screening tools (CPU only).

DAPE is represented by a small module that carries the REAL upstream tagging head (ram/models/bert_lora.py
BertModel in mode='tagging', self-attention deleted as RAMLora.del_selfattention does) and RAMLora's own
condition_forward / generate_tag, with a visual encoder that, like Swin-L, returns [pooled token, g*g tokens]
for a 384x384 input. No weights are downloaded.
"""
import json
import math
import types

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from cfa_seesr import MethodConfig, SemanticConfidenceConfig, SpatialSemanticConfig
from cfa_seesr.spatial_reliability import (SpatialMap, build_spatial, capture_dape, gaussian_smooth, grid_side,
                                           reliability_grid, resolve_mean_matched, tag_attention_maps, tile_regions,
                                           upstream_tag_indices)


# --------------------------------------------------------------------------- #
# Tiny DAPE with the real tagging head
# --------------------------------------------------------------------------- #
class _SwinLike(torch.nn.Module):
    """[B, 3, 384, 384] -> [B, 1 + 12*12, 24]: pooled token first, then the 12x12 grid (as Swin-L)."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 24, 32, 32)

    def forward(self, x):
        t = self.conv(x).flatten(2).transpose(1, 2)
        return torch.cat([t.mean(dim=1, keepdim=True), t], dim=1)


def tiny_dape_bert(n_class=30, dim=16, seed=0):
    pytest.importorskip("loralib")
    from ram.models.bert_lora import BertConfig, BertModel
    from ram.models.ram_lora import RAMLora
    torch.manual_seed(seed)

    class TinyDAPE(torch.nn.Module):
        generate_tag = RAMLora.generate_tag
        condition_forward = RAMLora.condition_forward
        generate_image_embeds = RAMLora.generate_image_embeds

        def __init__(self):
            super().__init__()
            self.visual_encoder = _SwinLike()
            self.image_proj = torch.nn.Linear(24, dim)
            self.wordvec_proj = torch.nn.Identity()
            self.label_embed = torch.nn.Parameter(torch.randn(n_class, dim))
            cfg = BertConfig(hidden_size=dim, num_hidden_layers=2, num_attention_heads=4, intermediate_size=32,
                             vocab_size=10, encoder_width=dim, add_cross_attention=True,
                             max_position_embeddings=16)
            self.tagging_head = BertModel(cfg, add_pooling_layer=False)
            for layer in self.tagging_head.encoder.layer:     # RAMLora.del_selfattention
                del layer.attention
            self.fc = torch.nn.Linear(dim, 1)
            torch.nn.init.normal_(self.fc.weight, std=4.0)
            self.num_class = n_class
            self.class_threshold = torch.rand(n_class) * 0.4 + 0.5
            self.tag_list = np.array([f"t{i}" for i in range(n_class)])
            self.tag_list_chinese = self.tag_list
            self.delete_tag_index = [2]

    return TinyDAPE().eval()


@pytest.fixture(scope="module")
def upstream_test_seesr():
    import sys
    sys.dont_write_bytecode = True
    import test_seesr
    return test_seesr


def _image(seed, size=64):
    from PIL import Image
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8))


def _lq(upstream, img):
    return upstream.ram_transforms(upstream.tensor_transforms(img).unsqueeze(0))


def _model_with_tags(upstream, min_tags=2):
    for seed in range(40):
        m = tiny_dape_bert(seed=seed)
        with torch.no_grad():
            _, logits, _ = m.condition_forward(_lq(upstream, _image(seed)), only_feature=False)
        if len(upstream_tag_indices(logits[0], m.class_threshold, m.delete_tag_index)) >= min_tags:
            return m, _image(seed)
    raise AssertionError("no tiny model with tags")


# --------------------------------------------------------------------------- #
# Attention capture
# --------------------------------------------------------------------------- #
def test_capture_shape_order_and_removal(upstream_test_seesr):
    m, img = _model_with_tags(upstream_test_seesr)
    x = _lq(upstream_test_seesr, img)
    with torch.no_grad(), capture_dape(m) as cap:
        embeds, logits, _ = m.condition_forward(x, only_feature=False)
    att = cap["attention"]
    assert att.shape == (1, 4, m.num_class, embeds.shape[1]) == (1, 4, 30, 145)
    assert torch.allclose(att.sum(-1), torch.ones(1, 4, 30))          # softmax over the image tokens
    assert torch.equal(cap["logits"], logits)
    assert grid_side(att.shape[-1]) == 12
    # hooks are gone
    dropout = m.tagging_head.encoder.layer[-1].crossattention.self.dropout
    assert not dropout._forward_hooks and not m.fc._forward_hooks
    # the captured tensor is the last layer's cross-attention: recompute it from its query/key projections
    layer = m.tagging_head.encoder.layer[-1].crossattention.self
    store = {}
    def keep_input(mod, inp, out):
        store["hidden"] = inp[0]          # returns None: the output is not replaced

    h = m.tagging_head.encoder.layer[-1].crossattention.register_forward_hook(keep_input)
    with torch.no_grad():
        m.condition_forward(x, only_feature=False)
    h.remove()
    q = layer.transpose_for_scores(layer.query(store["hidden"]))
    k = layer.transpose_for_scores(layer.key(m.image_proj(m.visual_encoder(x))))
    probs = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(layer.attention_head_size), dim=-1)
    assert torch.allclose(probs, att, atol=1e-6)


def test_capture_is_removed_on_error(upstream_test_seesr):
    m = tiny_dape_bert()
    with pytest.raises(RuntimeError):
        with capture_dape(m):
            raise RuntimeError("boom")
    assert not m.fc._forward_hooks
    assert not m.tagging_head.encoder.layer[-1].crossattention.self.dropout._forward_hooks


def test_dape_outputs_identical_with_and_without_capture(upstream_test_seesr):
    import cfa_seesr
    args = types.SimpleNamespace(prompt="")
    for seed in range(4):
        m, img = tiny_dape_bert(seed=seed), _image(seed)
        up_prompt, up_embeds = upstream_test_seesr.get_validation_prompt(args, img, m, device="cpu")
        for sem in (SemanticConfidenceConfig(), SemanticConfidenceConfig(enabled=True, min_confidence=0.73,
                                                                         soft_scale_min=0.5)):
            p0, e0, r0 = cfa_seesr.semantic_validation_prompt(upstream_test_seesr, args, img, m, sem)
            p1, e1, r1, sp = cfa_seesr.spatial_validation_prompt(
                upstream_test_seesr, args, img, m, sem, SpatialSemanticConfig(enabled=True, contrast=1.0), "k")
            assert (p0, r0) == (p1, r1) and torch.equal(e0, e1)
            assert sp.stats["n_tags"] == r1.n_upstream
        p, e, _, _ = cfa_seesr.spatial_validation_prompt(
            upstream_test_seesr, args, img, m, SemanticConfidenceConfig(), SpatialSemanticConfig(enabled=True), "k")
        assert p == up_prompt and torch.equal(e, up_embeds)   # A off: upstream prompt and embeddings


def test_upstream_indices_and_token_exclusion():
    logits = torch.tensor([3.0, -3.0, 3.0, 0.1, 2.0])
    thr = torch.full((5,), 0.6)
    assert upstream_tag_indices(logits, thr, [2]) == [0, 4]
    att = torch.zeros(2, 5, 1 + 4)
    att[:, :, 0] = 0.9                          # pooled token gets most of the mass
    att[:, :, 1] = 0.1                          # then grid cell 0
    maps = tag_attention_maps(att, [0, 4])
    assert maps.shape == (2, 2, 2)
    assert torch.allclose(maps.sum(dim=(1, 2)), torch.ones(2, dtype=torch.float64))
    assert maps[0, 0, 0] == 1.0                 # pooled token excluded, renormalised over the grid
    with pytest.raises(ValueError):
        grid_side(1 + 5)


# --------------------------------------------------------------------------- #
# Reliability grid
# --------------------------------------------------------------------------- #
def _maps(n, g, seed=0):
    t = torch.rand(n, g, g, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) ** 4
    return t / t.sum(dim=(1, 2), keepdim=True)


def test_reliability_and_coverage_by_hand():
    g = 2
    maps = torch.zeros(2, g, g, dtype=torch.float64)
    maps[0, 0, 0] = 1.0                      # tag 0 entirely on cell (0,0)
    maps[1] = 0.25                           # tag 1 uniform
    cfg = SpatialSemanticConfig(enabled=True, contrast=0.0, smoothing_sigma=0.0)
    w = [1.0, 0.2]
    R, st = reliability_grid(maps, w, 0.4, 0.4, cfg)
    mass = maps.sum(0)
    rho = (maps[0] * 1.0 + maps[1] * 0.2) / (mass + cfg.eps)
    kappa = (mass * 4 / 2).clamp(max=1)
    assert torch.allclose(kappa, torch.tensor([[1.0, 0.5], [0.5, 0.5]], dtype=torch.float64))
    assert abs(st["coverage_mean"] - float(kappa.mean())) < 1e-12
    r = kappa * rho + (1 - kappa) * 0.4
    cfg1 = SpatialSemanticConfig(enabled=True, contrast=1.0, smoothing_sigma=0.0)
    R1, st1 = reliability_grid(maps, w, 0.4, 0.4, cfg1)
    assert torch.allclose(R1, (0.4 + (r - r.mean())).clamp(0, 1))
    assert torch.allclose(R, torch.full((g, g), 0.4, dtype=torch.float64))   # lambda = 0: R = L


def test_bounds_mean_and_clipping_are_recorded():
    maps = _maps(5, 12)
    w = [0.0, 0.1, 0.5, 0.9, 1.0]
    for lam in (0.5, 1.0, 5.0, 50.0):
        R, st = reliability_grid(maps, w, 0.6, 0.6, SpatialSemanticConfig(enabled=True, contrast=lam))
        assert R.isfinite().all() and 0.0 <= R.min() and R.max() <= 1.0
        assert abs(st["mean_unclipped"] - 0.6) < 1e-12       # centring keeps the pre-clip mean at L
        assert st["clip_low_frac"] + st["clip_high_frac"] <= 1.0
        if st["clip_low_frac"] == st["clip_high_frac"] == 0:
            assert abs(st["mean"] - 0.6) < 1e-12
    assert st["clip_low_frac"] > 0 or st["clip_high_frac"] > 0   # lambda = 50 clips


def test_empty_tags_and_numerical_stability():
    R, st = reliability_grid(torch.zeros(0, 12, 12, dtype=torch.float64), [], 0.0, 0.0,
                             SpatialSemanticConfig(enabled=True, contrast=1.0))
    assert st["fallback"] == "no_tags" and torch.equal(R, torch.zeros(12, 12, dtype=torch.float64))
    tiny = torch.full((3, 12, 12), 1e-30, dtype=torch.float64)
    tiny[:, 0, 0] = 1.0
    R, st = reliability_grid(tiny, [1.0, 1.0, 1.0], 1.0, 1.0, SpatialSemanticConfig(enabled=True, contrast=2.0))
    assert R.isfinite().all() and 0 <= R.min() <= R.max() <= 1


def test_smoothing_is_a_convex_combination():
    grid = torch.rand(12, 12, dtype=torch.float64)
    for sigma in (0.0, 0.5, 1.0, 3.0, 6.0):
        s = gaussian_smooth(grid, sigma)
        assert s.shape == grid.shape and s.isfinite().all()
        assert grid.min() - 1e-12 <= s.min() and s.max() <= grid.max() + 1e-12
    assert torch.equal(gaussian_smooth(grid, 0.0), grid)
    const = torch.full((12, 12), 0.3, dtype=torch.float64)
    assert torch.allclose(gaussian_smooth(const, 2.0), const)


def test_shuffled_control_keeps_values_moves_cells():
    maps, w = _maps(4, 12, seed=3), [0.1, 0.4, 0.8, 1.0]
    base, _ = reliability_grid(maps, w, 0.5, 0.5, SpatialSemanticConfig(enabled=True, contrast=1.0))
    sh, _ = reliability_grid(maps, w, 0.5, 0.5, SpatialSemanticConfig(enabled=True, contrast=1.0,
                                                                      control="shuffled"), key="img1")
    assert torch.equal(base.flatten().sort().values, sh.flatten().sort().values)
    assert not torch.equal(base, sh)
    again, _ = reliability_grid(maps, w, 0.5, 0.5, SpatialSemanticConfig(enabled=True, contrast=1.0,
                                                                         control="shuffled"), key="img1")
    assert torch.equal(sh, again)          # deterministic per image key


@pytest.mark.parametrize("kwargs", [dict(contrast=-1), dict(contrast=float("inf")), dict(smoothing_sigma=-1),
                                    dict(soft_scale_min=0.8, soft_scale_max=0.5), dict(control="flip"),
                                    dict(temperature=0)])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        SpatialSemanticConfig(**kwargs)


# --------------------------------------------------------------------------- #
# build_spatial: modes, lambda = 0, controls
# --------------------------------------------------------------------------- #
def _spatial_for(upstream, sem, ssr):
    import cfa_seesr
    m, img = _model_with_tags(upstream, min_tags=3)
    return cfa_seesr.spatial_validation_prompt(upstream, types.SimpleNamespace(prompt=""), img, m, sem, ssr, "img")


def test_lambda_zero_is_component_a_exactly(upstream_test_seesr):
    sem = SemanticConfidenceConfig(enabled=True, min_confidence=0.6, soft_scale_min=0.3, soft_scale_max=0.9)
    _, _, result, sp = _spatial_for(upstream_test_seesr, sem, SpatialSemanticConfig(enabled=True, contrast=0.0))
    assert sp.mode == "scalar" and sp.grid is None and sp.soft_scale == result.soft_scale
    # A off: L = 1, s = s_max
    _, _, _, sp = _spatial_for(upstream_test_seesr, SemanticConfidenceConfig(),
                               SpatialSemanticConfig(enabled=True, contrast=0.0))
    assert sp.mode == "scalar" and sp.soft_scale == 1.0


def test_spatial_mode_and_mean_matched(upstream_test_seesr):
    sem = SemanticConfidenceConfig(enabled=True, soft_scale_min=0.5)
    _, _, result, sp = _spatial_for(upstream_test_seesr, sem, SpatialSemanticConfig(enabled=True, contrast=1.0))
    assert sp.mode == "spatial" and sp.grid.shape == (12, 12)
    assert sp.stats["level"] == result.reliability and abs(sp.stats["mean_unclipped"] - result.reliability) < 1e-9
    _, _, _, mm = _spatial_for(upstream_test_seesr, SemanticConfidenceConfig(),
                               SpatialSemanticConfig(enabled=True, contrast=1.0, control="mean_matched"))
    s = resolve_mean_matched(mm, 8, 8)
    m = SpatialMap(mm.grid, mm.s_min, mm.s_max)
    m.set_latent_size(8, 8)
    lv = float(m.latent.double().mean())
    assert abs(s - ((1 - lv) * 0.5 + lv * 1.0)) < 1e-12 and 0.5 <= s <= 1.0


# --------------------------------------------------------------------------- #
# Map at feature resolution and tiles
# --------------------------------------------------------------------------- #
def test_block_maps_resize_crop_and_broadcast():
    grid = torch.rand(12, 12, dtype=torch.float64)
    m = SpatialMap(grid, 0.5, 1.0)
    m.set_latent_size(64, 48)
    for H, W in ((64, 48), (32, 24), (16, 12), (8, 6)):
        s = m.block_scale(H, W, "cpu", torch.float32)
        assert s.shape == (1, 1, H, W) and 0.5 - 1e-6 <= s.min() and s.max() <= 1.0 + 1e-6
        if (H, W) != (64, 48):   # area resizing preserves the mean exactly for integer factors
            assert abs(float(m.block_map(H, W).mean()) - float(m.latent.mean())) < 1e-5
    x = torch.randn(2, 5, 16, 12)                        # CFG batch of 2
    s = m.block_scale(16, 12, "cpu", torch.float32)
    assert (s * x).shape == x.shape and torch.equal((s * x)[0], s[0] * x[0])
    m.set_region((8, 40, 4, 36))
    crop = m.block_map(32, 32)
    assert torch.equal(crop, m.latent[8:40, 4:36])
    assert torch.allclose(m.block_map(16, 16), torch.nn.functional.avg_pool2d(crop[None, None], 2)[0, 0])


def _upstream_tile_order(h, w, tile_size, tile_overlap):
    """Copy of the enumeration in pipeline_seesr (tiled branch), recording slices instead of running."""
    tile_size = min(tile_size, min(h, w))
    grid_rows = 0
    cur_x = 0
    while cur_x < w:
        cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
        grid_rows += 1
    grid_cols = 0
    cur_y = 0
    while cur_y < h:
        cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
        grid_cols += 1
    out = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size
            out.append((ofs_y, ofs_y + tile_size, ofs_x, ofs_x + tile_size))
    return out


@pytest.mark.parametrize("h,w,ts,ov", [(64, 64, 96, 4), (128, 96, 96, 4), (200, 150, 96, 4), (6, 6, 4, 1),
                                       (13, 7, 4, 2)])
def test_tile_regions_match_upstream_order(h, w, ts, ov):
    regions = tile_regions(h, w, ts, ov)
    assert regions == _upstream_tile_order(h, w, ts, ov)
    covered = np.zeros((h, w), bool)
    for y0, y1, x0, x1 in regions:
        assert 0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w
        covered[y0:y1, x0:x1] = True
    assert covered.all()


# --------------------------------------------------------------------------- #
# Hooks on the tiny real pipeline
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    pytest.importorskip("diffusers")
    from experiments.tests.tiny_pipeline import build_pipeline
    return build_pipeline(tmp_path_factory.mktemp("tok_spatial"))


def _run(pipe, img, steps=3, tile=96, overlap=4):
    from experiments.tests.tiny_pipeline import call_kwargs
    ram = torch.randn(1, 5, 512, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(7)
    kw = call_kwargs(img, steps, torch.Generator().manual_seed(7), ram)
    kw.update(latent_tiled_size=tile, latent_tiled_overlap=overlap)
    return pipe("a, clean", img, **kw).images[0]


def _install(pipe, state, overlap=4, lr=None):
    from cfa_seesr.hooks import MethodHooks
    hooks = MethodHooks()
    hooks.install_soft_prompt_scaling(pipe, state)
    if lr is not None:
        hooks.install_lr_schedule(pipe, lr, state)
    hooks.install_spatial_tracking(pipe, state, overlap)
    return hooks


def test_constant_map_equals_scalar_and_nonconstant_map_differs(pipe):
    from cfa_seesr.hooks import MethodState
    img = _image(5, 32)
    state = MethodState()
    hooks = _install(pipe, state)
    try:
        state.soft_scale = 0.7
        scalar = _run(pipe, img)
        # constant map with value M: s = (1 - M) * 0.5 + M * 1.0 = 0.7 at M = 0.4
        state.spatial = SpatialMap(torch.full((12, 12), 0.4, dtype=torch.float64), 0.5, 1.0)
        const = _run(pipe, img)
        state.spatial = SpatialMap(torch.rand(12, 12, generator=torch.Generator().manual_seed(0),
                                              dtype=torch.float64), 0.5, 1.0)
        varying = _run(pipe, img)
        assert state.tile_log and all(region is None for _, region in state.tile_log)   # untiled
    finally:
        hooks.remove()
    assert np.allclose(const, scalar, atol=1e-5)
    assert not np.allclose(varying, scalar, atol=1e-4)
    assert "forward" not in vars(pipe.controlnet) and "prepare_image" not in vars(pipe)


def test_every_image_attention_block_receives_the_map(pipe):
    from cfa_seesr.hooks import MethodState, image_attention_blocks
    img = _image(6, 32)
    state = MethodState()
    seen = []

    class Recording(SpatialMap):
        def block_scale(self, H, W, device, dtype):
            s = super().block_scale(H, W, device, dtype)
            seen.append((H, W, tuple(s.shape)))
            return s

    hooks = _install(pipe, state)
    try:
        state.spatial = Recording(torch.rand(12, 12, dtype=torch.float64), 0.5, 1.0)
        _run(pipe, img, steps=2)
    finally:
        hooks.remove()
    n_blocks = len(image_attention_blocks(pipe.unet)) + len(image_attention_blocks(pipe.controlnet))
    assert len(seen) == 2 * n_blocks                     # every block, every step (CFG batched in one call)
    assert all(s == (1, 1, H, W) for H, W, s in seen)


def test_tiled_sampling_crops_the_map_per_tile(pipe):
    from cfa_seesr.hooks import MethodState
    from cfa_seesr.lr_guidance import LRGuidanceConfig
    img = _image(8, 48)                                   # latent 6x6; tile 4, overlap 1 -> 4 tiles
    state = MethodState()
    lr = LRGuidanceConfig(enabled=True, schedule="power", scale_start=1.2, scale_end=0.8)
    hooks = _install(pipe, state, overlap=1, lr=lr)       # SSR tracking wraps the LR wrapper
    try:
        state.spatial = SpatialMap(torch.rand(12, 12, dtype=torch.float64), 0.5, 1.0)
        _run(pipe, img, steps=2, tile=4, overlap=1)
        regions = [r for _, r in state.tile_log]
        expected = tile_regions(6, 6, 4, 1)
        assert regions == expected * 2                    # every tile, in pipeline order, at both steps
        assert len(state.lr_log) == len(regions)          # B still applied to every call
        state.reset_call()
        with pytest.raises(RuntimeError, match="does not match|more ControlNet calls"):
            _run(pipe, img, steps=1, tile=4, overlap=3)   # tracking assumes overlap 1: refuses
    finally:
        hooks.remove()
    assert "forward" not in vars(pipe.controlnet) and "prepare_image" not in vars(pipe)


def test_runner_with_spatial_variant(pipe, upstream_test_seesr):
    from experiments import run_inference as ri
    from experiments.tests.tiny_pipeline import call_kwargs
    img = _image(9, 32)
    ref = _run(pipe, img)
    for ssr, expect_mode in ((dict(enabled=True, contrast=1.0), "spatial"),
                             (dict(enabled=True, contrast=1.0, control="mean_matched"), "mean_matched"),
                             (dict(enabled=True, contrast=0.0), "scalar")):
        method = MethodConfig.from_dict({"spatial_semantic": ssr})
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
            dape, _ = _model_with_tags(upstream_test_seesr, min_tags=3)
            prompt, embeds = mod.get_validation_prompt(types.SimpleNamespace(prompt=""), img, dape)
            up_prompt, _ = upstream_test_seesr.get_validation_prompt(types.SimpleNamespace(prompt=""), img, dape,
                                                                     device="cpu")
            assert prompt == up_prompt                     # A off: upstream prompt
            ram = torch.randn(1, 5, 512, generator=torch.Generator().manual_seed(1))
            torch.manual_seed(7)
            out = timed("a, clean", img, **call_kwargs(img, 3, torch.Generator().manual_seed(7), ram)).images[0]
            row = rec.rows[0]
            assert row["ssr_mode"] == expect_mode and row["ssr_n_tags"] >= 3
            assert 0 <= row["ssr_min"] <= row["ssr_mean"] <= row["ssr_max"] <= 1
            assert "n_tags_selected" not in row                    # component A is not reported as active
            assert rec.spatial_records and (expect_mode == "scalar") == (rec.spatial_grids == {})
            if expect_mode == "scalar":
                assert row["soft_scale"] == 1.0 and np.array_equal(out, ref)
            elif expect_mode == "mean_matched":
                assert 0.5 <= row["soft_scale"] <= 1.0
            else:
                assert row["ssr_tiled"] is False and row["ssr_controlnet_calls"] == 3
        finally:
            ri.remove_hooks(mod, rec)
            pipe.scheduler.step = original_step
        assert "forward" not in vars(pipe.controlnet) and "prepare_image" not in vars(pipe)
    assert np.array_equal(_run(pipe, img), ref)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_method_config_with_spatial_section():
    m = MethodConfig.from_dict({"spatial_semantic": {"enabled": True, "contrast": 1.0}})
    assert m.spatial_active and not m.semantic_active and not m.is_baseline
    assert m.soft_scale_range == (0.5, 1.0)
    a = MethodConfig.from_dict({"semantic_confidence": {"enabled": True, "soft_scale_min": 0.3},
                                "spatial_semantic": {"enabled": True, "soft_scale_min": 0.9}})
    assert a.soft_scale_range == (0.3, 1.0)                    # component A's range when A is on
    assert MethodConfig.from_dict({"spatial_semantic": {"enabled": False, "contrast": 3}}).is_baseline
    with pytest.raises(TypeError):
        MethodConfig.from_dict({"spatial_semantic": {"lambda": 1}})
    assert "spatial_semantic" in MethodConfig().to_dict()


def test_sweep_spatial_variants_and_defaults():
    from pathlib import Path

    from experiments.common import load_sweep, sweep_default_variants
    sweep = Path(__file__).resolve().parents[1] / "configs" / "sampling_sweep.yaml"
    assert sweep_default_variants(sweep) == ["original", "semantic_conf", "lr_schedule", "combined"]
    runs = {n: MethodConfig.from_dict(c["method"]) for n, c in load_sweep(sweep) if n.endswith("_ddpm20_s42")}
    get = lambda v: runs[f"{v}_ddpm20_s42"]  # noqa: E731
    assert not any(get(v).spatial_active for v in ("original", "semantic_conf", "lr_schedule", "combined"))
    assert get("spatial_sem").spatial_active and not get("spatial_sem").semantic_active
    for v in ("semantic_spatial", "spatial_shuffled"):
        assert get(v).semantic_confidence == get("semantic_conf").semantic_confidence
    assert get("spatial_shuffled").spatial_semantic.control == "shuffled"
    assert get("combined_spatial").lr_guidance == get("combined").lr_guidance
    mm, ss = get("global_meanmatched").spatial_semantic, get("spatial_sem").spatial_semantic
    assert mm.control == "mean_matched" and mm.contrast == ss.contrast and mm.soft_scale_min == ss.soft_scale_min


# --------------------------------------------------------------------------- #
# Gate 0
# --------------------------------------------------------------------------- #
def test_gate0_masks_and_controls():
    from experiments.tools import gate0_attention as g0
    a = torch.zeros(12, 12)
    a[2:5, 3:6] = 1.0
    top = g0.top_cells(a, 0.2)
    assert len(top) == 29 and set(r * 12 + c for r in range(2, 5) for c in range(3, 6)) <= set(top)
    shifted = g0.shifted_cell_sets(top, 12, 8, seed=1)
    assert len(shifted) == 8 and all(len(s) == len(top) and s != sorted(top) for s in shifted)
    scattered = g0.random_cell_sets(144, 29, 3, seed=1)
    assert all(len(set(s)) == 29 for s in scattered)
    mask = g0.cell_mask([0, 143], 12, 384)
    assert mask.sum() == 2 * 32 * 32 and mask[0, 0] == 1 and mask[-1, -1] == 1 and mask[0, 40] == 0


def test_gate0_statistics_and_decision():
    from experiments.tools import gate0_attention as g0
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        for t in range(4):   # tags of one image share an image-level effect
            d = 0.5 + rng.normal(0, 0.3)
            rows.append({"dataset": "D", "image": f"i{i}", "drop_top": 1 + d, "drop_rand": 1.0, "diff": d,
                         "entropy_norm": 0.8})
    s = g0.summarize(rows, 30)
    assert s["decision"] == "GO" and s["n_images_with_tags"] == 30 and s["n_tags"] == 120
    lo, hi = s["image_level"]["bootstrap_95ci_mean_diff"]
    assert 0 < lo < s["image_level"]["mean_diff"]["mean"] < hi
    null = [{**r, "diff": r["diff"] - 0.5, "drop_top": r["drop_top"] - 0.5} for r in rows]
    s0 = g0.summarize(null, 30)
    assert s0["decision"] == "NO-GO"
    few = g0.summarize(rows[:40], 10)                    # 10 images < min_images
    assert few["decision"] == "NO-GO" and not few["checks"]["enough_images"]
    assert g0.summarize([], 5)["decision"] == "NO-GO"
    # exact sign-flip test
    assert g0.sign_flip_p([1.0, 1.0, 1.0]) == 1 / 8
    assert len(g0.image_level(rows)) == 30


def test_gate0_end_to_end_on_tiny_dape(upstream_test_seesr, tmp_path):
    from experiments.tools import gate0_attention as g0
    m, img = _model_with_tags(upstream_test_seesr, min_tags=2)
    args = types.SimpleNamespace(layer=-1, fill="local_mean", top_frac=0.2, n_random=3, batch=4, seed=0,
                                 control="shifted")
    rows, grid = g0.analyse_image(m, _lq(upstream_test_seesr, img), "D", "x.png", args,
                                  overlays_dir=tmp_path, pil_image=img)
    assert rows and all(math.isfinite(r["diff"]) for r in rows)
    assert all(abs(r["diff"] - (r["drop_top"] - r["drop_rand"])) < 1e-9 for r in rows)
    assert (tmp_path / "D__x.png").exists() and grid.shape == (12, 12)
    args.control, args.fill = "scattered", "channel_mean"
    rows2, _ = g0.analyse_image(m, _lq(upstream_test_seesr, img), "D", "x.png", args)
    assert [r["logit"] for r in rows2] == [r["logit"] for r in rows]


def test_gate0_image_selection_is_deterministic(tmp_path):
    from experiments.tools import gate0_attention as g0
    for ds in ("A", "B"):
        (tmp_path / ds).mkdir()
        for i in range(5):
            _image(i, 8).save(tmp_path / ds / f"{i}.png")
    src = [("A", tmp_path / "A"), ("B", tmp_path / "B")]
    sel = g0.select_images(src, 6)
    assert sel == g0.select_images(src, 6) and len(sel) == 6
    assert sum(ds == "A" for ds, _ in sel) == 3


# --------------------------------------------------------------------------- #
# Screening decision
# --------------------------------------------------------------------------- #
def _fake(root, name, lp, ds):
    from experiments.common import write_csv, write_json
    d = root / name
    write_json(d / "run_metadata.json", {"status": "completed"})
    write_csv(d / "eval" / "paired_per_image.csv",
              [{"image": k, "lpips": v, "dists": v / 2, "status": "ok"} for k, v in lp.items()])
    write_json(d / "eval" / "paired_summary.json", {"metrics": {}})


def _screen(tmp_path, shuffled_gain):
    from experiments.tools import ssr_screening as sc
    base = {f"i{k}": 0.30 + 0.01 * k for k in range(20)}
    for steps in (20, 50):
        for v, gain in (("original", 0.0), ("semantic_conf", 0.0), ("spatial_sem", 0.02),
                        ("semantic_spatial", 0.02), ("spatial_shuffled", shuffled_gain)):
            _fake(tmp_path / "R", f"{v}_ddpm{steps}_s42", {k: x - gain for k, x in base.items()}, "R")
    out = tmp_path / "s.json"
    sc.main(["--results_root", f"R={tmp_path / 'R'}", "--output", str(out)])
    return json.loads(out.read_text())


def test_screening_continue_and_stop(tmp_path):
    assert _screen(tmp_path / "a", 0.0)["decision"] == "CONTINUE"
    s = _screen(tmp_path / "b", 0.015)                  # shuffled reproduces >= half the gain
    assert s["decision"] == "STOP" and not s["cells"]["R@20"]["checks"]["shuffled_does_not_reproduce"]
    s = _screen(tmp_path / "c", 0.0)
    assert s["decision"] == "CONTINUE"


def test_tile_verification_rejects_a_wrong_position(pipe):
    """Same number of tiles, different positions (overlap 0 vs 1 on a 10x10 latent): only the pixel check catches it."""
    from cfa_seesr.hooks import MethodState
    assert len(tile_regions(10, 10, 4, 0)) == len(tile_regions(10, 10, 4, 1))
    assert tile_regions(10, 10, 4, 0) != tile_regions(10, 10, 4, 1)
    state = MethodState()
    hooks = _install(pipe, state, overlap=0)
    try:
        state.spatial = SpatialMap(torch.rand(12, 12, dtype=torch.float64), 0.5, 1.0)
        with pytest.raises(RuntimeError, match="does not match"):
            _run(pipe, _image(10, 80), steps=1, tile=4, overlap=1)
    finally:
        hooks.remove()
