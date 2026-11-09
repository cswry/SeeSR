"""Spatial semantic reliability (SSR): where to trust the soft prompt.

Component A (semantic_confidence.py) gives one soft-prompt strength per image. SSR distributes that
strength over the image using information DAPE already computes: the cross-attention of each tag query
over the image tokens in DAPE's tagging head. No new model, no trainable parameter, no extra forward pass.

What DAPE computes (verified in ram/models/ram_lora.py, ram/models/bert_lora.py, swin_transformer_lora.py)
  * visual_encoder (Swin-L, 384x384 input) returns cat([avg-pooled token, 12x12 tokens]); after image_proj
    the image embeddings E have shape [1, 1 + g*g, 512] with g = 12. Token 0 is the pooled token.
  * tagging_head is a 2-layer BERT whose self-attention is deleted (RAMLora.del_selfattention); in
    mode='tagging' each layer runs only cross-attention of the K = 4585 tag queries over E.
  * BertSelfAttention.forward computes attention_probs = softmax(QK^T / sqrt(d)) of shape
    [B, heads, K, 1 + g*g] and passes it to self.dropout (identity in eval mode). A forward hook on
    tagging_head.encoder.layer[l].crossattention.self.dropout therefore receives the probabilities as its
    input, without any change to upstream code. Upstream's save_attention flag is not used: it registers
    a gradient hook, which fails under no_grad.

Formulation (g = grid side, G = grid cells, N = |G|; S0 = upstream-selected tags)
    a_i(g)  = mean over heads of P[h, i, 1 + g]                (pooled token excluded)
    A_i(g)  = a_i(g) / sum_g' a_i(g')                          per-tag spatial distribution over G
    c^_i    = 2 c_i - 1  in [0, 1]                             c_i: component A's confidence (c_i > 1/2 on S0)
    rho(g)  = sum_i A_i(g) c^_i / (sum_i A_i(g) + eps)         local reliability (convex combination)
    kappa(g)= min(1, N / |S0| * sum_i A_i(g))                  coverage: 1 where tags attend >= uniform
    r(g)    = kappa(g) rho(g) + (1 - kappa(g)) C               C: image-level reliability of component A
    r~      = Gaussian smoothing of r on the grid (sigma in cells, replicate padding, normalised kernel)
    R(g)    = clip(L + lambda (r~(g) - mean r~), 0, 1)         L = C if component A is on, else 1
    s(x)    = (1 - M(x)) s_min + M(x) s_max                    M: R resized to the feature map

Before clipping, mean(R) = L exactly, so lambda only moves strength between regions; clipping can shift the
mean, which is why the clipped fractions and the final mean are recorded. lambda = 0 gives R = L
everywhere, i.e. component A unchanged (or upstream SeeSR when A is off and s_max = 1): in that case no map
is built and the existing scalar path runs, so the equivalence is exact.

Controls (ablation only)
    shuffled      the 12x12 grid values are permuted (same value distribution, different locations)
    mean_matched  a single scalar equal to the mean of the latent-resolution map (spatial arrangement removed)
"""
import contextlib
import hashlib
import math
from dataclasses import asdict, dataclass

CONTROLS = ("none", "shuffled", "mean_matched")


@dataclass(frozen=True)
class SpatialSemanticConfig:
    enabled: bool = False
    contrast: float = 0.0             # lambda >= 0; 0 = no spatial redistribution
    smoothing_sigma: float = 1.0      # Gaussian sigma in grid cells (0 = no smoothing)
    attention_layer: int = -1         # index into tagging_head.encoder.layer (-1 = last)
    soft_scale_min: float = 0.5       # used only when semantic_confidence is disabled
    soft_scale_max: float = 1.0       # used only when semantic_confidence is disabled
    temperature: float = 1.0          # c_i temperature, used only when semantic_confidence is disabled
    control: str = "none"             # none | shuffled | mean_matched
    shuffle_seed: int = 0
    eps: float = 1e-8

    def __post_init__(self):
        if self.contrast < 0 or not math.isfinite(self.contrast):
            raise ValueError("spatial_semantic.contrast must be a finite value >= 0")
        if self.smoothing_sigma < 0:
            raise ValueError("spatial_semantic.smoothing_sigma must be >= 0")
        if not 0.0 <= self.soft_scale_min <= self.soft_scale_max:
            raise ValueError("spatial_semantic: need 0 <= soft_scale_min <= soft_scale_max")
        if self.temperature <= 0:
            raise ValueError("spatial_semantic.temperature must be > 0")
        if self.control not in CONTROLS:
            raise ValueError(f"spatial_semantic.control must be one of {CONTROLS}")
        if self.eps <= 0:
            raise ValueError("spatial_semantic.eps must be > 0")

    @classmethod
    def from_dict(cls, d):
        return cls(**(d or {}))


# --------------------------------------------------------------------------- #
# Attention capture (passive forward hooks; removed on exit)
# --------------------------------------------------------------------------- #
def attention_dropout_module(model, layer=-1):
    """The nn.Dropout that receives the cross-attention probabilities of one tagging-head layer."""
    import torch
    layers = model.tagging_head.encoder.layer
    blk = layers[layer]
    if not hasattr(blk, "crossattention"):
        raise RuntimeError("tagging-head layer has no crossattention: not a DAPE/RAM tagging head")
    attn = blk.crossattention.self
    if not isinstance(attn.dropout, torch.nn.Dropout):
        raise RuntimeError("expected crossattention.self.dropout to be torch.nn.Dropout")
    return attn.dropout


@contextlib.contextmanager
def capture_dape(model, layer=-1):
    """Capture tagging-head cross-attention probabilities and the tag logits during one DAPE pass.

    Yields a dict that receives 'attention' [B, heads, K, T] (detached, pre-dropout probabilities) and
    'logits' [B, K]. Hooks only read tensors; outputs are unchanged. Both hooks are removed on exit,
    also if the pass raises.
    """
    store = {}

    def on_attention(module, inputs, output):
        store["attention"] = inputs[0].detach()

    def on_fc(module, inputs, output):
        store["logits"] = output.detach().squeeze(-1)

    handles = [attention_dropout_module(model, layer).register_forward_hook(on_attention),
               model.fc.register_forward_hook(on_fc)]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def upstream_tag_indices(logits, thresholds, delete_index):
    """S0 in class-index order, with exactly the test of RAMLora.generate_tag / select_tags."""
    import torch
    probs = torch.sigmoid(logits.float())
    passed = probs > thresholds.float().to(probs.device)
    if len(delete_index):
        passed[list(delete_index)] = False
    return torch.nonzero(passed).flatten().tolist()


def grid_side(n_tokens):
    """g for T = 1 + g*g tokens (pooled token first); raises if the token count is not of that form."""
    g = math.isqrt(n_tokens - 1)
    if g * g != n_tokens - 1 or g < 1:
        raise ValueError(f"{n_tokens} image tokens is not 1 + g*g: cannot form a spatial grid")
    return g


def tag_attention_maps(attention, tag_index):
    """A_i over the spatial grid for the given tags.

    attention: [heads, K, T] probabilities for one image (T = 1 + g*g, token 0 pooled).
    Returns float64 tensor [n_tags, g, g] with each map summing to 1 (pooled token excluded).
    """
    import torch
    g = grid_side(attention.shape[-1])
    if not tag_index:
        return torch.zeros(0, g, g, dtype=torch.float64)
    rows = attention[:, tag_index, 1:].double().mean(dim=0)            # [n, g*g]
    rows = rows / rows.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return rows.reshape(len(tag_index), g, g)


# --------------------------------------------------------------------------- #
# Reliability grid
# --------------------------------------------------------------------------- #
def gaussian_smooth(grid, sigma):
    """Separable normalised Gaussian on a [g, g] grid, replicate padding (a convex combination of cells)."""
    import torch
    import torch.nn.functional as F
    if sigma <= 0:
        return grid.clone()
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    t = grid[None, None]
    t = F.pad(t, (radius, radius, radius, radius), mode="replicate")
    t = F.conv2d(t, k.view(1, 1, 1, -1))
    t = F.conv2d(t, k.view(1, 1, -1, 1))
    return t[0, 0]


def _shuffle_grid(grid, seed, key):
    import torch
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    gen = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF)
    flat = grid.flatten()
    return flat[torch.randperm(flat.numel(), generator=gen)].reshape(grid.shape)


def reliability_grid(maps, conf_hat, global_reliability, level, cfg, key=""):
    """Compute R on the attention grid. Returns (R [g, g] float64, stats dict).

    maps: [n, g, g] A_i (each sums to 1); conf_hat: [n] values c^_i in [0, 1];
    global_reliability: C in [0, 1]; level: L (C when component A is on, 1 otherwise).
    """
    import torch
    n, g = maps.shape[0], maps.shape[-1]
    C = float(min(1.0, max(0.0, global_reliability)))
    L = float(min(1.0, max(0.0, level)))
    stats = {"n_tags": n, "grid": g, "level": L, "global_reliability": C, "contrast": cfg.contrast,
             "control": cfg.control}
    if n == 0:   # no tag: no spatial evidence, R = L everywhere
        R = torch.full((g, g), L, dtype=torch.float64)
        stats.update(fallback="no_tags", coverage_mean=0.0)
        return R, _finish(R, R, stats)
    w = torch.as_tensor(conf_hat, dtype=torch.float64).clamp(0.0, 1.0)
    mass = maps.sum(dim=0)                                                  # sum_i A_i(g)
    rho = (maps * w[:, None, None]).sum(dim=0) / (mass + cfg.eps)
    kappa = (mass * (g * g) / n).clamp(max=1.0)
    r = kappa * rho + (1.0 - kappa) * C
    r_s = gaussian_smooth(r, cfg.smoothing_sigma)
    unclipped = L + cfg.contrast * (r_s - r_s.mean())
    R = unclipped.clamp(0.0, 1.0)
    stats.update(fallback=None, coverage_mean=float(kappa.mean()), rho_mean=float(rho.mean()),
                 r_min=float(r.min()), r_max=float(r.max()))
    if cfg.control == "shuffled":
        R = _shuffle_grid(R, cfg.shuffle_seed, key)
    return R, _finish(R, unclipped, stats)


def _finish(R, unclipped, stats):
    if not bool(R.isfinite().all()):
        raise FloatingPointError("spatial reliability map is not finite")
    n = unclipped.numel()
    stats.update(mean_unclipped=float(unclipped.mean()), mean=float(R.mean()), min=float(R.min()),
                 max=float(R.max()), clip_low_frac=float((unclipped < 0).sum()) / n,
                 clip_high_frac=float((unclipped > 1).sum()) / n)
    return stats


# --------------------------------------------------------------------------- #
# Map at feature resolution (with tile cropping)
# --------------------------------------------------------------------------- #
class SpatialMap:
    """Turns the grid R into per-block soft-prompt scales s(x) = (1 - M) s_min + M s_max.

    The grid is first resized (bilinear) to the full latent size (h, w); a block of size (H, W) gets the
    latent map, cropped to the current tile if the pipeline is tiling, area-resized to (H, W). Area
    resizing averages cells, so values stay in [0, 1]. One definition for tiled and untiled paths.
    """

    def __init__(self, grid, s_min, s_max):
        self.grid = grid
        self.s_min, self.s_max = float(s_min), float(s_max)
        self.latent = None          # [h, w] float32
        self.region = None          # (y0, y1, x0, x1) in latent cells, None = whole image
        self._cache = {}

    def set_latent_size(self, h, w):
        import torch.nn.functional as F
        if self.latent is not None and tuple(self.latent.shape) == (h, w):
            return
        self.latent = F.interpolate(self.grid.float()[None, None], size=(h, w), mode="bilinear",
                                    align_corners=False)[0, 0].clamp(0.0, 1.0)
        self._cache.clear()

    def set_region(self, region):
        self.region = region

    def latent_mean(self):
        return float(self.latent.double().mean())

    def block_map(self, H, W):
        import torch.nn.functional as F
        if self.latent is None:
            raise RuntimeError("SpatialMap used before the latent size was set")
        crop = self.latent if self.region is None else \
            self.latent[self.region[0]:self.region[1], self.region[2]:self.region[3]]
        if tuple(crop.shape) == (H, W):
            return crop
        return F.adaptive_avg_pool2d(crop[None, None], (H, W))[0, 0]

    def block_scale(self, H, W, device, dtype):
        key = (self.region, H, W, str(device), dtype)
        s = self._cache.get(key)
        if s is None:
            m = self.block_map(H, W)
            s = ((1.0 - m) * self.s_min + m * self.s_max)[None, None].to(device=device, dtype=dtype)
            self._cache[key] = s
        return s


def tile_regions(h, w, tile_size, tile_overlap):
    """Latent tile coordinates (y0, y1, x0, x1) in exactly the order pipeline_seesr visits them.

    Re-implements the enumeration in StableDiffusionControlNetPipeline.__call__ (tiled branch): note that
    the loop variable `row` indexes x and `col` indexes y there, and that the last row/column is aligned
    to the image border. Only meaningful when h*w > tile_size**2 (otherwise the pipeline does not tile).
    """
    tile_size = min(tile_size, min(h, w))
    grid_rows, cur_x = 0, 0
    while cur_x < w:
        cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
        grid_rows += 1
    grid_cols, cur_y = 0, 0
    while cur_y < h:
        cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
        grid_cols += 1
    out = []
    ofs_x = ofs_y = 0
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


# --------------------------------------------------------------------------- #
# Per-image result
# --------------------------------------------------------------------------- #
@dataclass
class SpatialResult:
    mode: str                 # "spatial" | "scalar"
    soft_scale: float         # scalar used when mode == "scalar" (lambda = 0 or mean_matched)
    grid: object              # torch [g, g] float64 R (after control), or None
    s_min: float
    s_max: float
    stats: dict

    def summary(self):
        d = {k: v for k, v in asdict(self).items() if k != "grid"}
        return d


def build_spatial(attention, logits, model, sem_cfg, ssr_cfg, semantic_result, key=""):
    """SpatialResult for one image from the captured DAPE tensors.

    attention [1, heads, K, T], logits [1, K]; semantic_result: component A's SemanticResult for the same
    pass (its soft_scale and reliability are used unchanged when A is enabled).
    """
    from .semantic_confidence import aggregate_reliability, tag_confidence

    if attention.shape[0] != 1 or logits.shape[0] != 1:
        raise ValueError("build_spatial expects a single image")
    K = logits.shape[-1]
    if attention.shape[-2] != K:
        raise ValueError(f"attention has {attention.shape[-2]} query rows, expected {K} tags")
    idx = upstream_tag_indices(logits[0], model.class_threshold, model.delete_tag_index)
    expected = ", ".join(str(model.tag_list[i]) for i in idx)
    if expected != semantic_result.upstream_tags:
        raise RuntimeError("captured logits do not reproduce the upstream tag set")
    a_on = sem_cfg.enabled
    temperature = sem_cfg.temperature if a_on else ssr_cfg.temperature
    conf, _ = tag_confidence(logits[0], model.class_threshold, temperature)
    c = [float(conf[i]) for i in idx]
    conf_hat = [min(1.0, max(0.0, 2.0 * v - 1.0)) for v in c]
    C = semantic_result.reliability if a_on else aggregate_reliability(c, "mean")
    L = C if a_on else 1.0
    s_min, s_max = ((sem_cfg.soft_scale_min, sem_cfg.soft_scale_max) if a_on
                    else (ssr_cfg.soft_scale_min, ssr_cfg.soft_scale_max))
    maps = tag_attention_maps(attention[0], idx)
    R, stats = reliability_grid(maps, conf_hat, C, L, ssr_cfg, key)
    stats["attention_tokens"] = int(attention.shape[-1])
    stats["attention_heads"] = int(attention.shape[1])

    def scalar(level):
        if a_on and level == L:
            return semantic_result.soft_scale                     # component A's value, bit-identical
        lv = min(1.0, max(0.0, level))
        return (1.0 - lv) * s_min + lv * s_max

    if ssr_cfg.contrast == 0.0 and ssr_cfg.control != "mean_matched":
        return SpatialResult("scalar", scalar(L), None, s_min, s_max, {**stats, "reason": "contrast_zero"})
    if ssr_cfg.control == "mean_matched":
        # the mean of the latent-resolution map is only known once the latent size is; store the grid and
        # let the runner resolve it (SpatialResult.soft_scale is filled by resolve_mean_matched)
        return SpatialResult("mean_matched", float("nan"), R, s_min, s_max, stats)
    return SpatialResult("spatial", float("nan"), R, s_min, s_max, stats)


def resolve_mean_matched(result, h, w):
    """Scalar soft-prompt scale equal to the mean of the latent-resolution map (mean_matched control)."""
    m = SpatialMap(result.grid, result.s_min, result.s_max)
    m.set_latent_size(h, w)
    level = m.latent_mean()
    result.stats["latent_mean"] = level
    result.soft_scale = (1.0 - level) * result.s_min + level * result.s_max
    return result.soft_scale
