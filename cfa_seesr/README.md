# Confidence- and Fidelity-Aware Few-Step SeeSR

A training-free inference extension of SeeSR (Wu et al., CVPR 2024) with two independent components and
one optional extension:

* **A, confidence-aware semantic conditioning**: tag selection and soft-prompt strength driven by DAPE's
  own margin over its class thresholds;
* **B, LR-fidelity-aware guidance**: a timestep-dependent strength for the ControlNet (LR) residuals;
* **SSR, spatial semantic reliability** (§11): A's soft-prompt strength distributed over the image using
  DAPE's tagging-head attention. Used only if an attention-localisation check (Gate 0) passes.

All of them use the released weights unchanged; none adds a model, a trainable parameter or a network
pass. Upstream SeeSR files are not modified. With every component disabled no hook is installed, and
outputs are bit-identical to upstream; the tests check this.
This document states the hypotheses. No GPU result exists yet, and nothing here claims an improvement.

## 1. Original conditioning flow (verified in code)

For an LR image `x` (`test_seesr.py::main` → `pipelines/pipeline_seesr.py`):

| Path | Code | What it does |
|---|---|---|
| DAPE | `ram/models/ram_lora.py::RAMLora` | Swin-L encoder + projection → image embeddings `E ∈ R^{1×(1+N)×512}`; tagging head → logits `z ∈ R^K` (K = 4585) |
| Hard prompt | `RAMLora.generate_tag` | tags `{i : σ(z_i) > τ_i}` (per-class thresholds `τ_i` from `ram/data/ram_tag_list_threshold.txt`), joined as `"t1, t2, …"`, then `f"{tags}, {prompt},"` + `"clean, high-resolution, 8k"` → CLIP text encoder |
| Soft prompt | `RAMLora.generate_image_embeds` | `E` → `image_attentions` blocks (one `Transformer2DModel` after every text cross-attention in the UNet and the ControlNet) |
| LR structure | `models/controlnet.py::ControlNetModel.forward` | LR image → conditioning embedding → down/mid residuals × `conditioning_scale` (1.0) → added to UNet skip connections, at every step |
| Sampling | `DDPMScheduler`, 50 steps, CFG 5.5 | the soft prompt is the same in both CFG branches, so CFG only contrasts the text prompts |

During training, the hard prompt was RAM's tags on the **HR** image, dropped for 50% of samples
(`null_text_ratio=0.5`). At inference it is DAPE's tags on the **LR** image.

## 2. Limitations motivating the changes

1. **Every passing tag counts the same.** A tag 0.1% above its threshold enters the prompt with the same
   weight as a near-certain one. Wrong tags degrade fidelity more than having no tags (Chen et al., SRSR, 2025).
2. **The soft prompt is applied at full strength regardless of how reliable DAPE is on this image.** The
   image embeddings and the tag logits come from the same encoder, so when the tagging head is unsure the
   embeddings feeding the soft prompt are plausibly unreliable too. This is a hypothesis, tested in §7.
3. **The LR residuals have a constant strength at every timestep.** Structure is mostly decided at high-noise
   steps and texture at low-noise steps. A constant strength cannot shift influence between the two, and
   with few steps each step carries more weight.

## 3. Component A: confidence-aware semantic conditioning

**What DAPE provides.** Logits `z_i`, probabilities `p_i = σ(z_i)`, thresholds `τ_i`, image embeddings `E`.
`p_i` is **not** a calibrated probability of the object being present: DAPE was distilled to match RAM's
outputs on HR images, and `τ_i` are RAM's tuned operating points. So confidence is measured relative to the
operating point, in the model's own log-odds scale:

```
m_i = z_i − logit(τ_i)                     margin over the class threshold (nats of log-odds)
c_i = σ(m_i / T)                           c_i > ½  ⇔  p_i > τ_i
```

For `T = 1`, `c_i = odds(p_i) / (odds(p_i) + odds(τ_i))`. `T > 0` (log-odds units) sets how much margin
counts as decisive. `c_i` is a monotone, threshold-centred score, not a probability of correctness.

**Tag selection.** `S0 = {i : p_i > τ_i} \ deleted` (upstream). Selected tags are `S = {i ∈ S0 : c_i > c_min}`.
`c_min = ½` reproduces upstream exactly. `c_min > ½` raises every threshold by the same log-odds amount
`T·logit(c_min)`, which respects class-specific thresholds; a constant offset in probability space would
treat a τ = 0.5 class and a τ = 0.95 class inconsistently.

**Reliability and strength.**

```
C     = clip((agg_{i∈S0} c_i − ½) / ½, 0, 1)        agg = mean (default) or min; C = 0 if S0 = ∅
s_sem = (1 − C)·s_min + C·s_max                       0 ≤ s_min ≤ s_max
```

`C` is computed over `S0`, independent of `c_min`, so selection and strength are separate knobs. `s_sem`
scales the residual of every image-attention block: `h ← h + s_sem·(f(h) − h)`, where `f(h)` is the
block's output (the block already contains its input as a residual). `s_sem = 1` is the trained model and
`s_sem = 0` removes the soft-prompt pathway; a test checks it equals skipping those blocks. `s_max = 1`
by default, so the pathway is never amplified beyond its trained strength.

**Fallback.** If no tag survives, the hard prompt is empty, which is an in-distribution input because of the
50% prompt dropout in training. If `S0` is empty, `C = 0` and the soft prompt uses `s_min`.

| Parameter | Role | Default (disabled) | Initial setting in the ablation |
|---|---|---|---|
| `temperature` T | log-odds scale of the margin | 1.0 | 1.0 |
| `min_confidence` c_min | tag selection | 0.5 (upstream) | σ(1) ≈ 0.731: margin ≥ 1 nat, i.e. odds ≥ e × threshold odds |
| `soft_scale_min` s_min | soft-prompt strength at C = 0 | 1.0 | 0.5 (halved, never removed) |
| `soft_scale_max` s_max | soft-prompt strength at C = 1 | 1.0 | 1.0 (trained strength) |
| `aggregate` | combination of c_i over S0 | mean | mean |

These are initial settings chosen by the rules stated, not tuned. Tune them on a held-out set, never on the
test sets they are reported on.

## 4. Component B: LR-fidelity-aware guidance

With `t` the **actual** DDPM timestep (`T = num_train_timesteps − 1 = 999`; large `t` = early, noisy),
the ControlNet call receives

```
s_lr(t) = conditioning_scale · f(t),      f(t) = s_end + (s_start − s_end)·(t / T)^γ
```

`s_start` is the factor at `t = T`, `s_end` at `t = 0`, and `γ > 0` the shape. Because `f` depends on `t`, not
the loop index, a given noise level gets the same factor at 50, 30 or 20 steps. The pipeline calls the
ControlNet once per step (CFG-batched) and once per latent tile in the tiled path; both receive `t`. Only
the scalar changes, so there are no extra network evaluations.

| Schedule (variant) | s_start → s_end, γ | Mean factor over executed timesteps |
|---|---|---|
| constant (`original`) | 1 → 1 | 1 |
| early-strong / late-weaker (`lr_schedule`) | 1.25 → 0.75, 1 | 0.996 / 0.990 / 0.988 at 50 / 30 / 20 steps |
| early-weaker / late-strong (`lr_schedule_late`, ablation) | 0.75 → 1.25, 1 | ≈ 1.0 likewise |

Both non-constant schedules keep the average guidance near the trained strength. The schedule
redistributes guidance over time rather than changing its amount, so the two directions isolate
*when* LR guidance matters. Factors above 1 are outside the training regime; the bound 1.25 is a modest,
untuned choice.

## 5. Combined method

`combined` enables A and B with the settings above. A changes the hard prompt and the soft-prompt
strength (both fixed for an image); B changes the ControlNet strength along the trajectory. They act on
different inputs of the same UNet, so their effects need not add up; that is why each is also evaluated alone.

## 6. Computational overhead

| | Network evaluations | Other |
|---|---|---|
| A | **fewer**: one DAPE pass (`condition_forward`) instead of upstream's two (`generate_tag` + `generate_image_embeds`, each running the Swin-L encoder) | one element-wise `h + s·(f(h) − h)` per image-attention block when `s ≠ 1` |
| B | none | one scalar multiplication per ControlNet call |
| SSR | none | attention read from A's DAPE pass; per block a cached map instead of A's scalar (§11) |

Every run records the per-image prompt time (`prompt_s`) and diffusion time (`diffusion_s`) separately, so
A's DAPE saving and any diffusion-time overhead are measured, not assumed.

## 7. Expected advantages, failure cases, and what would disprove the hypotheses

**Expected if the hypotheses hold.** Fewer wrong objects and textures (A), better structure at fixed
perceptual quality (B), with the benefit larger at 20–30 steps than at 50.

**Failure cases.**
* Dropping correct but less confident tags removes useful semantics (rare objects, small regions).
* The empty-prompt fallback may oversmooth images that are hard to tag.
* Lowering the soft prompt may lose detail that DAPE embeddings would have supplied correctly.
* Strong early LR guidance can lock in degradations (noise, JPEG blocks) as structure.
* Weaker late guidance can let texture hallucination through.

**Evidence that would disprove each hypothesis.**

| Hypothesis | Disproved if |
|---|---|
| c_i ranks tag reliability | `tools/analyze_tag_confidence.py`: agreement with RAM-on-HR tags does not increase with c_i, or kept and dropped tags agree equally often |
| A reduces semantic errors | `semantic_conf` vs `original` at the same steps: no improvement in tag F1 (`evaluation/semantic_tags.py`) and no LPIPS/DISTS gain beyond the within-image seed std |
| Timing of LR guidance matters | `lr_schedule` and `lr_schedule_late` are indistinguishable from `original` (differences within the seed spread) |
| Early-strong is the right direction | `lr_schedule_late` is better than or equal to `lr_schedule` on PSNR/SSIM and LPIPS/DISTS |
| The method helps few-step sampling | method − original at 20/30 steps is no better than at 50 steps |

## 8. Ablation design

Variants × DDPM steps {50, 30, 20} × seeds {42, 123, 2026}, with per-image seeding
(`experiments/configs/sampling_sweep.yaml`):

| Variant | A | B |
|---|---|---|
| `original` | off | constant |
| `semantic_conf` | on | constant |
| `lr_schedule` | off | 1.25 → 0.75 |
| `combined` | on | 1.25 → 0.75 |
| `lr_schedule_late` (optional) | off | 0.75 → 1.25 |

SSR variants (§11; not in the default set, run only after Gate 0 and the screening): `spatial_sem`,
`semantic_spatial`, `combined_spatial`, `spatial_shuffled`, `global_meanmatched`.

`experiments/tools/compare_results.py sampling` pairs runs by (image, seed). It reports each variant
against `original` at the same step count (method effect) and against `original` at 50 steps (combined
effect of steps and method), and rejects any comparison where anything other than the method, the steps
and the seed differs. The upstream-seeded 50-step reproduction (track A) is kept separate.

## 9. Related work (verified) and what is not new

| Work | Relation |
|---|---|
| **SeeSR**: Wu et al., CVPR 2024, arXiv:2311.16518 | base method: DAPE tags (hard) and embeddings (soft) |
| **SRSR**: Chen et al., NeurIPS 2025, arXiv:2510.22534 | filters DAPE tags with a visual-grounding confidence threshold and restricts each tag's cross-attention to its region; studies the threshold trade-off on SeeSR. **Tag filtering by confidence is prior work.** Ours uses DAPE's own margin over its class thresholds (no grounding model) and also scales the soft prompt |
| **StableSR**: Wang et al., IJCV 2024, arXiv:2305.07015 | argues that LR guidance should be adaptive per step, stronger early and weaker later; implemented with a trained time-aware encoder. **Early-strong LR guidance is a prior idea**; ours is a training-free scalar schedule on SeeSR's ControlNet |
| **DiffBIR**: Lin et al., ECCV 2024, arXiv:2308.15070 | training-free region-adaptive restoration guidance during sampling (fidelity vs quality trade-off) |
| **PASD**: Yang et al., ECCV 2024, arXiv:2308.14469 | pixel-aware conditioning plus high-level (classification/detection/captioning) information; adjustable noise schedule |
| **PiSA-SR**: Sun et al., CVPR 2025, arXiv:2412.03017 | separate pixel-level and semantic-level guidance scales adjustable at inference (trained dual LoRA). **Inference-time balancing of fidelity and semantics is prior work** |
| **ControlNet** (Zhang et al., ICCV 2023) and diffusers `control_guidance_start/end` | per-step on/off windows for ControlNet guidance as a fraction of steps. **Timestep-dependent control strength is established practice**; ours is a continuous schedule on actual timesteps |
| **CFG weight schedulers**: Wang et al., TMLR 2024, arXiv:2404.13040 | time-varying guidance weights (for CFG, not ControlNet) improve text-to-image results |
| **CCSR** (Sun et al., arXiv:2401.00877) and **OSEDiff** (Wu et al., NeurIPS 2024) | few-step / one-step SR via training. Out of scope here (training-free) |

**What this work claims.** Neither component is claimed to be novel in isolation. The contribution under
test is narrower:
* a training-free combination inside SeeSR, namely confidence from DAPE's own threshold margins controlling
  both tag selection and soft-prompt strength, plus a timestep schedule on the existing ControlNet residuals;
* evaluated under controlled few-step DDPM sampling, with paired multi-seed statistics and ablations.

Whether that combination helps is an empirical question that only the GPU experiments can answer.

## 10. Code map

```
cfa_seesr/
  __init__.py              MethodConfig, semantic_validation_prompt (drop-in for upstream get_validation_prompt)
  semantic_confidence.py   Component A: confidence, selection, reliability, soft-prompt scale
  lr_guidance.py           Component B: schedule f(t)
  spatial_reliability.py   SSR: attention capture, reliability grid, feature-resolution maps, tile regions
  hooks.py                 forward hooks on image_attentions (scalar or spatial); ControlNet / prepare_image
                           wrappers (LR schedule, SSR tile tracking); clean removal in reverse order
```

Configuration: the `method` section of `experiments/configs/baseline.yaml` (disabled) and the `variants`
in `experiments/configs/sampling_sweep.yaml`. Tests: `experiments/tests/test_method.py`,
`experiments/tests/test_spatial.py`.

## 11. Spatial semantic reliability (SSR)

**What it does.** Component A gives one soft-prompt strength per image. SSR keeps A's level but lets it vary
over the image: lower where the tags DAPE is unsure about are located, higher where its confident tags are.
A decides *how much*, B decides *when*, SSR decides *where*. It reuses the information A already uses (the
same DAPE pass); it is a spatial extension of A, not a new information source or a new encoder.

**DAPE attention (verified in code).** DAPE's Swin-L returns `cat([avg-pooled token, 12×12 tokens])`, so the
image embeddings are `[1, 145, 512]` with token 0 pooled. The tagging head is a 2-layer BERT whose
self-attention is deleted (`RAMLora.del_selfattention`); in `mode='tagging'` the 4585 tag queries only
cross-attend to the 145 image tokens, 4 heads per layer. `BertSelfAttention` passes the softmax
probabilities `[B, 4, 4585, 145]` to `self.dropout` (identity in eval mode), so a forward hook on
`tagging_head.encoder.layer[-1].crossattention.self.dropout` reads them without modifying upstream code
(upstream's `save_attention` registers a gradient hook and fails under `no_grad`). A hook on `fc` reads the
logits of the same pass. Both hooks only read and are removed right after the pass; tests check that tags,
logits and embeddings are identical with and without them, and that the captured tensor equals the attention
recomputed from the layer's own query/key projections.

**Formulation** (g = 12, G = grid cells, N = 144; S0 = upstream-selected tags; c_i from §3):

```
a_i(g)   = mean over heads of P[h, i, 1 + g]              pooled token excluded
A_i(g)   = a_i(g) / Σ_g' a_i(g')                          tag i's distribution over the grid
c^_i     = 2 c_i − 1 ∈ [0, 1]                              tag weight (c_i > ½ on S0)
ρ(g)     = Σ_i A_i(g) c^_i / (Σ_i A_i(g) + ε)             local reliability (convex combination)
κ(g)     = min(1, N/|S0| · Σ_i A_i(g))                    coverage: 1 where tags attend ≥ uniform
r(g)     = κ(g) ρ(g) + (1 − κ(g)) C                       C: A's image-level reliability
r~       = Gaussian(r, σ cells)                            normalised kernel, replicate padding
R(g)     = clip(L + λ (r~(g) − mean r~), 0, 1)             L = C with A on, 1 with A off
M        = R resized to the latent (bilinear), cropped to the tile, area-resized to each block
h ← h + s(x)·(f(h) − h),  s(x) = (1 − M(x)) s_min + M(x) s_max     every UNet and ControlNet image-attention block
```

Before clipping, mean(R) = L exactly, so λ only moves strength between regions; clipping can move the mean,
so the clipped fractions and the final mean are recorded per image (`spatial_reliability.jsonl`) and per run
(`run_metadata.json → method.spatial_summary`). **λ = 0 is component A exactly** (or upstream SeeSR with A off
and s_max = 1): no map is built and A's scalar is used unchanged. No tag: R = L. The map is the same for both
CFG branches (the soft prompt is identical in both). Text conditioning and the ControlNet residuals are not
touched; with B on, B still scales the residuals.

| Parameter | Role | Initial value in the SSR variants |
|---|---|---|
| `contrast` λ | strength of the spatial redistribution | 1.0 (a reliability difference of 0.1 moves the map by 0.1) |
| `smoothing_sigma` | Gaussian σ on the 12×12 grid | 1 cell |
| `attention_layer` | tagging-head layer read | last (−1) |
| `soft_scale_min/max`, `temperature` | used only with A off | 0.5 / 1.0 / 1.0 (A's values when A is on) |
| `control` | `shuffled` (cells permuted) / `mean_matched` (one scalar = mean of the latent map) | ablation controls |

Untuned; tune on held-out images, never on reported test sets.

**Tiled sampling.** For outputs above 768 px the pipeline tiles the latent (96×96, overlap 4) and passes no tile
coordinates. SSR re-implements the pipeline's tile enumeration (`tile_regions`, tested against a copy of the
upstream loop), assigns tiles by call order within a timestep, and **verifies every assignment** by comparing the
ControlNet's pixel input with the same slice of the full conditioning image; on any mismatch it raises rather
than apply a misaligned map. The map is cropped from the latent-resolution map, so overlapping tiles see the same
values in their overlap and the pipeline's Gaussian tile blending is unaffected.

**Overhead.** No network pass, no parameter. Extra work: reading one `[1, 4, 4585, 145]` tensor (≈ 10 MB fp32,
released after the rows of S0 are taken), a few operations on a 12×12 grid, and per block a cached `[1, 1, H, W]`
map multiplied into a residual that A's hook already computes. Measured per run as prompt and diffusion time.

**Gate 0 (must pass before any SSR diffusion run).** `experiments/tools/gate0_attention.py` runs DAPE on 50 LR
images. For every upstream tag it occludes the top 20 % of cells by that tag's attention and, as control, the same
cell pattern circularly shifted to random other positions (same size and clustering); occlusion fills a cell with
the image's local mean. d = (logit drop, attended) − (logit drop, control), averaged per image (tags of one image
are not independent). GO requires ≥ 20 images with tags, a 95 % bootstrap CI of the image-level mean above 0, a
one-sided sign-flip p < 0.05 and ≥ 60 % of images with d > 0. Limitations: Swin-L's receptive field is not
confined to a cell, occlusion is a distribution shift (shared by both arms), scene-level tags have no location,
and circular shifts wrap at borders. NO-GO means SSR is not used in the main experiments.

**Screening (if GO).** 20 RealSR + 20 DRealSR images, seed 42, 20 and 50 steps; `original`, `semantic_conf`,
`spatial_sem`, `semantic_spatial`, `spatial_shuffled`. `experiments/tools/ssr_screening.py` continues to the full
evaluation only if `semantic_spatial` beats `semantic_conf` and `spatial_sem` beats `original` on LPIPS and DISTS in
every dataset × step cell (mean and ≥ 60 % of images), and the shuffled control does not reproduce the gain.

**Ablations.** `spatial_sem` (SSR alone) vs `original` and vs `global_meanmatched` (same mean, no arrangement);
`semantic_spatial` vs `semantic_conf` (same mean strength, only *where* differs) and vs `spatial_shuffled` (same
values, wrong places); `combined_spatial` vs `combined`.

**Failure cases.** Diffuse or misplaced attention (scene tags, small objects on a 12×12 grid) makes SSR a random
reweighting — Gate 0 and the shuffled control test this. The least reliable regions are often the most degraded,
where the soft prompt may help most, so lowering it there can blur (watch MUSIQ/CLIPIQA and the low-reliability
regions). Bilinear upsampling of a 12×12 grid may leave soft region boundaries; σ smooths them.

**Relation to prior work.** SRSR (Chen et al., NeurIPS 2025, arXiv:2510.22534) is the closest: it constrains each
tag's text cross-attention to a segmentation mask from Grounded-SAM and disables text CFG in ungrounded pixels
(STCFG). **Spatially restricting semantic conditioning is SRSR's idea.** SSR differs in its source (DAPE's own
tagging attention, no grounding or segmentation model), its form (continuous, confidence-weighted, mean-preserving
redistribution instead of binary masks) and its target (the soft-prompt image-embedding pathway, which SRSR does
not modify). SAM-DiffSR (Wang et al., arXiv:2402.17133) uses segmentation masks for region-specific noise, but only
during training; UPSR (Zhang et al., CVPR 2025) controls noise per region from an uncertainty estimate, with a
trained model. SSR is an incremental, training-free variant; its value is an empirical question that Gate 0 and the
screening decide.
