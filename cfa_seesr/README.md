# Confidence- and Fidelity-Aware Few-Step SeeSR

A training-free inference extension of SeeSR (Wu et al., CVPR 2024) with two independent components and
one optional extension:

* **A, confidence-aware semantic conditioning**: tag selection and soft-prompt strength driven by DAPE's
  own margin over its class thresholds;
* **B, LR-fidelity-aware guidance**: a timestep-dependent strength for the ControlNet (LR) residuals;
* **SSR, spatial semantic reliability** (§11): A's soft-prompt strength distributed over the image using
  DAPE's tagging-head attention. Used only if an attention-localisation check (Gate 0) passes.
* **V2, SFR-OC** (§12): frequency-selective observation consistency inside the scheduler step, weighted by a
  fusion of LR-observation reliability and semantic reliability. Architecture diagrams: `docs/ARCHITECTURE.md`.

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
  observation_reliability.py  V2 part A: heuristic, noise-compensated observation-reliability indicators and map O
  frequency.py             V2: Gaussian / padded-FFT low-pass, two-pass split, decomposition, leakage diagnostic
  observation_consistency.py  V2 parts B and C: config, gamma(t), fusion W (+ controls), two-pass Delta,
                           output-space coefficient c0(t), deferred checks, CUDA-event timer, per-call controller
  lornet.py                V2: LORNet model, features, checkpoint format and validated loading
  latent_targets.py        V2: upstream-identical LR resizing, VAE encoding, teacher target, counterfactual benefit
  hooks.py                 forward hooks on image_attentions (scalar or spatial); ControlNet / prepare_image
                           wrappers (LR schedule, SSR tile tracking); clean removal in reverse order
```

Configuration: the `method` section of `experiments/configs/baseline.yaml` (disabled) and the `variants`
in `experiments/configs/sampling_sweep.yaml` (V1) and `v2_ablation.yaml` (V2). Tests: `experiments/tests/test_method.py`,
`test_spatial.py` (V1), `test_sfroc.py`, `test_sfroc_remediation.py`, `test_lornet.py` (V2). LORNet training:
`experiments/training/`.

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


## 12. V2: SFR-OC (frequency-selective observation consistency)

**Purpose.** Keep trustworthy *low-frequency* structure of the LR observation during denoising, where it is
trustworthy, while leaving high-frequency detail to the generative prior. No UNet/ControlNet pass is added; off by
default (`method.sfr_oc.enabled: false`, `strength: 0`). Related work and the precise hypothesis:
`docs/related_work_v2.md` (closest prior idea: ILVR's low-frequency consistency; SFR-OC is not claimed novel on its
own). Formulation version: `cfa_seesr.METHOD_VERSION` (2.1.1), recorded with `signature: cfa-seesr-v2` in every
`run_metadata.json`.

### Part A: observation reliability O (`observation_reliability.py`, heuristic; or LORNet, §13)

The runner passes the **original LR image** (before upstream resizing); the analysis runs on it, so the noise
estimate and every statistic refer to the same pixels. Without it (direct pipeline use) the conditioning image is
area-downsampled by `analysis_scale`. Y = BT.601 luminance; Gaussian windows σ_w = `window`/4.

*Noise.* With the LR (white model): σ_n = 1.4826·median|L|/6 over low-gradient pixels (Immerkær operator L; the mask
is the gradient of G₂∗Y at or below its median, so it follows structure rather than the noise). Each statistic's
noise contribution f_k follows from σ_n² and its filter. Without the LR the resampling makes the noise correlated:
each f_k is measured directly as the median of statistic k over the low-gradient pixels (limitation: an image with no
flat region gets floors that contain signal and is suppressed toward 0).

*Indicators* from compensated statistics x_s = max(x − κ_k f_k, 0), κ_k = 1 + z·(relative std of statistic k under
the noise model; z = `significance` = 2), normalised by an absolute significance ratio
SNR(x_s) = x_s / (x_s + κ_k f_k + a_k) (a_k: fixed reference energy from `reference_level`; no per-image rescaling):

| Indicator | Definition | Meaning |
|---|---|---|
| edge_coherence | SNR(λ₁′)·((λ₁′ − λ₂′)/(λ₁′ + λ₂′ + f + floor))², λ′: compensated structure-tensor eigenvalues | clean oriented edges |
| frequency_energy | SNR of the window energy of G₁∗Y − G₂∗Y | LR still carries band-pass content |
| contrast | SNR of the local variance | local signal contrast |
| noise | (var_s + floor²)/(var_s + σ_n² + floor²), floor = one 8-bit level | share of local variance that is signal |
| sharpness | (√G − √G_b)/(√G + √(κf_G) + floor) from compensated gradient energies of Y_s = G_pre∗Y before/after a σ 1.5 re-blur | low where the LR is already blurred, and for noise |

O = Σ w_k I_k / Σ w_k, Gaussian-smoothed (σ 1 px), resized to latent resolution. Tested property: the mean of O does
not increase with added white noise (averaged over noise draws; flat, textured, edge and mixed scenes), without any
clipping to a reference. O is a heuristic score, **not a probability**. Limitations: the indicators are correlated;
compensation assumes the noise model; JPEG blocks still look like structure.

### Part C: reliability fusion (`observation_consistency.fuse`)

At latent resolution (h, w) = (H/8, W/8); S = SSR map M (if SSR is active), else component A's C, else
`semantic_default` (logged as `semantic_source`):

```
W = clip(w_obs · O + w_sem · (1 − S) + bias, w_min, w_max)      map = fused (default w_obs = w_sem = 0.5)
W = mean(W_fused) everywhere                                    map = mean_matched  (control: same mean, no spatial variation)
W = 1;   W = O;   W = 1 − S                                      map = uniform / observation / semantic
control = shuffled: W permuted in bands (rows, then columns; any size; never the identity; same values and mean)
```

### Part B: correction in the scheduler step (`observation_consistency`, `hooks.install_observation_consistency`)

For `scheduler.step(model_output, t, x_t, ...)`, ᾱ = `alphas_cumprod[t]`, T = num_train_timesteps − 1:

```
γ(t)  = strength · f(t/T)                          f: constant | (t/T)^power | window; 0 outside [t_min, t_max]
x̂₀    = (x_t − √(1−ᾱ) ε̂)/√ᾱ  |  √ᾱ x_t − √(1−ᾱ) v̂  |  model_output        (epsilon | v_prediction | sample)
r     = z_lr − x̂₀                                   z_lr = posterior mean of VAE(2I − 1) · scaling_factor
Δ     = γ(t) · LPF_h(W ⊙ LPF_h(r))                  band low (default);   Δ = γ(t) · W ⊙ r   band full
x_{t−1}' = step(model_output, t, x_t) + c₀(t) · Δ  (= the step with x̂₀ + Δ; precision 'output', default)
   DDPM: c₀ = √ᾱ_prev · β_t / (1 − ᾱ),  β_t = 1 − ᾱ/ᾱ_prev
   DDIM: c₀ = √ᾱ_prev − √(1 − ᾱ_prev − s²) · √ᾱ/√(1 − ᾱ),  s = η·√variance(t, prev)
```

*Two-pass low-pass.* LPF_h is one pass of the documented filter applied twice: Gaussian with σ/√2 per pass
(variances add, so the composite is the documented σ), or the ideal FFT mask with the same cutoff (idempotent). For
uniform W this equals γ·W·LPF(r) (tested); for a spatially varying W, multiplying by W creates high frequencies that
the outer pass removes (tested with the Fourier energy above a threshold, odd and even sizes). The Gaussian is not
band-limited: its amplitude response exp(−2π²σ²f²) halves at a period of ≈ 5.3σ, i.e. **≈ 16 latent px ≈ 128 output
px** for σ = 3 (×8 VAE). The FFT path reflect-pads by half the size before filtering (no wrap-around between borders).

*Output-space application.* DDPM and DDIM are linear in x̂₀ for fixed x_t and noise when clipping and thresholding are
off (enforced), so the unchanged step runs on the **unchanged model output** (same dtype, same RNG draw as V1) and
c₀Δ is added to its result in float32, then cast to the output dtype; `pred_original_sample` becomes x̂₀ + Δ. Tested
equal (float64, fixed generator) to stepping with x̂₀ + Δ for DDPM (all prediction types, including the last step) and
DDIM (η = 0, 0.5, 1). The earlier formulation (rewrite ε̂ so the scheduler re-derives x̂₀ + Δ) is kept as
`precision: model_output`; with fp16 model outputs it amplifies the rounding of ε̂′ by √(1−ᾱ)/√ᾱ (≈ 13× at t = 981).
Measured on random fp16 inputs (`test_sfroc_remediation.py`, |Δ| = 0.03): relative error of the realised displacement
0.18 (output) vs 0.38 (model_output) at t = 951, 0.04 vs 0.09 at t = 501. The remaining error is the final fp16
rounding of x_{t−1}, the same floor V1's own update has; `precision: output_carry` keeps that remainder in float32
and adds it at the next corrected step (error feedback; preserves the accumulated displacement to one rounding;
an option, not the default, because the remainder is added after, not before, the next model evaluation).

*Exact no-op.* Where γ(t) = 0 the original step is called with the original arguments (tested bit-for-bit through
the runner: `v2_zero` / `v2_heur_sem_zero` == `combined_spatial`).

*Safeguards.* Supported: `DDPMScheduler`, `DDIMScheduler`; prediction types epsilon / v_prediction / sample. Rejected
at installation and at configuration time: other schedulers, learned variance (`variance_type: learned*`),
`thresholding: true`, `clip_sample: true`. The reference latent is the posterior mean of one VAE encode of batch
item 0 of the conditioning image (`reference: extra_encode`, default since 2.1.1; one VAE encoder pass per image);
the CFG copies of the conditioning image are checked to be identical before item 0 is used. `reference: pipeline`
reuses the pipeline's own LR-start posterior instead (no extra pass) and checks its CFG rows: in the V2.1 heuristic
pilot (T4, V1 enabled) those rows were not bit-identical (cause not isolated), so it is opt-in. Tiled sampling: the step sees the blended full latent (tested).

*Synchronisation and timing.* During denoising nothing is converted to a Python value: the timestep is read from a
host copy of `scheduler.timesteps` (taken once per call; agreement with the passed tensor is a deferred device check),
diagnostics stay on the device, and all checks (finite maps, identical CFG copies, timestep agreement) are evaluated
once in `summary()` after the call, which raises if one failed. Module time: CUDA events on CUDA (GPU time from the
first to the last SFR-OC kernel; UNet/ControlNet work queued earlier is excluded; read once after the call),
`perf_counter` on CPU; `profile: true` adds explicit synchronisation (benchmarks only). The CUDA path is not
exercised on this development machine (no GPU); the CPU path and the absence of host conversions are tested.

| Parameter | Default (`baseline.yaml`) | V2 ablation | Role |
|---|---|---|---|
| `strength` | 0.0 | screened (0.25 / 0.5 / 1.0), placeholder 0.5 | γ_max |
| `band` | low | low (full in optional variants) | frequency selectivity |
| `lowpass`, `sigma`, `cutoff` | gaussian, 3 latent px (half amplitude at ≈ 128 output px), 0.25 | same | bandwidth |
| `schedule`, `power`, `t_min`, `t_max` | power, 1, 0, 1000 | same | γ(t) = strength·t/T |
| `precision` | output | output | how Δ reaches the step |
| `reliability` | heuristic | heuristic / learned / blend | source of O (§13) |
| `fusion.*` | fused, 0.5 / 0.5, bias 0 | fused / observation / uniform / mean_matched / shuffled | map |
| `observation.*` | window 7, equal weights, z = 2, reference 0.02 | same | part A |

All values are initial and untuned except the strength, which is selected on held-out validation images by the
pre-registered rule of `experiments/tools/sfroc_screening.py`.

**Overhead.** No UNet/ControlNet pass (asserted in tests). Per image: one VAE encoder pass for z_lr, one map analysis
(or one LORNet pass) plus,
per corrected step, two low-passes of a `[1,4,h,w]` latent and element-wise operations. CPU reference on this
development machine (not a GPU claim): 512² image, latent 64²: heuristic analysis ≈ 16 ms, LORNet ≈ 37 ms,
correction ≈ 0.8 ms per step. GPU numbers: `experiments/tools/benchmark_sfroc.py` and `sfroc_*_ms` in `per_image.csv`.

**Expected failure cases.** Pulling toward a degraded LR latent can re-introduce blur, colour casts or JPEG structure
in the low band; the LR latent of a ×4 bicubic upscale is itself smooth, so a full-band correction (v2_global) is
expected to blur; the map may be high exactly where detail should be generated; overlap with the upstream AdaIN colour
fix may make part of the effect redundant. Measure, do not assume.

## 13. V2: LORNet (learned observation reliability)

**Why.** The heuristic indicators measure local signal activity; they cannot tell an LR edge that agrees with the HR
image from one that does not (ringing, JPEG blocks, misregistration and blur all produce "structure"). Whether the LR
observation is *faithful* at a location is a property of the LR–HR pair, so it can be learned from paired data — in
the same latent space in which SFR-OC corrects.

**Supervision** (`latent_targets.py`, `experiments/training/`). On aligned training pairs (RealSR V3 and DRealSR
*training* splits; validation split held out by image; test folders refused; any stem shared with the evaluation
folders is an error): the LR crop is resized exactly as `test_seesr.py` does, both images are encoded by the frozen
SeeSR VAE (posterior mean × scaling factor of 2I − 1), and

```
E = LocalAverage_3( mean_c |LPF(z_lr − z_hr)| )          LPF = the SFR-OC Gaussian (σ = 3 latent px)
R_teacher = exp(−E / τ),   τ = median(E over training crops) / ln 2   (median pixel → 0.5; saved with the model)
L = Charbonnier(σ(logit), R_teacher) + λ_TV · TV(σ(logit))           (λ_TV = 0.05)
```

Charbonnier (robust regression) is the default: the target is a deterministic soft score, not a label, and
large discrepancies come from misregistration as much as from degradation. Implemented alternatives: soft-label BCE
(calibrated-probability view) and a Gaussian NLL with an uncertainty head (heteroscedastic); use them only as
ablations. **The target is a proxy**: agreement of LR and HR latents does not prove that pulling the diffusion state
toward z_lr helps; `experiments/tools/sfroc_map_validity.py` tests that separately (counterfactual latent benefit on
V1 outputs) and the ablation measures it.

**Network** (`lornet.py`): input features y (luminance), gradient magnitude, local contrast, high-frequency energy
(4 channels; `rgb` optional; compact by design — each is one of the heuristic's measurements, so the learned model is
compared to the heuristic on equal information); conv 3×3/2 stem (16) → depthwise-separable stages /4 (24), /8 (32),
/16 (48) with residual DS blocks → multi-dilation context (d = 1, 2, 4) → bilinear ×2, skip from /8, 1×1 fuse,
residual block → 3×3 head → sigmoid at /8 = latent resolution. Any H, W multiple of 8 (internally padded to 16);
GroupNorm, so batch-size independent. **33 049 parameters, 143 M multiply-accumulates at 512²** (counted), ≈ 37 ms
on this CPU at 512² (GPU time: `train_lornet.py evaluate`, `benchmark_sfroc.py`).

**Use in SFR-OC.** `sfr_oc.reliability: learned` (O = LORNet), `blend` (O = b·O_LORNet + (1 − b)·O_heuristic) or
`heuristic`. The checkpoint (`sfr_oc.learned.checkpoint`) is validated before any model is loaded (format, version,
configuration, τ, finite weights, strict state-dict match); without a valid one the run does not start — a map is
never produced by random weights. Its path, SHA-256 and τ are recorded per run. The frozen backbone is unchanged;
nothing is back-propagated through SeeSR (training uses cached latents only).

**Why the backbone stays frozen.** The research question is whether observation reliability can steer an existing
restoration prior; retraining UNet/ControlNet would confound it and is not feasible on a T4. LORNet trains in minutes
to hours on cached latents (estimate; measure with `train_lornet.py`).

**Limitations and distribution shift.** Trained on RealSR/DRealSR camera degradations (Canon/Nikon/DRealSR sensors);
other degradations (RealLR200, JPEG-heavy web images) are out of distribution. RealSR LR images registered at HR size
are downscaled ×4 to produce the network input (as the RealSR test crops are); misregistration in the pairs enters the
target. The teacher is defined with the posterior mean; the pipeline's LR start samples the posterior.
