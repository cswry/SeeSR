# Methodology

CFA-SeeSR adds reliability estimates to SeeSR at inference time. Version 1 (V1) acts on the semantic conditioning;
version 2.1 (V2.1) adds an observation-consistency correction inside the sampler, with a heuristic or a learned
estimate of where the LR observation can be trusted. The SeeSR networks are never modified or fine-tuned. This page
gives the reasoning and the equations that matter; the complete derivations, parameter tables and verified code
paths are in [`cfa_seesr/README.md`](../cfa_seesr/README.md) (section numbers below refer to it).

## Starting point: what SeeSR conditions on

For each LR image, DAPE (a RAM-based tagger fine-tuned on degraded inputs) produces tag logits, per-class
thresholds and image embeddings. Tags above threshold form the *hard prompt* (CLIP text); the embeddings form the
*soft prompt*, injected by image-attention blocks in the UNet and ControlNet. The ControlNet also receives the
upscaled LR image, and sampling starts from the noised VAE latent of that image (LR-embedded start at t = 999).
Every passing tag counts equally, the soft prompt always has full strength, and the LR residuals have the same
strength at every timestep (§1–2).

## V1: semantic reliability

**A. Semantic confidence (§3).** Confidence is measured relative to each class's operating point, in log-odds:
c_i = σ((z_i − logit τ_i)/T), so c_i > ½ exactly when the upstream rule selects tag i. Tags with c_i ≤ c_min are
dropped (c_min = ½ reproduces upstream). An image-level reliability C = clip((mean c_i − ½)/½, 0, 1) sets the
soft-prompt strength s = (1 − C)·s_min + C·s_max, applied as h ← h + s·(f(h) − h) in every image-attention block
(s = 1 is the trained model).

**B. Timestep-dependent LR guidance (§4).** The ControlNet residual scale becomes conditioning_scale·f(t) with
f(t) = s_end + (s_start − s_end)(t/T)^γ over the *actual* DDPM timestep, so a given noise level gets the same
factor at 50, 30 or 20 steps. The default schedule (1.25 → 0.75) redistributes guidance toward early steps while
keeping its average near 1.

**SSR. Spatial semantic reliability (§11).** DAPE's tagging head attends from each tag query to a 12 × 12 grid of
image tokens. SSR turns this into a map: each cell's reliability is the confidence-weighted average of the tags that
attend to it, blended with C where few tags attend, smoothed, and centred on C. The soft-prompt strength then varies
over the image: s(x) = (1 − M(x))·s_min + M(x)·s_max. Contrast λ = 0 reduces SSR to component A exactly. A
pre-registered check (Gate 0) tests whether the attention actually localises tags before SSR is used.

## V2.1: frequency-selective observation consistency (SFR-OC)

**Idea.** The LR image constrains the low frequencies of the output well and the high frequencies poorly. SFR-OC
pulls the low-frequency part of the predicted clean latent toward the LR latent during sampling, and only where the
observation is judged reliable and the semantic prior is judged unreliable (§12).

**Correction.** At a reverse step with timestep t, model output, and current latent x_t:

```
x̂₀       = predicted clean latent (from the scheduler's prediction type)
z_lr     = VAE posterior mean of the conditioning image × scaling_factor
Δ        = γ(t) · LPF_h( W ⊙ LPF_h(z_lr − x̂₀) )
x_{t−1}' = scheduler.step(model_output, t, x_t) + c₀(t) · Δ
```

* **Low-pass.** LPF_h is half of a Gaussian low-pass (σ/√2 per pass, σ = 3 latent px overall); filtering again
  after multiplying by W removes the high frequencies that a spatially varying W would create. The Gaussian is a
  smooth roll-off, not a band limit: amplitude halves at a period of ≈ 16 latent px (≈ 128 output px).
* **Exactness.** DDPM and DDIM updates are linear in x̂₀ (clipping and thresholding off, which is enforced), so
  adding c₀(t)·Δ to the step output equals stepping with x̂₀ + Δ. DDPM: c₀ = √ᾱ_prev·β_t/(1 − ᾱ_t); DDIM:
  c₀ = √ᾱ_prev − √(1 − ᾱ_prev − s²)·√(ᾱ_t/(1 − ᾱ_t)). The step still runs on the original model output, so the
  dtype and the random draw are those of V1.
* **Schedule.** γ(t) = strength·(t/T): strongest at high noise, zero at the end. Strength 0 calls the original step
  with the original arguments — an exact no-op, tested bit for bit through the runner.
* **No extra network passes.** The correction uses tensors the step already has. Per image it adds one VAE encoder
  pass (z_lr) and one reliability-map computation.

**Reliability fusion.**

```
W = clip( w_obs · O + w_sem · (1 − S) + bias, w_min, w_max )       default w_obs = w_sem = 0.5
```

O is observation reliability (below), S is SSR's map (or C when SSR is off). Controls defined for the planned ablation (not yet run): W = 1
(uniform), W ≡ mean(W) (mean-matched: same average strength, no spatial variation), and W permuted in bands
(shuffled: same values, wrong places).

## Observation reliability: heuristic and learned

**Heuristic O (`observation_reliability.py`, §12 part A).** Computed on the original LR image. The noise level is
estimated with Immerkær's operator on the flattest pixels; five local statistics (structure-tensor coherence,
band-pass energy, local variance, signal share of the variance, sharpness after light pre-smoothing) are corrected
for their expected noise contribution and normalised by a fixed reference, then averaged. Tested property: added
white noise does not raise the mean of O (averaged over noise draws, flat to textured test scenes). It needs no training, but it measures local *signal activity*, not whether the LR
is *faithful*: ringing, JPEG blocks or misregistration also look like structure.

**Learned O: LORNet (`lornet.py`, §13).** Faithfulness is a property of the LR–HR pair, so it can be supervised:

```
E          = LocalAverage₃( mean_c | LPF(z_lr − z_hr) | )      z from the frozen VAE, same encoding as inference
R_teacher  = exp(−E / τ),  τ = median(E_train) / ln 2           τ from the training split only
L          = Charbonnier( σ(logit), R_teacher ) + 0.05 · TV( σ(logit) )
```

The network takes four cheap per-pixel features (luminance, gradient magnitude, local contrast, high-frequency
energy) of the conditioning image, uses depthwise-separable convolutions down to 1/16 resolution with a
multi-dilation context block, decodes with a skip connection from 1/8, and outputs a sigmoid map at the latent
resolution: 33,049 parameters, 143 M multiply-accumulates at 512 × 512. It sees the same kind of information as
the heuristic, so any gain would come from the learned combination, not from a larger model.

**Training.** Only LORNet is trained (`experiments/training/`). The VAE encodes each training crop once into a
cache; training never touches the UNet, ControlNet or DAPE and nothing is back-propagated through SeeSR. Test
folders are refused as sources, image stems shared with the evaluation folders stop the run, and validation is held
out by image. Selection: lowest validation loss with early stopping. The pilot checkpoint's numbers are in
[`results.md`](results.md).

**Caveat.** The teacher is a proxy: agreement of LR and HR latents does not prove that pulling the diffusion state
toward z_lr improves the output. `experiments/tools/sfroc_map_validity.py` tests that directly on finished V1
outputs (does the map predict where the low-frequency correction lowers the latent error?), and the ablation measures
the effect on images.

## Safeguards built into the implementation

* Unsupported schedulers, learned-variance DDPM, thresholding and sample clipping are rejected before a run.
* CFG copies of the conditioning image are checked identical before batch item 0 is used; all checks are deferred to
  after the pipeline call, so the sampling loop has no host synchronisation.
* LORNet is used only from a validated checkpoint (format, version, configuration, τ, finite weights); without one
  the run does not start.
* Every run records the method version (`2.1.1`), a signature (`upstream`, `cfa-seesr-v1`, `cfa-seesr-v2`) and,
  for learned runs, the checkpoint's SHA-256 and τ.
