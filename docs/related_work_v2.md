# CFA-SeeSR V2 (SFR-OC + LORNet): related work and hypothesis

Scope: how SFR-OC (`cfa_seesr/observation_reliability.py`, `frequency.py`, `observation_consistency.py`) and LORNet
(`cfa_seesr/lornet.py`, `experiments/training/`) relate to existing work. Statements about other papers are limited
to what their abstracts / published descriptions state; where a detail (for example whether a component is trained)
was not checked against the full paper, the text says so. Nothing here claims that V2 is better than any of them.
No V2 result exists yet.

## What V2 does (two lines)

In every supported reverse step, move the low-frequency part of the predicted clean latent toward the VAE latent of the
upscaled LR image, weighted per position by a fused map of LR-observation reliability and (inverse) semantic
reliability, with a timestep schedule; no extra UNet/ControlNet evaluation, frozen SeeSR. The observation-reliability
map is either a training-free heuristic or LORNet, a 33 k-parameter CNN trained on cached latents of paired data to
predict where the LR and HR latents agree.

## Comparison

| Work | What it does (as published) | Overlap with SFR-OC | Difference |
|---|---|---|---|
| **SeeSR** — Wu et al., CVPR 2024, arXiv:2311.16518 | Degradation-aware prompt extractor (DAPE) supplies hard (tags) and soft (embeddings) semantic prompts; ControlNet carries the LR image; LR-embedded start (LRE) at t = 999. | SFR-OC reuses SeeSR's own VAE latent of the LR image (the LRE encode) as its reference and SeeSR's semantic reliability (V1) in its fusion. | SeeSR uses the LR latent only to initialise sampling; SFR-OC uses it again inside every corrected step, only in a low-frequency band and with a spatial weight. |
| **TFDSR** — Li et al., IJCAI 2025, "A Timestep-Adaptive Frequency-Enhancement Framework for Diffusion-based Image Super-Resolution" | Studies frequency behaviour of diffusion SR over sampling timesteps and enhances different frequency components at different timesteps; needs (minimal) fine-tuning. | Timestep-dependent, frequency-specific intervention. | TFDSR enhances frequency components through trained components; SFR-OC is training-free, acts only on the low band of the predicted clean latent, and pulls it toward the observation rather than enhancing it. |
| **DASP-SR** — Ji, Chen, Xue, Hu, arXiv:2604.11470 (2026), "Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution" | As described in its abstract: Degradation-aware Token Injection (lightweight degradation statistics from the LR input) and Spatially Asymmetric Noise Injection to preserve structural regions. (Where and how each component is trained was not verified here.) | Degradation/structure statistics computed from the LR image; spatially non-uniform treatment of structural regions. | DASP-SR feeds such statistics into the generation process; V2 uses a reliability map (heuristic, or LORNet trained separately with the backbone frozen) only to weight a closed-form latent correction at inference. |
| **OE-Diff** — Zhai et al., Expert Systems with Applications 324 (2026), doi:10.1016/j.eswa.2026.132520 | As described in its abstract: Observation-Embedded Diffusion embeds the observation (LR) into the forward and reverse processes, with closed-form guidance and an inverse-root schedule. (Its training procedure was not verified here.) | Uses the observation explicitly in the reverse process, in closed form, with a schedule. | OE-Diff changes the diffusion formulation itself; V2 keeps SeeSR's trained process and scheduler unchanged and adds c₀(t)·Δ to the output of the existing step. |
| **ILVR** — Choi et al., ICCV 2021, arXiv:2108.02938 | Iterative latent variable refinement: at each step, replace the low-frequency content of the sample with that of the (noised) reference. | **Closest prior idea**: low-pass observation consistency during sampling. | ILVR works in pixel space on the noisy sample with a fixed down/up-sampling filter and no spatial weighting; SFR-OC works on the predicted clean latent of a latent diffusion model, with a configurable band, a timestep schedule and a reliability map. |
| **DDNM** — Wang et al., ICLR 2023, arXiv:2212.00490 | Range-null space decomposition: replace the range-space part of the predicted x0 with the observation's, for known linear degradations. | Correcting the predicted x0 toward the observation inside each step. | DDNM requires a known linear operator; real SR degradations are unknown, so SFR-OC uses a soft, partial low-pass pull in latent space instead. |
| **DiffBIR** — Lin et al., ECCV 2024, arXiv:2308.15070 | Training-free region-adaptive restoration guidance during sampling (gradient of a fidelity loss through the decoder). | Region-adaptive, training-free fidelity guidance. | DiffBIR's guidance needs decoder gradients (extra computation); SFR-OC is a closed-form latent edit with no gradient and no network pass. |
| **StableSR** — Wang et al., IJCV 2024, arXiv:2305.07015 | Time-aware encoder for adaptive LR guidance; controllable feature wrapping. | LR fidelity that varies with timestep (also related to V1's component B). | Trained modules; SFR-OC is training-free and acts on x0, not on features. |

## Learned observation reliability: what exists and what differs

| Line of work | Examples (verified references only) | Relation to LORNet |
|---|---|---|
| Blind degradation estimation | KernelGAN (Bell-Kligler et al., NeurIPS 2019), DAN (Luo et al., NeurIPS 2020): estimate the blur kernel of an LR image | They estimate *how* the image was degraded (a global kernel); LORNet predicts *where* the LR still agrees with the HR image, as a spatial map, in a diffusion model's latent space. It does not estimate a kernel. |
| Degradation-aware conditioning | DAPE in SeeSR; DASP-SR's degradation tokens | Degradation information enters the generator's conditioning; LORNet's output never enters the UNet/ControlNet, it only weights the correction. |
| Observation consistency in sampling | ILVR, DDNM, DiffBIR guidance, OE-Diff | They enforce agreement with the observation (globally, via a known operator, or via gradients); V2 enforces it in the low band and only where a map says the observation is reliable. |

Why a learned estimator at all: the heuristic indicators (edge coherence, band energy, contrast, noise share,
sharpness) measure local *signal activity*. Ringing, JPEG blocks, demosaicking artefacts and misregistration also
produce activity, and a blurred but correct region produces little; none of these tells whether the LR observation is
faithful there. Faithfulness is a property of the LR–HR pair, so it can be supervised on paired data
(`cfa_seesr/README.md` §13). LORNet is deliberately small and fed with the same kind of measurements, so a gain over
the heuristic would come from learning the combination, not from a larger model.

What is **not** claimed: that the latent-agreement target identifies where the correction helps (tested separately by
the counterfactual map-validity tool), that LORNet transfers beyond RealSR/DRealSR camera degradations, or novelty of
any single ingredient.

## What overlaps, and the specific hypothesis

Low-frequency observation consistency (ILVR), x0 correction inside the step (DDNM, DiffBIR's guidance in spirit),
timestep- and frequency-dependent treatment (TFDSR, OE-Diff's scheduling) and degradation/structure statistics from
the LR input (DASP-SR) all exist. **SFR-OC is a combination of these ideas inside a frozen SeeSR; it is not
claimed to be novel on its own.** The hypothesis under test is narrower:

> With few DDPM steps (20–30), structural drift of SeeSR outputs is mostly a low-frequency latent error that a
> spatially weighted pull toward the LR latent can reduce, (i) without adding network evaluations and (ii) without
> removing the high-frequency detail the generative prior adds, and the weighting matters: correcting where the LR
> observation looks reliable and the semantic prior looks unreliable is better than correcting everywhere.

Falsified (per dataset and step count) if, against `combined_spatial` (V1) on the same images and seeds:
* `v2_learned_sem` / `v2_heur_sem` do not improve PSNR/SSIM or the LR-consistency diagnostic beyond the seed spread,
  or improve them only with a clear LPIPS/DISTS/MUSIQ/CLIPIQA loss;
* `v2_learned_sem` is not better than `v2_meanmatched` (spatial variation adds nothing beyond the mean level) or than
  `v2_shuffled` (the arrangement of the map does not matter);
* `v2_learned_sem` is not better than `v2_heur_sem` (learning adds nothing over the heuristic);
* (optional) `v2_global` (all frequencies) is as good as `v2_uniform` (frequency selection adds nothing).
`v2_zero` must equal V1 image for image; any difference is a bug, not a result.

## References

* Wu, R. et al. SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution. CVPR 2024. arXiv:2311.16518.
* Li, Y., Zhao, H., Zhou, J., Xu, G., Hu, T., Chen, G., Wang, H. A Timestep-Adaptive Frequency-Enhancement Framework
  for Diffusion-based Image Super-Resolution. IJCAI 2025. https://www.ijcai.org/proceedings/2025/0168.pdf
* Ji, Y., Chen, Z., Xue, Z., Hu, J. Degradation-Aware and Structure-Preserving Diffusion for Real-World Image
  Super-Resolution. arXiv:2604.11470, 2026. Code: https://github.com/jiyang0315/DASP-SR
* Zhai, Q. et al. OE-Diff: Observation-Embedded Diffusion with Closed-Form Guidance
  and Inverse-Root Scheduling for Real-World Image Super-Resolution. Expert Systems with Applications 324, 2026.
  doi:10.1016/j.eswa.2026.132520
* Choi, J. et al. ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models. ICCV 2021. arXiv:2108.02938.
* Wang, Y., Yu, J., Zhang, J. Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model. ICLR 2023.
  arXiv:2212.00490.
* Lin, X. et al. DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior. ECCV 2024. arXiv:2308.15070.
* Wang, J. et al. Exploiting Diffusion Prior for Real-World Image Super-Resolution (StableSR). IJCV 2024.
  arXiv:2305.07015.
* Bell-Kligler, S., Shocher, A., Irani, M. Blind Super-Resolution Kernel Estimation using an Internal-GAN
  (KernelGAN). NeurIPS 2019. arXiv:1909.06581.
* Luo, Z., Huang, Y., Li, S., Wang, L., Tan, T. Unfolding the Alternating Optimization for Blind Super Resolution (DAN).
  NeurIPS 2020. arXiv:2010.02631.
* Immerkaer, J. Fast Noise Variance Estimation. Computer Vision and Image Understanding 64(2), 1996 (noise indicator).
* Crete-Roffet, F. et al. The Blur Effect: Perception and Estimation with a New No-Reference Perceptual Blur Metric.
  SPIE Human Vision and Electronic Imaging, 2007 (sharpness indicator, in spirit).
