# Limitations

**Evidence.** GPU evidence is one image and one seed (`results.md`), from runs that differ in V1 setting, reference
latent and software stack as well as reliability source. The dataset-level, controlled ablation is the next step.

**Training data and generalisation.** LORNet was trained on RealSR V3 Canon and Nikon training images (400 pairs;
32 for validation). Other cameras, DRealSR's sensors, JPEG-heavy web images and synthetic degradations are out of
distribution. Misregistration in the training pairs enters the teacher as "unreliable observation". The selected
epoch rests on a 32-image validation split. The pilot checkpoint was trained on the `Train` split with `exclude_dirs: []`; the
pilot image is the byte-verified StableSR test crop with no byte duplicate among the 800 training files, while the
crop-level overlap search has so far covered 42 of ≈ 400 training LR images (see [results](results.md)). Set
`exclude_dirs` for checkpoints used in the ablation.

**Proxy target.** LORNet learns where LR and HR latents agree. That is evidence that the observation is faithful
there, not proof that correcting the diffusion state toward it improves the output; the counterfactual map test and
the ablation controls (mean-matched, shuffled, uniform) exist to check this and have not been run.

**Calibration.** Neither the heuristic O nor LORNet's output is a calibrated probability. The fusion weights
(0.5 / 0.5), the low-pass width (σ = 3 latent px) and the schedule are initial, untuned choices; only the strength has
a pre-registered selection procedure, which has not been run yet.

**Numerics on GPU.** The upstream script runs the pipeline under CUDA autocast, so parts of SFR-OC's analysis and
filtering execute in fp16 there. The fp16 rounding of the latent bounds how small a per-step correction can be
represented (`precision: output_carry` is an option that carries the remainder; not used in the pilots). The
CUDA-specific code paths (event timing, device placement) are covered by tests that only run with a GPU.

**Reference latent.** The default (and the heuristic pilot) uses one extra VAE encode of the conditioning image as
the reference; reusing the pipeline's own encode failed a strict identical-rows check in that run for reasons not
isolated. The learned pilot reused the pipeline's encode, so the two pilots differ here too.

**Cost.** By construction: no extra UNet/ControlNet pass; one VAE encode and one map computation per image; two
latent low-passes per corrected step. On a T4 the pilots recorded ≈ 13–20 ms of correction over 30 steps; systematic
GPU latency and memory benchmarks are still to be run.

**Scope.** SFR-OC supports DDPM and DDIM only. SeeSR's turbo variant and other samplers are not covered. The SeeSR
backbone is never retrained, so gains are bounded by what inference-time steering can achieve.
