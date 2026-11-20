# SeeSR experiment framework

Infrastructure for the *Confidence- and Fidelity-Aware Few-Step SeeSR* study. It runs the **unmodified**
upstream SeeSR inference (`test_seesr.main`) as recorded experiments, sweeps the number of DDPM steps and the
method variants, and evaluates the outputs with paired, no-reference and LR-consistency metrics. Nothing here
changes upstream code; the original SeeSR implementation, README and license are untouched (Apache-2.0,
© the SeeSR authors).

The training-free method itself lives in `cfa_seesr/` at the repository root (description, formulations and
related work: `cfa_seesr/README.md`). It is attached at run time through hooks and is **off by default**: with
the `method` section disabled no hook is installed and the run is upstream SeeSR. The repository contains **no
results**; every number you see later comes from files written by these scripts.

## Layout

```text
experiments/
  README.md                 this file
  requirements-colab.txt    inference environment (upstream pins + resolved extras)
  requirements-metrics.txt  pyiqa, installed into the same environment
  common.py                 config loading, sweep expansion, upstream-default extraction, JSON/CSV, provenance
  run_inference.py          wrapper around upstream test_seesr.main(); installs cfa_seesr hooks when enabled
  configs/
    baseline.yaml           original SeeSR settings (each value tagged upstream / exp / user); method disabled
    sampling_sweep.yaml     track B grid: method variants × 50 / 30 / 20 DDPM steps × seeds, per-image seeding
    v2_ablation.yaml        V2 hybrid ablation grid (SFR-OC, heuristic / LORNet reliability), track-B protocol (5b)
    v2_strength_screening.yaml  V2: SFR-OC strength 0.25 / 0.5 / 1.0 on held-out validation images (5b)
    efficiency_sweep.yaml   separate sampler experiment: DDPM vs DDIM vs DPM-Solver++ at fewer steps (section 5c)
    published_seesr.yaml    SeeSR paper values (Table 2), for the reproduction check only
  evaluation/
    evaluate.py             paired (PSNR, SSIM, LPIPS, DISTS), no-reference (MUSIQ, CLIPIQA), LR consistency
    metrics.py              metric registry on top of pyiqa
    lr_consistency.py       LR-consistency diagnostic (bicubic-downsampled SR vs LR input)
    semantic_tags.py        optional RAM tag-agreement check (see caveats)
  tools/
    validate_inputs.py      checkpoint and dataset checks (read-only)
    compare_results.py      per-dataset comparison tables with comparability checks
    analyze_tag_confidence.py   does DAPE's confidence score rank tag reliability? (component A check)
    gate0_attention.py      Gate 0: does DAPE's tagging attention localise its tags? (decides whether SSR is used)
    ssr_screening.py        SSR screening decision (CONTINUE / STOP) from the 40-image screening runs
    benchmark_sfroc.py      V2: SFR-OC / LORNet module cost on synthetic inputs, separate from the pipeline
    sfroc_screening.py      V2: pre-registered strength selection from the screening runs
    sfroc_map_validity.py   V2: do the maps predict where the LR correction lowers the latent error? (V1 outputs)
  training/                 V2: LORNet training (README.md there): data/leakage rules, latent cache, train,
                            evaluate, make_screening_set.py, configs/lornet.yaml, configs/lornet_realsr_v3.yaml
    pilot_report.py         V2.1 single-image pilot: README/CSV check, metadata sanitising, figures
  notebooks/
    seesr_evaluation.ipynb  Colab: GPU check, pinned env, validation, smoke tests, tracks A and B, evaluation, V2
    lornet_training.ipynb   Colab (T4): LORNet cache / train / evaluate, V2 screening images
  scripts/
    run_inference.sh        baseline | sweep, with log files
    run_evaluation.sh       evaluate several finished experiments
  tests/                    CPU-only tests (pytest)
  results/  logs/           outputs (git-ignored except .gitkeep)
```

## 1. Environment

| | |
|---|---|
| Python | **3.10** (upstream `.pyc` files are cpython-310; torch 2.0.1 has no wheels for Python ≥ 3.12, Colab's default) |
| GPU | NVIDIA with CUDA 11.8 driver support. T4 (16 GB) runs the fp16 defaults; L4 / A100 are faster |
| Inference stack | torch 2.0.1+cu118, torchvision 0.15.2, xformers 0.0.20, diffusers 0.21.0, transformers 4.25.0 |
| Extra pins found necessary | `huggingface_hub==0.25.2` (diffusers 0.21 imports `cached_download`, removed in 0.26); `numpy<2`; `setuptools<81` for pyiqa's CLIP dependency |
| Metrics | pyiqa 0.1.10 (the newest version compatible with transformers 4.25 / timm 0.9.12) |

```bash
uv venv --seed --python 3.10 .venv            # or any Python 3.10 virtualenv
.venv/bin/python -m pip install -r experiments/requirements-colab.txt -r experiments/requirements-metrics.txt
.venv/bin/python -m pytest -q experiments/tests   # CPU-only, no weights needed
```

On Colab, run `experiments/notebooks/seesr_evaluation.ipynb` top to bottom; it creates this environment in `/content/seesr-env`.

## 2. Checkpoints (not downloaded automatically)

Upstream code loads weights from `preset/models/` relative to the repository root. The RAM path is hard-coded in
`test_seesr.load_tag_model`, so keep the layout below. On Colab, `preset/models` is symlinked to Drive.

```text
preset/models/
  stable-diffusion-2-base/   diffusers folder (scheduler/, text_encoder/, tokenizer/, unet/, vae/, feature_extractor/)
  seesr/                     unet/, controlnet/            (SeeSR release, links in upstream README)
  DAPE.pth                                                  (SeeSR release)
  ram_swin_large_14m.pth                                    (RAM release on HuggingFace)
```

pyiqa downloads its own metric weights (LPIPS, DISTS, MUSIQ, CLIPIQA) on first use into
`~/.cache/torch/hub/pyiqa`; that needs internet access on the GPU machine.

## 3. Datasets and evaluation protocol

| Dataset | Pairs | HR GT | Metrics | Notes |
|---|---|---|---|---|
| **RealSR** | paired (×4, real camera pairs) | yes | PSNR, SSIM, LPIPS, DISTS + MUSIQ, CLIPIQA | SeeSR follows StableSR: LR center-cropped to 128×128, GT 512×512. Use the StableSR test-set copy linked in the upstream README |
| **DRealSR** | paired (×4) | yes | same as RealSR | same StableSR protocol (128 → 512 crops) |
| **RealLR200** | unpaired | **no** | MUSIQ, CLIPIQA only (set `dataset.gt_dir: null`) | 200 real LR images of varying size from SeeSR; no fidelity claims possible |
| DIV2K-Val (optional) | synthetic paired | yes | all | SeeSR paper: 3K random 512×512 crops degraded with the training pipeline. Degradations are random, so **generate once, store, reuse**; never regenerate between experiments |

Rules for every comparison:

1. **Same images.** The wrapper records the image count and a SHA-256 of the sorted filenames
   (`dataset.filenames_sha256` in `run_metadata.json`); compare only runs with identical hashes.
2. **Same preprocessing.** Upstream `test_seesr.main` handles resizing: it upsamples inputs whose short side is
   under `process_size/upscale` and always resizes the output to `input size × upscale`. Do not pre-resize LR images.
3. **Matching.** SR files are named `<LR stem>.png`. GT files are matched by stem after removing the suffixes in
   `dataset.strip_suffixes` (for example `["_HR"]`). Unmatched or duplicate stems stop the evaluation unless
   `--allow_missing` is given, and they are always listed in the summary.
4. **Same metric settings.** PSNR/SSIM on the Y channel with `crop_border: 0` by default (`evaluation` section of
   `baseline.yaml`). LPIPS uses AlexNet v0.1 (pyiqa default, as in the SeeSR paper).
5. **Sizes must match exactly.** A mismatch is an error for that image; nothing is resized during evaluation.

No-reference scores (MUSIQ, CLIPIQA) estimate perceived quality only. They say nothing about fidelity to the GT
or semantic correctness and can reward hallucinated detail, so never use them alone to claim an improvement.

## 4. Baseline inference

Paths in configs are relative to the repository root. Override anything with `--set key=value`.

```bash
# check paths and print the exact arguments passed to upstream main(); runs nothing
python experiments/run_inference.py --config experiments/configs/baseline.yaml --dry_run

# original SeeSR: DDPM, 50 steps, CFG 5.5, fp16, upstream tiling and AdaIN colour fix
bash experiments/scripts/run_inference.sh baseline \
    --set dataset.lr_dir=preset/datasets/RealSR/test_LR --set dataset.gt_dir=preset/datasets/RealSR/test_HR
```

How it works: `run_inference.py` builds the `argparse.Namespace` that upstream `main()` expects, starting from the
argparse defaults **read from `test_seesr.py` itself**, then applying the config. It wraps the pipeline object and
`get_validation_prompt` to time each image, calls `test_seesr.main(args, enable_xformers_memory_efficient_attention=...)`
unchanged, and writes metadata. Guards that keep the baseline honest:

* `inference.start_steps` other than 999 is rejected, because upstream `main()` never passes it to the pipeline.
* `runtime.scheduler` other than `DDPMScheduler` is rejected; no alternative scheduler has been verified.
* Unknown `inference` keys, missing checkpoints and non-image files in the LR folder fail before models load.
* A non-empty experiment folder is never overwritten without `--overwrite`.
* `runtime.enable_xformers: true` (upstream behaviour) fails if xformers is missing; `auto` falls back to PyTorch
  SDPA (`AttnProcessor2_0`, the diffusers 0.21 default) and records which backend ran. SDPA and xformers are not
  bitwise identical, so compare runs that used the same backend.

Before the first GPU run, check inputs (read-only; exit 2 on any problem):

```bash
python experiments/tools/validate_inputs.py checkpoints --config experiments/configs/baseline.yaml
python experiments/tools/validate_inputs.py dataset --name RealSR --lr_dir <LR> --gt_dir <HR> --scale 4 \
    --output <RESULTS>/RealSR/dataset_report.json
```

The checkpoint check covers required files, git-lfs pointer files, `use_image_cross_attention: true` in both SeeSR
configs, `prediction_type: epsilon`, and the DDPM timesteps the real scheduler config yields for 50 / 30 / 20 steps.
The dataset check covers image counts, sizes and modes, LR↔HR filename matching, HR = 4 × LR, duplicate names or
content, non-image files, and how many LR images upstream will upsample (short side < 128 px).

Each run also records what actually executed: the timesteps passed to `scheduler.step()` per image
(`denoising_steps`, `first_t`, `last_t` in `per_image.csv`), the dtypes and attention-processor classes of the loaded
models (`runtime_verified`), and the seeding mode.

The baseline differs from bare `python test_seesr.py` defaults in two recorded ways: `seed: 42` (upstream is
unseeded) and `save_prompts: true` (writes prompt text files only). `prompt: ""` matches the argparse default.
Note that `scripts/test_seesr.sh` passes `--prompt None`, which inserts the literal word "None" into every prompt;
this framework does not.

## 5. Two tracks: reproduction (A) and controlled comparison (B)

| track | experiments | seeding | used for |
|---|---|---|---|
| **A** reproduction | `baseline_ddpm50` (`baseline.yaml`) | upstream: one generator seeded once (`reseed_per_image: false`) | comparison with the published SeeSR numbers only |
| **B** controlled comparison | `{variant}_ddpm{50,30,20}_s{42,123,2026}` (`sampling_sweep.yaml`) | per-image: `reseed_per_image: true` | step-count and method comparison, paired by (image, seed) |

Track B variants (the `method` section is the only difference between them):

| variant | component A (semantic confidence) | component B (LR guidance schedule) |
|---|---|---|
| `original` | off | constant (upstream) |
| `semantic_conf` | on | constant |
| `lr_schedule` | off | early-strong 1.25 → 0.75 |
| `combined` | on | 1.25 → 0.75 |
| `lr_schedule_late` (not in the default set) | off | 0.75 → 1.25 (direction ablation) |

Spatial semantic reliability (SSR, `cfa_seesr/README.md` §11) adds five variants that are **not** in the default
set and are run only after two gates (section 5a):

| variant | A | B | SSR | purpose |
|---|---|---|---|---|
| `spatial_sem` | off | constant | λ = 1 | SSR alone |
| `semantic_spatial` | on | constant | λ = 1 | vs `semantic_conf`: same mean strength, only *where* differs |
| `combined_spatial` | on | 1.25 → 0.75 | λ = 1 | full method |
| `spatial_shuffled` | on | constant | λ = 1, cells permuted | control: same values, wrong places |
| `global_meanmatched` | off | constant | one scalar = map mean | control for `spatial_sem`: no arrangement |

The tracks are never compared with each other. A single track-A run against a single track-B run says nothing
about seed variation.

**Why track B reseeds.** Upstream seeds one `torch.Generator` once, and `DDPMScheduler.step` draws noise from it at
every step. The noise for image *k* therefore depends on the step counts of all earlier images, so with upstream
seeding a 30-step and a 50-step run start every image after the first from different noise. In track B, before
each image the wrapper sets both RNG sources the pipeline uses to `image_seed(seed, image name)` (a sha256-derived
value): the pipeline generator (initial latents, every DDPM step) and the global torch RNG (the LR latent sample and
`torch.randint` in `pipeline_seesr.py`). An image's noise is then identical at every step count and independent of
image order and of which other images are in the run. `tests/test_rng_pipeline.py` checks this on a tiny instance of
the real pipeline.

```bash
R=<RESULTS>/RealSR            # one results root per dataset: experiment names repeat across datasets
S="--set dataset.lr_dir=<LR> --set dataset.gt_dir=<HR>"
bash experiments/scripts/run_inference.sh baseline --results_root $R $S                 # track A
bash experiments/scripts/run_inference.sh sweep    --results_root $R $S --seeds 42 --variants original  # upstream method first
bash experiments/scripts/run_inference.sh sweep    --results_root $R $S --seeds 42      # default four variants
bash experiments/scripts/run_inference.sh sweep    --results_root $R $S --seeds 123 2026  # expand later
bash experiments/scripts/run_inference.sh sweep    --results_root $R $S --only lr_schedule_late_ddpm20_s42
```

`--variants` defaults to the sweep's `default_variants` (`all` selects every variant); `--only` names exact runs
and bypasses that default. `--dry_run` lists the expanded runs without loading models.

Within a variant and seed, only `num_inference_steps` differs; across seeds, only `seed`; across variants, only
the `method` section. With one seed no seed variation is
estimated; with three seeds the comparison reports it (section 6).

With diffusers 0.21's `DDPMScheduler` and the SD-2-base scheduler config (`steps_offset: 1`, `leading` spacing), the
executed timesteps start at 981 / 958 / 951 for 50 / 30 / 20 steps and all end at 1. The LR latent is always noised
to t = 999, so the gap between that noise level and the first denoising step grows slightly as steps are reduced.
This is upstream behaviour, recorded and not corrected. Fewer steps is a configuration experiment, not a new
sampling method. No alternative scheduler is used.

## 5a. SSR gates: Gate 0 and screening

**Gate 0** (DAPE only, minutes on a GPU) tests whether DAPE's tagging attention localises tag evidence:

```bash
python experiments/tools/gate0_attention.py --images RealSR=<RealSR LR> --images DRealSR=<DRealSR LR> \
    --n_images 50 --output_dir <RESULTS>/gate0
```

For each image and upstream tag: occlude the top 20 % of grid cells by the tag's attention, and, as control, the
same pattern circularly shifted to random positions (8 shifts); compare the tag-logit drops. The image is the
unit of analysis (tags of one image share it). Output: `gate0_per_tag.csv`, `gate0_per_image.csv`,
`gate0_summary.json` (decision, checks, CI, tests) and `overlays/`. **GO** needs ≥ 20 images with tags, a 95 %
bootstrap CI of the image-level mean difference above 0, a one-sided sign-flip p < 0.05 and ≥ 60 % of images
positive; otherwise **NO-GO**, and no SSR variant is run. The docstring lists the masking limitations.

**Screening** (only after GO): 20 RealSR + 20 DRealSR images (hash-selected, symlinked), seed 42, 20 and 50
steps, variants `original`, `semantic_conf`, `spatial_sem`, `semantic_spatial`, `spatial_shuffled`, evaluated
like any run, then

```bash
python experiments/tools/ssr_screening.py --results_root RealSR=<R>/screening/RealSR \
    --results_root DRealSR=<R>/screening/DRealSR --steps 20 50 --seed 42 --output <R>/screening/ssr_screening.json
```

**CONTINUE** to the full SSR evaluation only if, in every dataset × step cell, `semantic_spatial` beats
`semantic_conf` and `spatial_sem` beats `original` on LPIPS and DISTS (mean and ≥ 60 % of images) and the shuffled
control does not reproduce the gain (not consistent, and less than half of `semantic_spatial`'s mean gain);
otherwise **STOP**. The notebook applies both gates automatically.

Useful metrics per component: A, tag F1 (`semantic_tags.py`), LPIPS, DISTS; B, PSNR, SSIM and LR consistency;
SSR, LPIPS and DISTS against `semantic_conf` / `original` and the two controls, with MUSIQ/CLIPIQA to catch
blurring where the map is low.

## 5b. V2: SFR-OC with heuristic or learned (LORNet) reliability

Design, maths and limitations: `cfa_seesr/README.md` §12-13; diagrams: `docs/ARCHITECTURE.md`; related work:
`docs/related_work_v2.md`. Status (2.1.1): step 1 has produced the pilot checkpoint `lornet_v1`; steps 2-4 are
planned and configured but have not been run (see `docs/results.md`). Order of work (each step only after the
previous one):

1. **Train LORNet** (`training/README.md`, `notebooks/lornet_training.ipynb`): training splits only; writes
   `checkpoints/best.pt` and the V2 screening images (its *validation* split; never test images).
2. **Screen the strength** on those images: `configs/v2_strength_screening.yaml` (original, V1, training-free fused
   SFR-OC at γ = 0.25 / 0.5 / 1.0; 20 and 50 steps; seed 42), evaluate, then
   `tools/sfroc_screening.py` (rule fixed in its docstring: largest mean ΔPSNR vs V1 with ΔLPIPS and ΔDISTS
   ≤ +0.005 in every cell; ties within 0.05 dB → smaller γ; none admissible → NONE, report and stop tuning).
3. **Screen the maps**: `tools/sfroc_map_validity.py` on the V1 screening outputs (frozen VAE; counterfactual benefit
   of the low-frequency LR correction; Spearman and quartile contrast with bootstrap CIs over images). If no fused
   map is SUPPORTED, spatial fusion is unsupported and `v2_uniform` stays the candidate.
4. **Ablation** `configs/v2_ablation.yaml` with the selected strength and the one checkpoint:

| # | variant | reliability O | map W | notes |
|---|---|---|---|---|
| 1 | `original` | – | – | SeeSR as released (same run as track B) |
| 2 | `combined_spatial` | – | – | V1 = A + B + SSR (same run as track B) |
| 3 | `v2_heur` | heuristic | O | |
| 4 | `v2_learned` | LORNet | O | |
| 5 | `v2_heur_sem` | heuristic | fuse(O, S) | training-free V2 |
| 6 | `v2_learned_sem` | LORNet | fuse(O, S) | primary hybrid |
| 7 | `v2_uniform` | – | 1 | |
| 8 | `v2_meanmatched` | LORNet | mean of 6's W | same mean strength, no spatial variation |
| 9 | `v2_shuffled` | LORNet | 6's W band-permuted | same values and mean, wrong places |
| 10 | `v2_zero` | LORNet | 6's W | strength 0: must reproduce 2 exactly |

All use Δ = γ·LPF_h(W ⊙ LPF_h(z_lr − x̂₀)). Optional: `v2_global`, `v2_spatial` (band full), `v2_heur_sem_meanmatched`,
`v2_heur_sem_shuffled`, `v2_heur_sem_zero`, `v2_blend_sem`, `v2_sem_only`. Without a checkpoint run the training-free
subset (`--variants original combined_spatial v2_heur v2_heur_sem v2_uniform v2_heur_sem_meanmatched
v2_heur_sem_shuffled v2_heur_sem_zero`); the runner refuses learned variants without a valid checkpoint. If Gate 0 or
the SSR screening said NO-GO, add `--set method.spatial_semantic.enabled=false` and use a separate results folder.

```bash
python experiments/run_inference.py --sweep experiments/configs/v2_ablation.yaml --seeds 42 \
    --results_root $R/RealSR --set dataset.lr_dir=... --set dataset.gt_dir=... \
    --set method.sfr_oc.strength=<selected> --set method.sfr_oc.learned.checkpoint=<best.pt>
python experiments/tools/compare_results.py sampling --results_root $R/RealSR --dataset RealSR \
    --experiments $(ls $R/RealSR | grep -E '^(original|combined_spatial|v2_)')
```

Paired comparisons (per image, bootstrap CIs; `compare_results.py` writes `paired_differences.csv`):
`v2_learned_sem` vs `v2_heur_sem` (learned vs handcrafted), vs `v2_meanmatched` (spatial variation beyond the mean),
vs `v2_shuffled` (placement); `v2_uniform` vs `v2_meanmatched` (mean level); every V2 variant vs V1. Metrics:
PSNR/SSIM and the LR-consistency diagnostic (fidelity), LPIPS/DISTS (perceptual cost), MUSIQ/CLIPIQA (blurring);
`diffusion_s`, `sfroc_*_ms`, peak VRAM and `denoising_steps` per image (cost). Module cost alone:
`python experiments/tools/benchmark_sfroc.py --device cuda --lornet_checkpoint <best.pt> --output $R/benchmark_sfroc.json`.

## 5c. Efficiency experiment (separate): alternative samplers

`configs/efficiency_sweep.yaml` runs the original method with DDPM (50/20/10), DDIM (20/10) and DPM-Solver++ (20/10)
steps, selected by `runtime.sampler_override` (`runtime.scheduler` stays the upstream DDPMScheduler and every main
config keeps `sampler_override: null`). Compatibility with the upstream pipeline (step signature, generator, LRE
`add_noise`, step count, determinism) is tested on the tiny pipeline; quality is what the experiment measures, so no
sampler replaces DDPM without these results. SFR-OC supports DDPM and DDIM only (DPM-Solver++ is rejected).

**Stochastic vs deterministic samplers.** DDPM adds fresh noise at every step; DDIM with η = 0 and DPM-Solver++ are
deterministic given the start latent. With per-image seeding the *start* is identical across samplers, but the
trajectories are not comparable sample by sample: a DDPM–DDIM difference mixes the effect of fewer steps with the
removal of per-step noise (DDPM output varies with the seed, DDIM η = 0 much less). Report each sampler's seed spread
(seeds 42 / 123 / 2026) next to its mean, compare samplers on distributions of metrics rather than on pixels, and do
not read a lower variance of a deterministic sampler as a quality gain. SFR-OC's correction is exact for both (output
space, DDIM with any η), but a V2 gain measured with one sampler is not evidence for the other.

```bash
python experiments/run_inference.py --sweep experiments/configs/efficiency_sweep.yaml --results_root $R/efficiency/RealSR ...
python experiments/tools/compare_results.py sampling --allow_sampler_differences \
    --results_root $R/efficiency/RealSR --dataset RealSR --reference_steps 50
```

## 6. Evaluation

```bash
python experiments/evaluation/evaluate.py --paired --noref \
    --sr_dir experiments/results/baseline_ddpm50/sr/sample00 \
    --gt_dir preset/datasets/RealSR/test_HR \
    --config experiments/configs/baseline.yaml \
    --output_dir experiments/results/baseline_ddpm50/eval

# unpaired data: no-reference metrics only, no GT directory
python experiments/evaluation/evaluate.py --noref --sr_dir <...>/sr/sample00 --output_dir <...>/eval

# LR-consistency diagnostic (works with or without GT)
python experiments/evaluation/evaluate.py --lrc --lr_dir <LR> --scale 4 --sr_dir <...>/sr/sample00 --output_dir <...>/eval

# several experiments at once (LR_DIR adds --lrc)
GT_DIR=preset/datasets/RealSR/test_HR LR_DIR=preset/datasets/RealSR/test_LR bash experiments/scripts/run_evaluation.sh \
    experiments/results/RealSR baseline_ddpm50 original_ddpm50_s42 original_ddpm30_s42 original_ddpm20_s42

# track A: reproduction check against the paper (single run, no error bars)
python experiments/tools/compare_results.py reproduction --results_root experiments/results/RealSR --dataset RealSR

# track B: controlled comparison, paired by (image, seed)
python experiments/tools/compare_results.py sampling --results_root experiments/results/RealSR --dataset RealSR \
    --reference_steps 50 [--experiments original_ddpm50_s42 original_ddpm20_s42 combined_ddpm20_s42 ...]
```

`compare_results.py reproduction` checks that the run used upstream RNG and upstream defaults (apart from the recorded
`seed` and `save_prompts`), fp16, DDPMScheduler and 50 executed steps, and tabulates reproduced vs published values.
It exits 2 if a check fails.

`compare_results.py sampling` first **rejects** the comparison (exit 2, no tables; every difference written to
`sampling_comparison.json`) unless all runs are completed, use per-image seeding, and are identical in everything
except method variant, steps and seed. Without `--experiments` it discovers track-B runs itself and prints every
folder it leaves out, with the reason. That covers image set (names, LR-folder fingerprint, count), checkpoint fingerprints,
precision (configured and the loaded dtypes), attention backend and processors, CFG, scheduler class and config,
preprocessing, tiling, prompts, colour correction, upscale, GPU, package versions, git commit and uncommitted files,
metric settings, GT folder and evaluated-image counts. It also rejects duplicate (variant, steps, seed) runs and a variant whose runs disagree on the method
configuration. If they pass, it writes `sampling_summary.csv`, `sampling_paired_differences.csv`,
`sampling_comparison.{json,md}` with, per variant, step count and metric:

* the mean over all (image, seed) pairs, the std across seeds of the dataset mean, and the **within-image seed
  std** (per image, std across seeds at a fixed step count, averaged over images): the seed-variation estimate,
  available only with at least 2 seeds;
* paired differences against `original` at the reference step count (combined effect of steps and method) and,
  for the other variants, against `original` at the **same** step count (method effect alone): per image, Δ
  averaged over the paired seeds, then mean,
  sd and standard error over images, the share of images that improved, and the per-seed mean Δ with its sd
  across seeds;
* runtime with the warm-up image excluded: diffusion and total seconds per image, total inference time, peak
  allocated and reserved VRAM, step reduction, diffusion-latency change vs the reference, and the mean
  scheduled LR-guidance factor.

No composite score is formed, and no significance test is applied. Pairs within one image across seeds are not
independent, so standard errors are computed over images.

Exit codes: `0` everything evaluated; `2` validation problems (unmatched or duplicate files, size mismatch,
unreadable image, per-image failure), with partial results still written; `3` a metric could not be loaded
(pyiqa missing or weights not downloadable).

**LR consistency (diagnostic).** `--lrc` downsamples each SR image to the LR size with bicubic interpolation (PIL)
and reports PSNR and MAE (RGB, 0–255) against the LR input: `lrc_per_image.csv`, `lrc_summary.json`. It assumes
the LR was produced roughly by bicubic downsampling of the scene, which is false for real degradations (blur,
noise, JPEG), so absolute values are not meaningful and an SR image that reproduces the LR's noise scores well.
Use it only to compare variants on the same images, e.g. to see whether component B moves outputs toward or away
from the LR input. It is never a quality metric.

**Optional semantic check.** `evaluation/semantic_tags.py` tags SR and HR images with the original RAM model
(same preprocessing as `utils_data/make_tags.py`) and reports tag precision, recall, F1 and Jaccard against the HR
tags. HR tags are model predictions, not ground truth. Because DAPE is distilled from RAM, the score is biased toward
SeeSR's own semantic prior, so use it only to compare SeeSR variants with each other. It needs the RAM checkpoint and a GPU.

**Component A check.** `tools/analyze_tag_confidence.py --run <semantic_conf run> --reference <run>/eval/semantic_per_image.csv`
bins DAPE's per-tag confidence `c_i` (from `semantic_confidence.jsonl`) and reports how often RAM on the HR image
predicts the same tag, per bin and for tags kept or dropped by `c_min` (`tag_confidence_bins.csv`,
`tag_confidence_summary.json`). If agreement does not rise with `c_i`, component A's premise fails on that dataset.

## 7. Output files

```text
<results_root>/<dataset>/dataset_report.json      validate_inputs.py dataset
<results_root>/<dataset>/comparison/reproduction.{json,md}                     track A
<results_root>/<dataset>/comparison/sampling_{comparison.json,comparison.md,summary.csv,paired_differences.csv}  track B
<results_root>/gate0/gate0_{per_tag.csv,per_image.csv,summary.json}, overlays/*.png                       Gate 0
<results_root>/screening/<dataset>/<variant>_ddpm{20,50}_s42/, screening/ssr_screening.json               SSR screening
<results_root>/<dataset>/<experiment>/
  run_metadata.json        provenance, config, upstream args, environment, scheduler, runtime_verified, timing, status
  per_image.csv            image, process size, prompt_s, diffusion_s_mean, total_s, denoising_steps, first_t, last_t,
                           seed, peak_vram_mb, peak_vram_reserved_mb, hard_prompt; with a method enabled also
                           n_tags_upstream, n_tags_selected, semantic_reliability, soft_scale,
                           lr_factor_first/last/mean, controlnet_calls; with SSR ssr_mode, ssr_n_tags,
                           ssr_mean/min/max, ssr_clip_low/high, ssr_coverage, ssr_tiled, ssr_controlnet_calls;
                           with SFR-OC sfroc_reference, sfroc_steps_corrected, sfroc_gamma_mean,
                           sfroc_delta_abs_mean, sfroc_W_mean/min/max, sfroc_O_mean, sfroc_S_mean,
                           sfroc_analysis_ms, sfroc_correction_ms, sfroc_encode_ms, sfroc_timing_method,
                           sfroc_reliability, sfroc_O_learned_mean
  semantic_confidence.jsonl   component A only: per image, every upstream tag with its confidence, kept or not
  spatial_reliability.jsonl   SSR only: per image, mode, scalar (if any) and map statistics (mean, clipping, coverage)
  spatial_maps.npz            SSR only: the 12x12 reliability grid per image (after the shuffle control, if any)
  observation_consistency.jsonl  V2 only: per pipeline call, SFR-OC settings, map statistics (O, S, W and every
                              indicator mean), semantic and reference source, overhead and the per-step gamma/Delta log
  observation_maps.npz        V2 only: W, O and the indicator maps per image (heuristic: five indicators at analysis
                              resolution; learned: component_learned = the LORNet map at latent resolution)
<results_root>/v2_screening/<dataset>/..., sfroc_screening.json, map_validity_<dataset>.json     V2 screening
<lornet_output_dir>/cache/, checkpoints/{best,last}.pt, train_log.jsonl, train_summary.json, evaluation.json   LORNet
  sr/sample00/<stem>.png   upstream outputs
  sr/txt/<stem>.txt        prompts (save_prompts)
  eval/paired_per_image.csv   eval/paired_summary.json
  eval/noref_per_image.csv    eval/noref_summary.json
  eval/lrc_per_image.csv      eval/lrc_summary.json          (--lrc)
  eval/tag_confidence_*.{csv,json}                           (tools/analyze_tag_confidence.py)
  eval/semantic_*.{csv,json}  (optional)
```

Summaries contain mean, population standard deviation (ddof=0), min, max and n per metric over successfully
evaluated images, plus the metric settings, package versions, missing files and every problem encountered.

## 8. Reproducibility metadata

`run_metadata.json` records: experiment name, variant and description; the resolved `method` configuration with
`is_baseline`, `semantic_active`, `lr_active` and the mean scheduled LR factor; start and finish timestamps; git commit, branch,
whether tracked files were modified and the list of uncommitted files; the full resolved config; the exact upstream
argument namespace; checkpoint paths, sizes and fingerprints (sha256 of size + first and last 16 MiB of every file;
cheap on Drive, detects different or truncated files, not a full-content hash); dataset name, split, LR/GT
folders, image count, filename hash and LR-folder fingerprint; seeding mode (per-image seeds in `per_image.csv`);
scheduler class, config, scheduled timesteps and the step counts actually passed to `scheduler.step()`; loaded
dtypes and attention processors (`runtime_verified`); attention backend; GPU name, total and free VRAM and compute
capability; Python, PyTorch, CUDA build, cuDNN, diffusers, transformers, accelerate, xformers and pyiqa versions;
timing; status; and missing output files.

**Status.** The metadata is written at the start (`status: running`) and again at the end. A run is `completed`
only if `test_seesr.main()` returned **and** every expected `sr/sampleXX/<stem>.png` exists; otherwise it is
`incomplete: …` or `failed: …`, and the CLI exits non-zero. A session killed mid-run stays `running`. Re-running
with `--overwrite` renames the old folder to `<name>__replaced_<timestamp>` instead of writing into it.

**Timing.** `torch.cuda.synchronize()` is called before and after the prompt step (DAPE tagging and soft-prompt
embedding, `prompt_s`) and around the pipeline call (VAE encode, all denoising steps, VAE decode, `diffusion_s`).
Model loading, image I/O, colour correction, saving and all metric computation are excluded; evaluation runs in a
separate process. Aggregates: per-image means with and without the first (warm-up) image, total prompt + diffusion
time, total diffusion time, wall clock, and peak allocated and reserved VRAM (peak statistics are reset before each
image).

Runs are seeded but **not bitwise deterministic**: cuDNN and xformers kernels and fp16 autocast are not forced into
deterministic mode. Expect small numeric differences between repeated runs and across GPU types.

## 9. Known limitations

* Commit `experiments/` before benchmark runs. Otherwise `git.uncommitted_files` in the metadata is non-empty and
  the commit hash alone does not identify the code that ran.
* The method components have only been tested on CPU with tiny randomly initialised models (behaviour, hook
  installation and removal, baseline equivalence). Whether they load and run with the real weights is checked by
  the notebook smoke test; whether they help is unknown until track B has been run.
* SSR has not been validated on real DAPE weights: whether DAPE's attention localises tags is exactly what Gate 0
  measures, and no Gate 0 result exists yet. The SSR tile tracking is verified per call against the pipeline's
  pixels and tested on the tiny pipeline; the real tiled path (outputs > 768 px, e.g. some RealLR200 images) is
  first exercised on Colab.
* V2 (SFR-OC) has only run on the tiny CPU pipeline: correctness of the maths, the exact no-op, absence of extra
  UNet/ControlNet calls and the tiled path are tested; whether the reference latent and the maps behave sensibly on
  real images, and whether SFR-OC helps, is unknown until the V2 ablation has run. The observation indicators are
  uncalibrated heuristics.
* Not yet run on a GPU. The tests cover config handling, the wrapper (with a stand-in for `test_seesr`), file
  matching, validation, aggregation and real pyiqa PSNR/SSIM on CPU. LPIPS, DISTS, MUSIQ and CLIPIQA weights could
  not be downloaded in the development sandbox, so those four metrics are first exercised on Colab.
* `xformers==0.0.20` is built for torch 2.0.1; whether its kernels load on the Colab GPU is checked in notebook
  section 5. If not, use `runtime.enable_xformers: auto` for every run, so all runs use the same backend.
* Timing covers one warm-up image only; GPU clocks, Drive I/O contention and other Colab tenants add variance that
  repeated seeds do not remove. Compare timings only between runs on the same GPU type (the sampling comparison
  rejects mixed GPUs).
* Seed variation is estimated from at most three seeds, so its sd is itself imprecise. The comparison reports
  standard errors over images but no significance tests or confidence claims.
* `inference.vae_*_tiled_size` and `latent_tiled_*` change outputs; keep them identical across compared runs even
  on larger GPUs.
* Upstream commits its `__pycache__/*.pyc` files. Importing upstream modules under another Python build rewrites
  them and makes the tree look modified (verified: 21 files on Python 3.10). The wrapper, scripts, tests and notebook
  disable bytecode writing. Set `PYTHONDONTWRITEBYTECODE=1` yourself when running upstream scripts directly.
* Upstream quirks are left as is: unused `--blending_alpha`, disconnected `--start_steps`, the `--prompt None`
  shell script and the bug in the `train_seesr.sh` line continuation.
