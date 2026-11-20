# Experiments

What has been run, how, and what is planned. Operational details (every flag, folder layout, sweep files) are in
[`experiments/README.md`](../experiments/README.md); this page is the protocol summary.

## Completed

| Experiment | Where | Outcome |
|---|---|---|
| CPU test suite | `experiments/tests/` | Colab V2.1 environment, CPU-only: 294 passed, 1 skipped, 1 deselected (`test_dry_run_reports_missing_checkpoints`) |
| LORNet training (`lornet_v1`) | Colab, Tesla T4 | 12 epochs, best epoch 3, validation loss 0.111512 ([`results.md`](results.md)) |
| Single-image pilot, three runs | Colab, Tesla T4 | Canon_001_LR4: original SeeSR, heuristic SFR-OC (with V1), learned SFR-OC (without V1) ([`results.md`](results.md)) |

## Common protocol

* **Runner.** `experiments/run_inference.py` builds the upstream `test_seesr.main()` arguments from a YAML config,
  calls the upstream loop unchanged, installs the CFA-SeeSR hooks when the `method` section enables them, and writes
  `run_metadata.json` (configuration, environment, method version and signature, timings) and `per_image.csv`.
  A run counts as completed only when every output image exists.
* **Inference settings** (`experiments/configs/baseline.yaml`, upstream values unless marked): SD2-base +
  SeeSR ControlNet, DDPMScheduler, `start_point: lr`, CFG 5.5, `process_size` 512, ×4, fp16, AdaIN colour fix,
  tiled VAE / latent settings of upstream; prompt suffix and negative prompt as upstream.
* **Seeds.** `inference.seed: 42` by default. In sweeps (`runtime.reseed_per_image: true`) the generator is reseeded
  per image, so an image's noise does not depend on its position in the folder; additional seeds are 123 and 2026.
* **Metrics** (`experiments/evaluation/evaluate.py`): paired PSNR and SSIM on the Y channel without border crop,
  LPIPS and DISTS (pyiqa); no-reference MUSIQ and CLIPIQA; an LR-consistency diagnostic. The same settings for every
  run that is compared.
* **Statistics** (`experiments/tools/compare_results.py`): per-image paired differences against a reference run,
  with bootstrap confidence intervals, only between runs whose configuration signatures match except for the
  intended difference.

## LORNet training protocol

`experiments/training/train_lornet.py` with `experiments/training/configs/lornet_realsr_v3.yaml`:
RealSR V3 Canon and Nikon *training* folders; validation split by sha256 of the image id (10 %); up to four 512 px
HR crops per image on a regular grid, with the matching LR crop resized exactly as at inference; frozen SD2 VAE
(fp16) for the latent cache; seed 2026; AdamW (lr 2·10⁻³, weight decay 10⁻⁴), cosine schedule over 60 epochs,
batch 16, fp16 autocast; early stopping after 8 epochs without validation improvement; `best.pt` = lowest
validation loss. Evaluation (`train_lornet.py evaluate`) reports agreement with the teacher (MAE, Spearman, ECE)
next to the heuristic map on the same crops.

## Single-image pilot

Three runs on `Canon_001_LR4` (StableSR RealSR test crop, SHA-256 verified — see [results](results.md); 512 × 512
output), each evaluated afterwards with
`evaluate.py --paired` in the pinned environment:

| Run | Method section | Code / environment |
|---|---|---|
| `realsr_single_image_pilot` | none (upstream SeeSR) | `77e664f`; torch 2.0.1, xformers |
| `realsr_heuristic_pilot` | V1 (A + B + SSR) + SFR-OC, `reliability: heuristic`, strength 1.0 | `19ac39e` + runtime fixes; torch 2.0.1, PyTorch attention |
| `realsr_lornet_learned_pilot` | SFR-OC only, `reliability: learned` (`lornet_v1`), strength 1.0 | `77e664f`; torch 2.14.1, PyTorch attention |

Shared: DDPM 30 steps, CFG 5.5, seed 42 (single generator), fp16, LR-embedded start, AdaIN. The full settings are in
each `run_metadata.json` under `assets/pilots/canon_001/`; `python experiments/tools/pilot_report.py check` prints
them. They are observed pilot results; the runs differ in more than the reliability source, so attribution is left to
the ablation below ([`results.md`](results.md)).

## Planned (configured, not yet run on GPU)

Order matters: each step decides whether the next one is worth running.

1. **SFR-OC strength screening** — `experiments/configs/v2_strength_screening.yaml`: γ ∈ {0.25, 0.5, 1.0} on held-out
   validation images (LORNet's validation split, written by `experiments/training/make_screening_set.py`), 20 and 50
   steps, seed 42; selection by the rule fixed in `experiments/tools/sfroc_screening.py` (largest mean ΔPSNR against
   V1 with ΔLPIPS and ΔDISTS ≤ +0.005 in every cell).
2. **Map validity** — `experiments/tools/sfroc_map_validity.py`: Spearman correlation between each map and the
   per-location benefit of the low-frequency correction on V1 outputs, with bootstrap CIs over images.
3. **V2.1 ablation** — `experiments/configs/v2_ablation.yaml`, RealSR (100) and DRealSR (93), 50/30/20 steps:
   original, V1, heuristic O, learned O, heuristic + semantic, learned + semantic, uniform W, mean-matched W,
   shuffled W, zero strength (must equal V1 exactly). Paired comparisons: learned vs heuristic, primary vs
   mean-matched and shuffled controls, every variant vs V1.
4. **Seeds** 123 and 2026 for the cells that matter; **efficiency** (`efficiency_sweep.yaml`) for samplers, kept
   separate from the method comparison.

The V1 evaluation framework (tracks A and B, Gate 0, SSR screening) is configured in the same way
(`sampling_sweep.yaml`, `experiments/README.md` §4–5a).
