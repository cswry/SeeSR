# Reproducibility

## Environment

Python 3.10 with the pinned stack in `experiments/requirements-colab.txt` (torch 2.0.1 + CUDA 11.8 wheels,
torchvision 0.15.2, xformers 0.0.20, diffusers 0.21.0, transformers 4.25.0, huggingface_hub 0.25.2, NumPy < 2) and,
for metrics, `experiments/requirements-metrics.txt` (pyiqa). diffusers 0.21 needs huggingface_hub < 0.26; torch 2.0.1
needs NumPy 1.x.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
python -m pip install -r experiments/requirements-colab.txt -r experiments/requirements-metrics.txt
python -m pytest -q experiments/tests          # CPU; no weights needed
```

On Colab, `experiments/notebooks/seesr_evaluation.ipynb` creates the same environment with `uv` (≈ 5 min) and
records it; `experiments/notebooks/lornet_training.ipynb` does the same for training. Recorded environments: the
original and heuristic pilots and all three evaluations used the pinned stack (Python 3.10.22, torch 2.0.1+cu118,
xformers 0.0.20, pyiqa 0.1.10); the learned pilot ran with torch 2.14.1 (CUDA 13) and NumPy 2.2.6 — use the pinned
stack for new runs (`docs/results.md`).

## Weights (not in the repository)

Download from the links in [`upstream_seesr_README.md`](upstream_seesr_README.md) and place under `preset/models/`
(the upstream code hard-codes this path; a symlink to another disk works):

```
preset/models/
├── stable-diffusion-2-base/     scheduler/ text_encoder/ tokenizer/ vae/ feature_extractor/  (diffusers layout)
├── seesr/                       unet/ controlnet/
├── DAPE.pth
└── ram_swin_large_14m.pth
```

`python experiments/tools/validate_inputs.py checkpoints --config experiments/configs/baseline.yaml` checks the
files, git-lfs pointer files and the scheduler configuration (`validate_inputs.py dataset ...` checks a dataset folder). A LORNet checkpoint is produced by `train_lornet.py train`
(`<output_dir>/checkpoints/best.pt`); point `method.sfr_oc.learned.checkpoint` at it.

## Data (not in the repository)

```
datasets/
├── StableSR_testsets/RealSRVal_crop128/{test_LR,test_HR}     RealSR evaluation: 100 pairs, LR 128, HR 512
├── StableSR_testsets/DrealSRVal_crop128/{test_LR,test_HR}    DRealSR evaluation: 93 pairs
├── RealLR200/                                                no ground truth
└── RealSR_V3/{Canon,Nikon}/Train/4/*_HR.png, *_LR4.png       LORNet training only
```

Pass folders with `--set dataset.lr_dir=... --set dataset.gt_dir=...` (inference) or the `data.sources` entries of
the training config. Never list an evaluation folder as a training source; add it to `data.exclude_dirs` so that a
shared image name stops the cache step.

## Commands

```bash
# validate a configuration without running anything
python experiments/run_inference.py --config experiments/configs/baseline.yaml --dry_run

# original SeeSR, one dataset
python experiments/run_inference.py --config experiments/configs/baseline.yaml --name baseline_ddpm50 \
    --set dataset.lr_dir=<LR> --set dataset.gt_dir=<HR> --results_root <results>/RealSR

# V2.1 variants (one seed)
python experiments/run_inference.py --sweep experiments/configs/v2_ablation.yaml --seeds 42 \
    --variants combined_spatial v2_heur_sem v2_learned_sem \
    --set method.sfr_oc.learned.checkpoint=<best.pt> \
    --set dataset.lr_dir=<LR> --set dataset.gt_dir=<HR> --results_root <results>/RealSR

# metrics for one run
python experiments/evaluation/evaluate.py --sr_dir <run>/sr/sample00 --gt_dir <HR> --output_dir <run>/eval \
    --paired --noref --config experiments/configs/baseline.yaml

# paired comparison of several runs
python experiments/tools/compare_results.py sampling --results_root <results>/RealSR --dataset RealSR \
    --experiments <run names> --reference_steps 50

# LORNet
python experiments/training/train_lornet.py cache    --config experiments/training/configs/lornet_realsr_v3.yaml \
    --set data.sources.0.hr_dir=<Canon Train/4> --set data.sources.0.lr_dir=<Canon Train/4> \
    --set data.sources.1.hr_dir=<Nikon Train/4> --set data.sources.1.lr_dir=<Nikon Train/4>
python experiments/training/train_lornet.py train    --config experiments/training/configs/lornet_realsr_v3.yaml
python experiments/training/train_lornet.py evaluate --config experiments/training/configs/lornet_realsr_v3.yaml

# pilot test-set evidence (results.md): SHA-256 against the StableSR test files, the runs' lr_dir_fingerprint,
# byte duplicates among all training files, and a crop search (NCC) in every training LR image; the verdict
# requires full LR coverage (read-only; writes only --out)
python experiments/tools/check_pilot_holdout.py --pilot-lr-dir <pilot LR> --pilot-gt-dir <pilot GT> \
    --test-lr-dir <RealSRVal_crop128/test_LR> --test-hr-dir <RealSRVal_crop128/test_HR> \
    --train-dir <Canon Train/4> --train-dir <Nikon Train/4> \
    --expect-lr-fingerprint aecdd6dd33825d3692b4700bb42c39ad715bc470d37d66e68b1f068a1f775e08 \
    --out pilot_holdout_evidence.json
```

The `--set` overrides of the `cache` step must be repeated for `train` and `evaluate` (or edited into a copy of the
config); `train` refuses a cache built with a different target definition.

## Determinism

* Inference: per-image reseeding in sweeps; with fixed seed, weights and attention backend, reruns on the same GPU
  type give the same images up to non-deterministic CUDA kernels. The zero-strength control (`v2_zero`) must
  reproduce V1 byte for byte on the same machine; compare metrics, not pixels, across machines.
* Training: seeds for Python, NumPy, torch and DataLoader workers; cuDNN deterministic. Some CUDA backward kernels
  used by LORNet (replicate padding, bilinear upsampling) have no deterministic implementation and emit warnings,
  so GPU training is reproducible only approximately; CPU training is exactly repeatable (tested).
* Every run records the method version and signature, the configuration, the environment and (for learned runs) the
  LORNet checkpoint's SHA-256.
