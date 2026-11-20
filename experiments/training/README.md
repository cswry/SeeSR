# LORNet training (CFA-SeeSR V2)

LORNet is the only trainable component of CFA-SeeSR. It predicts, at latent resolution, where the LR observation
agrees with the HR image in the frozen SeeSR VAE's latent space. Design and maths: `cfa_seesr/README.md` §13;
network and training diagrams: `docs/ARCHITECTURE.md` (C, D). Colab: `experiments/notebooks/lornet_training.ipynb`.

Nothing of SeeSR is trained or back-propagated through. `cache` loads only the SD2 VAE (frozen, `no_grad`, fp16
on CUDA); `train` and `evaluate` read the cache.

Configurations: `configs/lornet.yaml` (RealSR V3 + DRealSR training splits) and `configs/lornet_realsr_v3.yaml`
(RealSR V3 only: the configuration of the pilot checkpoint `lornet_v1`; outcome in `assets/training/`).

## Data and leakage rules

| Rule | Where enforced |
|---|---|
| Training splits only (RealSR V3 `*/Train/4`, DRealSR `Train_x4`) | `data.discover_pairs` refuses folders named like a test split (`Test`, `test_LR`, `Test_x4`, ...) |
| No evaluation image in training/validation | every stem in `data.exclude_dirs` (the evaluation LR folders) is collected; any overlap is an error |
| Validation held out by image | `data.split_pairs`: sha256 of the pair id, `val_fraction` 0.1; all crops of an image on one side |
| τ from training data only | `train_lornet.py cache`: τ = median(E over training crops) / ln 2, stored in the manifest and checkpoints |
| V2 screening images ⊂ validation split | `make_screening_set.py` (strength and map screening never see test images) |

Pairs are matched by a normalised stem (`_HR`, `_LR4`, `_x4` suffixes removed). An LR registered at the HR size
(RealSR V3 `LR4`) is bicubic-downscaled ×4 to the LR input SeeSR sees (as the 128 px RealSR test crops are); a ¼-size
LR (DRealSR) is used as is. The mode is recorded per split in `cache/manifest.json`.

## Target

For each HR crop (default 512 px, up to 4 per image on an even grid) and its LR crop:

```
cond  = upstream resizing of the LR crop (test_seesr.py: enlarge if small, ×4, floor to 8; bicubic)
z_lr  = VAE.encode(2·cond − 1).latent_dist.mean · scaling_factor        (same for z_hr from the HR crop)
E     = LocalAverage_3( mean_c |LPF(z_lr − z_hr)| ),  LPF = Gaussian σ 3 latent px (the SFR-OC low-pass)
R     = exp(−E / τ)
```

Cached per crop: the LR crop (uint8) and E (float16); the conditioning image is rebuilt on the fly. The SeeSR
pipeline encodes through its tiled-VAE wrapper, which does not tile images up to 1024 px, so a 512 px crop is encoded
identically.

## Model and loss

`cfa_seesr/lornet.py`: features (y, gradient, local contrast, HF energy) → depthwise-separable multi-scale encoder
(/2 … /16) with dilated context → decoder with a skip from /8 → sigmoid map at /8. 33 049 parameters, 143 M MACs at
512² (counted). Loss `charbonnier` (default; robust regression) + `tv_weight`·TV; `bce` (soft labels) and
`gaussian_nll` (with `model.uncertainty: true`) are implemented as ablations. AdamW (lr 2e-3, wd 1e-4), cosine
schedule, fp16 autocast on CUDA, early stopping on the validation loss (patience 8), `best.pt` = lowest validation
loss, `last.pt` = last epoch. Seeds: Python, NumPy, torch, DataLoader generator and workers; cuDNN deterministic.

## Commands

```bash
CFG=experiments/training/configs/lornet.yaml          # edit paths, or pass --set key=value (list items by index)
python experiments/training/train_lornet.py cache    --config $CFG --set output_dir=/path/lornet \
       --set vae.pretrained_model_path=preset/models/stable-diffusion-2-base
python experiments/training/train_lornet.py train    --config $CFG --set output_dir=/path/lornet
python experiments/training/train_lornet.py evaluate --config $CFG --set output_dir=/path/lornet
python experiments/training/make_screening_set.py    --config $CFG --set output_dir=/path/lornet \
       --n 20 --crop_hr 512 --output /path/v2_screening_data
```

Outputs: `cache/` (manifest, train.pt, val.pt), `checkpoints/{best,last}.pt`, `train_log.jsonl`,
`train_summary.json` (best epoch, parameters, MACs), `evaluation.json`.

## Evaluation (validation split)

`evaluate` reports, for LORNet against the teacher: MAE, RMSE, Pearson, pooled and per-crop Spearman (with a
bootstrap CI over crops), ECE (10 bins), and E(bottom predicted quartile) − E(top quartile) (> 0: high predicted
reliability goes with low discrepancy). The heuristic map O on the same crops gets the rank metrics only (it is not
calibrated). Plus parameters, MACs and inference time at 512² on the device used.

These numbers measure agreement with the proxy target. Whether the learned map helps SR is tested by
`experiments/tools/sfroc_map_validity.py` (counterfactual latent benefit on V1 outputs of the screening images) and by
the V2 ablation (`v2_learned_sem` vs `v2_heur_sem`, `v2_meanmatched`, `v2_shuffled`).

## Cost (estimates, not measured on a GPU here)

T4: the cache step is one VAE encode per crop pair (≈ 2 × 0.1 s at 512² fp16 → minutes for ~2 k crops); training a
33 k-parameter network on 512² crops at batch 16 is dominated by data loading and feature computation (minutes per
epoch). Memory: well under 4 GB. Measure with the logs (`train_log.jsonl` seconds per epoch).

## Known limitations

* The teacher is a proxy for observation fidelity, defined with the posterior mean and the SFR-OC low-pass.
* Misregistration in the training pairs enters E as "unreliable observation".
* Trained on RealSR/DRealSR camera degradations; RealLR200 and web images are out of distribution.
* Small validation split (by image); confidence intervals are over crops, which are correlated within an image.
