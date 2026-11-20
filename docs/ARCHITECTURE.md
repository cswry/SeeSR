# CFA-SeeSR architecture: V1 (preserved) and V2.1 (SFR-OC + LORNet)

Legend used in every diagram:

| Style | Meaning |
|---|---|
| dashed grey | **frozen pretrained** networks (SeeSR UNet, ControlNet, DAPE, CLIP, SD2 VAE): never trained, never back-propagated through |
| light grey | upstream SeeSR code and tensors (unmodified) |
| blue | **V1** components (`cfa_seesr/` A, B, SSR), inference-only, training-free |
| orange | **V2** inference-only code (SFR-OC: reliability analysis, fusion, correction) |
| green | **LORNet**, the only trainable component (trained offline; loaded frozen from a validated checkpoint at inference) |
| purple | **training-only** operations (data guards, latent cache, targets, loss, checkpoint selection) |

Sources: `docs/architecture/*.mmd` (Mermaid, version-controlled), rendered `*.svg` next to them (re-render with
`mmdc -i X.mmd -o X.svg -b white`). Every inference component is attached at run time by `cfa_seesr/hooks.py`; no
upstream file is edited, and each component is off unless its config section enables it.

Related work and hypotheses: [`related_work_v2.md`](related_work_v2.md). Method details:
[`../cfa_seesr/README.md`](../cfa_seesr/README.md) (§12 SFR-OC, §13 LORNet). Experiments:
[`../experiments/README.md`](../experiments/README.md); training: [`../experiments/training/README.md`](../experiments/training/README.md).

## Implementation map

| File | Role | Used by |
|---|---|---|
| `cfa_seesr/__init__.py` | `MethodConfig` (all sections), `METHOD_VERSION`, `method_signature` | runner, hooks |
| `cfa_seesr/hooks.py` | attaches / removes every component on a loaded pipeline (wrappers and forward hooks only) | runner |
| `cfa_seesr/semantic_confidence.py` | A: tag confidence, selection, reliability C, soft-prompt scale | hooks, SSR |
| `cfa_seesr/lr_guidance.py` | B: f(t) for the ControlNet residual scale | hooks |
| `cfa_seesr/spatial_reliability.py` | SSR: attention capture, reliability grid, latent-resolution map, tiles | hooks, Gate 0 tool |
| `cfa_seesr/observation_reliability.py` | heuristic O, noise estimate | SFR-OC controller, tools, LORNet evaluation |
| `cfa_seesr/lornet.py` | LORNet model, features, validated checkpoint loading | SFR-OC controller, training |
| `cfa_seesr/latent_targets.py` | upstream-identical resizing, VAE encoding, teacher target, counterfactual benefit | training, map-validity tool |
| `cfa_seesr/frequency.py` | Gaussian / padded-FFT low-pass, two-pass split | SFR-OC |
| `cfa_seesr/observation_consistency.py` | SFR-OC config, fusion and controls, Δ, c₀(t), deferred checks, timers, per-call controller | hooks |
| `experiments/run_inference.py` | config → upstream `test_seesr.main()`, hook installation, metadata | users, notebooks |
| `experiments/training/` | LORNet data guards, cache, training, evaluation, screening-set export | users, notebook |
| `experiments/evaluation/`, `experiments/tools/` | metrics, comparisons, screening, benchmarks, pilot report | users, notebooks |

Interaction per image: the runner calls upstream `test_seesr.main()`; the prompt hook (A, SSR) runs during DAPE;
`prepare_image` triggers the SFR-OC map (heuristic and/or LORNet, fused with SSR's map); every ControlNet call gets
B's factor; every image-attention block gets the soft-prompt scale; every `scheduler.step` gets the SFR-OC
correction; after the call the runner reads each component's summary (deferred checks, timings) into the metadata.

## A. CFA-SeeSR V1 (preserved)

![V1](architecture/cfa_seesr_v1.svg)

Source: [`architecture/cfa_seesr_v1.mmd`](architecture/cfa_seesr_v1.mmd). Only the legend (frozen networks marked)
changed; V1 code paths, configs (`sampling_sweep.yaml`) and run names are unchanged by V2.

| Component | Hook point (code) | Acts on |
|---|---|---|
| A, semantic confidence | `test_seesr.get_validation_prompt` replaced by `cfa_seesr.semantic_validation_prompt` (runner) | tag selection; scalar soft-prompt strength |
| B, LR schedule | `controlnet.forward` wrapper | `conditioning_scale x f(t)` per actual timestep |
| SSR | `image_attentions` forward hooks; `prepare_image` + `controlnet.forward` wrappers (tile tracking) | spatial soft-prompt strength map |

## B. CFA-SeeSR V2 complete inference architecture

![V2](architecture/cfa_seesr_v2.svg)

Source: [`architecture/cfa_seesr_v2.mmd`](architecture/cfa_seesr_v2.mmd).

| Part | File | Runs | Output |
|---|---|---|---|
| Heuristic observation reliability | `observation_reliability.py` | once per pipeline call (`prepare_image` wrapper); noise estimated on the original LR | O_h in [0, 1] + indicator maps |
| Learned observation reliability | `lornet.py` (checkpoint from `experiments/training/`) | once per call, `no_grad`, batch item 0 | O_l in (0, 1) at latent resolution |
| Reliability fusion | `observation_consistency.fuse` | once per call | W at latent resolution (fused / observation / uniform / mean_matched / shuffled) |
| Frequency-selective correction | `observation_consistency.correction_delta`, `x0_coefficient`, `apply_to_step_output`; `scheduler.step` wrapper | every reverse step with γ(t) > 0 | x_{t−1} + c₀(t)·Δ, Δ = γ LPF_h(W ⊙ LPF_h(z_lr − x̂₀)) |

The reference latent z_lr is the posterior **mean** of one VAE encode of the conditioning image (batch item 0),
times `scaling_factor`: the same VAE and scaling as the pipeline's LR-embedded start, deterministic, no extra RNG draw
(`sfr_oc.reference: extra_encode`, the default since 2.1.1, used by the heuristic pilot). Reusing the pipeline's own
LR-start posterior is available as `reference: pipeline` (what the learned pilot used); in the heuristic pilot that
posterior's two CFG rows were not bit-identical and failed the deferred check, which led to the change. With `reliability: learned` the run refuses
to start unless the LORNet checkpoint validates (format, version, configuration, τ, finite weights, strict match).

## C. LORNet internal network

![LORNet](architecture/lornet_network.svg)

Source: [`architecture/lornet_network.mmd`](architecture/lornet_network.mmd). 33 049 parameters and 143 M
multiply-accumulates at 512² with the defaults (counted by `cfa_seesr.lornet.parameter_count` / `conv_macs`).

## D. Offline LORNet training workflow

![LORNet training](architecture/lornet_training.svg)

Source: [`architecture/lornet_training.mmd`](architecture/lornet_training.mmd). Separate from diffusion inference:
only the VAE encoder is loaded (frozen, `no_grad`) to build the latent cache; training reads the cache; gradients
reach LORNet only.

## E. SFR-OC data flow and its place in a reverse step

![SFR-OC data flow](architecture/sfroc_dataflow.svg)

Source: [`architecture/sfroc_dataflow.mmd`](architecture/sfroc_dataflow.mmd).

![SFR-OC in the timestep loop](architecture/sfroc_timesteps.svg)

Source: [`architecture/sfroc_timesteps.mmd`](architecture/sfroc_timesteps.mmd).

The correction sits after classifier-free guidance (and, in the tiled path, tile blending): the unchanged scheduler
step runs on exactly the model output it would have used, on the full latent, and c₀(t)·Δ is added to its result. No
UNet or ControlNet call is added (asserted in `tests/test_sfroc.py`). Nothing is converted to a host value during the
loop; deferred checks and CUDA-event timings are read once after the call.

## V1 versus V2

| | V1 | V2 |
|---|---|---|
| Information used | DAPE logits / attention (A, SSR), timestep (B) | + LR statistics or LORNet's map, and the LR latent in every corrected step |
| Where it acts | prompt selection, soft-prompt residual (A, SSR), ControlNet residual scale (B) | + output of `scheduler.step` (equivalently the predicted clean latent) |
| Frequency selectivity | none | two-pass Gaussian (or padded FFT) low-pass; half amplitude at ≈ 16 latent px ≈ 128 output px for σ = 3 |
| Spatial selectivity | SSR (semantic reliability) | + fused map of observation and semantic reliability, with mean-matched and shuffled controls |
| Extra network passes | none (A saves one DAPE pass) | no UNet/ControlNet pass; one VAE encoder pass per image (reference latent); one LORNet pass per image if learned (33 k parameters) |
| Extra computation | negligible | analysis or LORNet once per image + two low-passes per corrected step (`tools/benchmark_sfroc.py`) |
| Trainable parameters | 0 | 0 (heuristic) or 33 049 (LORNet, trained offline; SeeSR frozen) |
| Default | off | off (`sfr_oc.enabled: false`, `strength: 0`, `reliability: heuristic`) |
| Version recorded | `method.version` 2.1.1, `signature: cfa-seesr-v1` | `signature: cfa-seesr-v2`, formulation string, LORNet path / SHA-256 / τ |

## Configuration map

| Section (`experiments/configs/baseline.yaml` → `method`) | Purpose | Default |
|---|---|---|
| `semantic_confidence` | V1 A | disabled |
| `lr_guidance` | V1 B | disabled |
| `spatial_semantic` | V1 SSR | disabled |
| `sfr_oc` (+ `observation`, `learned`, `fusion`) | V2 SFR-OC | disabled, strength 0, heuristic |
| `runtime.sampler_override` | separate efficiency experiment only | null (upstream DDPMScheduler) |

Sweeps: `sampling_sweep.yaml` (V1 track B, unchanged), `v2_strength_screening.yaml` (V2 strength, validation images),
`v2_ablation.yaml` (V2), `efficiency_sweep.yaml` (samplers). LORNet: `experiments/training/configs/lornet.yaml`.
