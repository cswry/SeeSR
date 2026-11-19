"""Benchmark SFR-OC's own cost, separately from the SeeSR pipeline (no checkpoint needed).

Times, on synthetic inputs of realistic size, the parts SFR-OC adds per image / per step:
  analysis    heuristic observation-reliability analysis + fusion (once per pipeline call)
  lornet      LORNet forward pass on the conditioning image (once per call; learned reliability only)
  correction  x0 prediction, two-pass low-pass, Delta, c0(t) and the addition to the step output (precision
              'output'; once per corrected step; the scheduler step itself is V1's cost and not included)
with CUDA synchronisation around each timed block (benchmark only; the pipeline hooks use CUDA events).
LORNet timing uses randomly initialised weights unless --lornet_checkpoint is given: the cost of a forward pass
does not depend on the weights; such weights are never used to produce a map. The UNet/ControlNet cost is NOT included: SFR-OC adds no
evaluation of either; its total cost per image is analysis + n_corrected_steps * correction (+ one VAE encode
only with start_point 'noise'). The in-pipeline numbers are logged per run (sfroc_*_ms in per_image.csv).

  python experiments/tools/benchmark_sfroc.py --device cuda --sizes 512 1024 --output <R>/benchmark_sfroc.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _timeit(fn, device, repeats, warmup=2):
    import torch
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        out.append(1000 * (time.perf_counter() - t0))
    return {"median_ms": statistics.median(out), "min_ms": min(out), "max_ms": max(out), "repeats": repeats}


def run(device="cpu", sizes=(512,), repeats=10, dtype="float16", cfg_dict=None, lornet_checkpoint=None):
    import torch
    from diffusers import DDPMScheduler

    from cfa_seesr.lornet import LORNet, conv_macs, load_lornet, parameter_count
    from cfa_seesr.observation_consistency import (SFROCConfig, SFROCController, apply_to_step_output,
                                                   correction_delta, x0_coefficient)
    from experiments.tests.tiny_pipeline import SD2_BASE_SCHEDULER

    cfg = SFROCConfig.from_dict({"enabled": True, "strength": 0.5, **(cfg_dict or {})})
    sched = DDPMScheduler.from_config({k: v for k, v in SD2_BASE_SCHEDULER.items() if k != "_class_name"})
    sched.set_timesteps(50)
    dt = getattr(torch, dtype) if device.startswith("cuda") else torch.float32
    if lornet_checkpoint:
        net, _ = load_lornet(lornet_checkpoint, device)
        lornet_weights = "checkpoint"
    else:
        net = LORNet().to(device).eval()
        lornet_weights = "random (timing only)"
    results = []
    for size in sizes:
        g = torch.Generator(device="cpu").manual_seed(0)
        image = torch.rand(2, 3, size, size, generator=g).to(device)
        h = w = size // 8
        sample = torch.randn(1, 4, h, w, generator=g).to(device, dt)
        eps = torch.randn(1, 4, h, w, generator=g).to(device, dt)
        z_ref = torch.randn(1, 4, h, w, generator=g).to(device)
        S = torch.full((h, w), 0.5, device=device)
        ctrl = SFROCController(cfg, 999, "epsilon")
        analysis = _timeit(lambda: ctrl.build_maps(image, (h, w), S), device, repeats)
        W = ctrl.W
        t = int(sched.timesteps[10])
        a_t = float(sched.alphas_cumprod[t])
        prev = sample.clone()

        def corr():
            delta, x0, _ = correction_delta(eps, sample, a_t, z_ref, W, 0.5, cfg, "epsilon")
            return apply_to_step_output((prev,), delta, x0_coefficient(sched, t), x0)

        step = _timeit(corr, device, repeats)
        with torch.no_grad():
            lor = _timeit(lambda: net(image[0:1]), device, repeats)
        results.append({"image_size": size, "latent_size": h, "analysis": analysis, "lornet": lor,
                        "correction_per_step": step,
                        "per_image_ms_at_50_steps_heuristic": analysis["median_ms"] + 50 * step["median_ms"],
                        "per_image_ms_at_20_steps_heuristic": analysis["median_ms"] + 20 * step["median_ms"],
                        "per_image_ms_at_50_steps_learned": lor["median_ms"] + 50 * step["median_ms"],
                        "lornet_macs": conv_macs(net, size, size)})
    env = {"device": device, "dtype": str(dt), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else None}
    return {"environment": env, "settings": cfg_dict or {}, "results": results,
            "lornet": {"parameters": parameter_count(net), "weights": lornet_weights},
            "note": "module cost only; compare with diffusion_s in per_image.csv of a run on the same GPU"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--dtype", default="float16", help="dtype of model_output/sample on CUDA (CPU uses float32)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="SFR-OC setting, e.g. lowpass=fft")
    p.add_argument("--lornet_checkpoint", default=None)
    p.add_argument("--output")
    a = p.parse_args(argv)
    import torch
    device = a.device if (not a.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    import yaml
    cfg = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    out = run(device, a.sizes, a.repeats, a.dtype, cfg, a.lornet_checkpoint)
    text = json.dumps(out, indent=2)
    print(text)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
