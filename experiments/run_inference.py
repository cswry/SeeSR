"""Run original SeeSR inference (upstream ``test_seesr.main``) as a recorded experiment.

The upstream loop is executed unchanged. This wrapper only:
  * builds the argparse Namespace that ``test_seesr.main`` expects from a YAML
    config (missing keys fall back to the upstream argparse defaults),
  * redirects outputs into an experiment-specific directory,
  * wraps the pipeline object and the prompt extractor to time each image,
  * optionally reseeds the RNG per image (track B) and, when the ``method`` section
    enables them, installs the cfa_seesr hooks (removed again after the run),
  * writes ``run_metadata.json`` and ``per_image.csv``.

Examples (run from anywhere; paths in configs are relative to the repo root):
  python experiments/run_inference.py --config experiments/configs/baseline.yaml
  python experiments/run_inference.py --sweep experiments/configs/sampling_sweep.yaml --seeds 42 --variants original
  python experiments/run_inference.py --config experiments/configs/baseline.yaml \
      --set inference.num_inference_steps=30 --name my_run --dry_run
"""
import argparse
import functools
import hashlib
import os
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path

# Upstream tracks __pycache__/*.pyc in git; importing its modules would rewrite those files.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (  # noqa: E402
    REPO_ROOT, environment_info, file_record, git_info, list_images, load_sweep, load_yaml, portable_relpath,
    sweep_default_variants,
    now_iso, parse_override, resolve_path, set_dotted, sha256_of_names, upstream_defaults,
    write_csv, write_json,
)

SUPPORTED_SCHEDULER = "DDPMScheduler"
CONNECTED_START_STEPS = 999  # test_seesr.main() never forwards --start_steps; the pipeline default is 999.


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Config -> upstream Namespace
# --------------------------------------------------------------------------- #
def build_upstream_args(cfg, output_dir):
    """Translate an experiment config into the Namespace used by test_seesr.main()."""
    defaults = upstream_defaults()
    inference = dict(cfg["inference"])
    unknown = sorted(set(inference) - set(defaults))
    if unknown:
        raise ConfigError(f"inference keys not accepted by upstream test_seesr.py: {unknown}")

    if inference.get("start_steps", CONNECTED_START_STEPS) != CONNECTED_START_STEPS:
        raise ConfigError(
            "inference.start_steps is not forwarded to the pipeline by upstream test_seesr.main(); "
            f"only {CONNECTED_START_STEPS} reflects what actually runs. Changing it needs a pipeline "
            "extension, which is outside the baseline framework."
        )
    scheduler = cfg.get("runtime", {}).get("scheduler", SUPPORTED_SCHEDULER)
    if scheduler != SUPPORTED_SCHEDULER:
        raise ConfigError(f"runtime.scheduler={scheduler!r}: only {SUPPORTED_SCHEDULER} (upstream) is supported.")
    if cfg.get("runtime", {}).get("reseed_per_image") and inference.get("seed", defaults["seed"]) is None:
        raise ConfigError("runtime.reseed_per_image requires inference.seed to be set")
    if int(inference.get("num_inference_steps", defaults["num_inference_steps"])) < 1:
        raise ConfigError("inference.num_inference_steps must be >= 1")

    method_config(cfg)  # validate early, also in --dry_run
    models = cfg["models"]
    args = dict(defaults)
    args.update(inference)
    args.update(
        seesr_model_path=str(resolve_path(models["seesr_model_path"])),
        ram_ft_path=str(resolve_path(models["ram_ft_path"])),
        pretrained_model_path=str(resolve_path(models["pretrained_model_path"])),
        image_path=str(resolve_path(cfg["dataset"]["lr_dir"])),
        output_dir=str(Path(output_dir) / "sr"),
    )
    return Namespace(**args)


def method_config(cfg):
    from cfa_seesr import MethodConfig
    try:
        return MethodConfig.from_dict(cfg.get("method"))
    except (TypeError, ValueError) as e:
        raise ConfigError(f"invalid method section: {e}") from e


def preflight(cfg, args):
    """Check inputs before any model is loaded. Returns the list of LR images."""
    problems = []
    models = cfg["models"]
    for key in ("pretrained_model_path", "seesr_model_path", "ram_ft_path"):
        if not resolve_path(models[key]).exists():
            problems.append(f"models.{key} not found: {resolve_path(models[key])}")
    ram_path = resolve_path(models["ram_pretrained_path"])
    if ram_path != REPO_ROOT / "preset/models/ram_swin_large_14m.pth":
        problems.append("models.ram_pretrained_path must stay preset/models/ram_swin_large_14m.pth "
                        "(hard-coded in upstream test_seesr.load_tag_model)")
    elif not ram_path.exists():
        problems.append(f"RAM weights not found: {ram_path}")

    lr_dir = Path(args.image_path)
    if not lr_dir.is_dir():
        problems.append(f"dataset.lr_dir is not a directory: {lr_dir}")
        images = []
    else:
        images = list_images(lr_dir)
        # Upstream globs '<dir>/*.*' and opens every match as an image.
        others = sorted(p.name for p in lr_dir.glob("*.*") if p.is_file() and p not in images)
        if others:
            problems.append(f"dataset.lr_dir contains non-image files that upstream would try to open: {others[:5]}")
        if not images:
            problems.append(f"no images found in {lr_dir}")
    return images, problems


def upstream_image_order(image_path):
    """Exactly the file order used by test_seesr.main()."""
    import glob
    return sorted(glob.glob(f"{image_path}/*.*"))


# --------------------------------------------------------------------------- #
# Timing hooks (no upstream code is modified)
# --------------------------------------------------------------------------- #
def image_seed(base_seed, image_stem, sample_index=0):
    """Per-image seed derived from (run seed, image name, sample index).

    Independent of image order, of which other images are in the run and of the step count, so the
    same image gets the same starting noise in every configuration that uses the same base seed.
    """
    digest = hashlib.sha256(f"{base_seed}:{image_stem}:{sample_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


class Recorder:
    def __init__(self, names, reseed_base=None, method=None):
        from cfa_seesr import MethodConfig
        from cfa_seesr.hooks import MethodState

        self.names = [Path(n).stem for n in names]
        self.rows = []
        self.pipeline = None
        self.reseed_base = reseed_base  # None = upstream behaviour (one RNG stream for all images)
        self.step_timesteps = []        # timesteps seen by scheduler.step() during the current call
        self.method = method or MethodConfig()
        self.method_state = MethodState()
        self.method_hooks = None
        self.semantic_records = []      # per-image tag confidences (Component A), written as JSONL
        self.spatial_records = []       # per-image SSR statistics, written as JSONL
        self.spatial_grids = {}         # image stem -> SSR grid (numpy), written as spatial_maps.npz
        self.tile_overlap = 4           # latent_tiled_overlap of the run (SSR tile tracking)

    def _sync(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start_image(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.rows.append({"image": self.names[len(self.rows)], "prompt_s": None, "diffusion_s": []})

    def peak_vram_mb(self):
        import torch
        if not torch.cuda.is_available():
            return None, None
        return round(torch.cuda.max_memory_allocated() / 2**20), round(torch.cuda.max_memory_reserved() / 2**20)


class TimedPipeline:
    """Forwards everything to the real pipeline, timing each ``__call__``."""

    def __init__(self, pipe, recorder):
        self._pipe = pipe
        self._rec = recorder

    def __call__(self, *a, **k):
        rec = self._rec
        row = rec.rows[-1]
        if rec.reseed_base is not None:
            # RNG consumers in pipeline_seesr: initial latents and every DDPMScheduler.step use the
            # pipeline generator; the LR latent sample (latent_dist.sample) and torch.randint use the
            # global RNG. Reseed both so this image's noise does not depend on any other image.
            import torch
            seed = image_seed(rec.reseed_base, row["image"], len(row["diffusion_s"]))
            k["generator"].manual_seed(seed)
            torch.manual_seed(seed)
            row["seed"] = seed
        rec.step_timesteps = []
        rec.method_state.reset_call()
        rec._sync()
        t0 = time.perf_counter()
        out = self._pipe(*a, **k)
        rec._sync()
        row["diffusion_s"].append(time.perf_counter() - t0)
        row["height"], row["width"] = k.get("height"), k.get("width")
        row["peak_vram_mb"], row["peak_vram_reserved_mb"] = rec.peak_vram_mb()
        row["denoising_steps"] = len(rec.step_timesteps)
        row["first_t"] = rec.step_timesteps[0] if rec.step_timesteps else None
        row["last_t"] = rec.step_timesteps[-1] if rec.step_timesteps else None
        if rec.method.lr_active:
            factors = [f for _, f in rec.method_state.lr_log]
            row["lr_factor_first"], row["lr_factor_last"] = factors[0], factors[-1]
            row["lr_factor_mean"] = sum(factors) / len(factors)
            row["controlnet_calls"] = len(factors)
        if rec.method.spatial_active:
            log = rec.method_state.tile_log
            row["ssr_tiled"] = any(region is not None for _, region in log)
            row["ssr_controlnet_calls"] = len(log)
            res = rec.method_state.spatial_result
            if res is not None and res.mode == "mean_matched":
                row["soft_scale"] = res.soft_scale
        return out

    def __getattr__(self, name):
        return getattr(self._pipe, name)


HOOKED = ("load_seesr_pipeline", "get_validation_prompt", "load_tag_model")


def install_hooks(test_seesr, recorder, timings):
    """Replace three upstream functions with timing wrappers around the ORIGINAL functions.

    The originals are kept on the module the first time, so several experiments in one process
    (a sweep) never wrap each other's wrappers. Call remove_hooks() afterwards.
    """
    if not hasattr(test_seesr, "_experiment_originals"):
        test_seesr._experiment_originals = {n: getattr(test_seesr, n) for n in HOOKED}
    original_load = test_seesr._experiment_originals["load_seesr_pipeline"]
    original_prompt = test_seesr._experiment_originals["get_validation_prompt"]
    original_tag_model = test_seesr._experiment_originals["load_tag_model"]

    def load_pipeline(*a, **k):
        t0 = time.perf_counter()
        pipe = original_load(*a, **k)
        timings["pipeline_load_s"] = time.perf_counter() - t0
        recorder.pipeline = pipe
        original_step = pipe.scheduler.step

        # functools.wraps keeps the signature visible to inspect.signature(); the upstream
        # pipeline uses it to decide whether to pass `generator` and `eta` to step().
        @functools.wraps(original_step)
        def counting_step(model_output, timestep, *sa, **sk):  # counts denoising steps actually executed
            recorder.step_timesteps.append(int(timestep))
            return original_step(model_output, timestep, *sa, **sk)

        pipe.scheduler.step = counting_step
        install_method(pipe, recorder)
        return TimedPipeline(pipe, recorder)

    def load_tag_model(*a, **k):
        t0 = time.perf_counter()
        model = original_tag_model(*a, **k)
        timings["tag_model_load_s"] = time.perf_counter() - t0
        return model

    def get_prompt(*a, **k):
        recorder.start_image()
        recorder._sync()
        t0 = time.perf_counter()
        spatial = None
        if recorder.method.spatial_active:
            import cfa_seesr
            from cfa_seesr.spatial_reliability import SpatialMap
            args, image, model = a[:3]
            method = recorder.method
            # with component A off, its (disabled) settings must not touch the prompt: use the default
            # config, whose selection is exactly upstream's
            sem_cfg = method.semantic_confidence if method.semantic_active else cfa_seesr.SemanticConfidenceConfig()
            prompt, embeds, result, spatial = cfa_seesr.spatial_validation_prompt(
                test_seesr, args, image, model, sem_cfg, method.spatial_semantic,
                key=recorder.names[len(recorder.rows) - 1])
            state = recorder.method_state
            state.spatial_result = spatial
            if spatial.mode == "spatial":
                state.spatial = SpatialMap(spatial.grid, spatial.s_min, spatial.s_max)
                state.soft_scale = float("nan")   # never used while a map is set
            else:
                state.spatial = None
                state.soft_scale = spatial.soft_scale   # scalar (lambda = 0); mean_matched: set at prepare_image
            if not method.semantic_active:
                result = None   # component A is off: do not report it as active
        elif recorder.method.semantic_active:
            import cfa_seesr
            args, image, model = a[:3]
            prompt, embeds, result = cfa_seesr.semantic_validation_prompt(
                test_seesr, args, image, model, recorder.method.semantic_confidence)
            recorder.method_state.soft_scale = result.soft_scale
        else:
            prompt, embeds = original_prompt(*a, **k)
            result = None
        recorder._sync()
        row = recorder.rows[-1]
        row["prompt_s"] = time.perf_counter() - t0
        row["hard_prompt"] = prompt
        if result is not None:
            row.update(n_tags_upstream=result.n_upstream, n_tags_selected=result.n_selected,
                       semantic_reliability=result.reliability, soft_scale=result.soft_scale)
            recorder.semantic_records.append({"image": row["image"], **result.summary(),
                                              "tag_scores": result.tag_scores})
        if spatial is not None:
            st = spatial.stats
            row.update(ssr_mode=spatial.mode, ssr_n_tags=st["n_tags"], ssr_mean=st["mean"], ssr_min=st["min"],
                       ssr_max=st["max"], ssr_clip_low=st["clip_low_frac"], ssr_clip_high=st["clip_high_frac"],
                       ssr_coverage=st["coverage_mean"])
            if spatial.mode == "scalar":
                row["soft_scale"] = spatial.soft_scale
            recorder.spatial_records.append({"image": row["image"], **spatial.summary()})
            if spatial.grid is not None:
                recorder.spatial_grids[row["image"]] = spatial.grid.numpy()
        return prompt, embeds

    test_seesr.load_seesr_pipeline = load_pipeline
    test_seesr.load_tag_model = load_tag_model
    test_seesr.get_validation_prompt = get_prompt


def install_method(pipe, recorder):
    """Attach the enabled method components to a freshly loaded pipeline (no-op for the baseline)."""
    from cfa_seesr.hooks import MethodHooks

    method = recorder.method
    if method.is_baseline:
        return
    hooks = MethodHooks()
    soft_range_is_identity = method.soft_scale_range == (1.0, 1.0)
    if (method.semantic_active or method.spatial_active) and not soft_range_is_identity:
        hooks.install_soft_prompt_scaling(pipe, recorder.method_state)
    if method.lr_active:
        hooks.install_lr_schedule(pipe, method.lr_guidance, recorder.method_state)
    if method.spatial_active and not soft_range_is_identity:
        # installed after the LR wrapper, so it wraps it: the tile is identified before B rescales
        hooks.install_spatial_tracking(pipe, recorder.method_state, recorder.tile_overlap)
    recorder.method_hooks = hooks


def remove_hooks(test_seesr, recorder=None):
    for name, fn in getattr(test_seesr, "_experiment_originals", {}).items():
        setattr(test_seesr, name, fn)
    if recorder is not None and recorder.method_hooks is not None:
        recorder.method_hooks.remove()
        recorder.method_hooks = None


def attention_backend(requested):
    from diffusers.utils.import_utils import is_xformers_available
    available = is_xformers_available()
    if requested == "auto":
        use = available
    else:
        use = bool(requested)
        if use and not available:
            raise ConfigError("runtime.enable_xformers=true but xformers is not importable "
                              "(upstream would raise the same error). Use 'auto' or false.")
    return use, ("xformers" if use else "pytorch_sdpa_default")


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_one(cfg, results_root, overwrite=False, dry_run=False):
    name = cfg["experiment"]["name"]
    out_dir = Path(results_root) / name
    args = build_upstream_args(cfg, out_dir)
    images, problems = preflight(cfg, args)

    if dry_run:
        print(f"[dry-run] {name}: output -> {out_dir}")
        for key in sorted(vars(args)):
            print(f"    {key} = {getattr(args, key)!r}")
        print(f"    images found: {len(images)}")
        for p in problems:
            print(f"    PROBLEM: {p}")
        return not problems

    if problems:
        raise ConfigError("Preflight failed:\n  " + "\n  ".join(problems))
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise ConfigError(f"{out_dir} is not empty; use --overwrite or a new experiment name.")
        # never mix outputs of two runs: keep the old folder under a new name instead of writing into it
        stale = out_dir.with_name(f"{out_dir.name}__replaced_{time.strftime('%Y%m%d_%H%M%S')}")
        out_dir.rename(stale)
        print(f"moved previous outputs to {stale}")
    out_dir.mkdir(parents=True, exist_ok=True)

    use_xformers, backend = attention_backend(cfg.get("runtime", {}).get("enable_xformers", True))

    # Upstream code resolves 'preset/models/...' relative to the working directory.
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    import test_seesr  # noqa: E402  (upstream module, imported unmodified)

    order = upstream_image_order(args.image_path)
    reseed = bool(cfg.get("runtime", {}).get("reseed_per_image", False))
    method = method_config(cfg)
    recorder = Recorder(order, reseed_base=args.seed if reseed else None, method=method)
    recorder.tile_overlap = int(args.latent_tiled_overlap)
    timings = {}
    install_hooks(test_seesr, recorder, timings)

    meta = {
        "experiment": cfg["experiment"],
        "status": "running",
        "started_at": now_iso(),
        "git": git_info(),
        "environment": environment_info(),
        "attention_backend": backend,
        "method": {**method.to_dict(), "is_baseline": method.is_baseline,
                   "semantic_active": method.semantic_active, "lr_active": method.lr_active,
                   "spatial_active": method.spatial_active},
        "seeding": (
            f"reseed_per_image: before each pipeline call the generator and the global RNG are set to "
            f"image_seed(seed={args.seed}, image stem, sample index) (sha256-derived; see per_image.csv)" if reseed else
            f"upstream: one generator seeded once with {args.seed}; per-image noise depends on image order "
            "and on the number of steps of earlier images"
        ),
        "config": cfg,
        "upstream_args": vars(args),
        "checkpoints": {k: file_record(resolve_path(v)) for k, v in cfg["models"].items()},
        "dataset": {
            **cfg["dataset"],
            "n_images": len(order),
            "filenames_sha256": sha256_of_names(order),
            "lr_dir_fingerprint": file_record(args.image_path).get("fingerprint"),
        },
        "determinism": (
            "Seeded torch generator, but not bitwise deterministic: cuDNN/xformers kernels and fp16 "
            "autocast are not forced into deterministic mode."
        ),
    }
    write_json(out_dir / "run_metadata.json", meta)

    t0 = time.perf_counter()
    try:
        test_seesr.main(args, enable_xformers_memory_efficient_attention=use_xformers)
        missing = missing_outputs(out_dir, order, args.sample_times)
        meta["missing_outputs"] = missing
        # a run counts as completed only if every expected output image exists
        meta["status"] = "completed" if not missing else f"incomplete: {len(missing)} output image(s) missing"
    except BaseException as exc:  # record the failure, then re-raise
        meta["status"] = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        wall = time.perf_counter() - t0
        rows = finalize_rows(recorder.rows)
        meta["finished_at"] = now_iso()
        meta["timing"] = summarize_timing(rows, wall, timings)
        if recorder.pipeline is not None:
            sched = recorder.pipeline.scheduler
            steps = [r["denoising_steps"] for r in rows if r["denoising_steps"] is not None]
            meta["scheduler"] = {
                "class": type(sched).__name__,
                "config": {k: v for k, v in dict(sched.config).items()},
                "scheduled_timesteps": [int(t) for t in sched.timesteps],
                "denoising_steps_per_call": sorted(set(steps)),
            }
            meta["runtime_verified"] = runtime_snapshot(recorder.pipeline)
            if method.lr_active:
                from cfa_seesr import mean_factor
                meta["method"]["lr_mean_factor_scheduled"] = mean_factor(
                    sched.timesteps, method.lr_guidance, sched.config.num_train_timesteps - 1)
        remove_hooks(test_seesr, recorder)
        if recorder.semantic_records:
            write_jsonl(out_dir / "semantic_confidence.jsonl", recorder.semantic_records)
        if recorder.spatial_records:
            write_jsonl(out_dir / "spatial_reliability.jsonl", recorder.spatial_records)
            meta["method"]["spatial_summary"] = spatial_summary(recorder.spatial_records)
        if recorder.spatial_grids:
            import numpy as np
            np.savez_compressed(out_dir / "spatial_maps.npz", **recorder.spatial_grids)
        meta["outputs"] = {"sr_dir": str(out_dir / "sr" / "sample00"), "per_image_csv": str(out_dir / "per_image.csv")}
        write_csv(out_dir / "per_image.csv", rows)
        write_json(out_dir / "run_metadata.json", meta)  # written last; status is only final here
    if meta["status"] != "completed":
        raise RuntimeError(f"{name}: {meta['status']}")
    return True


def write_jsonl(path, records):
    import json
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def missing_outputs(out_dir, order, sample_times):
    """Output files test_seesr.main() should have written: sr/sampleXX/<stem>.png.

    Returned relative to out_dir with '/' separators on every OS (portable run metadata).
    """
    missing = []
    for s in range(sample_times):
        for name in order:
            f = Path(out_dir) / "sr" / f"sample{str(s).zfill(2)}" / f"{Path(name).stem}.png"
            if not f.is_file():
                missing.append(portable_relpath(f, out_dir))
    return missing


def runtime_snapshot(pipe):
    """What was actually loaded, read from the live pipeline objects."""
    snap = {}
    for name in ("unet", "controlnet", "vae", "text_encoder"):
        module = getattr(pipe, name, None)
        if module is not None:
            snap[f"{name}_dtype"] = str(getattr(module, "dtype", None))
            snap[f"{name}_class"] = f"{type(module).__module__}.{type(module).__name__}"
    for name in ("unet", "controlnet"):
        procs = getattr(getattr(pipe, name, None), "attn_processors", None)
        if procs:
            snap[f"{name}_attention_processors"] = sorted({type(p).__name__ for p in procs.values()})
    snap["unet_use_image_cross_attention"] = getattr(pipe.unet.config, "use_image_cross_attention", None)
    return snap


METHOD_ROW_FIELDS = ("n_tags_upstream", "n_tags_selected", "semantic_reliability", "soft_scale",
                     "lr_factor_first", "lr_factor_last", "lr_factor_mean", "controlnet_calls",
                     "ssr_mode", "ssr_n_tags", "ssr_mean", "ssr_min", "ssr_max", "ssr_clip_low", "ssr_clip_high",
                     "ssr_coverage", "ssr_tiled", "ssr_controlnet_calls")


def spatial_summary(records):
    """Run-level SSR statistics: how much the map was clipped and how far its mean moved from L."""
    st = [r["stats"] for r in records]
    n = len(st)

    def mean(key):
        vals = [s[key] for s in st if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {"n_images": n, "modes": sorted({r["mode"] for r in records}),
            "n_no_tag_fallback": sum(1 for s in st if s.get("fallback") == "no_tags"),
            "mean_map_mean": mean("mean"), "mean_level": mean("level"),
            "mean_abs_mean_shift": (sum(abs(s["mean"] - s["level"]) for s in st) / n) if n else None,
            "mean_clip_low_frac": mean("clip_low_frac"), "mean_clip_high_frac": mean("clip_high_frac"),
            "mean_coverage": mean("coverage_mean")}


def finalize_rows(rows):
    out = []
    for r in rows:
        diff = r.get("diffusion_s") or []
        out.append({
            "image": r["image"],
            "process_height": r.get("height"),
            "process_width": r.get("width"),
            "prompt_s": _round(r.get("prompt_s")),
            "diffusion_s_mean": _round(statistics.fmean(diff)) if diff else None,
            "n_samples": len(diff),
            "total_s": _round((r.get("prompt_s") or 0) + sum(diff)),
            "denoising_steps": r.get("denoising_steps"),
            "first_t": r.get("first_t"),
            "last_t": r.get("last_t"),
            "seed": r.get("seed"),
            "peak_vram_mb": r.get("peak_vram_mb"),
            "peak_vram_reserved_mb": r.get("peak_vram_reserved_mb"),
            "hard_prompt": r.get("hard_prompt"),
            **{k: _round(r[k]) if isinstance(r[k], float) else r[k] for k in METHOD_ROW_FIELDS if k in r},
        })
    return out


def summarize_timing(rows, wall, timings):
    totals = [r["total_s"] for r in rows if r["total_s"] is not None]
    diffusion = [r["diffusion_s_mean"] * r["n_samples"] for r in rows if r["diffusion_s_mean"] is not None]
    peaks = [r["peak_vram_mb"] for r in rows if r.get("peak_vram_mb") is not None]
    reserved = [r["peak_vram_reserved_mb"] for r in rows if r.get("peak_vram_reserved_mb") is not None]
    return {
        "wall_clock_s": _round(wall),
        **{k: _round(v) for k, v in timings.items()},
        "images_timed": len(totals),
        "per_image_total_s_mean": _round(statistics.fmean(totals)) if totals else None,
        "per_image_total_s_median": _round(statistics.median(totals)) if totals else None,
        "per_image_total_s_mean_excl_first": _round(statistics.fmean(totals[1:])) if len(totals) > 1 else None,
        "sum_per_image_total_s": _round(sum(totals)),
        "per_image_diffusion_s_mean_excl_first": _round(statistics.fmean(diffusion[1:])) if len(diffusion) > 1 else None,
        "sum_diffusion_s": _round(sum(diffusion)),
        "warmup_image_excluded": rows[0]["image"] if len(totals) > 1 else None,
        "peak_vram_allocated_mb_max": max(peaks) if peaks else None,
        "peak_vram_reserved_mb_max": max(reserved) if reserved else None,
        "note": "Timed with torch.cuda.synchronize() before and after each section. prompt_s = DAPE tagging + "
                "soft-prompt embedding; diffusion_s = pipeline call (VAE encode, all denoising steps, VAE decode). "
                "Model loading, image I/O, colour correction, saving and all metric computation are excluded. "
                "The first image includes CUDA warm-up; *_excl_first values drop it.",
    }


def _round(x, nd=4):
    return None if x is None else round(float(x), nd)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="single experiment YAML")
    src.add_argument("--sweep", help="sweep YAML (base_config + experiments list or grid)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. inference.num_inference_steps=30 (repeatable)")
    p.add_argument("--name", help="override experiment.name (single config only)")
    p.add_argument("--results_root", help="override output.results_root")
    p.add_argument("--only", nargs="*", help="with --sweep: run only these experiment names")
    p.add_argument("--seeds", nargs="*", type=int, help="with --sweep: run only runs whose inference.seed is listed")
    p.add_argument("--variants", nargs="*", help="with --sweep: method variants to run (default: the sweep's "
                                                  "default_variants; 'all' for every variant)")
    p.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty experiment dir")
    p.add_argument("--dry_run", action="store_true", help="validate config and paths, print upstream args, run nothing")
    return p.parse_args(argv)


def main(argv=None):
    opts = parse_args(argv)
    if opts.sweep:
        runs = load_sweep(opts.sweep)
        # explicit --only names select runs directly; otherwise default to the sweep's default_variants
        wanted = opts.variants or (None if opts.only else sweep_default_variants(opts.sweep))
        if wanted and wanted != ["all"]:
            known = {c["experiment"]["variant"] for _, c in runs}
            unknown = set(wanted) - known
            if unknown:
                raise ConfigError(f"--variants not in sweep: {sorted(unknown)}; available: {sorted(known)}")
            runs = [(n, c) for n, c in runs if c["experiment"]["variant"] in wanted]
        if opts.only:
            missing = set(opts.only) - {n for n, _ in runs}
            if missing:
                raise ConfigError(f"--only names not in sweep: {sorted(missing)}")
            runs = [(n, c) for n, c in runs if n in opts.only]
        if opts.seeds:
            unknown = set(opts.seeds) - {c["inference"]["seed"] for _, c in runs}
            if unknown:
                raise ConfigError(f"--seeds not in sweep: {sorted(unknown)}")
            runs = [(n, c) for n, c in runs if c["inference"]["seed"] in opts.seeds]
    else:
        cfg = load_yaml(opts.config)
        if opts.name:
            cfg["experiment"]["name"] = opts.name
        runs = [(cfg["experiment"]["name"], cfg)]

    ok = True
    for _, cfg in runs:
        for item in opts.set:
            set_dotted(cfg, *parse_override(item))
        root = resolve_path(opts.results_root or cfg["output"]["results_root"])
        ok = run_one(cfg, root, overwrite=opts.overwrite, dry_run=opts.dry_run) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
