"""CPU-only tests for configuration handling and the inference wrapper (no model loading)."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments import run_inference as ri  # noqa: E402
from experiments.common import load_sweep, load_yaml, set_dotted, upstream_defaults, write_json  # noqa: E402

CONFIGS = ROOT / "experiments" / "configs"
# Values the baseline config intentionally sets differently from upstream argparse defaults.
EXPERIMENT_SPECIFIC = {"seed", "save_prompts"}


def baseline():
    return load_yaml(CONFIGS / "baseline.yaml")


def test_upstream_defaults_parsed():
    d = upstream_defaults()
    assert d["num_inference_steps"] == 50
    assert d["guidance_scale"] == 5.5
    assert d["align_method"] == "adain"
    assert d["start_steps"] == 999
    assert d["save_prompts"] is False


def test_baseline_matches_upstream_defaults():
    d = upstream_defaults()
    inf = baseline()["inference"]
    for key, value in inf.items():
        if key in EXPERIMENT_SPECIFIC:
            continue
        assert value == d[key], f"baseline.yaml inference.{key}={value!r} differs from upstream {d[key]!r}"
    # every upstream inference knob (apart from paths) is stated explicitly
    path_keys = {"seesr_model_path", "ram_ft_path", "pretrained_model_path", "image_path", "output_dir"}
    assert set(d) - path_keys == set(inf)


def test_build_upstream_args(tmp_path):
    args = ri.build_upstream_args(baseline(), tmp_path / "exp")
    assert set(vars(args)) == set(upstream_defaults())
    assert args.num_inference_steps == 50
    assert args.output_dir == str(tmp_path / "exp" / "sr")
    assert Path(args.image_path).is_absolute()


@pytest.mark.parametrize("key,value", [
    ("inference.start_steps", 600),
    ("runtime.scheduler", "DDIMScheduler"),
    ("inference.num_inference_steps", 0),
])
def test_rejected_settings(tmp_path, key, value):
    cfg = baseline()
    set_dotted(cfg, key, value)
    with pytest.raises(ri.ConfigError):
        ri.build_upstream_args(cfg, tmp_path)


def test_unknown_inference_key(tmp_path):
    cfg = baseline()
    cfg["inference"]["not_an_upstream_arg"] = 1
    with pytest.raises(ri.ConfigError):
        ri.build_upstream_args(cfg, tmp_path)


def test_set_dotted_unknown_key():
    with pytest.raises(KeyError):
        set_dotted(baseline(), "inference.typo_steps", 3)


VARIANTS = ["original", "semantic_conf", "lr_schedule", "combined", "lr_schedule_late",
            "spatial_sem", "semantic_spatial", "combined_spatial", "spatial_shuffled", "global_meanmatched"]


def test_sweep_grid_variants_steps_seeds():
    runs = load_sweep(CONFIGS / "sampling_sweep.yaml")
    assert [n for n, _ in runs] == [f"{v}_ddpm{st}_s{sd}" for v in VARIANTS for st in (50, 30, 20)
                                    for sd in (42, 123, 2026)]
    ref = runs[0][1]
    for _, cfg in runs:
        assert cfg["runtime"]["reseed_per_image"] is True
        for section in ("models", "dataset", "runtime", "evaluation", "output"):
            assert cfg[section] == ref[section]
        strip = lambda c: {k: v for k, v in c["inference"].items() if k not in ("num_inference_steps", "seed")}
        assert strip(cfg) == strip(ref)
    # 'original' keeps the baseline method section (both components off)
    base = baseline()
    originals = [c for n, c in runs if c["experiment"]["variant"] == "original"]
    assert all(c["method"] == base["method"] for c in originals)
    # track B differs from the track-A baseline only in seed/steps, the reseed policy and the method
    assert {k: v for k, v in ref["inference"].items() if k != "seed"} == \
        {k: v for k, v in base["inference"].items() if k != "seed"}
    assert {k: v for k, v in ref["runtime"].items() if k != "reseed_per_image"} == \
        {k: v for k, v in base["runtime"].items() if k != "reseed_per_image"}
    assert base["runtime"]["reseed_per_image"] is False


def test_variant_method_sections():
    from cfa_seesr import MethodConfig
    by_variant = {c["experiment"]["variant"]: MethodConfig.from_dict(c["method"])
                  for _, c in load_sweep(CONFIGS / "sampling_sweep.yaml")}
    assert by_variant["original"].is_baseline
    assert by_variant["semantic_conf"].semantic_active and not by_variant["semantic_conf"].lr_active
    assert by_variant["lr_schedule"].lr_active and not by_variant["lr_schedule"].semantic_active
    assert by_variant["combined"].semantic_active and by_variant["combined"].lr_active
    assert by_variant["combined"].semantic_confidence == by_variant["semantic_conf"].semantic_confidence
    assert by_variant["combined"].lr_guidance == by_variant["lr_schedule"].lr_guidance
    late, early = by_variant["lr_schedule_late"].lr_guidance, by_variant["lr_schedule"].lr_guidance
    assert (late.scale_start, late.scale_end) == (early.scale_end, early.scale_start)


def _dry_run_names(capsys, *extra):
    ri.main(["--sweep", str(CONFIGS / "sampling_sweep.yaml"), "--dry_run", *extra])
    out = capsys.readouterr().out
    return [l.split()[1].rstrip(":") for l in out.splitlines() if l.startswith("[dry-run]")]


def test_sweep_filters(capsys):
    assert _dry_run_names(capsys, "--variants", "original", "--seeds", "42") == \
        ["original_ddpm50_s42", "original_ddpm30_s42", "original_ddpm20_s42"]
    default = _dry_run_names(capsys, "--seeds", "42")   # default_variants: the four ablations
    assert len(default) == 12 and not any(n.startswith("lr_schedule_late") for n in default)
    assert not any("spatial" in n or "meanmatched" in n for n in default)   # SSR is never run by default
    assert len(_dry_run_names(capsys, "--variants", "all")) == 90
    assert _dry_run_names(capsys, "--only", "lr_schedule_late_ddpm20_s123") == ["lr_schedule_late_ddpm20_s123"]
    with pytest.raises(ri.ConfigError):
        ri.main(["--sweep", str(CONFIGS / "sampling_sweep.yaml"), "--seeds", "7", "--dry_run"])
    with pytest.raises(ri.ConfigError):
        ri.main(["--sweep", str(CONFIGS / "sampling_sweep.yaml"), "--variants", "nope", "--dry_run"])


def test_invalid_method_section_rejected(tmp_path):
    cfg = baseline()
    cfg["method"]["lr_guidance"]["schedule"] = "cosine"
    with pytest.raises(ri.ConfigError):
        ri.build_upstream_args(cfg, tmp_path)
    cfg = baseline()
    cfg["method"]["semantic_confidence"]["soft_scale_min"] = 1.5   # > soft_scale_max
    with pytest.raises(ri.ConfigError):
        ri.build_upstream_args(cfg, tmp_path)


def test_reseed_requires_seed(tmp_path):
    cfg = baseline()
    cfg["runtime"]["reseed_per_image"] = True
    cfg["inference"]["seed"] = None
    with pytest.raises(ri.ConfigError):
        ri.build_upstream_args(cfg, tmp_path)


def test_dry_run_reports_missing_checkpoints(tmp_path, capsys):
    rc = ri.main(["--config", str(CONFIGS / "baseline.yaml"), "--dry_run", "--results_root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1  # checkpoints and dataset are not present in this checkout
    assert "num_inference_steps = 50" in out
    assert "PROBLEM: models.seesr_model_path not found" in out
    assert not any(tmp_path.iterdir())  # dry run writes nothing


def test_preflight_rejects_non_images(tmp_path):
    lr = tmp_path / "lr"
    lr.mkdir()
    (lr / "a.png").write_bytes(b"")
    (lr / "notes.txt").write_text("x")
    cfg = baseline()
    cfg["dataset"]["lr_dir"] = str(lr)
    _, problems = ri.preflight(cfg, ri.build_upstream_args(cfg, tmp_path / "o"))
    assert any("non-image files" in p for p in problems)


def test_json_metadata_roundtrip(tmp_path):
    data = {"path": tmp_path, "n": 3, "nested": {"x": [1.5, None]}}
    write_json(tmp_path / "m.json", data)
    assert json.loads((tmp_path / "m.json").read_text())["path"] == str(tmp_path)
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"v": math.nan})


def _stub_upstream(steps_log, skip_outputs=()):
    """Stand-in for test_seesr whose main() follows the upstream call order and RNG use."""
    import inspect
    import types

    import torch

    class FakeScheduler:
        config = {"_class_name": "PNDMScheduler", "steps_offset": 1}

        def set_timesteps(self, n):
            self.timesteps = (torch.arange(0, n) * (1000 // n)).flip(0) + 1  # diffusers "leading", steps_offset=1

        def step(self, model_output, timestep, sample, generator=None, return_dict=True):
            return (sample + torch.randn(sample.shape, generator=generator),)

    class FakePipe:
        def __init__(self):
            self.scheduler = FakeScheduler()
            self.unet = types.SimpleNamespace(dtype=torch.float16, config=types.SimpleNamespace(use_image_cross_attention=True))

        def __call__(self, prompt, image, generator=None, num_inference_steps=50, **kw):
            # mirrors pipeline_seesr: generator goes to step() only if its signature accepts it
            accepts = "generator" in inspect.signature(self.scheduler.step).parameters
            self.scheduler.set_timesteps(num_inference_steps)
            x = torch.randn((1, 4), generator=generator)  # initial noise of this image
            start = float(x.sum())
            for t in self.scheduler.timesteps:
                x = self.scheduler.step(x, t, x, **({"generator": generator} if accepts else {}))[0]
            steps_log.append((start, accepts))
            return types.SimpleNamespace(images=["img"])

    stub = types.ModuleType("test_seesr")
    stub.load_seesr_pipeline = lambda args, acc, xf: FakePipe()
    stub.load_tag_model = lambda args, device: "tagger"
    stub.get_validation_prompt = lambda args, image, model: (f"tags of {image}", None)

    def fake_main(args, enable_xformers_memory_efficient_attention=True):
        import glob
        import os
        pipe = stub.load_seesr_pipeline(args, None, enable_xformers_memory_efficient_attention)
        model = stub.load_tag_model(args, "cpu")
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        for name in sorted(glob.glob(f"{args.image_path}/*.*")):
            prompt, _ = stub.get_validation_prompt(args, name, model)
            for s in range(args.sample_times):
                pipe(prompt, None, generator=generator, height=512, width=512,
                     num_inference_steps=args.num_inference_steps)
                stem = os.path.splitext(os.path.basename(name))[0]
                if stem in skip_outputs:
                    continue  # simulate an image that was never saved
                out = f"{args.output_dir}/sample{str(s).zfill(2)}"
                os.makedirs(out, exist_ok=True)
                open(f"{out}/{stem}.png", "wb").close()

    stub.main = fake_main
    return stub


def _stub_setup(tmp_path, monkeypatch, steps_log, skip_outputs=()):
    repo = tmp_path / "repo"
    (repo / "preset" / "models").mkdir(parents=True)
    for d in ("sd2", "seesr"):
        (repo / d).mkdir()
    (repo / "dape.pth").write_bytes(b"x")
    (repo / "preset/models/ram_swin_large_14m.pth").write_bytes(b"x")
    lr = repo / "lr"
    lr.mkdir()
    for n in ("b", "a", "c"):
        (lr / f"{n}.png").write_bytes(b"")
    monkeypatch.setitem(sys.modules, "test_seesr", _stub_upstream(steps_log, skip_outputs))
    monkeypatch.setattr(ri, "REPO_ROOT", repo)
    monkeypatch.chdir(tmp_path)
    cfg = baseline()
    cfg["models"].update(pretrained_model_path=str(repo / "sd2"), seesr_model_path=str(repo / "seesr"),
                         ram_ft_path=str(repo / "dape.pth"),
                         ram_pretrained_path=str(repo / "preset/models/ram_swin_large_14m.pth"))
    cfg["dataset"]["lr_dir"] = str(lr)
    cfg["runtime"]["enable_xformers"] = False
    return cfg


def test_run_one_records_metadata_with_stub_upstream(tmp_path, monkeypatch):
    pytest.importorskip("diffusers")
    log = []
    cfg = _stub_setup(tmp_path, monkeypatch, log)
    cfg["inference"]["num_inference_steps"] = 20
    assert ri.run_one(cfg, tmp_path / "results")

    out = tmp_path / "results" / "baseline_ddpm50"
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["status"] == "completed"
    assert meta["attention_backend"] == "pytorch_sdpa_default"
    assert meta["dataset"]["n_images"] == 3
    assert meta["scheduler"]["denoising_steps_per_call"] == [20]
    assert meta["scheduler"]["scheduled_timesteps"][0] == 951
    assert meta["runtime_verified"]["unet_dtype"] == "torch.float16"
    assert meta["timing"]["images_timed"] == 3
    assert "upstream" in meta["seeding"]
    # the step-counting wrapper must keep `generator` visible to inspect.signature
    assert all(accepts for _, accepts in log)
    rows = list(__import__("csv").DictReader(open(out / "per_image.csv")))
    assert [r["image"] for r in rows] == ["a", "b", "c"]
    assert {r["denoising_steps"] for r in rows} == {"20"} and rows[0]["first_t"] == "951" and rows[0]["last_t"] == "1"
    assert meta["missing_outputs"] == []
    with pytest.raises(ri.ConfigError):  # refuses to overwrite a finished experiment
        ri.run_one(cfg, tmp_path / "results")
    # --overwrite moves the previous run aside instead of writing into it
    assert ri.run_one(cfg, tmp_path / "results", overwrite=True)
    replaced = [d.name for d in (tmp_path / "results").iterdir() if d.name.startswith("baseline_ddpm50__replaced_")]
    assert len(replaced) == 1


def test_missing_outputs_mark_run_incomplete(tmp_path, monkeypatch):
    pytest.importorskip("diffusers")
    cfg = _stub_setup(tmp_path, monkeypatch, [], skip_outputs=("b",))
    with pytest.raises(RuntimeError, match="incomplete"):
        ri.run_one(cfg, tmp_path / "results")
    meta = json.loads((tmp_path / "results" / "baseline_ddpm50" / "run_metadata.json").read_text())
    assert meta["status"].startswith("incomplete") and meta["missing_outputs"] == ["sr/sample00/b.png"]


def test_reseed_seeds_are_recorded_and_step_independent(tmp_path, monkeypatch):
    """Run-level check; the noise itself is tested on the real pipeline in test_rng_pipeline.py."""
    pytest.importorskip("diffusers")
    seeds = {}
    for reseed in (False, True):
        for steps in (50, 20):
            work = tmp_path / f"{reseed}_{steps}"
            cfg = _stub_setup(work, monkeypatch, [])
            cfg["inference"]["num_inference_steps"] = steps
            cfg["runtime"]["reseed_per_image"] = reseed
            ri.run_one(cfg, work / "results")
            rows = list(__import__("csv").DictReader(open(work / "results" / "baseline_ddpm50" / "per_image.csv")))
            seeds[(reseed, steps)] = [r["seed"] for r in rows]
    expected = [str(ri.image_seed(42, n)) for n in ("a", "b", "c")]
    assert seeds[(True, 50)] == seeds[(True, 20)] == expected
    assert set(seeds[(False, 50)]) == {""}


def test_sweep_in_one_process_does_not_stack_hooks(tmp_path, monkeypatch):
    """Several experiments in one process (a --sweep) must each wrap the original upstream functions."""
    pytest.importorskip("diffusers")
    cfg = _stub_setup(tmp_path, monkeypatch, [])
    stub = sys.modules["test_seesr"]
    original = stub.get_validation_prompt
    for steps in (50, 30, 20):
        cfg["inference"]["num_inference_steps"] = steps
        cfg["experiment"]["name"] = f"run{steps}"
        assert ri.run_one(cfg, tmp_path / "results")
        meta = json.loads((tmp_path / "results" / f"run{steps}" / "run_metadata.json").read_text())
        assert meta["scheduler"]["denoising_steps_per_call"] == [steps] and meta["timing"]["images_timed"] == 3
    assert stub.get_validation_prompt is original  # hooks removed after each run


def test_portable_relpath_uses_forward_slashes_on_every_platform():
    """Run metadata must not depend on the OS: Windows paths serialize with '/' like POSIX ones."""
    from pathlib import PurePosixPath, PureWindowsPath

    from experiments.common import portable_relpath
    win = PureWindowsPath(r"C:\results\RealSR\run\sr\sample00\b.png")
    assert portable_relpath(win, PureWindowsPath(r"C:\results\RealSR\run")) == "sr/sample00/b.png"
    assert portable_relpath(win, r"C:\results\RealSR\run") == "sr/sample00/b.png"
    posix = PurePosixPath("/results/RealSR/run/sr/sample01/c.png")
    assert portable_relpath(posix, "/results/RealSR/run") == "sr/sample01/c.png"
    with pytest.raises(ValueError):
        portable_relpath(win, PureWindowsPath(r"C:\other"))


def test_missing_outputs_are_portable_relative_paths(tmp_path):
    (tmp_path / "sr" / "sample00").mkdir(parents=True)
    (tmp_path / "sr" / "sample00" / "a.png").write_bytes(b"x")
    missing = ri.missing_outputs(tmp_path, ["a.png", "b.jpg"], 2)
    assert missing == ["sr/sample00/b.png", "sr/sample01/a.png", "sr/sample01/b.png"]
    assert all("\\" not in m for m in missing)
