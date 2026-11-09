"""Tests for validate_inputs.py and compare_results.py on synthetic files (CPU only)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.tools import compare_results as cr  # noqa: E402
from experiments.tools import validate_inputs as vi  # noqa: E402
from experiments.common import write_csv, write_json  # noqa: E402


def img(path, w, h, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((h, w, 3), value, np.uint8)).save(path)


# --------------------------------------------------------------------------- #
# validate_inputs: dataset
# --------------------------------------------------------------------------- #
def test_dataset_paired_ok(tmp_path):
    for i in range(3):
        img(tmp_path / "lr" / f"{i}.png", 128, 128, i)
        img(tmp_path / "hr" / f"{i}.png", 512, 512, i)
    r = vi.check_dataset(tmp_path / "lr", tmp_path / "hr", scale=4)
    assert r["problems"] == [] and r["n_pairs"] == 3 and r["lr_sizes"] == {"128x128": 3}


def test_dataset_detects_problems(tmp_path):
    img(tmp_path / "lr" / "a.png", 128, 128, 1)
    img(tmp_path / "lr" / "b.png", 128, 128, 1)          # byte-identical to a.png
    img(tmp_path / "lr" / "c.png", 128, 128, 3)          # no GT
    (tmp_path / "lr" / "readme.txt").write_text("x")     # upstream would try to open it
    img(tmp_path / "hr" / "a.png", 512, 512)
    img(tmp_path / "hr" / "b.png", 500, 512)             # wrong scale
    img(tmp_path / "hr" / "z.png", 512, 512)             # no LR
    r = vi.check_dataset(tmp_path / "lr", tmp_path / "hr", scale=4)
    text = " | ".join(r["problems"])
    assert "non-image files" in text
    assert "byte-identical" in text
    assert "without GT" in text and "without LR" in text
    assert "GT size != LR size x 4" in text
    assert vi.main(["dataset", "--name", "t", "--lr_dir", str(tmp_path / "lr"), "--gt_dir", str(tmp_path / "hr")]) == 2


def test_dataset_suffix_matching_and_unpaired(tmp_path):
    img(tmp_path / "lr" / "x_LR4.png", 100, 80)
    img(tmp_path / "hr" / "x_HR.png", 400, 320)
    r = vi.check_dataset(tmp_path / "lr", tmp_path / "hr", 4, ["_LR4", "_HR"])
    assert r["problems"] == [] and r["n_pairs"] == 1
    assert r["n_lr_upsampled_by_upstream"] == 1 and r["warnings"]  # side < 512/4
    out = tmp_path / "rep.json"
    assert vi.main(["dataset", "--name", "u", "--lr_dir", str(tmp_path / "lr"), "--output", str(out)]) == 0
    assert json.loads(out.read_text())["paired"] is False


# --------------------------------------------------------------------------- #
# validate_inputs: checkpoints
# --------------------------------------------------------------------------- #
def _component(folder, cfg=None, weight="diffusion_pytorch_model.safetensors", data=b"\0" * 2_000_000):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(cfg or {}))
    (folder / weight).write_bytes(data)


def test_checkpoints(tmp_path):
    sd, seesr = tmp_path / "sd2", tmp_path / "seesr"
    (sd / "scheduler").mkdir(parents=True)
    (sd / "scheduler" / "scheduler_config.json").write_text(json.dumps({
        "_class_name": "PNDMScheduler", "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085,
        "num_train_timesteps": 1000, "prediction_type": "epsilon", "steps_offset": 1}))
    for f in ("vocab.json", "merges.txt"):
        (sd / "tokenizer").mkdir(exist_ok=True)
        (sd / "tokenizer" / f).write_text("x")
    (sd / "feature_extractor").mkdir()
    (sd / "feature_extractor" / "preprocessor_config.json").write_text("{}")
    _component(sd / "text_encoder", weight="model.safetensors")
    _component(sd / "vae")
    _component(seesr / "unet", {"use_image_cross_attention": True})
    _component(seesr / "controlnet", {"use_image_cross_attention": True})
    (tmp_path / "dape.pth").write_bytes(b"\0" * 2_000_000)
    (tmp_path / "ram.pth").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")  # LFS pointer
    models = {"pretrained_model_path": str(sd), "seesr_model_path": str(seesr),
              "ram_ft_path": str(tmp_path / "dape.pth"), "ram_pretrained_path": str(tmp_path / "ram.pth")}
    report, problems = vi.check_checkpoints(models)
    assert problems == [f"git-lfs pointer, not the real file (run `git lfs pull`): {tmp_path / 'ram.pth'}"]
    if not isinstance(report["ddpm_timesteps"], str):  # diffusers available
        assert report["ddpm_timesteps"]["30"] == {"count": 30, "first": 958, "last": 1, "executed_with_start_steps_999": 30}

    _component(seesr / "unet", {})  # image cross-attention flag missing
    (tmp_path / "ram.pth").write_bytes(b"\0" * 2_000_000)
    _, problems = vi.check_checkpoints(models)
    assert len(problems) == 1 and "use_image_cross_attention" in problems[0]


# --------------------------------------------------------------------------- #
# compare_results
# --------------------------------------------------------------------------- #
import copy  # noqa: E402

from experiments.common import upstream_defaults  # noqa: E402

PSNR_DESC = {"pyiqa_name": "psnr", "lower_better": False, "test_y_channel": True, "crop_border": 0}
LPIPS_DESC = {"pyiqa_name": "lpips", "lower_better": True}


def base_meta(steps, seed, reseed=True):
    args = dict(upstream_defaults())
    args.update(num_inference_steps=steps, seed=seed, save_prompts=True, output_dir="/r/sr", image_path="/d/lr",
                seesr_model_path="/m/seesr", ram_ft_path="/m/DAPE.pth", pretrained_model_path="/m/sd2")
    return {
        "status": "completed", "upstream_args": args,
        "config": {"runtime": {"scheduler": "DDPMScheduler", "enable_xformers": True, "reseed_per_image": reseed}},
        "dataset": {"n_images": 3, "filenames_sha256": "abc", "lr_dir_fingerprint": "lrfp"},
        "checkpoints": {"seesr_model_path": {"path": "/m/seesr", "size_bytes": 10, "fingerprint": "ckpt"}},
        "environment": {"gpu_name": "NVIDIA L4", "packages": {"torch": "2.0.1", "diffusers": "0.21.0"}},
        "attention_backend": "xformers", "git": {"commit": "c0ffee", "uncommitted_files": []},
        "scheduler": {"class": "DDPMScheduler", "config": {"steps_offset": 1}, "denoising_steps_per_call": [steps]},
        "runtime_verified": {"unet_dtype": "torch.float16", "unet_attention_processors": ["XFormersAttnProcessor"]},
        "timing": {"per_image_diffusion_s_mean_excl_first": steps / 10, "per_image_total_s_mean_excl_first": steps / 10 + 0.5,
                   "sum_per_image_total_s": 3 * (steps / 10 + 0.5), "peak_vram_allocated_mb_max": 9000,
                   "peak_vram_reserved_mb_max": 11000},
    }


def fake_run(root, name, meta, psnr, lpips):
    d = root / name
    write_json(d / "run_metadata.json", meta)
    imgs = sorted(psnr)
    write_csv(d / "eval" / "paired_per_image.csv",
              [{"image": i, "psnr": psnr[i], "lpips": lpips[i], "status": "ok"} for i in imgs])
    write_json(d / "eval" / "paired_summary.json", {
        "n_evaluated_ok": len(imgs), "n_errors": 0, "gt_dir": "/d/hr",
        "metrics": {"psnr": PSNR_DESC, "lpips": LPIPS_DESC},
        "aggregate": {"psnr": {"mean": sum(psnr.values()) / 3}, "lpips": {"mean": sum(lpips.values()) / 3}}})


# per (steps, seed): psnr per image. 30 steps is 0.5 dB lower on image a and b, equal on c (seed 42);
# for seed 123 it is 1.0 / 0.0 / 0.5 lower.
PSNR = {
    (50, 42): {"a": 25.0, "b": 26.0, "c": 27.0}, (30, 42): {"a": 24.5, "b": 25.5, "c": 27.0},
    (50, 123): {"a": 25.2, "b": 26.4, "c": 26.8}, (30, 123): {"a": 24.2, "b": 26.4, "c": 26.3},
}


def make_sweep(root, seeds=(42, 123)):
    for (steps, seed), psnr in PSNR.items():
        if seed in seeds:
            lp = {k: 0.3 + (50 - steps) / 1000 for k in psnr}
            fake_run(root, f"sweep_ddpm{steps}_s{seed}", base_meta(steps, seed), psnr, lp)


def test_sampling_pairs_by_image_and_seed(tmp_path):
    make_sweep(tmp_path)
    assert cr.main(["sampling", "--results_root", str(tmp_path), "--dataset", "RealSR"]) == 0
    out = json.loads((tmp_path / "comparison" / "sampling_comparison.json").read_text())
    assert out["compatible"] and out["differences"] == []
    d = next(x for x in out["paired_differences"] if x["metric"] == "psnr")
    # per image, mean over seeds of (30 - 50): a: (-0.5-1.0)/2, b: (-0.5+0)/2, c: (0-0.5)/2
    per_image = [-0.75, -0.25, -0.25]
    assert d["mean_diff"] == pytest.approx(np.mean(per_image))
    assert d["sd_diff"] == pytest.approx(np.std(per_image, ddof=1))
    assert d["se_diff"] == pytest.approx(np.std(per_image, ddof=1) / np.sqrt(3))
    assert d["frac_images_better"] == 0.0 and d["n_images"] == 3 and d["paired_on_seeds"] == "42 123"
    assert d["sd_of_per_seed_mean_diff"] == pytest.approx(np.std([-1 / 3, -0.5], ddof=1))
    lp = next(x for x in out["paired_differences"] if x["metric"] == "lpips")
    assert lp["lower_better"] is True and lp["frac_images_better"] == 0.0  # lpips rose by 0.02
    s50 = next(r for r in out["summary"] if r["steps"] == 50)
    within = np.mean([np.std([25.0, 25.2], ddof=1), np.std([26.0, 26.4], ddof=1), np.std([27.0, 26.8], ddof=1)])
    assert s50["psnr_within_image_seed_sd"] == pytest.approx(within)
    assert s50["psnr_sd_between_seed_means"] == pytest.approx(np.std([26.0, 26.1333333], ddof=1), rel=1e-6)
    s30 = next(r for r in out["summary"] if r["steps"] == 30)
    assert s30["step_reduction_pct"] == pytest.approx(40.0)
    assert s30["diffusion_latency_change_pct_vs_ref"] == pytest.approx(-40.0)
    assert (tmp_path / "comparison" / "sampling_comparison.md").exists()


def test_single_seed_reports_no_seed_variation(tmp_path):
    make_sweep(tmp_path, seeds=(42,))
    assert cr.main(["sampling", "--results_root", str(tmp_path), "--dataset", "RealSR"]) == 0
    out = json.loads((tmp_path / "comparison" / "sampling_comparison.json").read_text())
    s50 = next(r for r in out["summary"] if r["steps"] == 50)
    assert s50["psnr_within_image_seed_sd"] is None and s50["psnr_sd_between_seed_means"] is None


def _set(meta, dotted, value):
    node = meta
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


@pytest.mark.parametrize("field,value", [
    ("upstream_args.guidance_scale", 4.0),             # CFG
    ("upstream_args.mixed_precision", "bf16"),         # precision
    ("runtime_verified.unet_dtype", "torch.float32"),  # actual precision
    ("upstream_args.align_method", "wavelet"),         # colour correction
    ("upstream_args.process_size", 768),               # preprocessing
    ("upstream_args.vae_decoder_tiled_size", 128),     # tiling
    ("upstream_args.upscale", 2),                      # output size
    ("attention_backend", "pytorch_sdpa_default"),
    ("scheduler.class", "DDIMScheduler"),
    ("config.runtime.scheduler", "DDIMScheduler"),
    ("checkpoints.seesr_model_path.fingerprint", "other"),
    ("dataset.filenames_sha256", "zzz"),
    ("dataset.lr_dir_fingerprint", "changed"),
    ("dataset.n_images", 2),
    ("environment.gpu_name", "NVIDIA A100"),
    ("git.commit", "deadbeef"),
    ("status", "incomplete: 1 output image(s) missing"),
    ("config.runtime.reseed_per_image", False),
])
def test_incompatible_runs_are_rejected(tmp_path, field, value):
    make_sweep(tmp_path)
    f = tmp_path / "sweep_ddpm30_s123" / "run_metadata.json"
    meta = json.loads(f.read_text())
    _set(meta, field, value)
    write_json(f, meta)
    names = sorted(d.name for d in tmp_path.iterdir() if d.name.startswith("sweep_"))
    assert cr.main(["sampling", "--results_root", str(tmp_path), "--dataset", "RealSR", "--experiments", *names]) == 2
    out = json.loads((tmp_path / "comparison" / "sampling_comparison.json").read_text())
    assert out["compatible"] is False and "summary" not in out
    text = " ".join(out["differences"])
    assert "sweep_ddpm30_s123" in text


def test_metric_settings_mismatch_is_rejected(tmp_path):
    make_sweep(tmp_path)
    f = tmp_path / "sweep_ddpm30_s42" / "eval" / "paired_summary.json"
    s = json.loads(f.read_text())
    s["metrics"]["psnr"]["test_y_channel"] = False
    write_json(f, s)
    assert cr.main(["sampling", "--results_root", str(tmp_path), "--dataset", "RealSR"]) == 2


def test_duplicate_steps_seed_rejected(tmp_path):
    make_sweep(tmp_path)
    fake_run(tmp_path, "sweep_ddpm30_s42_copy", base_meta(30, 42), PSNR[(30, 42)], {k: 0.3 for k in "abc"})
    assert cr.main(["sampling", "--results_root", str(tmp_path), "--dataset", "RealSR"]) == 2
    out = json.loads((tmp_path / "comparison" / "sampling_comparison.json").read_text())
    assert any("steps=30, seed=42" in d and "several runs" in d for d in out["differences"])


def test_reproduction_track(tmp_path):
    meta = base_meta(50, 42, reseed=False)
    fake_run(tmp_path, "baseline_ddpm50", meta, PSNR[(50, 42)], {k: 0.3 for k in "abc"})
    pub = tmp_path / "pub.yaml"
    pub.write_text("source: test\ndatasets:\n  RealSR: {psnr: 25.18, lpips: 0.3009}\n")
    assert cr.main(["reproduction", "--results_root", str(tmp_path), "--dataset", "RealSR", "--published", str(pub)]) == 0
    rep = json.loads((tmp_path / "comparison" / "reproduction.json").read_text())
    row = next(r for r in rep["rows"] if r["metric"] == "psnr")
    assert row["reproduced"] == pytest.approx(26.0) and row["difference"] == pytest.approx(26.0 - 25.18)
    assert rep["checks"] == []
    bad = copy.deepcopy(meta)
    bad["upstream_args"]["guidance_scale"] = 4.0
    bad["config"]["runtime"]["reseed_per_image"] = True
    write_json(tmp_path / "baseline_ddpm50" / "run_metadata.json", bad)
    assert cr.main(["reproduction", "--results_root", str(tmp_path), "--dataset", "RealSR", "--published", str(pub)]) == 2
    checks = " ".join(json.loads((tmp_path / "comparison" / "reproduction.json").read_text())["checks"])
    assert "guidance_scale" in checks and "reseed_per_image" in checks


def test_default_discovery_reports_excluded_runs(tmp_path, capsys):
    make_sweep(tmp_path)
    fake_run(tmp_path, "baseline_ddpm50", base_meta(50, 42, reseed=False), PSNR[(50, 42)], {k: 0.3 for k in "abc"})
    (tmp_path / "sweep_ddpm30_s42__replaced_20261110_120000").mkdir()
    assert cr.main(["sampling", "--results_root", str(tmp_path), "--dataset", "RealSR"]) == 0
    out = json.loads((tmp_path / "comparison" / "sampling_comparison.json").read_text())
    assert out["excluded"] == {"baseline_ddpm50": "upstream RNG (track A)",
                               "sweep_ddpm30_s42__replaced_20261110_120000": "replaced by a later run"}
    assert "not compared: baseline_ddpm50" in capsys.readouterr().err


def test_tag_confidence_analysis(tmp_path):
    from experiments.tools import analyze_tag_confidence as atc
    run = tmp_path / "run"
    run.mkdir()
    recs = [{"image": "a", "tag_scores": [["dog", 0.9, 0.8, 0.69], ["cat", 0.7, 0.65, 0.55], ["sky", 0.99, 0.6, 0.98]]},
            {"image": "b", "tag_scores": [["tree", 0.8, 0.7, 0.63]]},
            {"image": "c", "tag_scores": [["car", 0.9, 0.5, 0.9]]}]
    (run / "semantic_confidence.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    write_csv(tmp_path / "ref.csv", [{"image": "a", "ref_tags": "dog, sky"}, {"image": "b", "ref_tags": "grass"}])
    assert atc.main(["--run", str(run), "--reference", str(tmp_path / "ref.csv"), "--min_confidence", "0.65"]) == 0
    s = json.loads((run / "eval" / "tag_confidence_summary.json").read_text())
    assert s["kept"] == {"n": 2, "agreement": 1.0}            # dog (0.69), sky (0.98)
    assert s["dropped"] == {"n": 2, "agreement": 0.0}         # cat, tree
    assert s["images_without_reference"] == ["c"]
    bins = {b["c_low"]: b for b in s["bins"]}
    assert bins[0.5]["n_tags"] == 1 and bins[0.6]["n_tags"] == 2 and bins[0.95]["agreement_with_ram_hr"] == 1.0


def test_sampling_with_method_variants(tmp_path):
    method_on = {"semantic_confidence": {"enabled": True}, "lr_guidance": {"enabled": False}}
    for variant, delta, method in (("original", 0.0, {}), ("combined", 0.2, method_on)):
        for (steps, seed), psnr in PSNR.items():
            meta = base_meta(steps, seed)
            meta["config"]["experiment"] = {"variant": variant}
            meta["config"]["method"] = method
            fake_run(tmp_path, f"{variant}_ddpm{steps}_s{seed}", meta,
                     {k: v + delta for k, v in psnr.items()}, {k: 0.3 for k in psnr})
    assert cr.main(["sampling", "--results_root", str(tmp_path), "--dataset", "RealSR"]) == 0
    out = json.loads((tmp_path / "comparison" / "sampling_comparison.json").read_text())
    assert out["reference"] == {"variant": "original", "steps": 50}
    d = {(x["variant"], x["steps"], x["reference"], x["metric"]): x for x in out["paired_differences"]}
    assert d[("combined", 30, "original@30", "psnr")]["mean_diff"] == pytest.approx(0.2)     # method effect alone
    assert d[("combined", 50, "original@50", "psnr")]["mean_diff"] == pytest.approx(0.2)
    assert d[("combined", 30, "original@50", "psnr")]["mean_diff"] == pytest.approx(np.mean([-0.75, -0.25, -0.25]) + 0.2)
    assert ("original", 30, "original@30", "psnr") not in d
    assert [(r["variant"], r["steps"]) for r in out["summary"]] == [("original", 50), ("original", 30),
                                                                   ("combined", 50), ("combined", 30)]
    # one run of a variant with different method settings -> rejected
    f = tmp_path / "combined_ddpm30_s123" / "run_metadata.json"
    meta = json.loads(f.read_text())
    meta["config"]["method"]["semantic_confidence"]["min_confidence"] = 0.9
    write_json(f, meta)
    assert cr.main(["sampling", "--results_root", str(tmp_path), "--dataset", "RealSR"]) == 2
    out = json.loads((tmp_path / "comparison" / "sampling_comparison.json").read_text())
    assert any("variant 'combined' has 2 different method configurations" in x for x in out["differences"])
