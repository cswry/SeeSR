"""experiments/tools/pilot_report.py: README/CSV agreement, path sanitising, figures (no GPU, no real data)."""
import csv
import json

import pytest

from experiments.tools import pilot_report as pr

README_VALUES = {"original": (26.4415, 0.8763, 0.1795, 0.1714), "heuristic": (26.0001, 0.8843, 0.2023, 0.1953),
                 "learned": (26.7053, 0.8870, 0.1828, 0.1592)}


def _pilot(tmp_path, values=README_VALUES, meta=None):
    for k, v in values.items():
        d = tmp_path / k
        d.mkdir(parents=True, exist_ok=True)
        with (d / "paired_per_image.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "sr_file", "gt_file", "height", "width", "psnr", "ssim", "lpips", "dists",
                        "status", "notes"])
            w.writerow(["Canon_001", "Canon_001_LR4.png", "Canon_001_HR.png", 512, 512, *[f"{x:.6f}" for x in v],
                        "ok", ""])
        if meta:
            (d / "run_metadata.json").write_text(json.dumps(meta))
    return tmp_path


def test_readme_table_matches_the_verified_values():
    table = pr.readme_table((pr.REPO / "README.md").read_text(encoding="utf-8"))
    for k, v in README_VALUES.items():
        assert tuple(table[k][m] for m, _, _ in pr.METRICS) == v


def test_check_passes_and_detects_mismatch(tmp_path):
    assert pr.cmd_check(_pilot(tmp_path / "a")) == 0
    bad = dict(README_VALUES, learned=(26.7, 0.8870, 0.1828, 0.1592))
    assert pr.cmd_check(_pilot(tmp_path / "b", bad)) == 1
    assert pr.cmd_check(tmp_path / "missing") == 1


def test_sanitize_and_training_split_flag(tmp_path):
    meta = {"config": {"dataset": {"lr_dir": "/content/drive/MyDrive/seesr/datasets/StableSR_testsets/RealSRVal_crop128/test_LR"},
                       "inference": {"num_inference_steps": 50, "seed": 42}}, "method": {"version": "2.1.0"}}
    p = _pilot(tmp_path / "s", meta=meta)
    assert pr.cmd_check(p) == 1                                     # private Drive path
    pr.cmd_sanitize(p)
    text = (p / "learned" / "run_metadata.json").read_text()
    assert "/content/" not in text and "<DRIVE_ROOT>/datasets" in text
    assert pr.cmd_check(p) == 0
    meta["config"]["dataset"]["lr_dir"] = "<DRIVE_ROOT>/datasets/RealSR_V3/Canon/Train/4"
    assert pr.cmd_check(_pilot(tmp_path / "t", meta=meta)) == 1    # input from a training split


def test_figures(tmp_path):
    np = pytest.importorskip("numpy")
    from PIL import Image
    p = _pilot(tmp_path / "f")
    svg = pr.metrics_svg(pr.read_metrics(p)[0])
    assert svg.startswith("<svg") and "26.7053" in svg and "not a benchmark" in svg
    rng = np.random.default_rng(0)
    for k in README_VALUES:
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(p / k / "sr.png")
    Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(tmp_path / "lr.png")
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / "gt.png")
    out = pr.comparison_png(p, tmp_path / "lr.png", tmp_path / "gt.png", tmp_path / "cmp.png", crop=32)
    assert Image.open(out).size[0] > 5 * 256
