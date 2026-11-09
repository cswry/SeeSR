"""Evaluation-pipeline tests on tiny synthetic images.

A fake metric factory is injected so the tests check file matching, validation,
aggregation and output formats without network access or pyiqa weights. A
separate test checks real pyiqa PSNR when pyiqa is importable.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.evaluation import evaluate as ev  # noqa: E402
from experiments.evaluation.semantic_tags import parse_tags, tag_agreement  # noqa: E402

torch = pytest.importorskip("torch")


def fake_factory(spec, device, y_channel, crop_border):
    def compute(sr, gt=None):
        if gt is None:
            return float(sr.mean())
        return float((sr - gt).abs().mean())
    return compute


def save(path, value, size=(8, 6), mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    img = Image.fromarray(arr)
    if mode != "RGB":
        img = img.convert(mode)
    img.save(path)


def run(argv):
    return ev.run(ev.parse_args(argv), factory=fake_factory)


@pytest.fixture
def dirs(tmp_path):
    sr, gt, out = tmp_path / "sr", tmp_path / "gt", tmp_path / "out"
    for name, s, g in [("a", 10, 20), ("b", 30, 30)]:
        save(sr / f"{name}.png", s)
        save(gt / f"{name}.png", g)
    return sr, gt, out


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_match_pairs_with_suffixes(tmp_path):
    sr = [tmp_path / "Canon_001.png", tmp_path / "Canon_002.png"]
    gt = [tmp_path / "Canon_001_HR.png", tmp_path / "Canon_003_HR.png"]
    m = ev.match_pairs(sr, gt, ["_HR"])
    assert [k for k, _, _ in m["pairs"]] == ["Canon_001"]
    assert m["missing_gt"] == ["Canon_002.png"]
    assert m["missing_sr"] == ["Canon_003_HR.png"]


def test_match_pairs_duplicates(tmp_path):
    m = ev.match_pairs([tmp_path / "x.png", tmp_path / "x.jpg"], [tmp_path / "x.png"])
    assert m["duplicates"]["sr"] == {"x": ["x.png", "x.jpg"]}


def test_paired_and_noref_ok(dirs):
    sr, gt, out = dirs
    rc = run(["--sr_dir", str(sr), "--gt_dir", str(gt), "--output_dir", str(out), "--paired", "--noref",
              "--paired_metrics", "psnr", "--noref_metrics", "musiq", "--device", "cpu"])
    assert rc == 0
    rows = read_csv(out / "paired_per_image.csv")
    assert [r["image"] for r in rows] == ["a", "b"]
    assert float(rows[0]["psnr"]) == pytest.approx(10 / 255)
    s = json.loads((out / "paired_summary.json").read_text())
    agg = s["aggregate"]["psnr"]
    assert agg["n"] == 2 and agg["mean"] == pytest.approx(5 / 255) and agg["std"] == pytest.approx(5 / 255)
    assert s["missing_gt"] == [] and s["problems"] == []
    n = json.loads((out / "noref_summary.json").read_text())
    assert "disclaimer" in n and n["n_evaluated_ok"] == 2
    assert "gt_file" not in read_csv(out / "noref_per_image.csv")[0]


def test_missing_gt_blocks_by_default(dirs):
    sr, gt, out = dirs
    save(sr / "c.png", 50)
    rc = run(["--sr_dir", str(sr), "--gt_dir", str(gt), "--output_dir", str(out), "--paired",
              "--paired_metrics", "psnr", "--device", "cpu"])
    assert rc == 2
    s = json.loads((out / "paired_summary.json").read_text())
    assert s["missing_gt"] == ["c.png"] and s["evaluated"] is False and s["aggregate"]["psnr"]["n"] == 0


def test_allow_missing_evaluates_subset_and_reports(dirs):
    sr, gt, out = dirs
    save(sr / "c.png", 50)
    rc = run(["--sr_dir", str(sr), "--gt_dir", str(gt), "--output_dir", str(out), "--paired", "--allow_missing",
              "--paired_metrics", "psnr", "--device", "cpu"])
    assert rc == 0
    s = json.loads((out / "paired_summary.json").read_text())
    assert s["n_evaluated_ok"] == 2 and s["missing_gt"] == ["c.png"] and s["warnings"]


def test_size_mismatch_is_an_error(dirs):
    sr, gt, out = dirs
    save(gt / "b.png", 30, size=(9, 6))
    rc = run(["--sr_dir", str(sr), "--gt_dir", str(gt), "--output_dir", str(out), "--paired",
              "--paired_metrics", "psnr", "--device", "cpu"])
    assert rc == 2
    rows = {r["image"]: r for r in read_csv(out / "paired_per_image.csv")}
    assert rows["a"]["status"] == "ok"
    assert rows["b"]["status"] == "error" and "size mismatch" in rows["b"]["notes"]
    assert json.loads((out / "paired_summary.json").read_text())["aggregate"]["psnr"]["n"] == 1


def test_grayscale_is_converted_and_noted(dirs):
    sr, gt, out = dirs
    save(gt / "a.png", 20, mode="L")
    assert run(["--sr_dir", str(sr), "--gt_dir", str(gt), "--output_dir", str(out), "--paired",
                "--paired_metrics", "psnr", "--device", "cpu"]) == 0
    rows = {r["image"]: r for r in read_csv(out / "paired_per_image.csv")}
    assert "gt converted L->RGB" in rows["a"]["notes"]


def test_noref_without_gt(dirs):
    sr, _, out = dirs
    assert run(["--sr_dir", str(sr), "--output_dir", str(out), "--noref", "--noref_metrics", "clipiqa",
                "--device", "cpu"]) == 0
    assert not (out / "paired_summary.json").exists()


def test_paired_requires_gt(dirs):
    sr, _, out = dirs
    with pytest.raises(SystemExit):
        run(["--sr_dir", str(sr), "--output_dir", str(out), "--paired"])


def test_requires_a_mode(dirs):
    sr, _, out = dirs
    with pytest.raises(SystemExit):
        run(["--sr_dir", str(sr), "--output_dir", str(out)])


def test_no_silent_overwrite(dirs):
    sr, _, out = dirs
    argv = ["--sr_dir", str(sr), "--output_dir", str(out), "--noref", "--noref_metrics", "musiq", "--device", "cpu"]
    assert run(argv) == 0
    with pytest.raises(SystemExit):
        run(argv)
    assert run(argv + ["--overwrite"]) == 0


def test_unknown_metric_rejected(dirs):
    sr, gt, out = dirs
    with pytest.raises(KeyError):
        run(["--sr_dir", str(sr), "--gt_dir", str(gt), "--output_dir", str(out), "--paired",
             "--paired_metrics", "musiq", "--device", "cpu"])  # no-reference metric requested as paired


def test_config_settings_are_used(dirs, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("dataset: {strip_suffixes: [_HR]}\nevaluation: {paired_metrics: [ssim], y_channel: false, crop_border: 2}\n")
    sr, gt, out = dirs
    for f in list(gt.iterdir()):
        f.rename(gt / f"{f.stem}_HR.png")
    assert run(["--sr_dir", str(sr), "--gt_dir", str(gt), "--output_dir", str(out), "--paired",
                "--config", str(cfg), "--device", "cpu"]) == 0
    s = json.loads((out / "paired_summary.json").read_text())
    assert s["metrics"]["ssim"] == {"pyiqa_name": "ssim", "lower_better": False, "test_y_channel": False, "crop_border": 2}


def test_real_pyiqa_psnr_matches_formula():
    pyiqa = pytest.importorskip("pyiqa")  # noqa: F841
    from experiments.evaluation.metrics import PAIRED_METRICS, pyiqa_factory
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-5, 6, a.shape), 0, 255).astype(np.uint8)
    psnr = pyiqa_factory(PAIRED_METRICS["psnr"], "cpu", y_channel=False, crop_border=0)
    got = psnr(ev.to_tensor(a, "cpu"), ev.to_tensor(b, "cpu"))
    mse = np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)
    assert got == pytest.approx(10 * np.log10(1.0 / mse), abs=1e-3)


def test_tag_agreement():
    r = tag_agreement(parse_tags("dog, grass, sky"), parse_tags("dog, sky, tree, cloud"))
    assert r["tag_precision"] == pytest.approx(2 / 3)
    assert r["tag_recall"] == pytest.approx(2 / 4)
    assert r["tag_f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert r["extra_tags"] == "grass" and r["lost_tags"] == "cloud, tree"
    assert tag_agreement(set(), set())["tag_f1"] == 1.0


def test_lr_consistency_values():
    from experiments.evaluation.lr_consistency import lr_consistency
    lr = np.full((4, 5, 3), 100, np.uint8)
    flat = lr_consistency(np.full((16, 20, 3), 100, np.uint8), lr, 4)
    assert flat["lr_mae"] == 0.0 and flat["lr_psnr"] == float("inf")
    off = lr_consistency(np.full((16, 20, 3), 110, np.uint8), lr, 4)
    assert off["lr_mae"] == pytest.approx(10.0) and off["lr_psnr"] == pytest.approx(10 * np.log10(255 ** 2 / 100))
    with pytest.raises(ValueError):
        lr_consistency(np.zeros((16, 21, 3), np.uint8), lr, 4)


def test_lrc_mode_cli(tmp_path):
    sr, lr, out = tmp_path / "sr", tmp_path / "lr", tmp_path / "out"
    save(lr / "a.png", 50, size=(4, 3))
    save(sr / "a.png", 60, size=(16, 12))
    save(lr / "b.png", 50, size=(4, 3))
    save(sr / "b.png", 50, size=(15, 12))           # wrong size -> per-image error
    rc = run(["--sr_dir", str(sr), "--lr_dir", str(lr), "--output_dir", str(out), "--lrc", "--device", "cpu"])
    assert rc == 2
    s = json.loads((out / "lrc_summary.json").read_text())
    assert s["mode"] == "lrc" and "disclaimer" in s and s["n_evaluated_ok"] == 1 and s["n_errors"] == 1
    assert s["aggregate"]["lr_mae"]["mean"] == pytest.approx(10.0)
    assert s["metrics"]["lr_mae"]["lower_better"] is True
    assert not (out / "paired_summary.json").exists() and not (out / "noref_summary.json").exists()
    with pytest.raises(SystemExit):
        run(["--sr_dir", str(sr), "--output_dir", str(out), "--lrc"])   # needs --lr_dir
