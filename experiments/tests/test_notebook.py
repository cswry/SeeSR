"""Notebook structure and helper functions (executed on CPU without Colab)."""
import json
import os
import re
from pathlib import Path
from unittest import mock

import pytest

from experiments.common import load_sweep

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "experiments" / "notebooks" / "seesr_evaluation.ipynb"
SWEEP = REPO / "experiments" / "configs" / "sampling_sweep.yaml"


def _cells():
    return [c for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _cell(marker):
    found = ["".join(c["source"]) for c in _cells() if marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return found[0]


@pytest.fixture
def ns(tmp_path):
    """Namespace with the configuration and runner cells executed, RESULTS redirected to tmp_path."""
    namespace = {}
    with mock.patch.dict(os.environ):
        exec(_cell("DRIVE_ROOT ="), namespace)
    namespace["RESULTS"] = str(tmp_path)
    exec(_cell("def finished("), namespace)
    return namespace


def test_notebook_paths_exist():
    text = NB.read_text(encoding="utf-8")
    for rel in set(re.findall(r"experiments/[\w/.-]+\.(?:py|yaml|sh)", text)):
        assert (REPO / rel).exists(), rel
    assert "sweep_ddpm" not in text


def test_track_b_names_match_sweep(ns):
    runs = {name for name, _ in load_sweep(SWEEP)}
    variants = ["original", "semantic_conf", "lr_schedule", "combined", "lr_schedule_late"] + ns["SSR_VARIANTS"]
    names = ns["track_b"]([42, 123, 2026], variants)
    assert set(names) == runs and len(names) == len(runs) == 90


def test_track_b_defaults_one_seed(ns):
    assert ns["SEEDS"] == [42]
    assert len(ns["track_b"]()) == len(ns["VARIANTS"]) * 3


def test_finished_rejects_incomplete(ns, tmp_path):
    assert not ns["finished"]("RealSR", "original_ddpm50_s42")
    run = tmp_path / "RealSR" / "original_ddpm50_s42"
    run.mkdir(parents=True)
    (run / "run_metadata.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(RuntimeError, match="running"):
        ns["finished"]("RealSR", "original_ddpm50_s42")
    (run / "run_metadata.json").write_text(json.dumps({"status": "completed"}))
    assert ns["finished"]("RealSR", "original_ddpm50_s42")


def test_run_track_b_skips_finished_runs(ns, tmp_path):
    for name in ns["track_b"]()[:2]:
        d = tmp_path / "RealSR" / name
        d.mkdir(parents=True)
        (d / "run_metadata.json").write_text(json.dumps({"status": "completed"}))
    calls = []
    ns["sh"] = calls.append
    ns["run_track_b"]("RealSR")
    assert len(calls) == 1
    todo = calls[0].split("--only ")[1].split(" --set")[0].split()
    assert todo == ns["track_b"]()[2:]


def test_evaluate_requests_lr_consistency(ns, tmp_path):
    name = "original_ddpm50_s42"
    d = tmp_path / "RealSR" / name
    d.mkdir(parents=True)
    (d / "run_metadata.json").write_text(json.dumps({"status": "completed"}))
    calls = []
    ns["sh"] = calls.append
    ns["evaluate"]("RealSR", [name])
    assert "--lrc --lr_dir" in calls[0] and "--paired" in calls[0] and "--noref" in calls[0]


def test_ssr_sections_skip_without_go_and_subset_pairs_images(ns, tmp_path):
    from PIL import Image
    calls = []
    ns["sh"] = calls.append
    exec(_cell("def screening_subset("), ns)            # GATE0 is "NOT RUN": nothing may run
    assert ns["SCREEN"] == "NOT RUN" and calls == []
    exec(_cell('if SCREEN != "CONTINUE":'), ns)
    assert calls == []
    lr, gt = tmp_path / "lr", tmp_path / "gt"
    lr.mkdir(), gt.mkdir()
    for i in range(5):
        Image.new("RGB", (4, 4)).save(lr / f"im{i}.png")
        if i != 3:
            Image.new("RGB", (16, 16)).save(gt / f"im{i}_HR.png")
    ns["DATASETS"] = {"RealSR": {"lr": str(lr), "gt": str(gt), "strip": ["_HR"]}}
    data = ns["screening_subset"]("RealSR", n=3)
    lrs = sorted(p.name for p in Path(data["lr"]).iterdir())
    gts = sorted(p.name for p in Path(data["gt"]).iterdir())
    assert len(lrs) == 3 and "im3.png" not in lrs                     # LR without GT is never chosen
    assert [g.replace("_HR", "") for g in gts] == lrs
    assert data == ns["screening_subset"]("RealSR", n=3)              # deterministic, idempotent
