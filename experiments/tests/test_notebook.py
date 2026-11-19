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


def test_v2_and_efficiency_names_match_their_sweeps(ns):
    src = _cell("def v2_names(")
    defs = src.split("run_v2(\"RealSR\")")[0]                 # definitions only, no execution
    exec(defs, ns)
    v2 = {n for n, _ in load_sweep(REPO / "experiments" / "configs" / "v2_ablation.yaml")}
    ns["LORNET_CKPT"] = "/nonexistent/best.pt"                # no checkpoint: training-free subset
    free = ns["v2_names"]([42, 123, 2026])
    assert set(free) <= v2 and len(free) == len(ns["V2_TRAINING_FREE"]) * 3 * 3
    learned = ns["v2_names"]([42, 123, 2026], ns["V2_VARIANTS"])
    assert set(learned) <= v2 and len(learned) == len(ns["V2_VARIANTS"]) * 3 * 3 == 90
    from experiments.common import sweep_default_variants
    assert ns["V2_VARIANTS"] == sweep_default_variants(REPO / "experiments" / "configs" / "v2_ablation.yaml")
    from cfa_seesr import MethodConfig
    runs = dict(load_sweep(REPO / "experiments" / "configs" / "v2_ablation.yaml"))
    assert not any(MethodConfig.from_dict(runs[n]["method"]).sfr_oc.uses_learned for n in free)
    scr = _cell("def run_v2_screening(")
    screening = {n for n, _ in load_sweep(REPO / "experiments" / "configs" / "v2_strength_screening.yaml")}
    listed = set(re.findall(r'"((?:original|combined_spatial|v2_heur_sem_g\d+))"', scr))
    assert {f"{v}_ddpm{st}_s42" for v in listed for st in (20, 50)} == screening
    eff_src = _cell("def run_efficiency(")
    eff = {n for n, _ in load_sweep(REPO / "experiments" / "configs" / "efficiency_sweep.yaml")}
    listed = set(re.findall(r'"(original_[a-z]+\d+_s42)"', eff_src))
    assert listed == eff


TRAIN_NB = REPO / "experiments" / "notebooks" / "lornet_training.ipynb"


def test_training_notebook_paths_and_handoff():
    text = TRAIN_NB.read_text(encoding="utf-8")
    nb = json.loads(text)
    for rel in set(re.findall(r"experiments/[\w/.-]+\.(?:py|yaml|sh|txt)", text)):
        assert (REPO / rel).exists(), rel
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    ns = {}
    cfg_cell = next("".join(c["source"]) for c in nb["cells"] if "LORNET_OUT =" in "".join(c["source"]))
    exec(cfg_cell.split("def sh(")[0], ns)
    eval_ns = {}
    with mock.patch.dict(os.environ):
        exec(_cell("DRIVE_ROOT ="), eval_ns)
    # the evaluation notebook reads exactly what the training notebook writes
    assert eval_ns["LORNET_CKPT"] == f"{ns['LORNET_OUT']}/checkpoints/best.pt"
    assert eval_ns["V2_SCREEN_DATA"] == ns["V2_SCREEN_DATA"]
    assert all("Test" not in s["hr_dir"] and "test" not in s["lr_dir"] for s in ns["TRAIN_SOURCES"])
    assert "train_lornet.py cache" in src and "train_lornet.py train" in src and "make_screening_set.py" in src
