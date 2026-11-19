"""LORNet (learned observation reliability): model, checkpoints, training data pipeline and leakage guards,
teacher target, training/evaluation loop (tiny VAE, CPU), SFR-OC integration, map-validity and strength-selection
tools. CPU only."""
import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from cfa_seesr.lornet import (CHECKPOINT_FORMAT, LORNet, LORNetCheckpointError, compute_features,  # noqa: E402
                              conv_macs, in_channels, load_lornet, make_checkpoint, parameter_count,
                              validate_model_config)
from cfa_seesr.observation_consistency import SFROCConfig  # noqa: E402

TARGET = {"tau": 0.01, "lowpass": {"method": "gaussian", "sigma": 3.0, "cutoff": 0.25}, "window": 3}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_model_shapes_range_and_size():
    torch.manual_seed(0)
    m = LORNet().eval()
    n = parameter_count(m)
    assert 10_000 < n < 100_000                                   # small by design (33k with the defaults)
    for H, W in ((512, 512), (520, 776), (64, 40), (8, 8)):
        with torch.no_grad():
            out = m(torch.rand(2, 3, H, W))
        assert out.shape == (2, 1, H // 8, W // 8)
        assert torch.isfinite(out).all() and (out > 0).all() and (out < 1).all()
    with pytest.raises(ValueError, match="multiples of 8"):
        m(torch.rand(1, 3, 30, 32))
    assert conv_macs(m, 512, 512) > 0


def test_model_is_deterministic_and_batch_independent():
    torch.manual_seed(1)
    m = LORNet().eval()
    x = torch.rand(3, 3, 64, 96)
    with torch.no_grad():
        a, b = m(x), m(x)
        single = torch.cat([m(x[i:i + 1]) for i in range(3)])
    assert torch.equal(a, b) and torch.allclose(a, single, atol=1e-6)


@pytest.mark.parametrize("feats", [["y"], ["rgb"], ["y", "grad"], ["y", "grad", "contrast", "hf", "rgb"]])
def test_feature_sets(feats):
    f = compute_features(torch.rand(1, 3, 32, 32), feats)
    assert f.shape[1] == in_channels(feats) and torch.isfinite(f).all() and f.min() >= 0
    m = LORNet(features=feats)
    assert m(torch.rand(1, 3, 32, 32)).shape == (1, 1, 4, 4)


@pytest.mark.parametrize("cfg", [{"features": []}, {"features": ["y", "y"]}, {"features": ["depth"]},
                                 {"widths": [16, 24, 32]}, {"widths": [16, 24, 30, 48]}])
def test_model_config_validation(cfg):
    with pytest.raises(ValueError):
        validate_model_config(cfg)


def test_uncertainty_head():
    m = LORNet(uncertainty=True)
    logit, logvar = m.forward_features(compute_features(torch.rand(1, 3, 32, 32), m.config["features"]))
    assert logit.shape == logvar.shape == (1, 1, 4, 4)


# --------------------------------------------------------------------------- #
# Checkpoints: never a random map
# --------------------------------------------------------------------------- #
def _save(tmp_path, model=None, **edit):
    model = model or LORNet()
    ck = make_checkpoint(model, TARGET, {"seed": 1})
    ck.update(edit)
    p = tmp_path / "lornet.pt"
    torch.save(ck, p)
    return p, model


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(3)
    p, model = _save(tmp_path)
    loaded, meta = load_lornet(p)
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(loaded(x), model.eval()(x))
    assert meta["format"] == CHECKPOINT_FORMAT and meta["target"]["tau"] == 0.01 and "state_dict" not in meta
    assert meta["method_version"].startswith("2.")


@pytest.mark.parametrize("edit,match", [({"format": "other"}, "not a"), ({"format_version": 99}, "format_version"),
                                        ({"target": {"tau": 0.0}}, "tau"), ({"target": {"tau": float("nan")}}, "tau"),
                                        ({"model_config": {"features": ["y"]}}, "do not match")])
def test_invalid_checkpoints_are_rejected(tmp_path, edit, match):
    p, _ = _save(tmp_path, **edit)
    with pytest.raises(LORNetCheckpointError, match=match):
        load_lornet(p)


def test_missing_corrupt_and_nonfinite_checkpoints(tmp_path):
    for bad in (None, "", tmp_path / "nope.pt"):
        with pytest.raises(LORNetCheckpointError):
            load_lornet(bad)
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(LORNetCheckpointError):
        load_lornet(junk)
    m = LORNet()
    with torch.no_grad():
        next(m.parameters()).fill_(float("nan"))
    p, _ = _save(tmp_path, model=m)
    with pytest.raises(LORNetCheckpointError, match="non-finite"):
        load_lornet(p)
    ck = torch.load(p)
    ck["state_dict"].pop(next(iter(ck["state_dict"])))
    torch.save(ck, p)
    with pytest.raises(LORNetCheckpointError, match="do not match"):
        load_lornet(p)


def test_learned_reliability_requires_a_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        SFROCConfig(enabled=True, strength=0.5, reliability="learned")
    SFROCConfig(enabled=False, reliability="learned")                    # disabled: nothing is loaded
    from cfa_seesr.observation_consistency import SFROCController
    cfg = SFROCConfig.from_dict({"enabled": True, "strength": 0.5, "reliability": "learned",
                                 "learned": {"checkpoint": "x.pt"}})
    with pytest.raises(ValueError, match="LORNet"):
        SFROCController(cfg, 999, "epsilon")
    with pytest.raises(ValueError):
        SFROCConfig.from_dict({"reliability": "magic"})
    with pytest.raises(ValueError):
        SFROCConfig.from_dict({"learned": {"blend": 1.5}})


# --------------------------------------------------------------------------- #
# Latent targets and counterfactual
# --------------------------------------------------------------------------- #
def test_teacher_target_and_tau():
    from cfa_seesr.latent_targets import discrepancy, estimate_tau, local_average, teacher_target
    z_hr = torch.randn(2, 4, 16, 16, generator=torch.Generator().manual_seed(0))
    E0 = discrepancy(z_hr, z_hr)
    assert E0.shape == (2, 1, 16, 16) and float(E0.abs().max()) == 0.0
    noisy = z_hr + 0.5
    E = discrepancy(noisy, z_hr)
    assert torch.allclose(E, torch.full_like(E, 0.5), atol=1e-5)          # low-frequency offset passes the LPF
    tau = estimate_tau([E, 2 * E])
    assert abs(tau - float(torch.cat([E.flatten(), 2 * E.flatten()]).median()) / math.log(2)) < 1e-9
    R = teacher_target(E, tau)
    assert (R > 0).all() and (R <= 1).all() and float(teacher_target(E0, tau).min()) == 1.0
    assert torch.equal(local_average(E, 1), E)
    with pytest.raises(ValueError):
        estimate_tau([torch.zeros(4)])
    assert tuple(discrepancy(torch.randn(1, 4, 9, 12), torch.randn(1, 4, 8, 13)).shape[-2:]) == (8, 12)


def test_counterfactual_signs():
    from cfa_seesr.latent_targets import counterfactual
    g = torch.Generator().manual_seed(0)
    z_gt = torch.randn(1, 4, 32, 32, generator=g)
    z_sr = z_gt + 0.3                                                    # low-frequency error
    good_lr = z_gt.clone()                                               # LR agrees with GT: correction helps
    bad_lr = z_gt - 0.6                                                  # LR wrong the other way: correction hurts
    assert float(counterfactual(z_sr, z_gt, good_lr)["benefit"].mean()) > 0.25
    assert float(counterfactual(z_sr, z_gt, bad_lr)["benefit"].mean()) < 0


def test_upscale_like_upstream_matches_test_seesr_geometry():
    from PIL import Image

    from cfa_seesr.latent_targets import upscale_like_upstream
    assert upscale_like_upstream(Image.new("RGB", (128, 128)), 4, 512).size == (512, 512)
    assert upscale_like_upstream(Image.new("RGB", (128, 100)), 4, 512).size == (648, 512)   # short side 100 < 128
    assert upscale_like_upstream(Image.new("RGB", (60, 90)), 4, 512).size == (512, 768)   # enlarged first
    assert upscale_like_upstream(Image.new("RGB", (130, 131)), 4, 512).size == (520, 520)  # floor to 8


# --------------------------------------------------------------------------- #
# Data pipeline and leakage guards
# --------------------------------------------------------------------------- #
def _write_pairs(root, name, n, lr_same_size=False, size=128, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    d = root / name / "Train"
    d.mkdir(parents=True)
    for i in range(n):
        base = (rng.random((size // 4, size // 4, 3)) * 255).astype(np.uint8)
        hr = Image.fromarray(base).resize((size, size), Image.BICUBIC)
        hr.save(d / f"{name}_{i:03d}_HR.png")
        lr = hr.resize((size // 4, size // 4), Image.BICUBIC)
        (lr.resize((size, size), Image.BICUBIC) if lr_same_size else lr).save(d / f"{name}_{i:03d}_LR4.png")
    return d


def test_pair_discovery_split_and_leakage(tmp_path):
    from experiments.training import data as D
    da = _write_pairs(tmp_path, "cama", 10)
    db = _write_pairs(tmp_path, "camb", 10, lr_same_size=True)
    srcs = [{"name": n, "hr_dir": str(d), "lr_dir": str(d), "hr_pattern": "*_HR.png", "lr_pattern": "*_LR4.png"}
            for n, d in (("a", da), ("b", db))]
    pairs = D.discover_pairs(srcs)
    assert len(pairs) == 20
    train, val = D.split_pairs(pairs, 0.3)
    assert {p["id"] for p in train}.isdisjoint({p["id"] for p in val}) and len(train) + len(val) == 20
    assert D.split_pairs(pairs, 0.3) == (train, val)                     # deterministic
    # an evaluation folder sharing stems with the training data is an error
    ev = tmp_path / "eval_lr"
    ev.mkdir()
    (ev / "cama_003_LR4.png").write_bytes((da / "cama_003_LR4.png").read_bytes())
    with pytest.raises(ValueError, match="evaluation"):
        D.split_pairs(pairs, 0.3, exclude=D.excluded_stems([ev]))
    # test-split folders are refused as training sources
    t = tmp_path / "RealSR" / "Test" / "4"
    t.mkdir(parents=True)
    with pytest.raises(ValueError, match="test split"):
        D.discover_pairs([{"name": "t", "hr_dir": str(t), "lr_dir": str(t)}])
    # LR geometry
    lr, hr, mode = D.load_pair(pairs[0], 4)
    lr2, hr2, mode2 = D.load_pair([p for p in pairs if p["source"] == "b"][0], 4)
    assert mode == "native" and mode2 == "downscaled_from_hr_size"
    assert lr.size == (32, 32) and lr2.size == (32, 32) and hr.size == hr2.size == (128, 128)


def test_crop_boxes():
    from experiments.training.data import crop_boxes
    assert crop_boxes(100, 100, 128, 4) == []
    b = crop_boxes(1024, 512, 512, 4)
    assert b == [(0, 0), (512, 0)]
    assert len(crop_boxes(2048, 2048, 512, 4)) == 4
    assert all(x + 512 <= 2000 and y + 512 <= 1500 for x, y in crop_boxes(2000, 1500, 512, 6))


# --------------------------------------------------------------------------- #
# Training loop (tiny VAE, CPU)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_vae(tmp_path_factory):
    from experiments.tests.tiny_pipeline import build_pipeline
    return build_pipeline(tmp_path_factory.mktemp("tok_lornet")).vae


def _train_cfg(tmp_path, out="run"):
    from experiments.training import train_lornet as T
    cfg = T.load_config(Path(T.__file__).parent / "configs" / "lornet.yaml")
    a = _write_pairs(tmp_path, "cama", 6, size=128, seed=1) if not (tmp_path / "cama").exists() else tmp_path / "cama" / "Train"
    b = _write_pairs(tmp_path, "camb", 6, lr_same_size=True, size=128, seed=2) if not (tmp_path / "camb").exists() \
        else tmp_path / "camb" / "Train"
    cfg["output_dir"] = str(tmp_path / out)
    cfg["data"]["sources"] = [{"name": n, "hr_dir": str(d), "lr_dir": str(d), "hr_pattern": "*_HR.png",
                               "lr_pattern": "*_LR4.png"} for n, d in (("a", a), ("b", b))]
    cfg["data"].update(val_fraction=0.3, crop_hr=64, crops_per_image=2, process_size=64)
    cfg["train"].update(epochs=3, batch_size=4, num_workers=0, early_stopping_patience=1, amp=False)
    return cfg


def test_cache_train_evaluate_reproducible(tmp_path, tiny_vae):
    from experiments.training import train_lornet as T
    quiet = lambda *a: None  # noqa: E731
    cfg = _train_cfg(tmp_path, "run1")
    man = T.cmd_cache(cfg, vae=tiny_vae, log=quiet)
    assert man["tau"] > 0 and man["splits"]["train"]["n_crops"] > 0 and man["splits"]["val"]["n_crops"] > 0
    assert man["vae"]["encoding"].startswith("posterior mean")
    s1 = T.cmd_train(cfg, log=quiet)
    root = Path(cfg["output_dir"])
    assert (root / "checkpoints" / "best.pt").exists() and (root / "checkpoints" / "last.pt").exists()
    log = [json.loads(l) for l in (root / "train_log.jsonl").read_text().splitlines()]
    assert 1 <= len(log) <= 3 and s1["best_epoch"] == min(range(len(log)), key=lambda i: log[i]["val_loss"])
    model, meta = load_lornet(root / "checkpoints" / "best.pt")
    assert meta["target"]["tau"] == man["tau"] and meta["training"]["selection"] == "lowest val_loss"
    assert meta["training"]["splits"]["val"]["ids_sha256"] == man["splits"]["val"]["ids_sha256"]
    ev = T.cmd_evaluate(cfg, log=quiet)
    assert {"mae", "spearman_pooled", "ece"} <= set(ev["lornet"]) and "mae" not in ev["heuristic_O"]
    assert ev["n_parameters"] == parameter_count(model)
    # same seed and cache -> identical training
    cfg2 = _train_cfg(tmp_path, "run2")
    T.cmd_cache(cfg2, vae=tiny_vae, log=quiet)
    s2 = T.cmd_train(cfg2, log=quiet)
    log2 = [json.loads(l) for l in (Path(cfg2["output_dir"]) / "train_log.jsonl").read_text().splitlines()]
    assert [r["val_loss"] for r in log] == [r["val_loss"] for r in log2] and s1["best_val_loss"] == s2["best_val_loss"]


def test_training_config_validation(tmp_path):
    from experiments.training import train_lornet as T
    cfg = _train_cfg(tmp_path)
    cfg["train"]["loss"] = "gaussian_nll"
    with pytest.raises(ValueError, match="uncertainty"):
        T.validate_config(cfg)
    cfg["train"]["loss"] = "l7"
    with pytest.raises(ValueError):
        T.validate_config(cfg)
    c = T.load_config(Path(T.__file__).parent / "configs" / "lornet.yaml", ["data.sources.0.hr_dir=/x", "train.epochs=2"])
    assert c["data"]["sources"][0]["hr_dir"] == "/x" and c["train"]["epochs"] == 2
    with pytest.raises(KeyError):
        T.load_config(Path(T.__file__).parent / "configs" / "lornet.yaml", ["train.nonsense=1"])


@pytest.mark.parametrize("loss", ["charbonnier", "bce", "gaussian_nll"])
def test_losses(loss):
    from experiments.training.metrics import reliability_loss
    logit = torch.zeros(2, 1, 4, 4, requires_grad=True)
    logvar = torch.zeros(2, 1, 4, 4) if loss == "gaussian_nll" else None
    total, parts = reliability_loss(loss, logit, logvar, torch.full((2, 1, 4, 4), 0.8), 0.1)
    total.backward()
    assert torch.isfinite(total) and float(logit.grad.abs().sum()) > 0 and float(parts["tv"]) == 0.0


def test_agreement_metrics():
    from experiments.training.metrics import ece, quartile_contrast, spearman, summarise_maps
    a = torch.rand(100)
    assert abs(spearman(a, a) - 1) < 1e-9 and abs(spearman(a, -a) + 1) < 1e-9
    assert quartile_contrast(a, a) > 0.4 and ece(a, a) < 1e-9
    E = torch.rand(8, 8)
    s = summarise_maps([torch.exp(-E)], [torch.exp(-E)], [E])
    assert s["mae"] == 0 and s["E_bottom_minus_top_quartile"] > 0


# --------------------------------------------------------------------------- #
# SFR-OC integration
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    from experiments.tests.tiny_pipeline import build_pipeline
    return build_pipeline(tmp_path_factory.mktemp("tok_lornet_pipe"))


def _img(seed, size=32):
    from PIL import Image
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8))


def _run(pipe, img, steps=3):
    from experiments.tests.tiny_pipeline import call_kwargs
    ram = torch.randn(1, 5, 512, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(7)
    return pipe("a, clean", img, **call_kwargs(img, steps, torch.Generator().manual_seed(7), ram)).images[0]


def _with(pipe, cfg, img):
    from cfa_seesr.hooks import MethodHooks, MethodState
    hooks, state = MethodHooks(), MethodState()
    hooks.install_observation_consistency(pipe, cfg, state)
    try:
        return _run(pipe, img), state.sfroc.summary(), state.sfroc
    finally:
        hooks.remove()


def test_learned_reliability_in_the_pipeline(pipe, tmp_path):
    torch.manual_seed(5)
    p, model = _save(tmp_path)
    img = _img(2)
    ref = _run(pipe, img)
    learned = SFROCConfig.from_dict({"enabled": True, "strength": 0.6, "schedule": "constant",
                                     "reliability": "learned", "learned": {"checkpoint": str(p)},
                                     "fusion": {"map": "observation"}})
    out, s, ctrl = _with(pipe, learned, img)
    with torch.no_grad():
        expect = model.eval()(ctrl.image[0:1].float())[0, 0]
    assert torch.allclose(ctrl.O_latent, expect, atol=1e-6) and torch.allclose(ctrl.W[0, 0], expect, atol=1e-6)
    assert s["reliability"] == "learned" and len(s["lornet_sha256"]) == 64 and s["lornet_tau"] == 0.01
    assert not np.array_equal(out, ref)
    heur, _, _ = _with(pipe, SFROCConfig.from_dict({"enabled": True, "strength": 0.6, "schedule": "constant",
                                                    "fusion": {"map": "observation"}}), img)
    assert not np.array_equal(out, heur)
    blend = SFROCConfig.from_dict({"enabled": True, "strength": 0.6, "reliability": "blend",
                                   "learned": {"checkpoint": str(p), "blend": 0.25}})
    _, sb, cb = _with(pipe, blend, img)
    assert sb["reliability"] == "blend" and {"learned", "noise"} <= set(cb.components)
    # zero strength with LORNet installed is still exactly V1 (here: the unhooked pipeline)
    zero = SFROCConfig.from_dict({"enabled": True, "strength": 0.0, "reliability": "learned",
                                  "learned": {"checkpoint": str(p)}})
    z, sz, _ = _with(pipe, zero, img)
    assert np.array_equal(z, ref) and sz["steps_corrected"] == 0


def test_runner_rejects_missing_or_invalid_checkpoint(tmp_path):
    from experiments import run_inference as ri
    from experiments.common import load_yaml
    cfg = load_yaml(Path(ri.__file__).parent / "configs" / "baseline.yaml")
    cfg["method"]["sfr_oc"].update(enabled=True, strength=0.5, reliability="learned")
    cfg["method"]["sfr_oc"]["learned"]["checkpoint"] = str(tmp_path / "missing.pt")
    with pytest.raises(ri.ConfigError, match="LORNet"):
        ri.method_config(cfg)
    cfg["method"]["sfr_oc"]["learned"]["checkpoint"] = None
    with pytest.raises(ri.ConfigError, match="checkpoint"):
        ri.method_config(cfg)
    p, _ = _save(tmp_path)
    cfg["method"]["sfr_oc"]["learned"]["checkpoint"] = str(p)
    assert ri.method_config(cfg).sfr_oc.uses_learned


# --------------------------------------------------------------------------- #
# Tools: map validity and strength selection
# --------------------------------------------------------------------------- #
def test_map_validity_on_tiny_vae(tiny_vae, tmp_path):
    from PIL import Image

    from experiments.tools.sfroc_map_validity import aggregate, evaluate_image
    rng = np.random.default_rng(0)
    rows = []
    p, _ = _save(tmp_path)
    net, _ = load_lornet(p)
    for i in range(12):
        gt = Image.fromarray((rng.random((32, 32, 3)) * 255).astype(np.uint8)).resize((128, 128), Image.BICUBIC)
        lr = gt.resize((32, 32), Image.BICUBIC)
        sr = Image.fromarray(np.clip(np.asarray(gt) + rng.normal(0, 25, (128, 128, 3)), 0, 255).astype(np.uint8))
        row, names = evaluate_image(tiny_vae, sr, gt, lr, S=np.random.rand(8, 8), lornet=net, process_size=128)
        rows.append(row)
    assert set(names) == {"heuristic_O", "learned_O", "semantic_1mS", "fused_heuristic", "fused_learned"}
    agg = aggregate(rows, names)
    assert all(v["decision"] in ("SUPPORTED", "UNSUPPORTED") for v in agg.values())
    assert aggregate(rows[:3], names)["heuristic_O"]["decision"].startswith("INSUFFICIENT")
    assert all(math.isfinite(r["benefit_mean"]) for r in rows)


def test_map_validity_refuses_final_test(capsys):
    from experiments.tools import sfroc_map_validity as mv
    with pytest.raises(SystemExit):
        mv.main(["--run_dir", "x", "--lr_dir", "x", "--gt_dir", "x", "--split_label", "test", "--output", "o.json"])


def test_strength_selection_rule():
    from experiments.tools.sfroc_screening import CANDIDATES, select

    def cell(dpsnr, dlpips, ddists):
        return {name: {"psnr": {"mean_diff": dpsnr[i]}, "lpips": {"mean_diff": dlpips[i]},
                       "dists": {"mean_diff": ddists[i]}} for i, name in enumerate(CANDIDATES)}
    # all admissible; 0.5 and 1.0 within the tie margin of the best -> smallest of them (0.5)
    s, _ = select({"a": cell([0.10, 0.30, 0.32], [0, 0, 0], [0, 0, 0])})
    assert s == 0.5
    # 1.0 exceeds the perceptual budget in one cell -> excluded
    s, t = select({"a": cell([0.1, 0.2, 0.5], [0, 0, 0.01], [0, 0, 0]), "b": cell([0.1, 0.2, 0.5], [0, 0, 0], [0, 0, 0])})
    assert s == 0.5 and not t["v2_heur_sem_g100"]["admissible"]
    # nothing improves PSNR -> NONE
    s, _ = select({"a": cell([-0.1, 0.0, -0.2], [0, 0, 0], [0, 0, 0])})
    assert s is None


def test_screening_set_comes_from_the_validation_split(tmp_path):
    from PIL import Image

    from experiments.training import data as D
    from experiments.training.make_screening_set import build
    cfg = _train_cfg(tmp_path)
    m = build(cfg, n=5, crop_hr=64, output=tmp_path / "screen")
    pairs = D.discover_pairs(cfg["data"]["sources"])
    train, val = D.split_pairs(pairs, cfg["data"]["val_fraction"])
    ids = {i["id"] for v in m["images"].values() for i in v}
    assert ids and ids <= {p["id"] for p in val} and ids.isdisjoint({p["id"] for p in train})
    for src, items in m["images"].items():
        for it in items:
            stem = it["stem"]
            lr = Image.open(tmp_path / "screen" / src / "lr" / f"{stem}.png")
            gt = Image.open(tmp_path / "screen" / src / "gt" / f"{stem}.png")
            assert gt.size == (lr.size[0] * 4, lr.size[1] * 4)
