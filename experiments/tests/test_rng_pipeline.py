"""Behavioural RNG and wrapper tests on a tiny instance of the REAL upstream pipeline (CPU).

They exercise pipelines/pipeline_seesr.py end to end (CFG, LR-embedded start, DDPM loop, tiled
VAE hook) through the same hooks run_inference.py installs, and compare actual outputs.
"""
import functools
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from experiments import run_inference as ri  # noqa: E402
from experiments.tests.tiny_pipeline import build_pipeline, call_kwargs  # noqa: E402

STEPS_A, STEPS_B = 6, 3


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    return build_pipeline(tmp_path_factory.mktemp("tok"))


@pytest.fixture(scope="module")
def inputs():
    from PIL import Image
    rng = np.random.default_rng(0)
    imgs = {n: Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)) for n in ("img_a", "img_b")}
    ram = torch.randn(1, 5, 512, generator=torch.Generator().manual_seed(1))
    return imgs, ram


def hooked(pipe, ram, names, reseed_base=None):
    """Wrap `pipe` exactly as run_inference.install_hooks does (via a stand-in test_seesr module)."""
    rec = ri.Recorder(names, reseed_base=reseed_base)
    mod = types.ModuleType("fake_test_seesr")
    mod.load_seesr_pipeline = lambda *a, **k: pipe
    mod.load_tag_model = lambda *a, **k: None
    mod.get_validation_prompt = lambda *a, **k: ("a, clean", None)
    original_step = pipe.scheduler.step
    ri.install_hooks(mod, rec, {})
    timed = mod.load_seesr_pipeline()

    def run(image, steps, generator):
        mod.get_validation_prompt()  # starts the per-image record, as upstream main() does
        return timed("a, clean", image, **call_kwargs(image, steps, generator, ram)).images[0]

    def restore():
        ri.remove_hooks(mod)
        pipe.scheduler.step = original_step

    return rec, run, restore


def plain(pipe, image, steps, generator, ram):
    return pipe("a, clean", image, **call_kwargs(image, steps, generator, ram)).images[0]


def test_same_image_and_seed_is_repeatable(pipe, inputs):
    imgs, ram = inputs
    outs = []
    for _ in range(2):
        torch.manual_seed(7)
        outs.append(plain(pipe, imgs["img_a"], STEPS_A, torch.Generator().manual_seed(42), ram))
    assert np.array_equal(outs[0], outs[1])
    torch.manual_seed(7)
    other = plain(pipe, imgs["img_a"], STEPS_A, torch.Generator().manual_seed(43), ram)
    assert not np.array_equal(outs[0], other)


def test_every_scheduler_step_receives_the_pipeline_generator(pipe, inputs):
    """Spy below the wrapper: each real DDPM step call must get the generator passed to the pipeline."""
    imgs, _ = inputs
    seen = []
    real_step = pipe.scheduler.step

    @functools.wraps(real_step)
    def spy(model_output, timestep, sample, *a, **k):
        seen.append(k.get("generator"))
        return real_step(model_output, timestep, sample, *a, **k)

    pipe.scheduler.step = spy
    rec, run, restore = hooked(pipe, inputs[1], ["img_a"])
    try:
        g = torch.Generator().manual_seed(42)
        run(imgs["img_a"], STEPS_A, g)
    finally:
        restore()
        pipe.scheduler.step = real_step
    assert len(seen) == STEPS_A and all(x is g for x in seen)
    row = rec.rows[0]
    assert row["denoising_steps"] == STEPS_A and row["last_t"] == 1


def test_hooks_do_not_change_outputs(pipe, inputs):
    """Upstream mode: hooked pipeline output == unhooked output for the same RNG state."""
    imgs, ram = inputs
    torch.manual_seed(5)
    ref = plain(pipe, imgs["img_a"], STEPS_A, torch.Generator().manual_seed(42), ram)
    rec, run, restore = hooked(pipe, inputs[1], ["img_a"])
    try:
        torch.manual_seed(5)
        out = run(imgs["img_a"], STEPS_A, torch.Generator().manual_seed(42))
    finally:
        restore()
    assert np.array_equal(ref, out)


def test_reseed_controls_initial_latent_and_global_rng(pipe, inputs):
    """With reseeding, arbitrary prior generator/global RNG state must not change the output."""
    imgs, _ = inputs
    outs = []
    for disturb in (0, 1):
        rec, run, restore = hooked(pipe, inputs[1], ["img_a"], reseed_base=42)
        try:
            torch.manual_seed(1000 + disturb)          # global RNG (LR latent sample, torch.randint)
            g = torch.Generator().manual_seed(9 + disturb)  # generator (initial latent, DDPM noise)
            outs.append(run(imgs["img_a"], STEPS_A, g))
        finally:
            restore()
    assert np.array_equal(outs[0], outs[1])
    assert rec.rows[0]["seed"] == ri.image_seed(42, "img_a", 0)


def test_reseeding_one_image_does_not_affect_the_next(pipe, inputs):
    imgs, _ = inputs
    results = {}
    for label, sequence in {"alone": [("img_b", STEPS_B)],
                            "after_6": [("img_a", STEPS_A), ("img_b", STEPS_B)],
                            "after_3": [("img_a", STEPS_B), ("img_b", STEPS_B)]}.items():
        rec, run, restore = hooked(pipe, inputs[1], [n for n, _ in sequence], reseed_base=123)
        g = torch.Generator().manual_seed(123)  # one generator shared across images, as upstream main()
        try:
            for name, steps in sequence:
                out = run(imgs[name], steps, g)
        finally:
            restore()
        results[label] = out
    assert np.array_equal(results["alone"], results["after_6"])
    assert np.array_equal(results["alone"], results["after_3"])


def test_upstream_seeding_couples_images(pipe, inputs):
    """Documents the problem reseeding solves: with one RNG stream, image B depends on image A's step count."""
    imgs, ram = inputs
    outs = []
    for steps_a in (STEPS_A, STEPS_B):
        torch.manual_seed(42)
        g = torch.Generator().manual_seed(42)
        plain(pipe, imgs["img_a"], steps_a, g, ram)
        outs.append(plain(pipe, imgs["img_b"], STEPS_B, g, ram))
    assert not np.array_equal(outs[0], outs[1])


def test_image_seed_properties():
    assert ri.image_seed(42, "x") == ri.image_seed(42, "x")
    assert len({ri.image_seed(s, n) for s in (42, 123, 2026) for n in ("a", "b", "c")}) == 9
    assert ri.image_seed(42, "x", 0) != ri.image_seed(42, "x", 1)
    assert 0 <= ri.image_seed(2026, "y") < 2**31
