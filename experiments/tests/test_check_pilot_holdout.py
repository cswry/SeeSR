"""check_pilot_holdout: identity, byte-duplicate and crop-search evidence on synthetic folders."""
import json

import numpy as np
from PIL import Image

from experiments.common import file_record
from experiments.tools import check_pilot_holdout as ch


def _img(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _setup(tmp_path, leak):
    rng = np.random.default_rng(0)
    test_lr = tmp_path / "test_LR"
    big = rng.integers(0, 255, (96, 120, 3))          # a "test scene" at LR resolution
    crop = big[10:42, 20:52]
    _img(test_lr / "Canon_001_LR4.png", crop)
    _img(test_lr / "Nikon_002_LR4.png", rng.integers(0, 255, (32, 32, 3)))
    pilot = tmp_path / "pilot_lr"
    _img(pilot / "Canon_001_LR4.png", crop)
    train = tmp_path / "Train" / "4"
    for i in range(3):                                  # HR-size LR4 images (registered, 4x larger)
        _img(train / f"Canon_{i:03d}_LR4.png", rng.integers(0, 255, (384, 480, 3)))
        _img(train / f"Canon_{i:03d}_HR.png", rng.integers(0, 255, (384, 480, 3)))
    if leak:   # the test scene is in the training data, registered at HR size
        _img(train / "Canon_009_LR4.png",
             np.asarray(Image.fromarray(big.astype(np.uint8)).resize((480, 384), Image.BICUBIC)))
    return pilot, test_lr, train


def _run(tmp_path, pilot, test_lr, train, expect=None):
    out = tmp_path / "ev.json"
    args = ["--pilot-lr-dir", str(pilot), "--test-lr-dir", str(test_lr), "--train-dir", str(train),
            "--out", str(out)]
    if expect:
        args += ["--expect-lr-fingerprint", expect]
    return ch.main(args), json.loads(out.read_text())


def test_held_out_when_identical_and_absent(tmp_path):
    pilot, test_lr, train = _setup(tmp_path, leak=False)
    rc, ev = _run(tmp_path, pilot, test_lr, train, expect=file_record(pilot)["fingerprint"])
    assert rc == 0 and ev["verdict"]["held_out"]
    assert ev["lr_identity"][0]["identical_to_test_file"]
    assert ev["lr_dir_fingerprint"]["match"] is True
    c = ev["content_overlap"]["per_pilot_image"]["Canon_001_LR4.png"]
    assert c["n_searched"] == 3 and c["max_ncc"] < 0.5


def test_crop_of_training_image_is_found(tmp_path):
    pilot, test_lr, train = _setup(tmp_path, leak=True)
    rc, ev = _run(tmp_path, pilot, test_lr, train)
    c = ev["content_overlap"]["per_pilot_image"]["Canon_001_LR4.png"]
    assert rc == 1 and ev["verdict"]["training_overlap_found"]
    assert c["top"][0]["training_file"].endswith("Canon_009_LR4.png") and c["max_ncc"] > 0.95
    assert c["top"][0]["xy_lr"] == [20, 10]


def test_byte_duplicate_and_fingerprint_mismatch(tmp_path):
    pilot, test_lr, train = _setup(tmp_path, leak=False)
    (train / "copy.png").write_bytes((pilot / "Canon_001_LR4.png").read_bytes())
    rc, ev = _run(tmp_path, pilot, test_lr, train, expect="0" * 64)
    assert rc == 1
    assert ev["byte_duplicates"]["duplicates"][0]["training_file"].endswith("copy.png")
    assert ev["lr_dir_fingerprint"]["match"] is False


def test_different_file_is_not_identical(tmp_path):
    pilot, test_lr, train = _setup(tmp_path, leak=False)
    _img(pilot / "Canon_001_LR4.png", np.zeros((32, 32, 3)) + 7)
    rc, ev = _run(tmp_path, pilot, test_lr, train)
    assert rc == 1 and not ev["verdict"]["pilot_is_stablesr_test_file"]


def test_ncc_exact_location():
    rng = np.random.default_rng(1)
    img = rng.random((50, 60))
    s, xy = ch.ncc_max(img, img[7:27, 11:31])
    assert abs(s - 1.0) < 1e-6 and xy == [11, 7]


def test_private_paths_are_masked():
    assert ch.public("/content/drive/MyDrive/seesr/datasets/x") == "<DRIVE_ROOT>/datasets/x"


def test_lowercase_lr_names_are_searched(tmp_path):
    pilot, test_lr, train = _setup(tmp_path, leak=False)
    _img(train / "nikon_050_lr4.png", np.random.default_rng(5).integers(0, 255, (384, 480, 3)))
    _img(train / "nikon_050_hr.png", np.random.default_rng(6).integers(0, 255, (384, 480, 3)))
    rc, ev = _run(tmp_path, pilot, test_lr, train)
    c = ev["content_overlap"]["per_pilot_image"]["Canon_001_LR4.png"]
    assert rc == 0 and c["n_training_lr_images"] == 4 == c["n_searched"] == c["n_training_hr_images"]
    assert ev["byte_duplicates"]["n_training_files_hashed"] == 8


def test_missed_lr_files_fail_coverage(tmp_path):
    pilot, test_lr, train = _setup(tmp_path, leak=False)
    _img(train / "Canon_077_HR.png", np.random.default_rng(7).integers(0, 255, (384, 480, 3)))
    _img(train / "Canon_077_degraded.png", np.random.default_rng(8).integers(0, 255, (384, 480, 3)))
    rc, ev = _run(tmp_path, pilot, test_lr, train)
    c = ev["content_overlap"]["per_pilot_image"]["Canon_001_LR4.png"]
    assert c["n_training_lr_images"] == 3 and c["n_training_hr_images"] == 4
    assert rc == 1 and not ev["verdict"]["training_search_complete"] and not ev["verdict"]["held_out"]


def test_unsearchable_image_fails_coverage(tmp_path):
    pilot, test_lr, train = _setup(tmp_path, leak=False)
    _img(train / "Canon_088_LR4.png", np.zeros((20, 20, 3)) + 3)     # smaller than the 32 px template
    _img(train / "Canon_088_HR.png", np.zeros((20, 20, 3)) + 3)
    rc, ev = _run(tmp_path, pilot, test_lr, train)
    c = ev["content_overlap"]["per_pilot_image"]["Canon_001_LR4.png"]
    assert c["n_searched"] == 3 and c["not_searched"][0].endswith("Canon_088_LR4.png")
    assert rc == 1 and not ev["verdict"]["held_out"]
