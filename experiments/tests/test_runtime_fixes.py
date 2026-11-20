"""Regression tests for the GPU runtime fixes found in the CFA-SeeSR V2.1 Colab pilots (method version 2.1.1).

On a CPU-only machine the device half of each fix cannot fail, so each fix is also exercised through the dtype
(or autograd) path that fails the same way on CPU; the CUDA variants run only when a GPU is present.
"""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from cfa_seesr import SpatialSemanticConfig  # noqa: E402
from cfa_seesr.observation_reliability import analyse, low_gradient_mask  # noqa: E402
from cfa_seesr.spatial_reliability import gaussian_smooth, reliability_grid  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available (device path unverified here)")


def _maps(n=3, g=12, dtype=torch.float64, device="cpu", seed=0):
    m = torch.rand(n, g, g, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return (m / m.sum(dim=(1, 2), keepdim=True)).to(device=device, dtype=dtype)


# Fix 1: confidences follow the attention maps' device and dtype
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_reliability_grid_keeps_the_maps_dtype(dtype):
    R, st = reliability_grid(_maps(dtype=dtype), [0.2, 0.9, 0.5], 0.6, 0.6, SpatialSemanticConfig(enabled=True))
    assert R.dtype == dtype and torch.isfinite(R).all() and 0 <= st["min"] <= st["max"] <= 1


def test_reliability_grid_float64_result_unchanged():
    """Same numbers as the pre-fix float64 path (maps are float64 in the pipeline)."""
    maps, w = _maps(), [0.2, 0.9, 0.5]
    cfg = SpatialSemanticConfig(enabled=True, smoothing_sigma=1.0, contrast=1.0)
    R, _ = reliability_grid(maps, w, 0.6, 0.6, cfg)
    wt = torch.tensor(w, dtype=torch.float64)
    mass = maps.sum(0)
    rho = (maps * wt[:, None, None]).sum(0) / (mass + cfg.eps)
    kappa = (mass * 144 / 3).clamp(max=1.0)
    r = gaussian_smooth(kappa * rho + (1 - kappa) * 0.6, 1.0)
    assert torch.allclose(R, (0.6 + (r - r.mean())).clamp(0, 1), atol=1e-12)


@CUDA
def test_reliability_grid_on_cuda():
    R, _ = reliability_grid(_maps(device="cuda"), [0.2, 0.9, 0.5], 0.6, 0.6, SpatialSemanticConfig(enabled=True))
    assert R.device.type == "cuda"


# Fix 2: Gaussian kernel on the grid's device (and dtype)
def test_gaussian_smooth_follows_grid_dtype():
    g32 = torch.rand(12, 12)
    out = gaussian_smooth(g32, 1.5)                       # float64 kernel on a float32 grid used to fail
    assert out.dtype == torch.float32
    assert torch.allclose(out.double(), gaussian_smooth(g32.double(), 1.5), atol=1e-6)


@CUDA
def test_gaussian_smooth_on_cuda():
    assert gaussian_smooth(torch.rand(12, 12, device="cuda", dtype=torch.float64), 1.0).device.type == "cuda"


# Fix 3: SSR grid copied to the CPU (and detached) before NumPy serialisation
def test_grid_to_numpy_detaches_and_copies():
    from experiments.run_inference import grid_to_numpy
    g = torch.rand(12, 12, dtype=torch.float64, requires_grad=True)
    with pytest.raises(RuntimeError):
        g.numpy()                                         # what the old line would have hit for such a grid
    a = grid_to_numpy(g)
    assert isinstance(a, np.ndarray) and a.dtype == np.float64 and np.allclose(a, g.detach().numpy())


@CUDA
def test_grid_to_numpy_from_cuda():
    from experiments.run_inference import grid_to_numpy
    assert grid_to_numpy(torch.rand(4, 4, device="cuda")).shape == (4, 4)


# Fix 4: torch.quantile needs float32/float64 (half under CUDA autocast failed)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_low_gradient_mask_accepts_half_precision_gradients(dtype, monkeypatch):
    """Under CUDA autocast the Sobel convolutions return fp16. CPU kernels cannot run fp16 replicate padding or
    sqrt, so bfloat16 gradients (same quantile failure) are injected directly; fp16 runs in the CUDA test."""
    import cfa_seesr.observation_reliability as orl
    y = torch.rand(1, 1, 32, 32)
    with pytest.raises(RuntimeError):
        torch.quantile(y.to(dtype).flatten(), 0.5)        # the failure mode before the fix
    real = orl._sobel
    monkeypatch.setattr(orl, "_sobel", lambda t: tuple(g.to(dtype) for g in real(t)))
    m = low_gradient_mask(y, 0.5)
    monkeypatch.setattr(orl, "_sobel", real)
    assert m.dtype == torch.bool and m.shape == y.shape
    assert 0.4 < float(m.float().mean()) < 0.65            # about half of the pixels at q = 0.5


@CUDA
def test_low_gradient_mask_under_cuda_autocast():
    y = torch.rand(1, 1, 64, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        m = low_gradient_mask(y, 0.5)
    assert m.dtype == torch.bool


def test_analyse_with_a_half_precision_lr_tensor():
    img = torch.rand(1, 3, 64, 64)
    O, _, _, checks = analyse(img.half(), lr=torch.rand(1, 3, 16, 16).half())
    assert O.dtype == torch.float32 and torch.isfinite(O).all() and bool(checks["O_finite"])
