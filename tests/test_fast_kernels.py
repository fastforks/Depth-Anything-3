"""Tests of the custom-kernel path (depth_anything_3.fast): kernels against their plain PyTorch ops, FastDA3 against
DepthAnything3.inference, kernels on / off / unavailable.

Needs a GPU; the end-to-end tests also need DA3_WEIGHTS=/path/to/DA3MONO-LARGE (config.json + model.safetensors) and skip
without it. The flash-attention and cuDNN-convolution kernels need `pip install nvidia-cudnn-frontend` for the convolutions.
"""

import os
import warnings
import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

WEIGHTS = os.environ.get("DA3_WEIGHTS", "")
needs_weights = pytest.mark.skipif(
    not os.path.isfile(os.path.join(WEIGHTS, "model.safetensors")),
    reason="set DA3_WEIGHTS to a DA3 snapshot directory (model.safetensors + config.json)",
)
ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "examples", "SOH"
)


def _images():
    out = {}
    for name in ("000.png", "010.png"):
        im = cv2.cvtColor(cv2.imread(os.path.join(ASSETS, name)), cv2.COLOR_BGR2RGB)
        out[name[:3] + "_full"] = im
        out[name[:3] + "_500x375"] = cv2.resize(im, (500, 375), interpolation=cv2.INTER_AREA)
    base = out["000_full"]
    out["portrait_251x333"] = cv2.resize(base, (251, 333), interpolation=cv2.INTER_AREA)
    out["small_97x61"] = cv2.resize(base, (97, 61), interpolation=cv2.INTER_AREA)
    out["constant"] = np.full((300, 400, 3), 127, np.uint8)
    return out


IMAGES = _images()


# ----------------------------------------------------------------------------- kernels against plain PyTorch
@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize(
    "src,dst",
    [((24, 36), (48, 72)), ((13, 17), (26, 34)), ((36, 36), (72, 72)), ((7, 9), (5, 20))],
)
def test_bilinear_kernel_equals_interpolate(align, src, dst):
    from depth_anything_3.fast import bilerp

    if not bilerp.available():
        pytest.skip("bilinear kernel cannot be loaded here")
    torch.manual_seed(0)
    x = (
        torch.randn(2, 64, *src, device="cuda")
        .half()
        .contiguous(memory_format=torch.channels_last)
    )
    got = bilerp.interp_bilinear(x, dst, align)
    ref = F.interpolate(x, size=dst, mode="bilinear", align_corners=align)
    assert got.shape == ref.shape and got.is_contiguous(memory_format=torch.channels_last)
    assert (
        got.float() - ref.float()
    ).abs().max() <= 2e-3  # one fp16 rounding step of different evaluation order


def test_bilinear_kernel_fallback_for_other_layouts():
    from depth_anything_3.fast import bilerp

    x = torch.randn(1, 12, 8, 8, device="cuda").half()  # C % 8 != 0, NCHW
    got = bilerp.interp_bilinear(x, (16, 16), True)
    assert torch.equal(got, F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=True))


@pytest.mark.parametrize("n", [1, 63, 64, 777, 1296, 5000])
def test_flash_attention_hh_vs_sdpa(n):
    from depth_anything_3.fast.fastbb_vendored.attention import attention_bhnd

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 16, n, 64, device="cuda").half() for _ in range(3))
    got = attention_bhnd(q, k, v, variant="hh")
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    assert torch.isfinite(got).all()
    assert (got.float() - ref).abs().max() < 2e-2


@pytest.mark.parametrize("kind,interp", [("area", cv2.INTER_AREA), ("cubic", cv2.INTER_CUBIC)])
@pytest.mark.parametrize("hw", [(333, 251), (680, 1208), (150, 203), (900, 700)])
def test_cv2_resize_gather_equivalent(kind, interp, hw):
    from depth_anything_3.fast.resize import cv2_resize

    img = IMAGES["000_full"]
    h, w = hw
    if (kind == "area") != (h < img.shape[0] and w < img.shape[1]):
        pytest.skip("cv2 picks INTER_AREA only for downscaling")
    ref = cv2.resize(img, (w, h), interpolation=interp).astype(np.float32)
    got = cv2_resize(torch.from_numpy(img).permute(2, 0, 1).cuda(), h, w, kind)
    diff = np.abs(got.permute(1, 2, 0).cpu().numpy() - ref)
    assert diff.max() <= 1.0 and diff.mean() < 0.05


# ----------------------------------------------------------------------------- FastDA3 against upstream
@pytest.fixture(scope="module")
def ref_api():
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained(WEIGHTS).cuda().eval()


@pytest.fixture(scope="module")
def fast_kernels():
    from depth_anything_3.fast import load_fast

    return load_fast(WEIGHTS, kernels=True)


@pytest.fixture(scope="module")
def fast_plain():
    from depth_anything_3.fast import load_fast

    return load_fast(WEIGHTS, kernels=False)


def _rel_diff(a, b):
    return float((a - b).abs().mean() / b.abs().mean())


def _depth(fast, img, res):
    x = torch.from_numpy(img).permute(2, 0, 1).cuda()
    return fast.infer([x], process_res=res)["depth"][0]


@needs_weights
@pytest.mark.parametrize("name", list(IMAGES))
@pytest.mark.parametrize("res", [504, 1024])
def test_parity_vs_upstream(ref_api, fast_kernels, fast_plain, name, res):
    img = IMAGES[name]
    up = torch.from_numpy(ref_api.inference([img], process_res=res).depth[0]).cuda()
    for fast in (fast_kernels, fast_plain):
        mine = _depth(fast, img, res)
        assert mine.shape == up.shape and torch.isfinite(mine).all()
        d = _rel_diff(mine, up)
        print(f"{name} @{res} kernels={fast.kernels}: mean |diff| / mean |upstream| = {d:.5f}")
        assert d < 0.02


@needs_weights
def test_kernels_on_vs_off(fast_kernels, fast_plain):
    img = IMAGES["010_full"]
    assert _rel_diff(_depth(fast_kernels, img, 504), _depth(fast_plain, img, 504)) < 0.01


@needs_weights
def test_repeated_calls_and_sizes_are_stable(fast_kernels):
    """Different sizes in one process (no per-size state), and the same input twice gives the same output."""
    a = _depth(fast_kernels, IMAGES["000_500x375"], 504)
    _depth(fast_kernels, IMAGES["portrait_251x333"], 336)
    b = _depth(fast_kernels, IMAGES["000_500x375"], 504)
    assert _rel_diff(a, b) < 1e-3  # not bitwise: cuDNN may pick different algorithms between calls


@needs_weights
def test_fallback_when_kernels_unavailable(monkeypatch):
    import depth_anything_3.fast.da3 as da3
    from depth_anything_3.fast import load_fast

    monkeypatch.setattr(da3, "_probe_kernels", lambda: (False, False, False))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fast = load_fast(WEIGHTS, kernels=True)
    assert any("falling back to plain PyTorch" in str(x.message) for x in w)
    d = _depth(fast, IMAGES["000_500x375"], 504)
    assert torch.isfinite(d).all()


@needs_weights
def test_env_opt_out(monkeypatch):
    from depth_anything_3.fast import load_fast

    monkeypatch.setenv("DA3_FAST_KERNELS", "0")
    assert load_fast(WEIGHTS).kernels is False
