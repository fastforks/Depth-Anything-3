"""Tests of the plain-PyTorch fast path (utils/gpu_inference.py, pos-embed matrices, DPT out_conv commute).

Needs a GPU and a local DA3 snapshot: DA3_WEIGHTS=/path/to/DA3MONO-LARGE pytest tests/test_fast.py
(the weights are not part of this repository; the tests that do not need them run on any CUDA machine).
"""

import os
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
    """Scene photos from the repository's assets at several sizes, not all multiples of 14, plus a constant image."""
    out = {}
    for name in ("000.png", "010.png"):
        im = cv2.cvtColor(cv2.imread(os.path.join(ASSETS, name)), cv2.COLOR_BGR2RGB)
        out[name[:3] + "_full"] = im
        out[name[:3] + "_500x375"] = cv2.resize(im, (500, 375), interpolation=cv2.INTER_AREA)
    base = out["000_full"]
    out["portrait_251x333"] = cv2.resize(base, (251, 333), interpolation=cv2.INTER_AREA)
    out["small_97x61"] = cv2.resize(
        base, (97, 61), interpolation=cv2.INTER_AREA
    )  # upscaled to 504
    out["constant"] = np.full((300, 400, 3), 127, np.uint8)
    return out


IMAGES = _images()


# ----------------------------------------------------------------------------- resize equivalence (no weights needed)
@pytest.mark.parametrize("kind,interp", [("area", cv2.INTER_AREA), ("cubic", cv2.INTER_CUBIC)])
@pytest.mark.parametrize("hw", [(333, 251), (680, 1208), (150, 203), (900, 700)])
def test_cv2_resize_equivalent(kind, interp, hw):
    from depth_anything_3.utils.gpu_inference import cv2_resize

    img = IMAGES["000_full"]
    h, w = hw
    if (kind == "area") != (h < img.shape[0] and w < img.shape[1]):
        pytest.skip("cv2 picks INTER_AREA only for downscaling")
    ref = cv2.resize(img, (w, h), interpolation=interp).astype(np.float32)
    got = cv2_resize(torch.from_numpy(img).permute(2, 0, 1).cuda(), h, w, kind)
    diff = (got.permute(1, 2, 0).cpu().numpy() - ref).__abs__()
    assert diff.max() <= 1.0  # fixed-point rounding of cv2: at most 1 LSB
    assert diff.mean() < 0.05


def test_proc_sizes_multiples_of_14():
    from depth_anything_3.utils.gpu_inference import proc_sizes

    for H, W in [(680, 1208), (375, 500), (333, 251), (61, 97), (504, 504), (40, 1000)]:
        (h1, w1), (h2, w2) = proc_sizes(H, W, 504)
        assert max(h1, w1) == 504 or (H, W) == (h1, w1)
        assert h2 % 14 == 0 and w2 % 14 == 0 and h2 >= 14 and w2 >= 14


# ----------------------------------------------------------------------------- pos-embed matrices vs F.interpolate
@pytest.mark.parametrize("grid", [(36, 36), (36, 48), (21, 17), (73, 5)])
def test_pos_embed_matrices_equal_interpolate(grid):
    from depth_anything_3.model.dinov2.vision_transformer import _bicubic_matrix

    M, offset, dim = 37, 0.1, 8
    torch.manual_seed(0)
    emb = torch.randn(1, dim, M, M, device="cuda")
    h0, w0 = grid
    ry, rx = _bicubic_matrix(w0, offset, M, emb.device), _bicubic_matrix(h0, offset, M, emb.device)
    got = ry @ emb @ rx.t()
    ref = F.interpolate(
        emb,
        scale_factor=(float(w0 + offset) / M, float(h0 + offset) / M),
        mode="bicubic",
        antialias=False,
    )
    assert got.shape == ref.shape
    assert (got - ref).abs().max() < 1e-4


def test_export_native_grid_is_identity():
    """the dynamic export keeps the eager native-grid shortcut (37 x 37 tokens -> pos_embed unchanged)"""
    from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer

    m = DinoVisionTransformer(
        img_size=518, patch_size=14, embed_dim=32, depth=1, num_heads=2, interpolate_offset=0.1
    ).eval()
    m.pos_embed.data = torch.randn_like(m.pos_embed)

    class Wrap(torch.nn.Module):
        def forward(self, grid):  # [1, 1, gh, gw] token-grid carrier
            gh, gw = grid.shape[-2:]
            return m.interpolate_pos_encoding(torch.zeros(1, gh * gw + 1, 32), gw * 14, gh * 14)

    gh, gw = torch.export.Dim("gh", min=2, max=60), torch.export.Dim("gw", min=2, max=60)
    ep = torch.export.export(
        Wrap(), (torch.zeros(1, 1, 19, 28),), dynamic_shapes={"grid": {2: gh, 3: gw}}
    ).module()
    for g in ((19, 28), (37, 20), (37, 37)):
        x = torch.zeros(1, 1, *g)
        assert torch.allclose(ep(x), Wrap()(x), atol=1e-6)
    assert torch.equal(ep(torch.zeros(1, 1, 37, 37)), m.pos_embed)


def test_fp32_head_conv_overrides_autocast():
    from depth_anything_3.utils.gpu_inference import _Float32

    layer = _Float32(torch.nn.Conv2d(3, 1, 1))
    x = torch.randn(1, 3, 5, 7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = layer(x)
    assert out.dtype == torch.float32 and torch.equal(out, layer(x))


# ----------------------------------------------------------------------------- end-to-end parity vs upstream
@pytest.fixture(scope="module")
def models():
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils import gpu_inference as gi

    ref = DepthAnything3.from_pretrained(WEIGHTS).cuda().eval()
    fast = DepthAnything3.from_pretrained(WEIGHTS).cuda().eval()
    gi.prepare_half(fast.model)
    return ref, fast, gi


def _rel_diff(a, b):
    return float((a - b).abs().mean() / b.abs().mean())


@needs_weights
@pytest.mark.parametrize("name", list(IMAGES))
@pytest.mark.parametrize("res", [504, 1024])
def test_parity_vs_upstream(models, name, res):
    ref, fast, gi = models
    img = IMAGES[name]
    up = torch.from_numpy(ref.inference([img], process_res=res).depth[0]).cuda()
    mine = gi.infer_depth(fast.model, torch.from_numpy(img).permute(2, 0, 1).cuda(), res)
    assert mine.shape == up.shape
    assert torch.isfinite(mine).all()
    # fp16 vs bf16 autocast: the same network up to precision (a constant image has no content, but the network's
    # output on it is not flat, so it is compared to upstream like the others)
    d = _rel_diff(mine, up)
    print(f"{name} @{res}: mean |diff| / mean |upstream| = {d:.5f}")
    assert d < 0.01  # measured <= 0.008 (fp16 vs bf16 autocast)


@needs_weights
def test_preprocess_matches_input_processor(models):
    ref, _, gi = models
    for name in ("000_500x375", "portrait_251x333", "small_97x61", "constant"):
        img = IMAGES[name]
        want, _, _ = ref.input_processor([img], None, None, 504, "upper_bound_resize")
        got, hw = gi.preprocess(torch.from_numpy(img).permute(2, 0, 1).cuda(), 504)
        assert tuple(want.shape[-2:]) == hw
        # <= 1 LSB of the 8-bit image in normalized units
        assert (got[0].cpu() - want).abs().max() <= 1.0 / (0.224 * 255) + 1e-4


@needs_weights
def test_postprocess_core_split_equals_forward(models):
    _, fast, gi = models
    net = fast.model
    x, _ = gi.preprocess(torch.from_numpy(IMAGES["000_500x375"]).permute(2, 0, 1).cuda(), 504)
    with torch.inference_mode():
        torch.manual_seed(0)
        full = gi.network(net, x)["depth"]
        torch.manual_seed(0)
        out = gi.postprocess(net, gi.network_core(net, x.half()))["depth"]
    assert torch.allclose(full.float(), out.float(), rtol=1e-3, atol=1e-3)
