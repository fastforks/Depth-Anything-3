"""GPU-in / GPU-out monocular inference for DepthAnything3Net in plain PyTorch (network_core exports to ONNX / TensorRT).

    from depth_anything_3.utils.gpu_inference import prepare_half, infer_depth
    prepare_half(net)                                  # once: backbone + head fp16, last head convs fp32
    depth = infer_depth(net, img_u8, process_res=504)  # img_u8 [3, H, W] uint8 RGB on the GPU -> [h, w] fp32 depth

Same math as DepthAnything3.inference([img]) (upper_bound_resize, mono sky handling) with these changes:
  - preprocessing on the GPU: OpenCV's INTER_AREA (downscale) / INTER_CUBIC (upscale) resizes reproduced as two separable
    GEMMs with OpenCV's weights (uint8 rounding after each resize as cv2), ImageNet normalization on the GPU
  - pure fp16 backbone (upstream: bf16 autocast) and fp16 DPT head (upstream: fp32); the last 1x1 convs of the head in
    fp32; camera encoder / decoder fp32
  - the output stays a torch tensor at the processing resolution (no numpy round trip)
"""

import functools
import numpy as np
import torch
import torch.nn as nn

PATCH = 14
MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


# ------------------------------------------------------------------------------------------- cv2-equivalent resizes
def _area(n_in, n_out):
    s = n_in / n_out
    W = np.zeros((n_out, n_in))
    for o in range(n_out):
        a, b = o * s, (o + 1) * s
        for i in range(int(np.floor(a)), min(int(np.ceil(b)), n_in)):
            W[o, i] = max(0.0, min(b, i + 1) - max(a, i))
        W[o] /= W[o].sum()
    return W


def _cubic(n_in, n_out):
    s = n_in / n_out
    W = np.zeros((n_out, n_in))
    A = -0.75
    for o in range(n_out):
        x = (o + 0.5) * s - 0.5
        x0 = int(np.floor(x))
        t = x - x0
        ws = [
            ((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A,
            ((A + 2) * t - (A + 3)) * t * t + 1,
            ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1,
        ]
        ws.append(1 - sum(ws))
        for i, w in zip(range(x0 - 1, x0 + 3), ws):
            W[o, min(max(i, 0), n_in - 1)] += w
    return W


@functools.lru_cache(maxsize=64)
def _mat(n_in, n_out, kind, device):
    W = _area(n_in, n_out) if kind == "area" else _cubic(n_in, n_out)
    return torch.tensor(W, dtype=torch.float32, device=device)


def cv2_resize(img, h, w, kind):
    """img [..., H, W] uint8 / float on the GPU -> float32 [..., h, w] with uint8-rounded values (= cv2.resize with
    INTER_AREA ('area') or INTER_CUBIC ('cubic') on a uint8 image, up to cv2's fixed-point rounding: <= 1 LSB).
    """
    x = img.float()
    H, W = x.shape[-2:]
    if W != w:
        x = x @ _mat(W, w, kind, x.device).t()
    if H != h:
        x = _mat(H, h, kind, x.device) @ x
    return x.round_().clamp_(0, 255)


def proc_sizes(H, W, process_res):
    """InputProcessor upper_bound_resize: longest side -> process_res, then each side to the nearest multiple of 14."""
    s = process_res / float(max(W, H))
    w1, h1 = (
        (W, H)
        if max(W, H) == process_res
        else (max(1, int(round(W * s))), max(1, int(round(H * s))))
    )

    def nearest(x):
        d = (x // PATCH) * PATCH
        return d + PATCH if abs(d + PATCH - x) <= abs(x - d) else d

    return (h1, w1), (max(1, nearest(h1)), max(1, nearest(w1)))


def preprocess(img_u8, process_res=504, resize=cv2_resize):
    """[3, H, W] uint8 RGB on the GPU -> normalized [1, 1, 3, h, w] fp32 (as InputProcessor), (h, w).
    `resize(img, h, w, kind)` is pluggable (same semantics as cv2_resize)."""
    H, W = img_u8.shape[-2:]
    (h1, w1), (h2, w2) = proc_sizes(H, W, process_res)
    x = img_u8
    if (h1, w1) != (H, W):
        x = resize(x, h1, w1, "cubic" if process_res / float(max(W, H)) > 1.0 else "area")
    if (h2, w2) != (h1, w1):
        x = resize(x, h2, w2, "cubic" if (w2 > w1 or h2 > h1) else "area")
    mean = torch.tensor(MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(3, 1, 1)
    return ((x.float() - mean) / std)[None, None], (h2, w2)


# ------------------------------------------------------------------------------------------- fp16 network
class _Float32(nn.Module):
    """runs the wrapped module in fp32 (input cast up; output stays fp32); stays fp32 under parent .half()"""

    def __init__(self, m):
        super().__init__()
        self.m = m.float()

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.m(x.float())

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.m.float()
        return self


def _cast_inputs(dtype):
    def pre(mod, args, kwargs):
        c = lambda v: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
        return tuple(c(a) for a in args), {k: c(v) for k, v in kwargs.items()}

    return pre


def prepare_half(net):
    """DepthAnything3Net (or a nested net) in place: backbone + head fp16, last head 1x1 convs fp32, camera modules fp32."""
    nets = [net.da3, net.da3_metric] if hasattr(net, "da3") else [net]
    for n in nets:
        n.eval()
        n.backbone.half()
        n.backbone.register_forward_pre_hook(_cast_inputs(torch.float16), with_kwargs=True)
        n.head.half()
        s = n.head.scratch
        for name in ("output_conv2", "sky_output_conv2"):
            seq = getattr(s, name, None)
            if seq is not None and not isinstance(seq[-1], _Float32):
                seq[-1] = _Float32(seq[-1])
        ape = n.head._add_pos_embed  # (builds the embedding in fp32: keep the map dtype)
        n.head._add_pos_embed = lambda x, W, H, ratio=0.1, ape=ape: ape(x, W, H, ratio).to(x.dtype)
        n.head.register_forward_pre_hook(_cast_inputs(torch.float16), with_kwargs=True)
        for name in ("cam_dec", "cam_enc"):
            m = getattr(n, name, None)
            if m is not None:
                m.float().register_forward_pre_hook(_cast_inputs(torch.float32), with_kwargs=True)
    return net


def network(net, x):
    """eager mono forward incl. the sky handling (network_core: the exportable part): [1, 1, 3, h, w] -> output dict"""
    with torch.autocast("cuda", enabled=False):
        return net(x, None, None, [], False, False, "saddle_balanced")


@torch.inference_mode()
def infer_depth(net, img_u8, process_res=504):
    """[3, H, W] uint8 RGB (GPU) -> depth [h, w] fp32 at the processing resolution (upstream prediction.depth[0])."""
    x, _ = preprocess(img_u8, process_res)
    out = network(net, x)
    return out["depth"][0, 0].float()


def network_core(net, x):
    """exportable core of the mono forward: backbone + DPT head, without the data-dependent sky post-processing
    (random sampling + quantile; apply postprocess() to the result). x [1, 1, 3, h, w] -> dict of head outputs.
    """
    H, W = x.shape[-2], x.shape[-1]
    with torch.autocast("cuda", enabled=False):
        feats, _ = net.backbone(
            x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced"
        )
        return net._process_depth_head(feats, H, W)


def postprocess(net, out):
    """the rest of DepthAnything3Net.forward for a mono net (camera head absent): sky regions -> 99 % depth quantile."""
    return net._process_mono_sky_estimation(out)
