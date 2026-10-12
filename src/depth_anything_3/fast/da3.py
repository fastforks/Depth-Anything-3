"""Depth Anything 3 fast path with custom kernels (adapted from exp 004 mono_fast/da3.py @ 2171802): GPU in / GPU out,
any size, nothing compiled per size. Mono, metric, multi-view any-view and nested models.

Same network math as `DepthAnything3.inference`; changes:
- GPU in / GPU out: the upstream CPU preprocessing (cv2 INTER_AREA / INTER_CUBIC resize to the upper-bound size, a second
  resize to multiples of 14, ToTensor + ImageNet normalize) reproduced on the GPU (fast.resize: separable gathers with cv2
  weights, uint8 rounding after each resize like cv2); outputs stay torch tensors (no numpy round trip)
- backbone in pure fp16 (upstream: bf16 autocast) with residual + LayerScale as addcmul, fp16-accumulate GEMMs
  (`fp16_accumulate`), and the head_dim-64 flash-attention kernel with fp16 accumulation (`kernels`, vendored fastbb)
- DPT / DualDPT head in fp16 channels-last (upstream: fp32): FeatureFusionBlock does its 1x1 out_conv BEFORE the bilinear
  upsample (exact reorder: both linear, 4x fewer FLOPs); the final 1x1 conv + activation in fp32; with `kernels` the head
  convolutions are fp16-accumulate cuDNN graphs (pip install nvidia-cudnn-frontend) and the bilinear upsamples one NHWC kernel
- DualDPT: the auxiliary ray branch (only used with use_ray_pose; upstream deletes its outputs otherwise) is skipped

`kernels=False` (or the environment variable DA3_FAST_KERNELS=0) keeps plain PyTorch ops only (SDPA, F.interpolate, cuDNN
via PyTorch); a kernel that cannot be loaded on this machine falls back to its plain op with a warning.
"""

import contextlib
import os
import types
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_3.utils.gpu_inference import preprocess

from .resize import cv2_resize


@contextlib.contextmanager
def fp16_accumulate(enabled=True):
    """torch.backends.cuda.matmul.allow_fp16_accumulation (2x tensor-core rate on GeForce) for the duration."""
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


def _interp_plain(x, size, align_corners):
    return F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=align_corners)


def _cast_hook(dtype):
    def pre(mod, args, kwargs):
        c = lambda v: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
        return tuple(c(a) for a in args), {k: c(v) for k, v in kwargs.items()}

    return pre


def _block_forward(self, x, pos=None, attn_mask=None):
    a = self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask)
    x = torch.addcmul(x, a, self.ls1.gamma) if hasattr(self.ls1, "gamma") else x + a
    m = self.mlp(self.norm2(x))
    return x.addcmul_(m, self.ls2.gamma) if hasattr(self.ls2, "gamma") else x.add_(m)


def _attn_forward(self, x, pos=None, attn_mask=None):
    """dinov2 Attention.forward (eval) with the fastbb flash kernel instead of SDPA (head_dim 64, no mask)."""
    from .fastbb_vendored.attention import attention_bhnd

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None and pos is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)
    if attn_mask is not None:
        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=(
                attn_mask[:, None].repeat(1, self.num_heads, 1, 1)
                if attn_mask is not None
                else None
            ),
        )
    else:
        o = attention_bhnd(q, k, v, variant=self._fb_variant)
    return self.proj(o.transpose(1, 2).reshape(B, N, C))


_ROPE_N = 4096  # rope table length (positions are token coordinates; table rows do not depend on its length)


def _rope_forward(self, tokens, positions):
    """RotaryPositionEmbedding2D.forward without the per-call `int(positions.max())` device sync: one fp32 cos/sin table
    (computed as upstream does under autocast, where the qk-normed tokens are fp32) cast to the token dtype, cached.
    """
    key = ("fast", tokens.size(-1) // 2, tokens.device, tokens.dtype)
    if key not in self.frequency_cache:
        cos, sin = self._compute_frequency_components(
            tokens.size(-1) // 2, _ROPE_N, tokens.device, torch.float32
        )
        self.frequency_cache[key] = (cos.to(tokens.dtype), sin.to(tokens.dtype))
    cos_c, sin_c = self.frequency_cache[key]
    v, h = tokens.chunk(2, dim=-1)
    v = self._apply_1d_rope(v, positions[..., 0], cos_c, sin_c)
    h = self._apply_1d_rope(h, positions[..., 1], cos_c, sin_c)
    return torch.cat((v, h), dim=-1)


def _fusion_forward(self, *xs, size=None):
    """FeatureFusionBlock: residual merge, RCU, then out_conv (1x1) before the upsample (exact: both linear)."""
    y = xs[0]
    if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
        y = y + self.resConfUnit1(xs[1])
    y = self.resConfUnit2(y)
    y = self.out_conv(y)
    if size is None and self.size is None:
        size = (int(y.shape[-2] * 2), int(y.shape[-1] * 2))
    elif size is None:
        size = self.size
    return self._interp(y, size, self.align_corners)


def _rcu_forward(self, x):
    out = self.conv1(F.relu(x))
    out = self.conv2(F.relu_(out))
    return out.add_(x)


class _F32Conv(nn.Module):
    """final 1x1 conv computed in fp32 (fp16 input upcast)."""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv.float()

    def forward(self, x):
        return self.conv(x.float())


def _dualdpt_forward_impl(self, feats, H, W, patch_start_idx):
    """DualDPT._forward_impl without the auxiliary (ray) branch."""
    B, _, C = feats[0].shape
    ph, pw = H // self.patch_size, W // self.patch_size
    rs = []
    for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
        x = feats[take_idx][:, patch_start_idx:]
        x = self.norm(x)
        x = x.view(B, ph, pw, C).permute(0, 3, 1, 2)  # channels-last view, no copy
        x = self.projects[stage_idx](x)
        if self.pos_embed:
            x = self._add_pos_embed(x, W, H)
        rs.append(self.resize_layers[stage_idx](x))
    s = self.scratch
    l1, l2, l3, l4 = s.layer1_rn(rs[0]), s.layer2_rn(rs[1]), s.layer3_rn(rs[2]), s.layer4_rn(rs[3])
    out = s.refinenet4(l4, size=l3.shape[2:])
    out = s.refinenet3(out, l3, size=l2.shape[2:])
    out = s.refinenet2(out, l2, size=l1.shape[2:])
    out = s.refinenet1(out, l1)
    out = s.output_conv1(out)
    h_out, w_out = int(ph * self.patch_size / self.down_ratio), int(
        pw * self.patch_size / self.down_ratio
    )
    out = self._interp(out, (h_out, w_out), True)
    if self.pos_embed:
        out = self._add_pos_embed(out, W, H)
    fmap = s.output_conv2(out).permute(0, 2, 3, 1)
    return {
        self.head_main: self._apply_activation_single(fmap[..., :-1], self.activation).squeeze(-1),
        f"{self.head_main}_conf": self._apply_activation_single(
            fmap[..., -1], self.conf_activation
        ),
    }


def _dpt_forward_impl(self, feats, H, W, patch_start_idx):
    """DPT._forward_impl with channels-last views (no permute copies); same math."""
    B, _, C = feats[0].shape
    ph, pw = H // self.patch_size, W // self.patch_size
    rs = []
    for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
        x = feats[take_idx][:, patch_start_idx:]
        x = self.norm(x)
        x = x.view(B, ph, pw, C).permute(0, 3, 1, 2)
        x = self.projects[stage_idx](x)
        if self.pos_embed:
            x = self._add_pos_embed(x, W, H)
        rs.append(self.resize_layers[stage_idx](x))
    fused = self._fuse(rs)
    h_out, w_out = int(ph * self.patch_size / self.down_ratio), int(
        pw * self.patch_size / self.down_ratio
    )
    fused = self.scratch.output_conv1(fused)
    fused = self._interp(fused, (h_out, w_out), True)
    if self.pos_embed:
        fused = self._add_pos_embed(fused, W, H)
    main = self.scratch.output_conv2(fused)
    outs = {}
    if self.has_conf:
        fmap = main.permute(0, 2, 3, 1)
        outs[self.head_main] = self._apply_activation_single(
            fmap[..., :-1], self.activation
        ).squeeze(1)
        outs[f"{self.head_main}_conf"] = self._apply_activation_single(
            fmap[..., -1], self.conf_activation
        ).squeeze(1)
    else:
        outs[self.head_main] = self._apply_activation_single(main, self.activation).squeeze(1)
    if self.use_sky_head:
        outs[self.sky_name] = self._apply_sky_activation(
            self.scratch.sky_output_conv2(fused)
        ).squeeze(1)
    return outs


def _prep_head(head, interp, hacc=False):
    from depth_anything_3.model.dpt import FeatureFusionBlock, ResidualConvUnit
    from depth_anything_3.model.dualdpt import DualDPT

    head.half()
    for m in head.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
        if isinstance(m, FeatureFusionBlock):
            m._interp = interp
            m.forward = types.MethodType(_fusion_forward, m)
        if isinstance(m, ResidualConvUnit):
            m.forward = types.MethodType(_rcu_forward, m)
    s = head.scratch
    for name in ("output_conv2", "sky_output_conv2"):
        if hasattr(s, name):
            seq = getattr(s, name)
            seq[-1] = _F32Conv(seq[-1])
    ape = (
        head._add_pos_embed
    )  # upstream builds the embedding in fp32 (promotes fp16 maps): keep the map dtype
    head._add_pos_embed = lambda x, W, H, ratio=0.1: ape(x, W, H, ratio).to(x.dtype)
    if isinstance(head, DualDPT):
        head._forward_impl = types.MethodType(_dualdpt_forward_impl, head)
    else:
        head._forward_impl = types.MethodType(_dpt_forward_impl, head)
    head._interp = interp
    head.register_forward_pre_hook(_cast_hook(torch.float16), with_kwargs=True)
    if hacc:  # head convs as fp16-accumulate cuDNN graphs (precompiled plans only, no timing)
        from .fastbb_vendored.cudnn_conv import patch_convs

        patch_convs(head)


def _prep_net(net, gacc, interp, hacc=False, attn=None):
    """DepthAnything3Net in place."""
    from depth_anything_3.model.dinov2.layers.block import Block

    bb = net.backbone
    bb.half()
    from depth_anything_3.model.dinov2.layers.attention import Attention
    from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D

    for m in bb.modules():
        if attn and isinstance(m, Attention):
            m._fb_variant = attn
            m.forward = types.MethodType(_attn_forward, m)
        if isinstance(m, Block):
            m.forward = types.MethodType(_block_forward, m)
        if isinstance(m, RotaryPositionEmbedding2D):
            m.forward = types.MethodType(_rope_forward, m)
    bb.register_forward_pre_hook(_cast_hook(torch.float16), with_kwargs=True)
    if gacc:
        f = bb.forward

        def fwd(*a, **k):
            with fp16_accumulate(True):
                return f(*a, **k)

        bb.forward = fwd
    _prep_head(net.head, interp, hacc)
    for name in ("cam_dec", "cam_enc"):
        m = getattr(net, name, None)
        if m is not None:
            m.float().register_forward_pre_hook(_cast_hook(torch.float32), with_kwargs=True)


def _probe_kernels():
    """Which custom kernels can run here: (bilinear kernel, flash attention kernel, cuDNN-frontend fp16-accumulate convs)."""
    from . import bilerp

    bil = bilerp.available()
    try:
        from .fastbb_vendored import kernels as fk

        fk.Kernels.get()
        flash = True
    except Exception:
        flash = False
    try:
        from .fastbb_vendored import cudnn_conv

        conv = cudnn_conv.cudnn is not None
    except Exception:
        conv = False
    return bil, flash, conv


class FastDA3:
    def __init__(self, api, fp16_accumulate=True, kernels=None):
        """api: depth_anything_3.api.DepthAnything3 (on cuda, eval); the network is modified in place.
        fp16_accumulate: torch's allow_fp16_accumulation in the backbone GEMMs (a PyTorch flag, no custom kernel).
        kernels: custom kernels (flash attention with fp16 accumulation, fp16-accumulate cuDNN head convolutions, bilinear
        upsample kernel); None = on unless DA3_FAST_KERNELS=0. Each kernel that cannot be loaded falls back to its plain op.
        """
        self.api = api
        net = api.model
        if kernels is None:
            kernels = os.environ.get("DA3_FAST_KERNELS", "1") != "0"
        self.kernels = bool(kernels)
        interp, attn, hacc = _interp_plain, None, False
        if self.kernels:
            from . import bilerp

            bil, flash, conv = _probe_kernels()
            interp = bilerp.interp_bilinear if bil else _interp_plain
            attn = "hh" if flash else None
            hacc = conv
            missing = [
                n
                for n, ok in (
                    ("bilinear kernel", bil),
                    ("flash attention", flash),
                    ("cudnn-frontend convs", conv),
                )
                if not ok
            ]
            if missing:
                warnings.warn(
                    "depth_anything_3.fast: falling back to plain PyTorch for "
                    + ", ".join(missing)
                )
        if hasattr(net, "da3"):  # nested
            _prep_net(net.da3, fp16_accumulate, interp, hacc, attn)
            _prep_net(net.da3_metric, fp16_accumulate, interp, hacc, attn)
        else:
            _prep_net(net, fp16_accumulate, interp, hacc, attn)

    @torch.inference_mode()
    def infer(self, images, process_res=504, extrinsics=None, intrinsics=None):
        """images: list of [3, H, W] uint8 cuda (same size). extrinsics [N, 4, 4] w2c / intrinsics [N, 3, 3] (orig px), optional.
        Returns dict: depth [N, h, w] fp32 at processing res (+ conf, sky, extrinsics, intrinsics as upstream raw output).
        """
        H, W = images[0].shape[-2:]
        pre = [preprocess(im, process_res, resize=cv2_resize) for im in images]
        x = torch.cat([p[0] for p in pre], dim=1).half()
        h2, w2 = pre[0][1]
        ex = ix = None
        if intrinsics is not None:
            ix = torch.as_tensor(intrinsics, dtype=torch.float32, device="cuda").clone()
            ix[:, :1] *= w2 / float(W)
            ix[:, 1:2] *= h2 / float(H)
            ix = ix[None]
        if extrinsics is not None:
            ex = self.api._normalize_extrinsics(
                torch.as_tensor(extrinsics, dtype=torch.float32, device="cuda")[None]
            )
        with torch.autocast("cuda", enabled=False):
            out = self.api.model(x, ex, ix, [], False, False, "saddle_balanced")
        res = {
            k: out[k][0]
            for k in ("depth", "depth_conf", "sky", "extrinsics", "intrinsics")
            if k in out and out[k] is not None
        }
        res["depth"] = res["depth"].float()
        res["proc_hw"] = (h2, w2)
        return res


def _normalize_extrinsics(ex_t):
    """= DepthAnything3._normalize_extrinsics (api.py) without importing the api module."""
    if ex_t is None:
        return None
    from depth_anything_3.utils.geometry import affine_inverse

    transform = affine_inverse(ex_t[:, :1])
    ex_t_norm = ex_t @ transform
    c2ws = affine_inverse(ex_t_norm)
    dists = c2ws[..., :3, 3].norm(dim=-1)
    median_dist = torch.clamp(torch.median(dists), min=1e-1)
    ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
    return ex_t_norm


def load_fast(weights_dir, **opts):
    """Build the DA3 network straight from the model registry config (no `depth_anything_3.api` import: that pulls
    torchvision, export / visualization deps, ~3 s), initialize it on the GPU and load model.safetensors onto the GPU.
    Same weights / modules as DepthAnything3.from_pretrained(weights_dir).model."""
    import json
    import os
    import types as _t
    from safetensors.torch import load_file

    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY

    name = json.load(open(os.path.join(weights_dir, "config.json")))["model_name"]
    with torch.device("cuda"):
        net = create_object(load_config(MODEL_REGISTRY[name]))
    sd = load_file(os.path.join(weights_dir, "model.safetensors"), device="cuda")
    sd = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # upstream (PyTorchModelHubMixin, strict=False) ignores the same keys: aux-head layers of an older layout (unused)
    assert all("_aux" in k for k in unexpected), unexpected[:5]
    # missing in the checkpoints (upstream keeps them random too): mask_token, some aux (ray) head layers - unused here
    assert all("mask_token" in k or "_aux" in k for k in missing), missing[:5]
    net.eval()
    api = _t.SimpleNamespace(model=net, _normalize_extrinsics=_normalize_extrinsics)
    return FastDA3(api, **opts)
