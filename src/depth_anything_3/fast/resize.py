"""cv2-equivalent GPU resize as T-tap gathers (same weights as utils.gpu_inference.cv2_resize, which uses dense fp32 GEMMs
that run on SIMT kernels: 1.8 ms for a 1482x994 image vs ~0.1 ms here)."""

import functools
import torch

from depth_anything_3.utils.gpu_inference import _mat


@functools.lru_cache(maxsize=64)
def _taps(n_in, n_out, kind, device):
    """sparse form of the resize matrix: (idx [n_out, T] int64, wts [n_out, T] fp32), T = max nonzeros per output row."""
    M = _mat(n_in, n_out, kind, "cpu")
    nz = M != 0
    T = int(nz.sum(1).max())
    idx = torch.zeros(n_out, T, dtype=torch.long)
    wts = torch.zeros(n_out, T)
    for o in range(n_out):
        j = torch.nonzero(nz[o]).flatten()
        idx[o, : len(j)] = j
        wts[o, : len(j)] = M[o, j]
    return idx.to(device), wts.to(device)


def cv2_resize(img, h, w, kind):
    """img [..., H, W] uint8 / float on the GPU -> float32 [..., h, w] with uint8-rounded values (cv2.resize with
    INTER_AREA ('area') or INTER_CUBIC ('cubic'), up to cv2's fixed-point rounding: <= 1 LSB)."""
    x = img.float()
    H, W = x.shape[-2:]
    if W != w:
        idx, wts = _taps(W, w, kind, x.device)
        x = (x[..., idx] * wts).sum(-1)
    if H != h:
        idx, wts = _taps(H, h, kind, x.device)
        x = (x.transpose(-1, -2)[..., idx] * wts).sum(-1).transpose(-1, -2).contiguous()
    return x.round_().clamp_(0, 255)
