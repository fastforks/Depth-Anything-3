import pytest
import torch
import torch.nn as nn

from depth_anything_3.model.dpt import FeatureFusionBlock, custom_interpolate


@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("size", [None, (30, 22)])
def test_out_conv_before_upsample_equals_after(align_corners, size):
    """The 1x1 out_conv (with bias) commutes with the bilinear upsample: new forward == upsample, then conv."""
    torch.manual_seed(0)
    blk = FeatureFusionBlock(16, nn.ReLU(), align_corners=align_corners, expand=True).eval()
    x0, x1 = torch.randn(2, 16, 11, 9), torch.randn(2, 16, 11, 9)
    with torch.no_grad():
        got = blk(x0, x1, size=size)
        y = blk.resConfUnit2(blk.skip_add.add(x0, blk.resConfUnit1(x1)))
        kw = {"scale_factor": 2} if size is None else {"size": size}
        want = blk.out_conv(
            custom_interpolate(y, **kw, mode="bilinear", align_corners=align_corners)
        )
    assert got.shape == want.shape
    assert (got - want).abs().max() < 1e-5
