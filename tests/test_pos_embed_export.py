import pytest
import torch
import torch.nn.functional as F

from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer, _bicubic_matrix


@pytest.mark.parametrize("grid", [(36, 36), (36, 48), (21, 17), (73, 5)])
def test_matrices_equal_interpolate(grid):
    M, offset = 37, 0.1
    torch.manual_seed(0)
    emb = torch.randn(1, 8, M, M)
    h0, w0 = grid
    got = (
        _bicubic_matrix(w0, offset, M, emb.device)
        @ emb
        @ _bicubic_matrix(h0, offset, M, emb.device).t()
    )
    ref = F.interpolate(
        emb, scale_factor=(float(w0 + offset) / M, float(h0 + offset) / M), mode="bicubic"
    )
    assert got.shape == ref.shape
    assert (got - ref).abs().max() < 1e-4


def test_export_with_dynamic_image_size():
    """torch.export with dynamic H / W (before: guard `0.1 + W // 14 == 32.1` from the Python float scale factor)."""
    m = DinoVisionTransformer(
        img_size=518, patch_size=14, embed_dim=64, depth=1, num_heads=2, init_values=1.0
    ).eval()

    class Wrap(torch.nn.Module):
        def forward(self, x):
            return m.interpolate_pos_encoding(torch.zeros(1, 5, 64), x.shape[-1], x.shape[-2])

    H, W = torch.export.Dim("H", min=2, max=60), torch.export.Dim("W", min=2, max=60)
    ep = torch.export.export(
        Wrap(),
        (torch.zeros(1, 3, 336, 448),),
        dynamic_shapes={"x": {2: 14 * H, 3: 14 * W}},
        strict=True,
    )
    for h, w in ((504, 588), (336, 448), (644, 280)):
        x = torch.zeros(1, 3, h, w)
        got = ep.module()(x)
        assert got.shape == (1, 1 + (h // 14) * (w // 14), 64)
        assert torch.allclose(got, Wrap()(x), atol=1e-5)


def test_export_native_grid_is_identity():
    """the dynamic export keeps the eager native-grid shortcut (37 x 37 tokens -> pos_embed unchanged)"""
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
