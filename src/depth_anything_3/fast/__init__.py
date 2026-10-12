"""Custom-kernel inference path for Depth Anything 3 (cuda-kernels branch): `FastDA3`, `load_fast`.

    from depth_anything_3.fast import load_fast
    fast = load_fast("/path/to/DA3MONO-LARGE")          # snapshot dir with config.json + model.safetensors
    out = fast.infer([img_u8_chw_cuda], process_res=504)  # dict: depth [N, h, w] fp32 (+ sky, depth_conf, ...)

`FastDA3(api, kernels=False)` (or DA3_FAST_KERNELS=0) switches the custom kernels off and runs plain PyTorch ops.
"""

from .da3 import FastDA3, load_fast  # noqa: F401
