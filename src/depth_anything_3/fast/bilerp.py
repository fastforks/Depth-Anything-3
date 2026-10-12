"""NHWC fp16 bilinear resize as one prebuilt CUDA kernel (torch's NHWC bilinear kernel is several times slower).

Adapted from exp 004 mono_fast/kernels.py + ops.py @ 2171802. The kernel (da3_kernels.cu) ships as a fatbin (SASS for sm_80 / 86 /
89 / 90 / 120 + compute_80 PTX) in fast/prebuilt; without a matching file it is compiled once with nvcc (CUDA_HOME) and cached in
~/.cache/da3-fast. Loaded through the driver API and launched on torch's current stream; size-independent (no per-shape build).
`available()` is False when neither works; callers then use F.interpolate.
"""

import ctypes
import hashlib
import os
import subprocess
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "da3_kernels.cu")
_PREBUILT = os.path.join(_HERE, "prebuilt")
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "da3-fast")
# argument types: p pointer (tensor), i int32, f float32
_SIG = "piiipiiiffi"


def _srctag():
    return hashlib.sha1(open(_SRC, "rb").read()).hexdigest()[:16]


def build_prebuilt(archs=(80, 86, 89, 90, 120)):
    """Rebuild the shipped fatbin for the current source (needs nvcc in $CUDA_HOME/bin)."""
    os.makedirs(_PREBUILT, exist_ok=True)
    for f in os.listdir(_PREBUILT):
        if f.startswith("da3_kernels_"):
            os.remove(os.path.join(_PREBUILT, f))
    out = os.path.join(_PREBUILT, f"da3_kernels_{_srctag()}.fatbin")
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]
    gencode += ["-gencode=arch=compute_80,code=compute_80"]
    cmd = [os.path.join(cuda_home, "bin", "nvcc"), "-fatbin", "-O3", "-std=c++17", *gencode]
    subprocess.run([*cmd, f"-I{cuda_home}/include", "-o", out + ".tmp", _SRC], check=True)
    os.replace(out + ".tmp", out)
    return out


def _check(res, what):
    if res != 0:
        raise RuntimeError(f"CUDA driver error {res} in {what}")


class _Kernel:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        torch.cuda.init()
        major, minor = torch.cuda.get_device_capability()
        fat = os.path.join(_PREBUILT, f"da3_kernels_{_srctag()}.fatbin")
        image = fat
        if not os.path.exists(
            image
        ):  # no shipped binary for this source: compile for this GPU once
            tag = hashlib.sha1(
                open(_SRC, "rb").read() + f"sm_{major}{minor}".encode()
            ).hexdigest()[:16]
            image = os.path.join(_CACHE, f"da3_kernels_{tag}.cubin")
            if not os.path.exists(image):
                os.makedirs(_CACHE, exist_ok=True)
                cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
                cmd = [os.path.join(cuda_home, "bin", "nvcc"), "-cubin", "-O3", "-std=c++17"]
                cmd += [
                    f"-arch=sm_{major}{minor}",
                    f"-I{cuda_home}/include",
                    "-o",
                    image + ".tmp",
                    _SRC,
                ]
                subprocess.run(cmd, check=True)
                os.replace(image + ".tmp", image)
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuLaunchKernel.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        )
        torch.zeros(1, device="cuda")  # make sure the primary context is current
        self.module = ctypes.c_void_p()
        self._image = open(image, "rb").read()
        _check(
            self.lib.cuModuleLoadData(ctypes.byref(self.module), self._image), "cuModuleLoadData"
        )
        self.func = ctypes.c_void_p()
        _check(
            self.lib.cuModuleGetFunction(ctypes.byref(self.func), self.module, b"bilerp_nhwc"),
            "cuModuleGetFunction",
        )

    def launch(self, n_threads, *args):
        grid = (min(max((n_threads + 255) // 256, 1), 65535 * 8), 1, 1)
        holders = []
        for t, a in zip(_SIG, args):
            if t == "p":
                holders.append(ctypes.c_void_p(a.data_ptr()))
            else:
                holders.append((ctypes.c_int32 if t == "i" else ctypes.c_float)(a))
        arr = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
        )
        stream = torch.cuda.current_stream().cuda_stream
        _check(
            self.lib.cuLaunchKernel(self.func, *grid, 256, 1, 1, 0, stream, arr, None),
            "launch bilerp_nhwc",
        )


_AVAILABLE = [None]


def available():
    """True when the kernel can be loaded on this machine (CUDA driver present and a binary, shipped or compiled)."""
    if _AVAILABLE[0] is None:
        try:
            _Kernel.get()
            _AVAILABLE[0] = True
        except Exception:  # no driver library, no nvcc for an unshipped architecture, ...
            _AVAILABLE[0] = False
    return _AVAILABLE[0]


def interp_bilinear(x, size, align_corners):
    """F.interpolate(x, size, 'bilinear', align_corners) for fp16 channels-last x with C % 8 == 0 in one vectorized kernel
    (same arithmetic as PyTorch's: fp32 weights, one rounding); anything else goes to F.interpolate.
    """
    N, C, Hs, Ws = x.shape
    Hd, Wd = int(size[0]), int(size[1])
    if x.dtype != torch.float16 or C % 8 or not x.is_contiguous(memory_format=torch.channels_last):
        return F.interpolate(x, size=(Hd, Wd), mode="bilinear", align_corners=align_corners)
    if align_corners:
        sh = (Hs - 1) / (Hd - 1) if Hd > 1 else 0.0
        sw = (Ws - 1) / (Wd - 1) if Wd > 1 else 0.0
    else:
        sh, sw = Hs / Hd, Ws / Wd
    y = torch.empty(
        (N, C, Hd, Wd), device=x.device, dtype=x.dtype, memory_format=torch.channels_last
    )
    _Kernel.get().launch(
        N * Hd * Wd * C // 8,
        x,
        N,
        Hs,
        Ws,
        y,
        Hd,
        Wd,
        C,
        float(sh),
        float(sw),
        1 if align_corners else 0,
    )
    return y
