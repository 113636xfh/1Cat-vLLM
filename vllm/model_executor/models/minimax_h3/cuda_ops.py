# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy build of native H3 SM70 extensions for source development."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def w8a16_extension():
    try:
        from vllm import _h3_w8a16_C

        return _h3_w8a16_C
    except ImportError:
        pass
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[4]
    source = root / "csrc/sm70_turbomind/ops/h3_w8a16.cu"
    if not source.is_file():
        raise RuntimeError("H3 W8A16 extension requires the 1Cat source build")
    return load(
        name="onecat_h3_w8a16",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        extra_ldflags=["-lcublas"],
        verbose=False,
    )


@lru_cache(maxsize=1)
def flashinfer_extension():
    try:
        from vllm import _h3_flashinfer_C

        return _h3_flashinfer_C
    except ImportError:
        pass
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[4]
    return load(
        name="onecat_h3_flashinfer_sm70",
        sources=[str(root / "flashinfer-sm70/csrc/h3_noncausal_sm70.cu")],
        extra_include_paths=[str(root / "flashinfer-sm70/include")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
        verbose=False,
    )
