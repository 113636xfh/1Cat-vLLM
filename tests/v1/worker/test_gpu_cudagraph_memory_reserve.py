# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    get_explicit_cudagraph_memory_reserve,
)

pytestmark = pytest.mark.cpu_test


def test_explicit_cudagraph_memory_reserve(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_V2_CUDAGRAPH_MEM_MIB", "1536.5")

    assert get_explicit_cudagraph_memory_reserve(CUDAGraphMode.FULL) == int(
        1536.5 * 1024 * 1024
    )
    assert get_explicit_cudagraph_memory_reserve(CUDAGraphMode.NONE) == 0


def test_explicit_cudagraph_memory_reserve_rejects_negative(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_V2_CUDAGRAPH_MEM_MIB", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        get_explicit_cudagraph_memory_reserve(CUDAGraphMode.FULL)
