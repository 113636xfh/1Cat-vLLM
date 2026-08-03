# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm.models.deepseek_v4.sm70 import sparse_kernels


def _is_sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


@pytest.mark.skipif(not _is_sm70(), reason="requires NVIDIA V100/SM70")
@pytest.mark.parametrize(
    ("width", "compress_ratio"),
    ((128, 1), (256, 128), (640, 4)),
)
def test_sm70_sparse_prefill_hmma_matches_triton(
    width: int, compress_ratio: int
) -> None:
    torch.manual_seed(20260803 + width)
    device = torch.device("cuda")
    num_queries = 64
    num_kv = 256
    q = torch.randn((num_queries, 8, 512), dtype=torch.float16, device=device)
    kv = torch.randn((num_kv, 512), dtype=torch.float16, device=device)
    rows = torch.arange(num_queries, dtype=torch.int64, device=device)[:, None]
    columns = torch.arange(width, dtype=torch.int64, device=device)[None, :]
    compressed = torch.clamp(
        (rows[:, 0] + 1) // compress_ratio,
        max=max(0, width - 128),
    )
    if width == 128:
        compressed.zero_()
    lengths = (compressed + torch.clamp(rows[:, 0] + 1, max=128)).to(torch.int32)
    indices = ((rows * 131 + columns * 67) % num_kv).to(torch.int32)
    indices.masked_fill_(columns >= lengths[:, None], -1)
    sink = torch.linspace(-4.0, -1.0, 8, dtype=torch.float32, device=device)
    reference = torch.empty_like(q)
    candidate = torch.empty_like(q)

    with patch.object(
        sparse_kernels.envs,
        "VLLM_SM70_DSV4_SPARSE_PREFILL_HMMA",
        False,
    ):
        sparse_kernels.sm70_sparse_attention_gathered(
            q, kv, indices, lengths, 512**-0.5, sink, reference
        )
    sparse_kernels._sm70_sparse_attention_hmma(
        q, kv, indices, lengths, sink, candidate, 512**-0.5
    )
    torch.accelerator.synchronize()

    assert torch.isfinite(candidate).all()
    torch.testing.assert_close(candidate, reference, rtol=0, atol=2**-9)
