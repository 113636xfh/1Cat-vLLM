# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.utils import KVBlockZeroer


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_zeroer_supports_heterogeneous_page_sizes() -> None:
    device = torch.device(current_platform.device_type)
    small = torch.ones((3, 8), dtype=torch.int32, device=device)
    large = torch.ones((3, 20), dtype=torch.int32, device=device)
    zeroer = KVBlockZeroer(device, pin_memory=False)
    zeroer._id_cap = 8
    zeroer._ids_pinned = torch.empty(8, dtype=torch.int64)
    zeroer._ids_gpu = torch.empty(8, dtype=torch.int64, device=device)
    zeroer._meta = (
        torch.tensor(
            [small.data_ptr(), large.data_ptr()], dtype=torch.uint64, device=device
        ),
        torch.tensor([8, 20], dtype=torch.int64, device=device),
        torch.tensor([8, 20], dtype=torch.int64, device=device),
        3,
        8,
        2,
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    torch.testing.assert_close(small[0], torch.ones_like(small[0]))
    torch.testing.assert_close(small[1], torch.zeros_like(small[1]))
    torch.testing.assert_close(small[2], torch.ones_like(small[2]))
    torch.testing.assert_close(large[0], torch.ones_like(large[0]))
    torch.testing.assert_close(large[1], torch.zeros_like(large[1]))
    torch.testing.assert_close(large[2], torch.ones_like(large[2]))
