# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    _qsa_sparse_launch_profile,
)


def test_sm70_qsa_prefill_uses_four_warps():
    assert _qsa_sparse_launch_profile(512, 8, True) == (64, 4, 4)
    assert _qsa_sparse_launch_profile(8192, 8, True) == (64, 1, 4)


def test_non_sm70_qsa_prefill_keeps_gb300_profile():
    assert _qsa_sparse_launch_profile(512, 8, False) == (64, 4, 2)
    assert _qsa_sparse_launch_profile(8192, 8, False) == (64, 1, 2)
