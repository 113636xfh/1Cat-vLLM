# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.qwen4_exp.common.qsa_cache import QSAKeyStateCache
from vllm.v1.worker.utils import bind_kv_cache


def test_bind_qsa_key_cache_builds_key_and_mrope_views() -> None:
    prefix = "model.layers.0.self_attn.raw_key_cache"
    static_forward_context = {}
    layer = QSAKeyStateCache(
        head_size=128,
        dtype=torch.float16,
        cache_rope_positions=True,
        prefix=prefix,
        cache_config=SimpleNamespace(block_size=16),
        compress_ratio=4,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context=static_forward_context
            )
        ),
    )
    cache = torch.empty(2, 8, 1, layer.head_size, dtype=torch.float16)
    runner_kv_caches = []

    bind_kv_cache({prefix: cache}, static_forward_context, runner_kv_caches)

    assert layer.kv_cache is cache
    assert layer.key_cache.shape == (2, 8, 1, 128)
    assert layer.key_cache.untyped_storage().data_ptr() == cache.data_ptr()
    assert layer.rope_position_cache.shape == (2, 8, 1, 3)
    assert layer.rope_position_cache.dtype == torch.int64
    assert runner_kv_caches == [cache]
