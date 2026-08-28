# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    _qsa_indexer_cublas_shape_supported,
    _qsa_sparse_launch_profile,
    _qsa_xqa_page4_shape_supported,
)


def test_sm70_qsa_prefill_uses_narrow_tiles_and_four_warps():
    assert _qsa_sparse_launch_profile(511, 8, True) == (64, 4, 4)
    assert _qsa_sparse_launch_profile(512, 8, True) == (32, 4, 4)
    assert _qsa_sparse_launch_profile(8192, 8, True) == (32, 1, 4)


def test_non_sm70_qsa_prefill_keeps_gb300_profile():
    assert _qsa_sparse_launch_profile(512, 8, False) == (64, 4, 2)
    assert _qsa_sparse_launch_profile(8192, 8, False) == (64, 1, 2)


def test_qsa_indexer_cublas_accepts_only_exact_single_request_shape():
    query = torch.empty(8, 4, 128, dtype=torch.float16)
    cache = torch.empty(2, 400, 1, 128, dtype=torch.float16)
    page_table = torch.empty(1, 2, dtype=torch.int32)

    assert _qsa_indexer_cublas_shape_supported(query, cache, page_table)
    assert not _qsa_indexer_cublas_shape_supported(
        query.to(torch.bfloat16), cache, page_table
    )
    assert not _qsa_indexer_cublas_shape_supported(
        query, cache, page_table.expand(2, -1)
    )
    assert not _qsa_indexer_cublas_shape_supported(query[:, :3], cache, page_table)


def test_qsa_xqa_page4_accepts_only_exact_sm70_prefill_shape():
    query = torch.empty(8, 6, 256, dtype=torch.float16)
    key_cache = torch.empty(2, 400, 1, 256, dtype=torch.float16)
    value_cache = torch.empty_like(key_cache)
    indices = torch.empty(8, 2051, dtype=torch.int32)
    page_table = torch.empty(1, 2, dtype=torch.int32)
    token_to_request = torch.zeros(8, dtype=torch.int32)
    query_positions = torch.arange(8, dtype=torch.int64)
    sequence_lengths = torch.full((1,), 8, dtype=torch.int32)

    assert _qsa_xqa_page4_shape_supported(
        query,
        key_cache,
        value_cache,
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )
    strided_query = torch.empty(8, 6, 257, dtype=torch.float16)[..., :256]
    assert _qsa_xqa_page4_shape_supported(
        strided_query,
        key_cache,
        value_cache,
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )
    interleaved_cache = torch.empty(2, 2, 400, 1, 256, dtype=torch.float16)
    interleaved_key_cache, interleaved_value_cache = interleaved_cache.unbind(1)
    assert _qsa_xqa_page4_shape_supported(
        query,
        interleaved_key_cache,
        interleaved_value_cache,
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )
    assert not _qsa_xqa_page4_shape_supported(
        query.to(torch.bfloat16),
        key_cache,
        value_cache,
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )
    assert not _qsa_xqa_page4_shape_supported(
        query,
        key_cache[:, :398],
        value_cache[:, :398],
        indices,
        page_table,
        token_to_request,
        query_positions,
        sequence_lengths,
    )


def test_qsa_indexer_cublas_does_not_capture_decode_rows(monkeypatch):
    cache = torch.empty(2, 400, 1, 128, dtype=torch.float16)
    page_table = torch.empty(1, 2, dtype=torch.int32)
    monkeypatch.setattr(qsa_ops, "_SM70_INDEXER_CUBLAS", True)
    monkeypatch.setattr(qsa_ops, "_SM70_INDEXER_CUBLAS_MIN_ROWS", 256)
    monkeypatch.setattr(
        qsa_ops.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )

    assert qsa_ops._use_sm70_qsa_indexer_cublas(
        torch.empty(256, 4, 128, dtype=torch.float16), cache, page_table
    )
    assert not qsa_ops._use_sm70_qsa_indexer_cublas(
        torch.empty(255, 4, 128, dtype=torch.float16), cache, page_table
    )


def test_qsa_indexer_cublas_requires_enough_score_work(monkeypatch):
    monkeypatch.setattr(
        qsa_ops,
        "_SM70_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS",
        1024**2,
    )

    assert not qsa_ops._qsa_indexer_cublas_work_supported(1024, 512)
    assert qsa_ops._qsa_indexer_cublas_work_supported(2048, 512)
