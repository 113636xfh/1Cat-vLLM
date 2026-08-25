# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization import fp8


def _layer(
    suffix: str,
    tp_size: int,
    k_dim: int,
    n_dim: int,
    *,
    output_partition_sizes: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prefix=f"model.layers.0.{suffix}",
        tp_size=tp_size,
        input_size_per_partition=k_dim,
        output_size_per_partition=n_dim,
        output_partition_sizes=output_partition_sizes,
        weight_block_size=[128, 128],
        weight=torch.empty((n_dim, k_dim), device="meta"),
    )


@pytest.mark.parametrize(
    ("layer", "gated_silu", "expected"),
    [
        (_layer("attn.fused_wqa_wkv", 1, 4096, 1536), False, (32, 2, False)),
        (_layer("attn.wq_b", 4, 1024, 8192), False, (8, 2, False)),
        (_layer("attn.wo_b", 4, 2048, 4096), False, (16, 2, False)),
        (
            _layer(
                "ffn.shared_experts.gate_up_proj",
                4,
                4096,
                1024,
                output_partition_sizes=[512, 512],
            ),
            False,
            (32, 2, False),
        ),
        (
            _layer(
                "ffn.shared_experts.gate_up_proj",
                4,
                4096,
                1024,
                output_partition_sizes=[512, 512],
            ),
            True,
            (16, 2, False),
        ),
        (
            _layer("ffn.shared_experts.down_proj", 4, 512, 4096),
            False,
            (16, 2, False),
        ),
    ],
)
def test_dsv4_qpn8_exact_shape_configs(
    layer: SimpleNamespace,
    gated_silu: bool,
    expected: tuple[int, int, bool],
) -> None:
    assert fp8._sm70_dsv4_fp8_qpn8_config(layer, gated_silu=gated_silu) == expected


def test_dsv4_qpn8_rejects_wrong_tp_and_gate_partition() -> None:
    wrong_tp = _layer("attn.wq_b", 8, 1024, 8192)
    assert fp8._sm70_dsv4_fp8_qpn8_config(wrong_tp, gated_silu=False) is None

    concurrent_indexer = _layer("attn.indexer.wq_b", 1, 1024, 8192)
    assert fp8._sm70_dsv4_fp8_qpn8_config(concurrent_indexer, gated_silu=False) is None

    wrong_gate = _layer(
        "ffn.shared_experts.gate_up_proj",
        4,
        4096,
        1024,
        output_partition_sizes=[256, 768],
    )
    assert fp8._sm70_dsv4_fp8_qpn8_config(wrong_gate, gated_silu=True) is None


def test_dsv4_qpn8_runtime_and_workspace_contract() -> None:
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            tensor_parallel_size=4,
            enable_dbo=False,
            ubatch_size=0,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        speculative_config=None,
    )
    with patch.object(fp8, "get_current_vllm_config", return_value=config):
        assert fp8._is_sm70_dsv4_fp8_qpn8_runtime_contract()
        config.speculative_config = object()
        assert not fp8._is_sm70_dsv4_fp8_qpn8_runtime_contract()
        config.speculative_config = None
        config.parallel_config.pipeline_parallel_size = 1
        assert not fp8._is_sm70_dsv4_fp8_qpn8_runtime_contract()
        config.parallel_config.pipeline_parallel_size = 2
        config.parallel_config.enable_dbo = True
        assert not fp8._is_sm70_dsv4_fp8_qpn8_runtime_contract()
        config.parallel_config.enable_dbo = False
        config.parallel_config.ubatch_size = 2
        assert not fp8._is_sm70_dsv4_fp8_qpn8_runtime_contract()

    assert fp8._SM70_DSV4_FP8_QPN8_WORKSPACE_ELEMENTS * 2 == 16 * 1024 * 1024


def test_dsv4_qpn8_grouped_wo_a_dispatches_caller_groups() -> None:
    layer = _layer("attn.wo_a", 4, 4096, 2048)
    layer.is_bmm = True
    layer.bmm_batch_size = 2
    assert fp8._sm70_dsv4_fp8_qpn8_bmm_config(layer) == (32, 2, False)

    layer.sm70_fp8_turbomind = True
    layer.sm70_fp8_qpn8 = True
    layer.sm70_fp8_qpn8_bmm = True
    layer.sm70_fp8_bmm_groups = 2
    layer.sm70_fp8_bmm_output_size = 1024
    layer.sm70_fp8_qpn8_split_k = 32
    layer.sm70_fp8_qpn8_nacc = 2
    layer.sm70_fp8_qpn8_prefetch = False
    layer.sm70_fp8_prefill_exact_dense_workspace_ptr = 123
    layer.weight = torch.empty((2, 4096, 1024), device="meta")
    layer.weight_scale_inv = torch.empty((2, 256, 32), device="meta")

    calls: list[tuple[tuple[int, ...], int, int, bool, bool]] = []

    def fake_dispatch(
        out: torch.Tensor,
        workspace_ptr: int,
        input_: torch.Tensor,
        codes: torch.Tensor,
        scales: torch.Tensor,
        split_k: int,
        nacc: int,
        prefetch: bool,
        gated_silu: bool,
    ) -> None:
        del codes, scales
        calls.append(
            (tuple(input_.shape), workspace_ptr, split_k, prefetch, gated_silu)
        )
        out.fill_(len(calls))
        assert nacc == 2

    x = torch.zeros((1, 2, 4096), dtype=torch.float16)
    with patch.object(fp8.sm70_ops, "fp8_qpn8_dispatch_sm70_out", fake_dispatch):
        out = fp8.Fp8LinearMethod.apply(None, layer, x)

    assert calls == [
        ((1, 4096), 123, 32, False, False),
        ((1, 4096), 123, 32, False, False),
    ]
    assert out.shape == (1, 2, 1024)
    torch.testing.assert_close(out[:, 0], torch.ones_like(out[:, 0]))
    torch.testing.assert_close(out[:, 1], torch.full_like(out[:, 1], 2))
