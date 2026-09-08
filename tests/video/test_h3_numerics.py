# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.models.minimax_h3.attention import (
    Attention,
    AttentionMetadata,
    attention_backend,
    chunked_attention_reference,
)
from vllm.model_executor.models.minimax_h3.comfy_checkpoint import (
    inspect_comfy_checkpoint,
)
from vllm.model_executor.models.minimax_h3.ops import (
    convrot_reference,
    dequantize_int8_reference,
)
from vllm.model_executor.models.minimax_h3.time_request import (
    MINIMAX_H3_SHAPE_PLANNER,
    minimax_h3_time_shift_sigmas,
)
from vllm.model_executor.models.minimax_h3.transformer import (
    MiniMaxH3DiTModel,
    _reorder_grouped_qkv_to_qkv,
)


def test_signed_int8_restores_each_rows_scale():
    weight = torch.tensor([[-128, -1, 0, 127], [127, 0, -1, -128]], dtype=torch.int8)
    scale = torch.tensor([0.125, 0.0001234567], dtype=torch.float32)
    actual = dequantize_int8_reference(weight, scale)
    torch.testing.assert_close(
        actual, (weight.double() * scale.double()[:, None]).half(), rtol=0, atol=0
    )


def test_convrot_matches_kronecker_and_inverse():
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float64,
    )
    h = h4
    for _ in range(3):
        h = torch.kron(h, h4)
    h /= 16
    x = torch.randn(3, 512)
    expected = (x.double().reshape(-1, 256) @ h).reshape(x.shape).float()
    torch.testing.assert_close(convrot_reference(x), expected, atol=5e-7, rtol=2e-5)
    torch.testing.assert_close(
        convrot_reference(convrot_reference(x)), x, atol=8e-7, rtol=3e-5
    )


def test_original_qkv_reorder_keeps_weights_and_row_scales_together():
    rows = torch.arange(3 * 4 * 8)
    expected = rows.reshape(4, 3, 8).permute(1, 0, 2).reshape(-1)
    for value in (rows, rows[:, None].repeat(1, 3)):
        actual = _reorder_grouped_qkv_to_qkv(
            value, num_query_groups=4, heads_per_group=1, head_dim=8
        )
        torch.testing.assert_close(actual, value[expected])


def test_pruned_adaln_interpolation_endpoints_and_midpoints():
    from types import SimpleNamespace

    table = torch.tensor([[0.0, 4.0], [2.0, 0.0], [8.0, 6.0]])
    model = SimpleNamespace(
        arch=SimpleNamespace(adaln_curve_grid=3), adaln_t_table=table
    )
    actual = MiniMaxH3DiTModel._embed_timesteps(
        model, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    )
    torch.testing.assert_close(
        actual,
        torch.tensor([[0.0, 4.0], [1.0, 2.0], [2.0, 0.0], [5.0, 3.0], [8.0, 6.0]]),
    )


def test_checkpoint_metadata_and_partition_validation(tmp_path):
    path = tmp_path / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    marker = json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    ).encode()
    save_file(
        {
            "blocks.0.attn.qkv_proj.weight": torch.ones(12, 256, dtype=torch.int8),
            "blocks.0.attn.qkv_proj.weight_scale": torch.ones(12, 1),
            "blocks.0.attn.qkv_proj.comfy_quant": torch.tensor(
                list(marker), dtype=torch.uint8
            ),
            "adaln_t_table": torch.zeros(1025, 8),
        },
        path,
    )
    config = inspect_comfy_checkpoint(path, expected_partition="fl2va")
    assert config.arch_overrides == {"adaln_curve_grid": 1025, "adaln_curve_dim": 8}
    with pytest.raises(ValueError, match="cannot serve"):
        inspect_comfy_checkpoint(path, expected_partition="ref2va")


def test_primary_workload_schedule_and_shape():
    planner = MINIMAX_H3_SHAPE_PLANNER
    assert planner.align_frame_count(240) == 243
    assert planner.video_latent_t(243) == 72
    assert planner.audio_latent_t(243 / 24) == 405
    sigmas = minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=12)
    assert len(sigmas) - 1 == 49
    assert sigmas[0] == 1 and sigmas[-1] == 0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))


@pytest.mark.parametrize("used,padded", [(31, 32), (129, 256)])
def test_attention_padding_excludes_poisoned_suffix(used, padded):
    token = attention_backend.set("TORCH_SDPA")
    try:
        attention = Attention(
            num_heads=2, num_kv_heads=2, head_size=128, softmax_scale=128**-0.5
        )
    finally:
        attention_backend.reset(token)
    q, k, v = [torch.randn(1, padded, 2, 128) for _ in range(3)]
    expected = chunked_attention_reference(
        q[:, :used], k[:, :used], v[:, :used], scale=128**-0.5
    )
    k[:, used:] = float("nan")
    v[:, used:] = float("nan")
    actual = attention(q, k, v, AttentionMetadata(extra={"valid_kv_length": used}))
    torch.testing.assert_close(actual[:, :used], expected)
    assert torch.count_nonzero(actual[:, used:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("backend", ["FLASH_ATTN_V100", "FLASHINFER_SM70"])
def test_noncausal_backend_matches_fp32_reference(backend):
    token = attention_backend.set(backend)
    try:
        attention = Attention(
            num_heads=14, num_kv_heads=14, head_size=128, softmax_scale=128**-0.5
        )
    finally:
        attention_backend.reset(token)
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(1, 257, 14, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    expected = chunked_attention_reference(
        q[:, :243], k[:, :243], v[:, :243], scale=128**-0.5
    )
    actual = attention(q, k, v, AttentionMetadata(extra={"valid_kv_length": 243}))
    torch.testing.assert_close(actual[:, :243], expected, atol=8e-4, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_sm70_w8a16_uses_signed_scales_and_fp32_gemm_reduction():
    from vllm.model_executor.models.minimax_h3.cuda_ops import w8a16_extension

    ops = w8a16_extension()
    torch.manual_seed(42)
    x = torch.randn(65, 512, device="cuda", dtype=torch.float16)
    weight = torch.randint(-128, 128, (257, 512), device="cuda", dtype=torch.int8)
    scale = torch.rand(257, device="cuda") * 0.001 + 0.0001
    rotated = ops.rotate(x)
    decoded = ops.dequantize(weight, scale)
    torch.testing.assert_close(rotated, convrot_reference(x), atol=0, rtol=0)
    torch.testing.assert_close(
        decoded, dequantize_int8_reference(weight, scale), atol=0, rtol=0
    )
    result = ops.gemm(rotated, decoded)
    reference = (rotated.float() @ decoded.float().T).half()
    torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_encoder_causal_gqa_matches_fp32():
    from vllm.model_executor.models.minimax_h3.encoder import (
        _scaled_dot_product_attention,
    )

    q = torch.randn(1, 16, 127, 128, device="cuda", dtype=torch.float16)
    k, v = [
        torch.randn(1, 2, 127, 128, device="cuda", dtype=torch.float16)
        for _ in range(2)
    ]
    actual = _scaled_dot_product_attention(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(8, 1),
        v.float().repeat_interleave(8, 1),
        is_causal=True,
    ).half()
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)


def test_encoder_residual_rmsnorm_preserves_residual_rounding():
    from vllm.model_executor.models.minimax_h3.ops import RMSNorm

    norm = RMSNorm(128, eps=1e-6, dtype=torch.float16)
    x, residual = [torch.randn(3, 128, dtype=torch.float16) for _ in range(2)]
    expected_residual = residual + x
    value = expected_residual.float()
    expected = (
        value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
    ).half()
    output, updated = norm(x, residual)
    torch.testing.assert_close(updated, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(output, expected)


def test_encoder_uses_functional_all_reduce_return():
    from types import SimpleNamespace

    from torch import nn

    from vllm.model_executor.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLRowParallelLinear,
        MiniMaxH3Qwen3VLVocabParallelEmbedding,
    )

    # Native GroupCoordinator may return a new tensor without modifying input.
    group = SimpleNamespace(
        rank_in_group=0, world_size=2, all_reduce=lambda value: value + 3
    )
    embedding = MiniMaxH3Qwen3VLVocabParallelEmbedding(group, 8, 4, torch.float32)
    embedding.weight.data.fill_(1)
    actual = embedding(torch.tensor([0, 6]))
    torch.testing.assert_close(
        actual, torch.tensor([[4.0, 4.0, 4.0, 4.0], [3.0, 3.0, 3.0, 3.0]])
    )
    projection = MiniMaxH3Qwen3VLRowParallelLinear.__new__(
        MiniMaxH3Qwen3VLRowParallelLinear
    )
    nn.Module.__init__(projection)
    projection.input_is_parallel = True
    projection._tp_size = 2
    projection.group = group
    projection.output_dtype = torch.float16
    projection.quant_method = SimpleNamespace(apply=lambda layer, value: value * 2)
    value = torch.ones(2, 4, dtype=torch.float16)
    torch.testing.assert_close(projection(value), value * 5)
