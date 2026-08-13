# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization.awq import (
    _awq_exact_f16_weight,
    _has_sm70_awq_prefill_exact_dense_capacity,
    _is_sm70_awq_prefill_exact_dense_layer,
)


def test_awq_exact_f16_weight_matches_half_fma_rounding():
    qweight = torch.full((8, 2), 0x11111111, dtype=torch.int32)
    qzeros = torch.full((1, 2), 0x33333333, dtype=torch.int32)
    scales = torch.full((1, 16), 0.0001, dtype=torch.float16)

    actual = _awq_exact_f16_weight(qweight, scales, qzeros, group_size=8)
    quant = torch.ones((), dtype=torch.float16)
    zero = torch.full((), 3.0, dtype=torch.float16)
    scale = scales[0, 0]
    bias = -zero * scale
    expected = torch.addcmul(bias, quant, scale)
    naive = (quant - zero) * scale

    assert actual.shape == (8, 16)
    assert actual.is_contiguous()
    assert torch.equal(actual, torch.full_like(actual, expected))
    assert expected != naive


def test_awq_prefill_exact_dense_shape_gate_is_narrow():
    qweight = SimpleNamespace(shape=(5120, 1088))
    layer = SimpleNamespace(
        tp_size=4,
        prefix="model.language_model.layers.1.mlp.gate_up_proj",
        qweight=qweight,
    )

    assert _is_sm70_awq_prefill_exact_dense_layer(layer)

    layer.tp_size = 2
    assert not _is_sm70_awq_prefill_exact_dense_layer(layer)
    layer.tp_size = 4
    layer.prefix = "model.language_model.layers.1.self_attn.qkv_proj"
    assert not _is_sm70_awq_prefill_exact_dense_layer(layer)
    layer.prefix = "model.language_model.layers.1.mlp.gate_up_proj"
    layer.qweight = SimpleNamespace(shape=(5120, 1024))
    assert not _is_sm70_awq_prefill_exact_dense_layer(layer)


def test_awq_prefill_exact_dense_requires_32gb_v100(monkeypatch):
    layer = SimpleNamespace(qweight=SimpleNamespace(device="cuda:0"))

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=16 * 1024**3),
    )
    assert not _has_sm70_awq_prefill_exact_dense_capacity(layer)

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 * 1024**3),
    )
    assert _has_sm70_awq_prefill_exact_dense_capacity(layer)
