# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.minimax_h3.cuda_ops import (
    cached_weight_gemm_plan,
    w8a16_extension,
)
from vllm.model_executor.models.minimax_h3.weight_cache import FP16WeightCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


@pytest.mark.parametrize(
    "n,k,fp32",
    [(5376, 5376, False), (7168, 5376, False), (5376, 1792, True), (5376, 3584, True)],
)
def test_cached_gemm_matches_uncached_h3_padded_shapes(n, k, fp32):
    torch.manual_seed(42)
    x = torch.randn(12352, k, device="cuda", dtype=torch.float16)
    encoded = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
    scale = torch.rand(n, device="cuda") * 0.001 + 0.0001
    ops = w8a16_extension()
    weight = ops.dequantize(encoded, scale)
    plan = cached_weight_gemm_plan(x.shape[0], n, k, fp32, x.get_device())
    info = plan.algorithm_info()
    assert info[2] <= 1 and info[3] == 0
    actual = plan.run(x, weight.t().contiguous())
    torch.testing.assert_close(actual, ops.gemm(x, weight, fp32), atol=0, rtol=0)
    reference = x[:32].float() @ weight.float().t()
    torch.testing.assert_close(actual[:32].float(), reference, atol=0.005, rtol=0.001)


def test_cached_gemm_fp32_output_survives_fp16_overflow_and_graph_replay():
    x = torch.full((32, 512), 256, device="cuda", dtype=torch.float16)
    weight_t = torch.ones(512, 256, device="cuda", dtype=torch.float16)
    plan = cached_weight_gemm_plan(32, 256, 512, True, x.get_device())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run(x, weight_t)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = plan.run(x, weight_t)
    for multiplier in (1, -2):
        x.mul_(multiplier)
        graph.replay()
        expected = x.float() @ weight_t.float()
        assert torch.isfinite(actual).all() and actual.abs().max() > 65504
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_cached_gemm_rejects_wrong_layout_shape_and_dtype():
    x = torch.ones(32, 512, device="cuda", dtype=torch.float16)
    weight_t = torch.ones(512, 256, device="cuda", dtype=torch.float16)
    plan = cached_weight_gemm_plan(32, 256, 512, False, x.get_device())
    for a, b in [
        (x[:1], weight_t),
        (x, weight_t.t()),
        (x.float(), weight_t),
        (x, weight_t[:, :128]),
        (x.cpu(), weight_t.cpu()),
    ]:
        with pytest.raises(RuntimeError, match="this plan's FP16"):
            plan.run(a, b)
    unaligned = torch.ones(x.numel() + 1, device="cuda", dtype=x.dtype)[1:].view_as(x)
    with pytest.raises(RuntimeError, match="256-byte aligned"):
        plan.run(unaligned, weight_t)


def test_cache_preserves_encoded_weights_and_cleans_partial_preparation(monkeypatch):
    model = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for layer in model:
        layer.register_buffer(
            "weight",
            torch.randint(-128, 128, (256, 256), device="cuda", dtype=torch.int8),
        )
        layer.register_buffer("weight_scale", torch.rand(256, device="cuda") + 0.01)
    originals = [(layer.weight.clone(), layer.weight_scale.clone()) for layer in model]
    cache = FP16WeightCache(model, budget_gib=1, layers=("0", "1"))
    cache.prepare()
    for layer, (weight, scale) in zip(model, originals):
        assert layer.h3_fp16_weight.stride() == (1, 256)
        torch.testing.assert_close(
            layer.h3_fp16_weight,
            w8a16_extension().dequantize(weight, scale),
            atol=0,
            rtol=0,
        )
        assert torch.equal(layer.weight, weight) and torch.equal(
            layer.weight_scale, scale
        )
    cache.clear()
    responses = iter([(2**40, 2**40), (0, 2**40)])
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda _: next(responses))
    with pytest.raises(RuntimeError, match="available GPU memory"):
        cache.prepare()
    assert all(not hasattr(layer, "h3_fp16_weight") for layer in model)
