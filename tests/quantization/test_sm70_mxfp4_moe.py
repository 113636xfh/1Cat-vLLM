# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization import mxfp4_sm70_moe as mxfp4_moe
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.mxfp4 import (
    make_deepseek_v4_mxfp4_moe_method,
)
from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import (
    Mxfp4SM70MoEMethod,
    _mxfp4_indexed_prefill_enabled,
    _select_mxfp4_stage_dispatch,
    validate_mxfp4_sm70_moe_contract,
    validate_mxfp4_sm70_moe_weight_layout,
)


@pytest.mark.parametrize(
    ("num_tokens", "fully_replicated", "grouped", "expected"),
    [
        (1024, True, True, True),
        (8192, True, True, True),
        (1023, True, True, False),
        (1024, False, True, False),
        (1024, True, False, False),
    ],
)
def test_mxfp4_sm70_indexed_prefill_gate(
    monkeypatch, num_tokens, fully_replicated, grouped, expected
):
    monkeypatch.setattr(envs, "VLLM_SM70_MXFP4_MOE_INDEXED_PREFILL", True)
    monkeypatch.setattr(envs, "VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL", grouped)

    assert (
        _mxfp4_indexed_prefill_enabled(
            num_tokens=num_tokens,
            fully_replicated_experts=fully_replicated,
        )
        is expected
    )


def _v4_flash_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=256,
        experts_per_token=6,
        hidden_dim=4096,
        intermediate_size_per_partition=256,
        num_local_experts=256,
        num_logical_experts=256,
        activation=MoEActivation.SILU,
        device=torch.device("cuda"),
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=8,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=False,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        ),
        in_dtype=torch.float16,
        swiglu_limit=7.0,
    )


@pytest.mark.parametrize(
    ("tp_size", "intermediate_size_per_partition"),
    [(8, 256), (4, 512)],
)
def test_mxfp4_sm70_contract_accepts_v4_flash_tp_shapes(
    tp_size: int, intermediate_size_per_partition: int
):
    validate_mxfp4_sm70_moe_contract(
        global_num_experts=256,
        top_k=6,
        hidden_size=4096,
        intermediate_size_per_partition=intermediate_size_per_partition,
        tp_size=tp_size,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"global_num_experts": 128}, "256 global experts"),
        ({"top_k": 8}, "top-k=6"),
        ({"hidden_size": 8192}, "hidden size 4096"),
        ({"intermediate_size_per_partition": 512}, "intermediate size 2048"),
    ],
)
def test_mxfp4_sm70_contract_rejects_non_v4_flash_shapes(kwargs, match):
    values = {
        "global_num_experts": 256,
        "top_k": 6,
        "hidden_size": 4096,
        "intermediate_size_per_partition": 256,
        "tp_size": 8,
    }
    values.update(kwargs)
    with pytest.raises(NotImplementedError, match=match):
        validate_mxfp4_sm70_moe_contract(**values)


@pytest.mark.parametrize("intermediate_size", [256, 512])
def test_mxfp4_sm70_weight_layout_accepts_packed_v4_flash_tp_tensors(
    intermediate_size: int,
):
    local_experts = 256
    hidden_size = 4096
    validate_mxfp4_sm70_moe_weight_layout(
        local_num_experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        w13_weight=torch.empty(
            local_experts,
            2 * intermediate_size,
            hidden_size // 2,
            dtype=torch.uint8,
            device="meta",
        ),
        w13_weight_scale=torch.empty(
            local_experts,
            2 * intermediate_size,
            hidden_size // 32,
            dtype=torch.uint8,
            device="meta",
        ),
        w2_weight=torch.empty(
            local_experts,
            hidden_size,
            intermediate_size // 2,
            dtype=torch.uint8,
            device="meta",
        ),
        w2_weight_scale=torch.empty(
            local_experts,
            hidden_size,
            intermediate_size // 32,
            dtype=torch.uint8,
            device="meta",
        ),
    )


def test_mxfp4_sm70_weight_layout_rejects_wrong_ue8m0_scale_shape():
    with pytest.raises(ValueError, match="w2_weight_scale"):
        validate_mxfp4_sm70_moe_weight_layout(
            local_num_experts=1,
            hidden_size=4096,
            intermediate_size_per_partition=256,
            w13_weight=torch.empty(1, 512, 2048, dtype=torch.uint8, device="meta"),
            w13_weight_scale=torch.empty(1, 512, 128, dtype=torch.uint8, device="meta"),
            w2_weight=torch.empty(1, 4096, 128, dtype=torch.uint8, device="meta"),
            w2_weight_scale=torch.empty(1, 4096, 7, dtype=torch.uint8, device="meta"),
        )


def test_mxfp4_sm70_platform_gate_is_exact(monkeypatch):
    monkeypatch.setattr(sm70_tm.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        sm70_tm.current_platform,
        "is_device_capability",
        lambda capability: capability == (7, 0),
    )
    assert sm70_tm.is_exact_sm70_cuda_platform()

    monkeypatch.setattr(
        sm70_tm.current_platform,
        "is_device_capability",
        lambda capability: False,
    )
    assert not sm70_tm.is_exact_sm70_cuda_platform()


def test_mxfp4_sm70_factory_selects_native_route(monkeypatch):
    monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(sm70_tm, "should_use_mxfp4_moe_turbomind", lambda: True)

    method = make_deepseek_v4_mxfp4_moe_method(_v4_flash_moe_config())

    assert isinstance(method, Mxfp4SM70MoEMethod)
    assert not method.skip_forward_padding


def test_mxfp4_sm70_factory_rejects_marlin_or_emulation(monkeypatch):
    monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(sm70_tm, "should_use_mxfp4_moe_turbomind", lambda: False)

    with pytest.raises(NotImplementedError, match="Marlin"):
        make_deepseek_v4_mxfp4_moe_method(_v4_flash_moe_config())


def test_mxfp4_sm70_b1_dispatch_selects_six_runtime_experts(monkeypatch):
    buffers = {
        "compact_expert_offsets": torch.arange(7, dtype=torch.int32),
        "permuted_experts_id": torch.tensor(
            [3, 17, 42, 99, 128, 255], dtype=torch.int32
        ),
        "expert_offsets": torch.arange(257, dtype=torch.int32),
        "dense_expert_ids": torch.arange(256, dtype=torch.int32),
    }
    monkeypatch.setattr(mxfp4_moe, "_mxfp4_active_expert_b1_enabled", lambda: True)

    offsets, expert_ids, count = _select_mxfp4_stage_dispatch(
        buffers,
        num_tokens=1,
        num_experts=256,
        fully_replicated_experts=True,
    )

    assert offsets is buffers["compact_expert_offsets"]
    assert expert_ids is buffers["permuted_experts_id"]
    assert count == 6


def test_mxfp4_sm70_b1_dispatch_rejects_incompatible_permute_fastpath(
    monkeypatch,
):
    monkeypatch.setattr(envs, "VLLM_SM70_MXFP4_MOE_ACTIVE_EXPERT_B1", True)
    monkeypatch.setattr(envs, "VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH", True)

    assert not mxfp4_moe._mxfp4_active_expert_b1_enabled()


def test_mxfp4_sm70_b1_dispatch_rejects_generic_single_token_fastpath(
    monkeypatch,
):
    monkeypatch.setattr(envs, "VLLM_SM70_MXFP4_MOE_ACTIVE_EXPERT_B1", True)
    monkeypatch.setattr(envs, "VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH", True)

    assert not mxfp4_moe._mxfp4_active_expert_b1_enabled()


def test_mxfp4_sm70_b1_dispatch_rejects_expert_parallel_metadata(monkeypatch):
    buffers = {
        "compact_expert_offsets": torch.arange(7, dtype=torch.int32),
        "permuted_experts_id": torch.arange(6, dtype=torch.int32),
        "expert_offsets": torch.arange(257, dtype=torch.int32),
        "dense_expert_ids": torch.arange(256, dtype=torch.int32),
    }
    monkeypatch.setattr(mxfp4_moe, "_mxfp4_active_expert_b1_enabled", lambda: True)

    offsets, expert_ids, count = _select_mxfp4_stage_dispatch(
        buffers,
        num_tokens=1,
        num_experts=256,
        fully_replicated_experts=False,
    )

    assert offsets is buffers["expert_offsets"]
    assert expert_ids is buffers["dense_expert_ids"]
    assert count == 256


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_mxfp4_sm70_dispatch_retains_dense_fallback(monkeypatch, num_tokens):
    buffers = {
        "compact_expert_offsets": torch.arange(7, dtype=torch.int32),
        "permuted_experts_id": torch.arange(6, dtype=torch.int32),
        "expert_offsets": torch.arange(257, dtype=torch.int32),
        "dense_expert_ids": torch.arange(256, dtype=torch.int32),
    }
    monkeypatch.setattr(
        mxfp4_moe,
        "_mxfp4_active_expert_b1_enabled",
        lambda: num_tokens != 1,
    )

    offsets, expert_ids, count = _select_mxfp4_stage_dispatch(
        buffers,
        num_tokens=num_tokens,
        num_experts=256,
        fully_replicated_experts=True,
    )

    assert offsets is buffers["expert_offsets"]
    assert expert_ids is buffers["dense_expert_ids"]
    assert count == 256


def test_mxfp4_sm70_post_load_reads_bias_from_method_config(monkeypatch):
    for op_name in (
        "mxfp4_sm70_prepare",
        "mxfp4_moe_dense_stage_sm70_out",
        "awq_moe_build_strided_ptrs",
    ):
        monkeypatch.setattr(torch.ops._C, op_name, object(), raising=False)
    monkeypatch.setattr(
        torch.ops._moe_C, "moe_permute_with_scratch", object(), raising=False
    )
    method = Mxfp4SM70MoEMethod(_v4_flash_moe_config())
    layer = SimpleNamespace(
        activation=MoEActivation.GELU,
        apply_router_weight_on_input=False,
    )

    with pytest.raises(NotImplementedError, match="SwiGLU"):
        method.process_weights_after_loading(layer)
