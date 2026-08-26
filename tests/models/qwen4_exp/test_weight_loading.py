# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from torch import nn

from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
    Qwen4ExpSparseMoeBlock,
    _remap_qsa_cache_scale_name,
)


@pytest.mark.parametrize(
    ("checkpoint_name", "model_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.qkv_proj.weight",
            "q",
        ),
        (
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.qkv_proj.weight",
            "k",
        ),
        (
            "layers.1.linear_attn.in_proj_qkv.weight",
            "layers.1.linear_attn.in_proj_qkvz.weight",
            (0, 1, 2),
        ),
        (
            "layers.1.linear_attn.in_proj_z.weight",
            "layers.1.linear_attn.in_proj_qkvz.weight",
            3,
        ),
        (
            "layers.1.linear_attn.in_proj_b.weight",
            "layers.1.linear_attn.in_proj_ba.weight",
            0,
        ),
        (
            "layers.1.mlp.gate_proj.weight",
            "layers.1.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.1.mlp.experts.0.gate_proj.weight",
            "layers.1.mlp.experts.0.gate_proj.weight",
            None,
        ),
        (
            "layers.0.self_attn.indexer.index_qk_proj.weight",
            "layers.0.self_attn.indexer.index_qk_proj.weight",
            None,
        ),
        (
            "layers.0.attn_hyper_connection.input_mix_weight_down.weight",
            "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
            0,
        ),
        (
            "layers.0.attn_hyper_connection.block_inject_weight.weight",
            "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
            1,
        ),
        (
            "hyper_connection_mixer.input_mix_weight_down.weight",
            "hyper_connection_mixer.input_mix_weight_down.weight",
            None,
        ),
        (
            "layers.1.ple.ple_embedding.layer_multipliers",
            "layers.1.ple.ple_embedding.layer_multipliers",
            None,
        ),
    ],
)
def test_text_checkpoint_mapper_preserves_qwen4_exp_specific_weights(
    checkpoint_name: str,
    model_name: str,
    shard_id: str | int | tuple[int, ...] | None,
) -> None:
    assert Qwen4ExpModel.hf_to_vllm_mapper._map_name_with_shard(checkpoint_name) == (
        model_name,
        shard_id,
    )


def test_outer_checkpoint_mapper_selects_language_model_only_paths() -> None:
    mapper = Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper

    assert (
        mapper._map_name("model.language_model.layers.0.ple.key_proj.weight")
        == "language_model.model.layers.0.ple.key_proj.weight"
    )
    assert mapper._map_name("lm_head.weight") == "language_model.lm_head.weight"
    assert mapper._map_name("model.visual.blocks.0.attn.qkv.weight") == (
        "visual.blocks.0.attn.qkv.weight"
    )


def test_sparse_moe_attaches_private_recursive_loader_mapping(monkeypatch) -> None:
    class FakeExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expert_mapping = None

    def fake_qwen3_next_init(self, vllm_config, prefix="") -> None:
        del vllm_config, prefix
        nn.Module.__init__(self)
        self.n_routed_experts = 2
        self.n_redundant_experts = 0
        self.experts = FakeExperts()

    monkeypatch.setattr(
        Qwen3NextSparseMoeBlock,
        "__init__",
        fake_qwen3_next_init,
    )
    vllm_config = type(
        "VllmConfigStub",
        (),
        {
            "parallel_config": type(
                "ParallelConfigStub", (), {"use_sequence_parallel_moe": False}
            )(),
            "model_config": type(
                "ModelConfigStub",
                (),
                {
                    "hf_text_config": type(
                        "TextConfigStub",
                        (),
                        {"shared_expert_intermediate_size": 640},
                    )()
                },
            )(),
        },
    )()

    block = Qwen4ExpSparseMoeBlock(vllm_config, prefix="model.layers.0.mlp")

    assert block.experts.expert_mapping == [
        ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
        ("experts.w2_", "experts.0.down_proj.", 0, "w2"),
        ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
        ("experts.w13_", "experts.1.gate_proj.", 1, "w1"),
        ("experts.w2_", "experts.1.down_proj.", 1, "w2"),
        ("experts.w13_", "experts.1.up_proj.", 1, "w3"),
    ]


@pytest.mark.parametrize(
    ("checkpoint_name", "model_name"),
    [
        (
            "layers.0.self_attn.k_proj.k_scale",
            "layers.0.self_attn._k_scale",
        ),
        (
            "layers.0.self_attn.v_proj.output_scale",
            "layers.0.self_attn._v_scale",
        ),
        (
            "language_model.model.layers.0.self_attn.attn.k_scale",
            "language_model.model.layers.0.self_attn._k_scale",
        ),
        (
            "layers.0.self_attn.indexer.index_qk_proj.weight_scale",
            "layers.0.self_attn.indexer.index_qk_proj.weight_scale",
        ),
        (
            "layers.1.self_attn.k_proj.k_scale",
            "layers.1.self_attn.k_proj.k_scale",
        ),
    ],
)
def test_only_qsa_main_cache_scales_move_to_the_merged_owner(
    checkpoint_name: str,
    model_name: str,
) -> None:
    assert _remap_qsa_cache_scale_name(checkpoint_name, frozenset({0})) == model_name
