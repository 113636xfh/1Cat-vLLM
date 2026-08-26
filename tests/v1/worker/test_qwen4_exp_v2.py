# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu import model_runner as mrv2
from vllm.v1.worker.gpu.block_table import BlockTables


def test_qsa_circular_group_uses_one_block_and_custom_slot_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=False)
    runner.vllm_config = SimpleNamespace()
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cpu")

    circular_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )
    compressed_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["compressor_state"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8, kv_cache_specs={"compressor_state": circular_spec}
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["attention", "compressed"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={
                        "attention": attention_spec,
                        "compressed": compressed_spec,
                    },
                ),
            ),
        ],
    )

    monkeypatch.setattr(
        mrv2,
        "init_attn_backend",
        lambda *args: ([], SimpleNamespace(), [8, 16]),
    )
    captured: dict[str, object] = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(mrv2, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 16384]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an exact SM70 CUDA device",
)
def test_qsa_circular_group_emits_no_generic_slots_on_sm70() -> None:
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[8, 262144],
        max_num_reqs=1,
        max_num_batched_tokens=4,
        max_num_blocks_per_group=[1, 1],
        device=device,
        kernel_block_sizes=[8, 262144],
        slot_mapping_enabled=[False, True],
    )
    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([7], [12]),
        overwrite=True,
    )
    block_tables.apply_staged_writes()

    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping=torch.tensor([0], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=device),
        positions=torch.tensor([153797, 165757], dtype=torch.int64, device=device),
        num_tokens_padded=2,
    )
    torch.cuda.synchronize()

    assert slot_mappings[0].tolist() == [-1, -1]
    assert slot_mappings[1].tolist() == [
        12 * 262144 + 153797,
        12 * 262144 + 165757,
    ]
