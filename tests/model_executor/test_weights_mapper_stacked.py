# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.utils import WeightsMapper


def test_weights_mapper_preserves_stacked_shard_id() -> None:
    mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q.": (".qkv.", "q"),
            ".k.": (".qkv.", "k"),
        }
    )
    q = torch.ones(2, 4)
    k = torch.full((2, 4), 2.0)

    mapped = list(
        mapper.apply(
            [
                ("layer.q.weight", q),
                ("layer.k.weight", k),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        "layer.qkv.weight",
        "layer.qkv.weight",
    ]
    assert q.shard_id == "q"
    assert k.shard_id == "k"
