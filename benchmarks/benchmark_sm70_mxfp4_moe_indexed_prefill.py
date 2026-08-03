# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare materialized and indexed DeepSeek V4 MXFP4 W13 prefill."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
from benchmark_sm70_mxfp4_moe_prefill import (
    STAGES,
    _prepare_experts,
    _require_sm70,
)

from vllm import _sm70_ops as sm70_ops


def _make_scratch(
    prompt_tokens: int,
    top_k: int,
    num_experts: int,
    hidden_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    expanded_rows = prompt_tokens * top_k
    return {
        "token_expert_indices": torch.arange(
            expanded_rows, dtype=torch.int32, device=device
        ).view(prompt_tokens, top_k),
        "permuted_input": torch.empty(
            expanded_rows, hidden_size, dtype=torch.float16, device=device
        ),
        "expert_offsets64": torch.empty(
            num_experts + 1, dtype=torch.int64, device=device
        ),
        "expert_offsets": torch.empty(
            num_experts + 1, dtype=torch.int32, device=device
        ),
        "inv_permuted_idx": torch.empty(
            prompt_tokens, top_k, dtype=torch.int32, device=device
        ),
        "permuted_idx": torch.empty(expanded_rows, dtype=torch.int32, device=device),
        "permuted_experts_id": torch.empty(
            prompt_tokens, top_k, dtype=torch.int32, device=device
        ),
        "sorted_row_idx": torch.empty(
            prompt_tokens, top_k, dtype=torch.int32, device=device
        ),
        "topk_ids_for_sort": torch.empty(
            prompt_tokens, top_k, dtype=torch.int32, device=device
        ),
        "sort_workspace": torch.empty(
            torch.ops._moe_C.moe_permute_sort_workspace_size(
                expanded_rows, num_experts
            ),
            dtype=torch.int8,
            device=device,
        ),
    }


def _time(call, repeats: int) -> list[float]:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    if args.prompt_tokens < 1 or args.top_k < 1 or args.repeats < 3:
        raise ValueError("prompt-tokens/top-k must be positive and repeats >= 3")
    _require_sm70()
    for op_name in (
        "moe_permute_with_scratch",
        "moe_permute_indexed_with_scratch",
    ):
        if not hasattr(torch.ops._moe_C, op_name):
            raise RuntimeError(f"Required operator is missing: _moe_C::{op_name}")
    if not hasattr(torch.ops._C, "mxfp4_moe_indexed_dense_stage_sm70_out"):
        raise RuntimeError(
            "Required operator is missing: _C::mxfp4_moe_indexed_dense_stage_sm70_out"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    shape = STAGES["w13"]
    expanded_rows = args.prompt_tokens * args.top_k
    x = torch.randn(args.prompt_tokens, shape.k, dtype=torch.float16, device=device)
    scores = torch.rand(
        args.prompt_tokens, args.num_experts, dtype=torch.float32, device=device
    )
    topk_ids = scores.topk(args.top_k, dim=-1).indices.to(torch.int32)
    del scores

    weights, scales, ptrs_w, ptrs_s = _prepare_experts(shape, args.num_experts, device)
    scratch = _make_scratch(
        args.prompt_tokens, args.top_k, args.num_experts, shape.k, device
    )
    dense_ids = torch.arange(args.num_experts, dtype=torch.int32, device=device)
    reference_out = torch.empty(
        expanded_rows, shape.n, dtype=torch.float16, device=device
    )
    indexed_out = torch.empty_like(reference_out)

    def prepare_common() -> None:
        scratch["permuted_idx"].fill_(expanded_rows)

    def reference() -> None:
        prepare_common()
        torch.ops._moe_C.moe_permute_with_scratch(
            x,
            topk_ids,
            scratch["token_expert_indices"],
            None,
            args.num_experts,
            args.num_experts,
            args.top_k,
            scratch["permuted_input"],
            scratch["expert_offsets64"],
            scratch["inv_permuted_idx"],
            scratch["permuted_idx"],
            scratch["sort_workspace"],
            scratch["permuted_experts_id"],
            scratch["sorted_row_idx"],
            scratch["topk_ids_for_sort"],
        )
        scratch["expert_offsets"].copy_(scratch["expert_offsets64"])
        sm70_ops.mxfp4_moe_dense_stage_sm70_out(
            reference_out,
            scratch["permuted_input"],
            scratch["expert_offsets"],
            dense_ids,
            ptrs_w,
            ptrs_s,
            args.num_experts,
            shape.k,
            shape.n,
            32,
        )

    def indexed() -> None:
        prepare_common()
        torch.ops._moe_C.moe_permute_indexed_with_scratch(
            x,
            topk_ids,
            scratch["token_expert_indices"],
            args.num_experts,
            args.num_experts,
            args.top_k,
            scratch["permuted_input"],
            scratch["expert_offsets64"],
            scratch["inv_permuted_idx"],
            scratch["permuted_idx"],
            scratch["sort_workspace"],
            scratch["permuted_experts_id"],
            scratch["sorted_row_idx"],
            scratch["topk_ids_for_sort"],
        )
        scratch["expert_offsets"].copy_(scratch["expert_offsets64"])
        sm70_ops.mxfp4_moe_indexed_dense_stage_sm70_out(
            indexed_out,
            x,
            scratch["sorted_row_idx"].view(-1),
            scratch["expert_offsets"],
            dense_ids,
            ptrs_w,
            ptrs_s,
            args.num_experts,
            shape.k,
            shape.n,
            32,
        )

    grouped_env = "VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL"
    original_grouped = os.environ.get(grouped_env)
    try:
        os.environ[grouped_env] = "1"
        reference()
        torch.cuda.synchronize()
        expected_out = reference_out.clone()
        expected_offsets = scratch["expert_offsets64"].clone()
        expected_inv = scratch["inv_permuted_idx"].clone()
        expected_permuted = scratch["permuted_idx"].clone()

        indexed()
        torch.cuda.synchronize()
        output_equal = torch.equal(expected_out, indexed_out)
        max_abs = float((expected_out - indexed_out).abs().max().item())
        metadata_equal = {
            "expert_offsets": torch.equal(
                expected_offsets, scratch["expert_offsets64"]
            ),
            "inverse_permutation": torch.equal(
                expected_inv, scratch["inv_permuted_idx"]
            ),
            "expanded_permutation": torch.equal(
                expected_permuted, scratch["permuted_idx"]
            ),
            "source_row_indices": torch.equal(
                expected_permuted // args.top_k,
                scratch["sorted_row_idx"].view(-1),
            ),
        }

        eager_indexed = indexed_out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            indexed()
        graph.replay()
        torch.cuda.synchronize()
        graph_equal = torch.equal(eager_indexed, indexed_out)

        reference_samples = _time(reference, args.repeats)
        indexed_samples = _time(indexed, args.repeats)
    finally:
        if original_grouped is None:
            os.environ.pop(grouped_env, None)
        else:
            os.environ[grouped_env] = original_grouped

    reference_median = statistics.median(reference_samples)
    indexed_median = statistics.median(indexed_samples)
    payload = {
        "contract": {
            "model": "DeepSeek-V4-Flash",
            "tp": 8,
            "prompt_tokens": args.prompt_tokens,
            "top_k": args.top_k,
            "expanded_rows": expanded_rows,
            "num_experts": args.num_experts,
            "stage": "w13",
            "k": shape.k,
            "n": shape.n,
            "seed": args.seed,
        },
        "correctness": {
            "output_bitwise": output_equal,
            "output_max_abs": max_abs,
            "metadata_bitwise": metadata_equal,
            "graph_replay_bitwise": graph_equal,
        },
        "timing": {
            "reference_samples_ms": reference_samples,
            "indexed_samples_ms": indexed_samples,
            "reference_median_ms": reference_median,
            "indexed_median_ms": indexed_median,
            "speedup": reference_median / indexed_median,
            "projected_43_layer_savings_ms": (reference_median - indexed_median) * 43,
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    del weights, scales
    return 0 if output_equal and graph_equal and all(metadata_equal.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
