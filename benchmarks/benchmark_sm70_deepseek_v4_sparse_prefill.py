# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the DeepSeek V4 gathered sparse-prefill kernel on SM70."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.models.deepseek_v4.sm70.sparse_kernels import (
    _sm70_sparse_gathered_kernel,
)
from vllm.triton_utils import triton


def _cuda_device_module():
    if not torch.accelerator.is_available():
        raise RuntimeError("CUDA is required")
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None or accelerator.type != "cuda":
        raise RuntimeError(f"CUDA is required, got {accelerator}")
    return torch.get_device_module(accelerator)


def _launch(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    sink: torch.Tensor,
    out: torch.Tensor,
    *,
    block_h: int,
    block_k: int,
    num_warps: int,
) -> None:
    _sm70_sparse_gathered_kernel[(q.shape[0], triton.cdiv(q.shape[1], block_h))](
        q,
        kv,
        indices,
        lengths,
        sink,
        out,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[1],
        kv.shape[0],
        512**-0.5,
        INDEX_WIDTH=indices.shape[1],
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        BLOCK_D=512,
        num_warps=num_warps,
    )


def _measure(call, *, warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        call()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _make_inputs(
    *, q_tokens: int, num_heads: int, pattern: str, seed: int
) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    torch.manual_seed(seed)
    q = torch.randn((q_tokens, num_heads, 512), dtype=torch.float16, device=device)
    if pattern == "c4":
        compress_ratio, index_width = 4, 640
    elif pattern == "c128":
        compress_ratio, index_width = 128, 256
    else:
        compress_ratio, index_width = 1, 128
    compressed_tokens = 0 if pattern == "swa" else triton.cdiv(q_tokens, compress_ratio)
    num_kv = compressed_tokens + q_tokens
    kv = torch.randn((num_kv, 512), dtype=torch.float16, device=device)

    rows = torch.arange(q_tokens, dtype=torch.int64, device=device)[:, None]
    columns = torch.arange(index_width, dtype=torch.int64, device=device)[None, :]
    compressed_lens = torch.clamp(
        (rows[:, 0] + 1) // compress_ratio,
        max=index_width - 128,
    )
    if pattern == "swa":
        compressed_lens.zero_()
    swa_lens = torch.clamp(rows[:, 0] + 1, max=128)
    lengths = (compressed_lens + swa_lens).to(torch.int32)
    indices = ((rows * 131 + columns * 67) % num_kv).to(torch.int32)
    indices.masked_fill_(columns >= lengths[:, None], -1)

    sink = torch.full((num_heads,), -float("inf"), dtype=torch.float32, device=device)
    reference = torch.empty_like(q)
    candidate = torch.empty_like(q)
    return q, kv, indices, lengths, sink, reference, candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-tokens", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--candidate", choices=("triton", "hmma"), default="triton")
    parser.add_argument("--pattern", choices=("c4", "c128", "swa"), default="c4")
    parser.add_argument("--block-h", type=int, default=8)
    parser.add_argument("--block-k", type=int, default=16)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if _cuda_device_module().get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires an NVIDIA V100 (SM70)")
    if args.block_h not in (1, 2, 4, 8, 16):
        raise ValueError("--block-h must be one of 1, 2, 4, 8, or 16")
    if args.block_k != 16:
        raise ValueError("--block-k must be 16 on SM70")
    if args.q_tokens < 1 or args.num_heads < 1:
        raise ValueError("token and head counts must be positive")

    q, kv, indices, lengths, sink, reference, candidate = _make_inputs(
        q_tokens=args.q_tokens,
        num_heads=args.num_heads,
        pattern=args.pattern,
        seed=args.seed,
    )

    baseline_call = lambda: _launch(
        q,
        kv,
        indices,
        lengths,
        sink,
        reference,
        block_h=8,
        block_k=16,
        num_warps=4,
    )
    if args.candidate == "hmma":
        if args.num_heads != 8:
            raise ValueError("the HMMA candidate requires --num-heads 8")

        candidate_call = lambda: torch.ops._C.sm70_deepseek_v4_sparse_attention_hmma(
            q, kv, indices, lengths, sink, candidate, 512**-0.5
        )
    else:
        candidate_call = lambda: _launch(
            q,
            kv,
            indices,
            lengths,
            sink,
            candidate,
            block_h=args.block_h,
            block_k=args.block_k,
            num_warps=args.num_warps,
        )

    baseline_call()
    candidate_call()
    torch.accelerator.synchronize()
    difference = (reference.float() - candidate.float()).abs()
    baseline_samples = _measure(
        baseline_call, warmups=args.warmups, repeats=args.repeats
    )
    candidate_samples = _measure(
        candidate_call, warmups=args.warmups, repeats=args.repeats
    )
    baseline_median = statistics.median(baseline_samples)
    candidate_median = statistics.median(candidate_samples)

    result = {
        "shape": {
            "pattern": args.pattern,
            "q_tokens": args.q_tokens,
            "num_heads": args.num_heads,
            "head_dim": 512,
            "index_width": indices.shape[1],
            "kv_rows": kv.shape[0],
        },
        "baseline": {
            "block_h": 8,
            "block_k": 16,
            "num_warps": 4,
            "samples_ms": baseline_samples,
            "median_ms": baseline_median,
        },
        "candidate": {
            "backend": args.candidate,
            "block_h": args.block_h,
            "block_k": args.block_k,
            "num_warps": 8 if args.candidate == "hmma" else args.num_warps,
            "samples_ms": candidate_samples,
            "median_ms": candidate_median,
        },
        "speedup": baseline_median / candidate_median,
        "bitwise_equal": torch.equal(reference, candidate),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "mismatch_elements": int(torch.count_nonzero(difference).item()),
        "mismatch_fraction": float(torch.count_nonzero(difference).item())
        / difference.numel(),
        "all_finite": bool(torch.isfinite(candidate).all().item()),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["all_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
