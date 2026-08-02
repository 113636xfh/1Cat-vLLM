# SM70 DeepSeek V4 Sparse MLA Split-K

## Scope

- Base: `dd462e37f2552f3e038f1ed7128e62bd7b4ab0d7` (PR #159)
- Model: DeepSeek-V4-Flash, TP8 on 8 x V100-SXM2-32GB
- Decode: CUDA Graph, FP16 query/output, packed `fp8_ds_mla` KV
- Quantization: TurboMind MXFP4; Marlin is out of scope
- Route gates: `VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_C4` and
  `VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_C128`

Both gates remain default-off until same-contract full-model speed and quality
checks pass. C4 and C128 use separate gates so either route can be rejected
without weakening the other.

## Trace Baseline

The accepted active-expert baseline is exact 1024 input / 256 output,
official `temperature=1.0`, `top_p=1.0`, no MTP, and CUDA Graph. Unprofiled
TPOT is `76.268 ms/token` (`13.112 tok/s`). The graph-node trace records
`78.087 ms/token`; use it for composition only.

| Sparse MLA layer type | Layers | Mean/layer | Per-token service |
|---|---:|---:|---:|
| SWA-only | 2 | 0.548 ms | 1.095 ms |
| C4 | 21 | 1.651 ms | 34.666 ms |
| C128 | 20 | 0.566 ms | 11.326 ms |
| Total | 43 | - | 46.920 ms |

Raw graph-node artifacts:

```text
/home/fudanwl/v100-worktrees/runs/
  dsv4-tp8-active-expert-b1-nsys-i1024-o256-retry1-20260802/
```

## Root Cause

TP8 leaves eight query heads per rank. The baseline uses `BLOCK_H=8`, so each
layer launches one CTA and serially scans every sparse KV block on one of the
V100's 80 SMs.

Exact C4 NCU evidence:

| Metric | Baseline |
|---|---:|
| Grid / block | `1 CTA` / `128 threads` |
| Duration | 2.11 ms |
| Registers | 32/thread |
| Dynamic shared memory | 51.71 KiB/CTA |
| Achieved occupancy | 6.25% |
| SM throughput | 0.20% |
| DRAM throughput | 0.03% |
| Scheduler cycles with no eligible warp | 92.61% |
| Long-scoreboard share of issue interval | 56.92% |

The bottleneck is insufficient CTA parallelism and exposed KV/dequant latency,
not saturated HBM or tensor-core throughput.

## Implementation

The candidate uses Flash-Decoding-style KV partitioning:

1. One CTA handles one 16-token sparse KV block and writes FP32 partial
   `(max, sum, weighted-value)` state.
2. A second kernel combines partial states in FP32 and applies the attention
   sink before writing FP16 output.
3. The C4 1024-token shape launches 40 stage-1 CTAs and 64 reduction CTAs.
4. Scratch comes from the graph-safe worker workspace and is reused across
   layers; there is no hot-path allocation or host synchronization.
5. E4M3 normal values are decoded by exact IEEE-FP32 bit construction. The
   seven NOPE scales are loaded and decoded once per 64-element group, then
   broadcast instead of being redundantly expanded 64 times.

## Microbenchmark Evidence

Exact q=1, eight-head CUDA Graph measurements:

| Shape | Baseline | Candidate | Speedup | Max abs error |
|---|---:|---:|---:|---:|
| C4, main 128 + extra 320 | 1.957 ms | 0.101 ms | 19.4x | 1.53e-5 |
| C128, main 128 + extra 10 | 0.581 ms | 0.078 ms | 7.4x | 3.05e-5 |
| C128, extra 512 | 4.899 ms | 0.100 ms | 49.1x | 7.63e-6 |
| C128, extra 1024 | 9.265 ms | 0.119 ms | 77.9x | 7.63e-6 |
| C128, extra 2048 | 17.966 ms | 0.152 ms | 118.2x | 7.63e-6 |

Latest C4 NCU stage times are `87.74 us` for split-K and `5.22 us` for the
reducer. Stage-1 reaches 40 CTAs and 12.93% DRAM throughput. These are kernel
measurements, not an end-to-end claim.

Numerical and graph gates completed:

- all 254 finite E4M3FN byte encodings match the arithmetic decoder bitwise;
- both E4M3FN NaN encodings preserve the same NaN mask;
- realistic FP8 KV tests over multiple seeds have max absolute output error
  at or below `1.53e-5` for the main C4 target;
- q=1 and q=2 CUDA Graph capture/replay complete with finite output;
- C128 lengths through 2048 compressed tokens remain finite and within the
  recorded error bound.

Artifacts:

```text
/home/fudanwl/v100-worktrees/runs/
  dsv4-sm70-sparse-attn-micro-20260802/
```

## Rejected Variants

| Variant | C4 graph mean | Decision |
|---|---:|---|
| 2 stage-1 warps | 0.481 ms | Reject; too little intra-CTA latency hiding |
| 4 stage-1 warps | 0.174 ms before scale grouping | Keep |
| 8 stage-1 warps | 0.279 ms | Reject; extra warps increase cost |
| `BLOCK_H=4` | 0.278 ms | Reject; duplicates KV/HMMA work to fill 80 CTAs |
| `BLOCK_H=8` | 0.177 ms before scale grouping | Keep |

## Remaining Gates

1. Measure C4-only and C4+C128 TP8 1024/256 endpoint TPOT against the accepted
   active-expert baseline.
2. Verify worker logs select both split-K routes inside FULL CUDA Graph.
3. Compare deterministic tokens/logits and run official-sampling long output.
4. Re-profile one endpoint request to prove sparse MLA service moves out of the
   first position without shifting cost into another kernel family.
5. Sweep long-context decode before either gate becomes default-on.
