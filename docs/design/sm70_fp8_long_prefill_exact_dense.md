# SM70 FP8 Long-Prefill Exact-Dense Projections

Date: 2026-08-15

## Scope

This route accelerates Qwen3.8-27B-FP8 TP4 prefill on four V100-32GB GPUs.
The accepted contract uses Python 3.12, Torch 2.10.0+cu128, FP16 KV,
Flash-V100, prefix caching with Mamba align, CUDA graphs, no MTP, and
`max_num_batched_tokens=15680`.

## Root Cause

The earlier Flash-V100 gather/exact-D256 attention route was active. The
missing gain was in large-M FP8 projections: TurboMind's W8A16 kernel is the
correct decode and tail path, but the 32K Nsight trace attributed 68.4% of
summed FP8 projection time to it. A first Python `M >= 3920` branch was also
invalid because the dynamic compile range folded the branch during tracing;
the measured request still launched 3648 TurboMind calls and no dense GEMM.

The accepted implementation places the M decision in the opaque CUDA op
`fp8_gemm_sm70_prefill_dispatch_out`. It reconstructs TurboMind's K8/N32 FP8
layout into one shared 85 MiB FP16 KxN workspace, then uses `mm_out` or the
exact gated-SiLU epilogue. M below 3920 stays in TurboMind.

The default allowlist is TP4 plus the exact `(K,N)` shape for
`gate_up_proj`, `down_proj`, `out_proj`, and `o_proj`. `qkv_proj` and
`in_proj_qkvz` remain excluded because direct dense evaluation changed FP16
outputs. The environment rollback is
`VLLM_SM70_FP8_PREFILL_EXACT_DENSE=0`.

## Performance

Matched chunk-15680 control and candidate requests preserve output hashes.

| Input | Control prefill | Candidate prefill | Control tok/s | Candidate tok/s | Throughput gain |
|---:|---:|---:|---:|---:|---:|
| 32K | 11.024 s | 9.010 s | 2972.3 | 3636.7 | 22.36% |
| 128K | 60.507 s | 53.575 s | 2166.2 | 2446.5 | 12.94% |
| 256K* | 175.323 s | 163.472 s | 1493.7 | 1602.0 | 7.25% |

`256K*` uses 261888 prompt tokens. The final sustained sweep is:

| Input | Mean prefill | Prompt tok/s |
|---:|---:|---:|
| 1K | 0.407 s | 2515.1 |
| 4K | 1.065 s | 3847.8 |
| 8K | 2.199 s | 3725.4 |
| 16K | 4.199 s | 3901.6 |
| 32K | 9.533 s | 3437.2 |
| 64K | 22.981 s | 2851.7 |
| 128K | 53.575 s | 2446.5 |
| 256K* | 163.472 s | 1602.0 |

The remembered 128K/256K reference was Qwen3.6-27B-AWQ without prefix
caching: 2517.6/1655.0 tok/s, not a same-model Qwen3.8-FP8 result. The current
Qwen3.8 production contract is within 2.8%/3.2% of that cross-model reference.

## Correctness And Memory

- 72 real-weight rank/shape/M checks are bitwise equal for all admitted
  projections.
- 32K, 128K, and 256K candidate output hashes exactly match their controls.
  All short-sweep repeats also have stable hashes and `is_corrupted=false`.
- Model residency is 7.57 GiB/rank. The 85 MiB shared workspace leaves
  19.18 GiB/rank for KV and an estimated 4.70x 256K cache capacity.
- Compiled microbenchmarks prove both M=784 and M=3920 reach the same runtime
  dispatch op. M3920 and M7840 improve 1.31x and 1.46x with zero mismatch.

## Closed Paths

- Do not restore a Python dynamic-M branch; `torch.compile` can fold it.
- Do not include QKV shapes without a new bitwise proof.
- Extending dense split-KV3 from Q4096 to Q15680 gives only 0.85%-1.80%
  attention-kernel gain, needs about 290 MiB/rank of FP32 workspace, and is
  not bitwise. It is not promoted.

## Evidence

- Control: `/data/minimax-h3/task-cache/qwen38-130-prefill-recovery-20260815/results/chunk15680-long-sweep.json`
- Candidate short sweep: `/data/minimax-h3/task-cache/qwen38-130-prefill-recovery-20260815/candidate-dispatch/results/short-1k-64k.json`
- Candidate long sweep: `/data/minimax-h3/task-cache/qwen38-130-prefill-recovery-20260815/candidate-dispatch/results/long-128k-256k.json`
- Route trace: `/data/minimax-h3/task-cache/qwen38-130-prefill-recovery-20260815/candidate-dispatch/nsys/qwen38_fp8_dispatch_tp4_i32k_r2.nsys-rep`
