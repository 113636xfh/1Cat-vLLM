# SM70 FA2 D256 Split-D Prefill

## Status

Qwen3.6-27B full-attention prefill now uses a dedicated Volta D256 kernel for
both the first dense chunk and later chunks in the standard interleaved paged
KV cache. The kernel reads the existing layouts directly and writes into the
vLLM output tensor without a gather, repack, temporary output, or copy.

The route is enabled by default. Set
`VLLM_FLASH_V100_FA2_D256_PREFILL=0` to disable it. Dispatch is exact-only:
unsupported shapes return to the established `FLASH_ATTN_V100` path rather
than entering an unvalidated generic FA2 fallback. Decode, MTP small-query
attention, FP8 KV cache, sliding-window attention, and non-causal attention
are unchanged.

## Accepted Contract

The full-model acceptance workload is:

- model: `/home/ymzx/models/Qwen3.6-27B-AWQ`;
- GPU: 4x V100-32GB, TP4, CUDA 12.8, PyTorch 2.10.0+cu128;
- FP16 activations and KV, TurboMind AWQ, no MTP;
- `FLASH_ATTN_V100`, CUDA graph `FULL_DECODE_ONLY`, not eager;
- input/output: 8192/16, chunk size 4096, max model length 16384;
- one warmup and two measured requests;
- separate official-sampling quality runs at 1K/64 and 8K/128.

The operator requires SM70, FP16, causal full attention, head dimension 256,
unit inner strides, query length at least 1024, and query/KV lengths divisible
by 64. The paged route currently accepts one sequence per call and preserves
the runtime block table. All other cases retain the previous implementation.

## Kernel Design

The external source is pinned to
`zhinianqin/flash-attention-v100@c2eda5e6115b98c3ba4bfd181570668742eece22`.
The vendored patch adds exact dense and paged Torch operators. The final
operator ABI includes `n32`; missing exact N32 operators are treated as a
stale build and cause a safe fallback. This prevents an older,
quality-rejected N64 binary from being selected from a local cache.

The accepted specialization uses:

- `BLOCK_M=64`, `BLOCK_N=32`, and four D64 chunks;
- eight warps arranged as four two-warp groups;
- eight distinct Q rows per warp for QK, eliminating duplicated QK work;
- a shared N32 P tile per warp pair;
- D128 output ownership per warp for PV;
- the standard FA2 N32 online-softmax order;
- conflict-aware Q/K/V layouts and software register prefetch;
- one paged-KV address resolution per K/V N32 tile, with D64 pointers
  derived by constant column offsets;
- a Volta `TT` PV tensor-core mapping that consumes V as K-by-D directly;
- 128-bit conflict-aware V stores and paired 64-bit V loads, removing the
  previous transpose-path scalar loads and most operand permutation work;
- FP32 score, online-softmax, and output accumulation;
- 45,568 bytes of dynamic shared memory.

The N32 order is a quality requirement, not a shape-tuning choice. An earlier
N64 variant merged two N32 halves before softmax. Its standalone relative L2
error was only about `1.27e-4`, but repeated full-attention layers amplified
that error enough to change official sampled tokens. N32 reduces the operator
relative L2 error to about `4.6e-6` at 4K and restores model token parity.

Dense and paged kernels use independent Q, K, and V strides. A real TP rank
has:

```text
Q: shape (4096, 6, 256), stride (1536, 256, 1)
K: shape (4096, 1, 256), stride (256, 256, 1)
V: shape (4096, 1, 256), stride (3584, 256, 1)
```

The paged kernel reads the normal
`[block, K/V, page, head, dim]` allocation directly, including page size 784.
No KV-major allocation change is retained.

The final causal SM70 specialization uses 255 registers/thread with no spill
or local stack. It remains at one CTA per SM. The gain comes from useful D256
parallelism, lower operand-movement cost, and software scheduling, not higher
occupancy.

## Operator Results

Measurements use Qwen TP4 `Hq=6`, `Hkv=1`, D256, FP16, and causal attention.
`Causal TFLOP/s` counts attended pairs. `Full-square logical TFLOP/s` counts
the masked upper triangle and is included only to compare with tools that use
that convention; it is not hardware-executed work.

| Q=KV | Generic FA2 | N32 Split-D | Speedup | Causal TFLOP/s | Full-square logical |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.340 ms | 0.213 ms | 1.60x | 15.1 | 30.2 |
| 4096 | 2.190 ms | 1.779 ms | 1.23x | 29.0 | 57.9 |
| 8192 | 7.043 ms | 5.425 ms | 1.30x | 38.0 | 76.0 |

The 4K and 8K logical rates exceed the original 49-TOPS comparison target.
The physically meaningful causal rates are reported separately above.

The release comparison below uses the real chunked-prefill shape (`Q=4096`),
Qwen TP4 heads (`Hq=6`, `Hkv=1`), page size 784, and a fixed 1312 MHz V100
application clock. `Current 1Cat` is the Flash-V100 extension shipped in
1Cat-vLLM v1.2.2 at source revision `644d8a7cd0`, not an earlier kernel from
this research branch. Its extension SHA256 is
`8582b5b1a72d5ebfd9a35417f267298845195a6846285e26fff7ad9a5905f771`.
Each entry is the median of two alternating runs after six warmups; both
variants were measured at exactly the same clock.

| KV length | Current 1Cat v1.2.2 | Final Split-D TT | Latency reduction | Speedup |
|---:|---:|---:|---:|---:|
| 8K | 11.1255 ms | 5.0542 ms | 54.57% | 2.20x |
| 64K | 87.6001 ms | 50.4504 ms | 42.41% | 1.74x |
| 128K | 174.9729 ms | 103.8420 ms | 40.65% | 1.68x |
| 256K | 349.6960 ms | 210.7364 ms | 39.74% | 1.66x |

A separate production-clock cross-check at up to 1530 MHz measured 2.20x,
1.69x, 1.75x, and 1.64x respectively. The fixed-clock table is authoritative
because the long runs otherwise trigger clock drift.

The final clean build is bitwise identical to the previously accepted exact
N32 operator at all four lengths. Against the current 1Cat extension, relative
L2 error is `3.79e-4` at 8K and remains below `4.0e-4` through 256K.

NCU isolates the final TT operand path from the immediately preceding
address-reuse candidate. At 8K and 1312 MHz, kernel time falls from 5.18 ms to
5.06 ms while LSU thread instructions fall from 187.1M to 132.5M and shared
load wavefronts fall from 268.0M to 230.0M. Static SASS shrinks from 1,875 to
1,767 instructions: `LDS.U16` falls from 128 to zero and `PRMT` from 68 to
four. Tensor active reaches 32.06%. Shared-store conflicts increase, but their
cost is smaller than the removed transpose/load/permutation chain.

An adjacent real-model TP4 route check used Qwen3.6-27B-AWQ, 8K input,
16-token deterministic output, chunk size 4096, FP16 KV, FlashAttentionV100,
and non-eager FULL_DECODE_ONLY graphs. With an unrelated resident Ray service
left untouched and a 2 GiB KV reservation, the previous binary measured
2586.58 ms prefill and the candidate measured 2581.31 ms, a 5.27 ms (0.20%)
reduction. Every rank reported 48
`prefill_prefix_paged_splitd_d256` calls and output token hashes matched. The
resident service slowed both prefill and decode relative to the clean historic
baseline, so this result proves route translation but is not a replacement
release baseline.

## Numerical And Quality Gates

- Exact Split-D versus generic FA2 has max absolute error `2.4414e-4` and
  relative L2 error `4.58e-6` at 4K and `5.79e-6` at 8K.
- Real non-contiguous V and a contiguous copy are bitwise identical.
- A randomized non-sequential paged block table is bitwise identical to the
  corresponding dense exact operator.
- The 1K/64 official Qwen sample produces 64/64 identical token IDs against
  both original Flash-V100 and generic FA2.
- The 8K/128 official Qwen sample produces 128/128 identical token IDs and
  identical text against original Flash-V100.
- The deterministic warmup and both measured requests retain token hash
  `220e51bcf45e69de1a35817c9501aadfcae784ed195448b3bf37b05d4aa815a2`.
- Every TP rank reports 48 dense and 48 paged exact route hits in the stable
  warmup-plus-two-repeat benchmark.

Prompt logprobs are not bitwise stable across FP16 attention implementations;
generic FA2 also differs from the original backend. Therefore promotion uses
operator error bounds plus official sampled token parity, rather than treating
the original backend's prompt perplexity as an absolute numerical oracle.

## Full-Model Result

The final controlled A/B uses Qwen3.6-27B-AWQ, TP4, FP16 KV, chunk size 4096,
16 generated tokens, no prefix cache, and non-eager compile graphs. Both
variants use the same source and runtime configuration; only
`VLLM_FLASH_V100_FA2_D256_PREFILL` changes. Four V100s are fixed at 1312 MHz.
Each value is the mean of two requests after a per-length warmup.

| Input | Current 1Cat prefill | Final TT prefill | Latency reduction | Current tok/s | Final tok/s | Throughput gain |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 2.5722 s | 2.4440 s | 4.99% | 3184.8 | 3351.9 | 5.25% |
| 64K | 30.2812 s | 25.5058 s | 15.77% | 2164.2 | 2569.5 | 18.72% |
| 128K | 83.1879 s | 64.5516 s | 22.40% | 1575.6 | 2030.5 | 28.87% |
| 256K | 256.6722 s | 183.9463 s | 28.33% | 1021.1 | 1424.8 | 39.54% |

The 256K row uses 262,080 input tokens so the final 4,032-token chunk remains
divisible by 64 and the 16-token generation stays within the model's exact
262,144-token limit. All measured requests preserve the current 1Cat output
token hash at every length.

The corresponding no-MTP decode gate uses the model's official sampling
parameters (`temperature=1.0`, `top_k=20`, `top_p=0.95`) and 256 generated
tokens. TPOT excludes prefill and the first generated token. The 256K row uses
261,824 input tokens, leaving 64 tokens below the exact model limit.

| Input | Current 1Cat TPOT | Final TT TPOT | Current tok/s | Final tok/s | Throughput delta |
|---:|---:|---:|---:|---:|---:|
| 8K | 15.034 ms | 15.037 ms | 66.52 | 66.50 | -0.02% |
| 64K | 18.999 ms | 18.985 ms | 52.63 | 52.67 | +0.08% |
| 128K | 24.072 ms | 24.062 ms | 41.54 | 41.56 | +0.04% |
| 256K | 33.766 ms | 33.760 ms | 29.62 | 29.62 | +0.02% |

All four 256-token output hashes match. The differences are measurement noise:
the new D256 operator is gated to prefill queries of at least 1,024 tokens,
while q=1 decode continues to use the existing paged XQA path. Relative to
8K, final decode throughput falls 20.8% at 64K, 37.5% at 128K, and 55.5% at
256K. Reducing that decay requires a separate decode-attention change.

An earlier no-compile 8K route-development run measured:

| Route | Mean prefill | Delta from original | Deterministic tokens |
|---|---:|---:|---:|
| Original Flash-V100 | 2.7445 s | reference | exact |
| Generic SM70 FA2 | 2.4084 s | -12.25% | exact |
| N32 Split-D direct output | 2.3355 s | **-14.90%** | exact |

The accepted route saves 408.93 ms against the original implementation and
72.85 ms against generic FA2. Its mean differs by only 0.73 ms from the faster
but quality-rejected N64 experiment.

The independent 8K/128 official-sampling run measured prefill
`3.1962 -> 2.8814 s` (-9.85%) and steady decode
`60.04 -> 60.27 tok/s`, with exact 128-token parity.

## Bottleneck After Attention

The retained Nsight Systems trace was captured before the N32 numerical-order
fix, so its exact attention time must not be reused as a current kernel timing.
It remains valid for category prioritization because N32 changes full-model
prefill by less than 0.1% relative to that run:

| Category | TP wall-equivalent time | Share |
|---|---:|---:|
| TurboMind AWQ GEMM | 1577.127 ms | 68.17% |
| Norm, activation, elementwise | 250.836 ms | 10.84% |
| TP NCCL collectives | 170.101 ms | 7.35% |
| GDN / linear attention | 111.975 ms | 4.84% |
| Split-D full attention | 96.850 ms | 4.19% |

The next end-to-end bottleneck is TurboMind AWQ GEMM, not D256 attention. A
representative `4096x8704x5120` gate/up kernel reaches 58.87 tensor TOPS but
is limited to 12.5% occupancy by 248 registers/thread and 65.55 KiB shared
memory.

## Evidence

- Final CUDA gate:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/splitd_n32_final_cuda_gate.json`
- Stable full-model result:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_8k_splitd_n32_direct_out.json`
- Final fixed-clock full-model comparison:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_prefill_current_vs_splitd_tt_fixed1312.json`
- Final fixed-clock decode comparison:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_decode_current_vs_splitd_tt_fixed1312.json`
- Current and final full-model raw results:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_prefill_current_1cat_fixed1312.json`
  and
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_prefill_splitd_tt_fixed1312.json`
- Current and final decode raw results:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_decode_current_1cat_fixed1312.json`
  and
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_decode_splitd_tt_fixed1312.json`
- Official 8K quality result:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/official_quality/splitd_n32_official_i8k_o128.json`
- Final N32 ABI default-on 1K quality result:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/official_quality/splitd_n32_abi_default_official_i1k_o64_logprobs.json`
- Official 8K comparison:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/official_quality/compare_baseline_splitd_n32_i8k_o128.json`
- Nsight Systems report:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/profiles/qwen36_27b_awq_tp4_i8k_splitd_both_exact_prefill.nsys-rep`
- Fixed-clock current-1Cat comparison:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/current_1cat_v122_vs_tt_vload_fixed1312.json`
- Final clean-build latency and quality gates:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/tt_final_clean_gate.jsonl`
  and
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/tt_final_clean_quality.json`
- Validated runtime binary SHA256:
  `91bd8ec125459411da57d5f6d111e6760573a717d3c8ab0f2161752dc6cdb084`
- Independent clean-build binary SHA256:
  `91bd8ec125459411da57d5f6d111e6760573a717d3c8ab0f2161752dc6cdb084`
- Vendored patch SHA256:
  `7fc34a1fa9d25d7f1c6c1b77382717c4b0f9aba252b486eeb346f3f8cbe4826b`

## Closed Paths

- N64 combined-softmax Split-D is rejected: fast standalone, but official
  sampled tokens diverge because its FP16 online-softmax order differs.
- D256 ports that keep all output columns in one warp/CTA exceed practical
  Volta instruction-cost limits. A full-head, single-barrier prototype
  reduced barrier stall to 0.75%, long-scoreboard to 2.52%, and shared-load
  conflicts to three, but increased LSU instructions by 37% and integer
  instructions by 80%; its 4.707 ms 8K result loses to address reuse.
- Split-D warp-pair register-P exchange is rejected. Three bitwise variants
  ranged from 5.495 ms to 10.664 ms at 8K. The best eliminated all P shared
  conflicts without changing Tensor instructions, but raised LSU instructions
  by 16.5% and integer instructions by 26.6%; shuffle/mailbox cost exceeded
  the conflict reduction.
- A KV-major cache allocation did not improve the full model and expanded the
  cache-management blast radius, so it was removed.
- Chunk size 8192 was 1.50% slower end to end than 4096 in the earlier A/B.
- Further attention-only work has a small current-model ceiling. Future
  prefill work should first target AWQ GEMM and elementwise overhead.
