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

The final SM70 cubin uses 253 registers/thread for dense and 252 for paged,
with no spill or local stack. Both remain at one CTA per SM. The gain comes
from useful D256 parallelism and software scheduling, not higher occupancy.

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
- Official 8K quality result:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/official_quality/splitd_n32_official_i8k_o128.json`
- Final N32 ABI default-on 1K quality result:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/official_quality/splitd_n32_abi_default_official_i1k_o64_logprobs.json`
- Official 8K comparison:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/official_quality/compare_baseline_splitd_n32_i8k_o128.json`
- Nsight Systems report:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/profiles/qwen36_27b_awq_tp4_i8k_splitd_both_exact_prefill.nsys-rep`
- Validated runtime binary SHA256:
  `e31d84095f700aa001f9fe4bb84110cf80ba4ab5142ec8cf4ea75ec8fd0fa5bc`
- Independent clean-build binary SHA256:
  `58df1e3fd7061d8547fe8e021a77090ac68e35f2ae38b5a715009cf0976b0a9b`
- Vendored patch SHA256:
  `39c755d2cab8bbbb9503bc73e05fcee3c0bbdadcd9d50e6fcee18d21b31789ed`

## Closed Paths

- N64 combined-softmax Split-D is rejected: fast standalone, but official
  sampled tokens diverge because its FP16 online-softmax order differs.
- D256 ports that keep all output columns in one warp/CTA exceed practical
  Volta register or shared-memory limits.
- A KV-major cache allocation did not improve the full model and expanded the
  cache-management blast radius, so it was removed.
- Chunk size 8192 was 1.50% slower end to end than 4096 in the earlier A/B.
- Further attention-only work has a small current-model ceiling. Future
  prefill work should first target AWQ GEMM and elementwise overhead.
