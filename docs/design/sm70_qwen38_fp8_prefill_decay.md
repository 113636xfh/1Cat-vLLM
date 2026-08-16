# SM70 Qwen3.8 FP8 Prefill Decay

Date: 2026-08-15

## Contract

- Source base: `87ac589295ba64399695ef2237c37cffc0d8b71b`.
- Model: Qwen3.8-27B-FP8 local checkpoint.
- Model weights: FP8 E4M3. This is independent from the KV-cache dtype.
- FP8 KV cache on SM70/Flash-V100: FP8 E5M2. The `fp8` CLI shorthand
  resolves to E5M2 on this route; explicit E4M3 remains explicit E4M3.
- Hardware: TP4 on V100-SXM2-16GB GPUs 0-3.
- Runtime: Python 3.12, Torch 2.10.0+cu128, `FLASH_ATTN_V100`, prefix
  caching with Mamba align, CUDA graphs, and no MTP. The chunk baseline and
  128K trace use FP16 KV; the later regression and fixed-route sections use
  the explicitly stated FP8 KV dtype.
- Sampling: temperature 1.0, top-p 0.95, top-k 20, seed 20260815.

## Chunk Baseline

Every request resets the prefix cache and emits 32 tokens. Output hashes are
stable and identical between the two chunk configurations.

| Input | Chunk 15680 | Chunk 4096 | Chunk 15680 gain |
|---:|---:|---:|---:|
| 4K | 4160.0 tok/s | 4146.1 tok/s | 0.33% |
| 16K | 4290.1 tok/s | 3841.2 tok/s | 11.69% |
| 64K | 3420.2 tok/s | 2778.9 tok/s | 23.07% |
| 128K | 2702.9 tok/s | 2020.5 tok/s | 33.77% |

Chunk 15680 is retained. The greater-than-4K peak is present on the current
machine; the unresolved issue is long-context decay, not a missing dispatch.

## 128K Critical-Rank Trace

The profiled prefill wall is 49.821 seconds versus 48.492 seconds unprofiled.
Critical-rank kernel attribution is:

| Category | Kernel time | Profiled wall share |
|---|---:|---:|
| D256 exact-dense attention | 17.710 s | 35.55% |
| D256 direct-paged attention | 3.099 s | 6.22% |
| FP8 exact-dense projections | 12.787 s | 25.67% |
| TurboMind FP8 projections | 5.551 s | 11.14% |
| TP communication | 4.895 s | 9.82% |
| GDN / linear attention | 1.640 s | 3.29% |
| Other FP16 GEMM | 1.547 s | 3.11% |
| Norm / elementwise / sampling | 1.240 s | 2.49% |
| KV cache and gather | 0.042 s | 0.08% |
| Host and unattributed residual | 1.309 s | 2.63% |

The 128 exact-dense calls are eight full chunks times 16 full-attention
layers. Mean per-layer latency grows from 18.50 ms in the first group to
256.59 ms in the eighth group.

## Exact-Shape NCU

The isolated shape is `Q=15680, KV=125440, Hq=6, Hkv=1, D=256`. The
unprofiled accepted operator is 242.954 ms median, 46.63 causal TFLOP/s, and
has output SHA256
`71653010ebcad1ff38a231ad92cc970fe47491f14d570f29b81f963c3b794862`.

- 253-254 registers/thread, 45.57 KiB dynamic shared memory, one CTA/SM,
  12.5% occupancy, and 18.38 waves.
- Tensor pipe active: 38.67%; issue active: 36.75%.
- Schedulers have no eligible warp for 62.14% of cycles.
- DRAM throughput is only 9.06% (81.65 GB/s); L2 hit rate is 88.51%.
- MIO throttle is the largest per-issue stall. Eight P-operand `LDS.128`
  instructions each account for about 172.9 million excessive shared
  wavefronts.
- The dominant long-scoreboard PC is the next-K shared publication waiting
  for its global register load. Earlier publication is already a closed
  regression because it extends the fragment across the register-dense loop.

This is a CTA-local shared-memory and dependency problem, not an HBM or grid
parallelism limit.

## Rejected Pre-Fix FP8-Alias Run

The same TP4, no-MTP contract was rerun with `kv_cache_dtype=fp8` and
`max_num_batched_tokens=8192`. At that point, the generic `fp8` alias was
incorrectly left as E4M3. These numbers describe the bug and are not an E5M2
baseline.

| Input | Prefill | FP16 KV/chunk 15680 | Relative | Decode |
|---:|---:|---:|---:|---:|
| 4K | 3529.8 tok/s | 4160.0 tok/s | -15.1% | 53.57 tok/s |
| 16K | 2479.0 tok/s | 4290.1 tok/s | -42.2% | 52.27 tok/s |
| 64K | 859.3 tok/s | 3420.2 tok/s | -74.9% | 32.55 tok/s |
| 128K | 460.3 tok/s | 2702.9 tok/s | -83.0% | 22.91 tok/s |

The configuration provides 685,491 KV tokens and 2.61x nominal 256K request
capacity, versus 294,431 tokens and 1.12x for FP16 KV/chunk 15680. The severe
regression came from selecting the wrong KV format and therefore missing every
SM70 E5M2 fast path:

- E4M3 prefix chunks cannot enter the D256 exact-dense path because that path
  currently requires FP16 K/V cache tensors.
- The existing one-pass FP8-to-FP16 bridge supports E5M2 only. E4M3 therefore
  uses direct paged prefill with software dequantization in the attention loop.
- Decode selects the scalar-paged E4M3 route instead of XQA, causing additional
  long-context decay.
- Model E4M3 weights do not imply an E4M3 KV cache. Weight quantization must
  never be used as a KV dispatch condition.

The fix resolves the SM70/Flash-V100 `fp8` alias to E5M2 while preserving an
explicit `fp8_e4m3` request. E5M2 then enters the exact-dense prefill bridge,
native cache writer, and scalar/XQA decode routes. Current E5M2 decode has a
lower long-context slope than FP16 and is faster at 128K and 256K; the final
matched prefill/decode table is recorded after the full acceptance sweep.

## Rejected Candidate

The first candidate reused the conflict-free native PV matrix-A layout but
kept direct logical stores from the QK accumulator, avoiding the previously
rejected shuffle/repack function.

- PTXAS: 253 registers/thread, zero stack, zero spill, unchanged shared size.
- Quality: exact output hash.
- Wall: 242.954 ms control to 254.450 ms candidate, a 4.73% regression.

The extra address-generation/scalar-publication issue cost exceeds the saved
P-load replay. Do not retry this layout spelling or the earlier
shuffle/repack form.

## Artifacts

- Root: task-local `qwen38-fp8-prefill-decay-20260815` artifact directory.
- Nsight Systems:
  `profiles/qwen38-fp8-tp4-i128k-chunk15680-r2.nsys-rep`.
- Nsight Compute:
  `profiles/ncu-d256-exact-q15680-kv125440-baseline.ncu-rep`.
- Candidate binary:
  `experiments/p-scalar-native-build-r3/_vllm_fa2_C.abi3.so`.
- FP8 KV/chunk 8192 result:
  `results/fp8kv-chunk8192-tp4.json`.
