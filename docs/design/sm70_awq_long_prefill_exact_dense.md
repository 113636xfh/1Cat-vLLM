# SM70 AWQ Long-Prefill Exact-Dense Path

## Scope

This ledger covers the Qwen3.6-27B-AWQ TP4 `M=4096` prefill projection path
on 32 GB V100. Decode, partial prefill chunks, TP2, 16 GB V100, and unknown
AWQ shapes stay on TurboMind AWQ.

The path is enabled by default through
`VLLM_SM70_AWQ_PREFILL_EXACT_DENSE=1`. Set it to `0` to preserve memory.

## Reason For The Path

NCU on the dominant `M4096,N8704,K5120` gate/up kernel showed:

- 248 registers/thread and 65.55 KB shared memory;
- one CTA/SM and 12.5% achieved occupancy;
- 62.1% of scheduler cycles with no eligible warp;
- 9.36% DRAM throughput, so HBM was not the limit;
- dependency, barrier, math, and shared-load stalls around online AWQ unpack.

Existing registry autotuning selected the same `CTA128x256x16` kernel. Shape
tuning was therefore closed. The structural replacement keeps compact AWQ
weights for decode but materializes selected FP16 weights once for long
prefill, allowing cuBLAS to run the compute-dense `M=4096` projection.

## Numerical Contract

TurboMind forms each dequantized weight as:

```text
bias = fp16(-zero * scale)
weight = fp16_fma(q, scale, bias)
```

Using `(q - zero) * scale` is not equivalent and produced up to `0.008789`
output error in the first probe. The production helper preserves the FP16
bias rounding and single-FMA order. All accepted `M=4096` projection outputs
were bitwise equal to TurboMind AWQ.

The route is limited to AWQ group size 128 and the following validated TP4
`(K,N)` shapes:

| projection | K | N | layers | AWQ | exact dense | saved/chunk | resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP gate/up | 5120 | 8704 | 63 | 7.089 ms | 4.781 ms | 145.40 ms | 5.23 GiB |
| MLP down | 4352 | 5120 | 63 | 3.371 ms | 2.296 ms | 67.76 ms | 2.61 GiB |
| linear-attention QKVZ | 5120 | 4096 | 47 | 3.376 ms | 2.377 ms | 46.96 ms | 1.84 GiB |
| linear-attention out | 1536 | 5120 | 47 | 1.279 ms | 0.899 ms | 17.86 ms | 0.69 GiB |
| full-attention out | 1536 | 5120 | 16 | 1.276 ms | 0.900 ms | 6.01 ms | 0.23 GiB |

Total modeled saving is `283.99 ms` per full chunk with `10.60 GiB` of dense
weights per rank.

## End-To-End Acceptance

The comparison includes the default gather-to-exact-dense and dense split-KV3
attention paths, CUDA graph decode, FlashQLA, and custom TP all-reduce.

| Qwen3.6-27B-AWQ TP4, input 64K/output 16 | prefill | TPOT | token result |
| --- | ---: | ---: | --- |
| latest attention baseline | 25.514792 s | 21.351 ms | reference |
| all selected exact-dense projections | 20.988843 s | 21.390 ms | identical |
| delta | -4.525949 s (-17.74%) | noise | identical IDs/text/hash |

Prefill throughput improves by 21.56%. Model residency rises from 6.50 GiB to
17.36 GiB per rank. The default is therefore additionally gated to devices
with at least 30 GiB total memory. The explicit 3 GiB KV configuration used
for acceptance completed without OOM.

## Rejected Or Deferred Variants

- Fused gate/up plus SiLU epilogue was exact but saved only 0.096 ms/layer,
  below 0.4% modeled end-to-end. Do not prioritize it over exact-dense.
- Ordinary `(q-zero)*scale` pre-expansion is numerically wrong for this route.
- Dense weights are not used for decode or partial chunks because those shapes
  lack a bitwise and performance gate.
- Extending the shape list requires per-rank bitwise operator evidence and a
  matching full-model token gate; suffix-only expansion is not accepted.

## Evidence

- `awq_prefill_exact_dense_gateup_tp4_allranks.json`
- `awq_prefill_exact_dense_all_shapes_tp4_allranks_exactness.json`
- `awq_prefill_exact_dense_all_shapes_m4096_tp4_rank0.json`
- `awq-prefill-dense-fullmodel/candidate_i8192.json`
- `awq-prefill-dense-fullmodel/candidate_i65536.json`
- `awq-prefill-dense-fullmodel/candidate_all_i65536.json`
- `ncu_awq_gateup_m4096_tp4.ncu-rep`

The artifacts are under
`/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/`.
