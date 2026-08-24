# SM70 DFlash2 long-context verification decay

## Scope

This task measures, attributes, and reduces the context-length growth of one
MRV2 DFlash2 verification round on V100. It is stacked on private DFlash2
verifier head `e8daa778f55cf16df311f8ce8dd032e2281f1680`. It does not replace or
retune DFlash1, `dflash_ddtree`, Eagle, or MTP.

Quality promotion remains score based. Greedy token identity is diagnostic;
the production gates are probabilistic task scores, PPL, acceptance length,
and output health against the same target-only checkpoint.

## Frozen workload

- Target: local Qwen3.8-27B mixed NVFP4/FP8 checkpoint, TP4.
- Draft: `incoai/Qwen3.8-27B-DFlash2`, FP16 draft KV.
- Target KV: `fp8_e5m2`.
- Attention: `FLASH_ATTN_V100` target and draft routes.
- Execution: FULL CUDA Graph, sequential batch one.
- Sampling: temperature 1.0, top-p 0.95, target top-k 20, probabilistic
  proposals, fixed seed.
- Selector top-K: checkpoint contract 16; speculative tokens: 7.
- Context sweep: 1K, 32K, 128K, and 256K boundary.

All comparisons must keep model, TP/GPU set, graph mode, KV dtypes, sampling,
and route gates identical. Prefill/TTFT are reported separately from steady
decode and verification-round wall time.

## Existing evidence to reproduce

The parent task recorded the following unprofiled paired measurements:

| Context | Prior complete round | Grouped complete round | Prior target phase | Grouped target phase |
| --- | ---: | ---: | ---: | ---: |
| 32K | 36.311 ms | 26.361 ms | 30.248 ms | 20.452 ms |
| 128K | 69.855 ms | 40.778 ms | 63.708 ms | 34.647 ms |

From 32K to 128K, the grouped complete round grows by 14.417 ms and its target
phase grows by 14.195 ms. This strongly localizes the observed growth to target
verification, but it does not yet distinguish paged attention service,
dependency gaps, launch geometry, communication placement, or trace overhead.

The retained grouped-attention microbenchmark reported 0.0686/0.3564/1.2145/
2.3962 ms at 1K/32K/128K/256K. These values must be re-run and reconciled with
the full-model critical-path wall before using them as a closed decomposition.

## Closed trace attribution

The retained 32K and 128K Nsight Systems traces use the same grouped one-pass
kernel and target CUDA Graph route. Summing the partial and combine kernels,
then dividing by the number of target graph replays across all four TP ranks,
gives the following service time per target verification graph:

| Context | Partial average | Partial calls | Combine average | Attention per target graph |
| --- | ---: | ---: | ---: | ---: |
| 32K | 0.313606 ms | 1,344 | 0.008282 ms | 5.150214 ms |
| 128K | 1.198762 ms | 1,088 | 0.008295 ms | 19.312912 ms |

There are sixteen full-attention layers per target graph. The attention service
therefore grows by 14.162698 ms between 32K and 128K, while the measured target
phase grows by 14.270586 ms. Grouped attention explains 99.24% of target-phase
growth and 95.83% of complete-round growth in this pair. Draft, selector,
sampling, TP exchange, and host residuals are not material causes of this
context-length slope.

Both traces launch the partial kernel with grid `(2, 40, 1)`, block size 256,
128 registers per thread, and 39,424 bytes of static shared memory. The fixed
80-CTA grid maps one CTA to each V100 SM even as the KV length quadruples. The
kernel's `__launch_bounds__(256, 2)` and resource arithmetic permit two resident
CTAs per SM (65,536 registers and 78,848 shared-memory bytes), so the first
bounded experiment is 40 versus 80 context splits. This is a hypothesis about
exposed parallelism, not yet a performance claim: an exact-shape paired
microbenchmark must distinguish occupancy benefit from memory-bandwidth
saturation and extra split-reduction cost.

The experiment follows the same split-KV principle used by the official
[FlashInfer scheduler](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/scheduler.cuh),
which derives its maximum grid from CUDA occupancy, and by
[Flash-Decoding](https://princeton-nlp.github.io/flash-decoding/). Recent
[PersistentKV](https://arxiv.org/abs/2606.26666) results likewise motivate
adaptive sequence splitting and page-aware scheduling for long-context decode.
These sources guide the search space; only same-machine measurements can
promote a V100-specific default.

## Split-count microbenchmark

The first candidate doubles the maximum context splits from 40 to 80. With
two GQA head groups this changes the fixed partial launch from 80 to 160 CTAs,
matching the kernel's two-block-per-SM launch bound. No attention arithmetic,
mask, FP8 decode, or MRV2 scheduling code changes.

An A/B/A CUDA-Graph run on physical GPU 1 used the production Qwen3.8 target
shape (Q8/H6/KV1/D256, E5M2 KV, page size 3,296), 40 warmups and 200 measured
rounds per point:

| Context | Split 40 A | Split 80 | Split 40 A2 | Split 80 versus A/A2 |
| --- | ---: | ---: | ---: | ---: |
| 1K | 0.058368 ms | 0.067584 ms | 0.058368 ms | +15.79% / +15.79% |
| 32K | 0.319488 ms | 0.261120 ms | 0.322560 ms | -18.27% / -19.05% |
| 128K | 1.204736 ms | 0.955392 ms | 1.206272 ms | -20.70% / -20.80% |
| 256K | 2.377728 ms | 1.864704 ms | 2.378752 ms | -21.58% / -21.61% |

At sixteen full-attention layers, the operator delta projects to -0.934 ms at
32K, -3.989 ms at 128K, and -8.208 ms at 256K per verification graph. The 1K
operator regression projects to +0.147 ms, or about +0.74% of the retained
approximately 20 ms complete short-context round. This remains a projection
until the TP4 full-model A/B is complete.

The split-80 extension passes all six targeted grouped-verifier tests: five
random physical-page cases against the FP32 reference, and CUDA Graph replay
while mutating runtime sequence length. These are operator correctness gates;
acceptance length, PPL, and task scores remain required before promotion.

## Measurement order

1. Reproduce low-overhead full-model B1 baselines at 1K/32K/128K/256K.
2. Record synchronized target, draft, selector/sampling, communication, and
   host residual timing from the same requests.
3. Capture short Nsight Systems graph-node traces at 32K and 128K, aligned by
   replay ordinal across all four TP workers.
4. Classify target graph nodes into attention, quantized GEMM, GDN/norm,
   LM-head/selector, collectives, and dependency gaps.
5. Build the smallest exact-shape microbenchmark for the confirmed growing
   kernel family. Run Nsight Compute only after the runtime route is proved.

## Promotion gates

- Reduce geometric-mean 32K/128K complete verification-round wall without a
  regression greater than 1% at either length.
- Do not increase 1K complete-round wall by more than 1%.
- Preserve request-mean acceptance length within 0.05 and pooled acceptance
  within sampling noise.
- Preserve the existing task-score and WikiText PPL gates before enabling a
  numerical or scheduling change by default.
- Validate the 256K boundary for capacity, output health, and latency after a
  candidate passes at 128K.

## Current status

The 32K/128K trace attribution is closed: target grouped attention is the
long-context verification bottleneck. The split-40/80 operator A/B is complete
and split 80 is the current candidate. Full-model TP4 timing and quality gates
are pending; no end-to-end split-count speedup is claimed yet.
