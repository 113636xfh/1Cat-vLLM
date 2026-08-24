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

## Initial status

Baseline reproduction and attribution are pending. No new performance claim
is made by this document.
