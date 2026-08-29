# SM70 DFlash2 NVFP4 Prefill Promotion

## Scope

Promote the existing bounded QPN2-packed NVFP4 prefill operator for the
quality-audited Qwen3.8-27B DFlash2 contract. The change does not alter kernel
arithmetic, the DFlash2 draft model, target verification, sampling, attention,
or prefix-cache semantics.

Integration base: `onecat/main` at
`62ad1e02693f4c857f3b7547cef1860ee54e8053`.

## Retained evidence

No target-only baseline is rerun in this campaign. Historical evidence is:

| Contract | 32K prompt tok/s | 64K prompt tok/s |
| --- | ---: | ---: |
| DFlash2 before the D256 closure | 3125.12 | 2588.53 |
| DFlash2 with the D256 sidecar | 3476.53 | 3103.02 |
| DFlash2 with D256 and QPN2-packed prefill | 3959.14 | 3450.29 |

The last row used the same NVFP4 target, official BF16 DFlash2 draft, TP4
V100, FP8 E5M2 target KV, FP16 draft KV, prefix cache, Mamba alignment, and
CUDA Graph decode contract. All retained requests were uncorrupted and kept
the same first-token hash. Raw artifacts are under
`/data/minimax-h3/task-cache/v100-dflash2-prefill-32k64k-20260827/`.

The often-quoted 5170.96 tok/s exact-8K and 2438.89 tok/s 256K results are
FP8 target-only contracts. They remain useful upper-bound references but are
not relabeled as NVFP4 DFlash2 measurements.

## Dispatch change

The QPN2 decode and bounded prefill implementations are already in main but
both environment gates default to off. They now resolve to on only when all of
the following hold:

- speculative method is `dflash` with seven draft tokens;
- PP1/TP4, `max_num_seqs=1`, and no DBO or microbatching;
- the existing exact Qwen3.8 dense NVFP4 layer-shape gate passes.

Explicit `VLLM_SM70_NVFP4_QPN2=0` and
`VLLM_SM70_NVFP4_QPN2_PREFILL=0` remain hard rollbacks. Other speculative
methods, target-only service, different TP, and concurrent service retain the
existing route.

## Promotion gates

- focused resolver, shape, zero-copy-layout, and dispatch tests;
- real-weight eager and graph numerical equality at production prefill widths;
- latest-main DFlash2 route hit plus cold 32K/64K throughput;
- unchanged short decode path, acceptance behavior, and structured-output
  health before default promotion.
