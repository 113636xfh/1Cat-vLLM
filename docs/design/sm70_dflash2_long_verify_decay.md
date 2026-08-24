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

## Full-model 32K and 128K confirmation

A TP4 FULL-Graph Nsight run used the same 128K prompt, 128 generated tokens,
probabilistic sampling seed, model/KV dtypes, and route gates as retained
split-40 run `v80`. The only implementation change was the split-80 Flash-V100
extension:

| Phase | Split 40 | Split 80 | Delta |
| --- | ---: | ---: | ---: |
| Complete verification round | 40.778072 ms | 36.683712 ms | -4.094360 ms (-10.04%) |
| Target | 34.646866 ms | 30.557332 ms | -4.089534 ms (-11.80%) |
| Draft | 4.604373 ms | 4.627743 ms | +0.023370 ms |
| Draft to target | 0.230538 ms | 0.225575 ms | -0.004963 ms |
| Target to draft | 1.296296 ms | 1.273062 ms | -0.023233 ms |

The partial kernel changed from grid `(2, 40, 1)` to `(2, 80, 1)`. Its mean
service time fell from 1.198762 ms to 0.944041 ms per full-attention layer
(-21.25%); combine rose from 0.008295 ms to 0.010430 ms. Across sixteen layers,
the net attention prediction is -4.041 ms, explaining 98.8% of the measured
target-phase improvement. The earlier operator projection was -3.989 ms, only
about 0.10 ms from the full target result.

The fixed diagnostic request retained identical accepted-token counts
`[17, 17, 17, 16, 16, 15, 15]`, mean acceptance length 7.647059, and all 128
output token IDs. This is trajectory evidence, not a substitute for the
required score/PPL gates.

The matching 32K TP4 FULL-Graph trace independently confirms the shorter
context point:

| Phase | Split 40 | Split 80 | Delta |
| --- | ---: | ---: | ---: |
| Complete verification round | 26.361447 ms | 25.238442 ms | -1.123005 ms (-4.26%) |
| Target | 20.451644 ms | 19.468326 ms | -0.983318 ms (-4.81%) |
| Draft | 4.365014 ms | 4.257568 ms | -0.107446 ms |
| Draft to target | 0.260419 ms | 0.235765 ms | -0.024654 ms |
| Target to draft | 1.284369 ms | 1.276783 ms | -0.007586 ms |

The split-80 partial kernel averages 0.248999 ms at 32K, versus 0.313606 ms
for split 40. The measured target reduction is within 0.05 ms of the operator
projection. Across the paired full-model endpoints, the 32K-to-128K complete
round growth falls from 14.416625 ms to 11.445270 ms (-20.61%), while target
growth falls from 14.195222 ms to 11.089005 ms (-21.88%). The remaining slope
is still serious: target accounts for 96.89% of the optimized complete-round
growth, so subsequent work remains scoped to grouped paged attention.

## Physical-page and tile-balanced follow-up

Two address-equivalent scheduling changes pass the same six operator tests:

- Specialize the actual FP8 kernel-page sizes (1,648 and 3,296) at compile
  time, replacing runtime integer division with constant arithmetic. The
  Qwen3.8 hybrid cache reports a 1,648-token logical attention block, while the
  grouped verifier receives the 3,296-element physical FP8 view; the final
  trace proves the `<page=3296>` specialization is selected.
- Divide the sequence by complete 32-token WMMA tiles, distributing the
  remainder across splits. This removes per-split padded tail tiles without
  changing visibility, loaded KV bytes, softmax, or WMMA arithmetic.

Repeated CUDA-Graph A/B/A runs show no 32K regression and a further
approximately 1.7% operator reduction at 128K versus plain split 80. The final
TP4 FULL-Graph endpoints are:

| Phase | Split 80 at 32K | Final at 32K | Split 80 at 128K | Final at 128K |
| --- | ---: | ---: | ---: | ---: |
| Complete verification round | 25.238442 ms | 25.237987 ms | 36.683712 ms | 36.482220 ms |
| Target | 19.468326 ms | 19.455442 ms | 30.557332 ms | 30.354906 ms |
| Draft | 4.257568 ms | 4.269400 ms | 4.627743 ms | 4.606652 ms |

At 128K, the partial kernel falls from 0.944041 ms to 0.926568 ms per
full-attention layer; sixteen layers predict -0.280 ms and the measured target
delta is -0.202 ms. At 32K, the partial kernel is 0.245912 ms and complete wall
is unchanged within 0.001 ms. The final 32K-to-128K complete-round slope is
11.244233 ms, 22.01% below split 40 and 1.76% below plain split 80. The final
target slope is 10.899464 ms, 23.22% below split 40. Target attention still
accounts for 96.93% of the residual complete-round slope.

A logical-work roofline proxy narrows the remaining operator bottleneck. The
two padded 32-row GQA groups perform about 2.15/8.59 GFLOP per layer at
32K/128K; the measured partial times correspond to only 8.73/9.27 logical
TFLOP/s. Their requested E5M2 K/V payload corresponds to about 136/145 GB/s
before accounting for duplicate-group L2 reuse. Both rates stay nearly flat as
context grows and are far below V100 peak figures. This is not an NCU hardware
counter claim, but it rules out treating the residual as unavoidable saturated
HBM traffic and prioritizes flat-QK/PV utilization, softmax synchronization,
and GQA packing over further page-address arithmetic.

The final 128K diagnostic request preserves the split-80 accepted-token counts
and 7.647059 mean acceptance length. At 32K, the single-request diagnostic
changes from 6.142857 to 6.095238 (-0.047619), inside the 0.05 diagnostic gate;
dataset-level acceptance and model-quality gates remain authoritative.

An attempted compact mapping for the two-head GQA tail passed the six operator
correctness tests but was 6.3-7.0% slower than split 80 across the micro sweep.
The run also observed a late external-worker ownership race, so its absolute
timings are not promotion evidence; the uniformly negative direction was
sufficient to revert the experiment completely. Nsight Compute hardware
counters are unavailable to the current user (`ERR_NVGPUCTRPERM`), so no NCU
counter claim is made.

A second exact-address experiment replaced sixteen redundant page divisions,
modulos, and block-table loads per D=256 row with one half-warp calculation and
broadcast. It preserved register count and passed all six operator correctness
tests, but regressed the one-pass kernel by 3.51%/2.36%/2.47%/2.63% at
1K/32K/128K/256K in a production-page-size A/B/A run. On this Volta path, the
independent address arithmetic is hidden better than the added shuffle
dependency. The candidate was reverted completely.

## Score and PPL promotion evidence

The final split-80/page-specialized/tile-balanced binary was evaluated with
probabilistic proposals, target sampling at temperature 1.0/top-p 0.95/top-k
20, fixed seed, a 2,048-token cap, and natural EOS. The mixed dataset supplies
32 GSM8K, 32 HumanEval, and 32 MATH500 cases. Its embedded MBPP prompts have an
incompatible function-name contract and are excluded; the independent
`mbpp-tests-32` dataset supplies the valid MBPP result.
The tested extension SHA256 is
`4a925265cfed934ebd7772be2a1eccaa64163d5324f87d31ba8ee99d10c3a044`.

| Suite | Target only | Prior grouped | Final candidate |
| --- | ---: | ---: | ---: |
| GSM8K | 29/32 | 30/32 | 30/32 |
| HumanEval | 28/32 | 28/32 | 27/32 |
| MATH500 | 26/32 | 25/32 | 26/32 |
| MBPP, independent tests | 25/32 | 25/32 | 26/32 |
| Valid total | 108/128 | 108/128 | 109/128 |

The suite-level movement is one HumanEval case down and a net two MATH/MBPP
cases up; the valid aggregate is therefore one case above both controls, not a
quality-regression claim. WikiText prompt-logprob/PPL output is exactly equal
to the prior grouped candidate across all eight samples (zero maximum and mean
difference). Against target only, the maximum PPL difference is 0.003952 and
the mean absolute prompt-logprob difference is 0.007519.

Acceptance also remains inside the frozen 0.05 gate. On the 128-request mixed
run, request-mean completion tokens per verification step change from 4.238430
to 4.218840 (-0.019589), and pooled acceptance length changes from 3.749104 to
3.714972 (-0.034132). On independent MBPP they change from 3.527945 to
3.509266 (-0.018679), while pooled acceptance changes from 3.421634 to
3.431016 (+0.009382). The final runs produce 101,312 and 28,070 output tokens,
respectively; 103/128 and 24/32 requests terminate naturally before the cap.

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
long-context verification bottleneck. The split-40/80 operator A/B and both
32K and 128K full-model traces confirm the predicted critical-path
improvement. Split 80 reduces, but does not eliminate, the long-context slope.
The physical-page specialization and tile-balanced partitioning also pass
operator and full-model performance gates, reducing the final slope by 22.01%
relative to split 40. PPL, score, and dataset-level acceptance gates pass. The
1K full-model boundary check remains pending before promotion. Two bounded
follow-ups (tail compaction and page-address broadcast) are recorded as
rejected paths and must not be resurrected without new evidence.
