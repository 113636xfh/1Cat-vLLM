# SM70 DeepSeek V4 PP2 x TP4 No-Spec Decode

## Objective

Reach at least 100 steady decode tokens/s, or at most 10 ms TPOT, for
`DeepSeek-V4-Flash-0731` on eight V100-SXM2-32GB GPUs with pipeline parallel
size 2, tensor parallel size 4, and no speculative decoding. The result must
retain the model's output quality and the existing SM70 numerical contract.

This campaign does not use DSpark to reach the first milestone. It also does
not replace TurboMind, weaken FP8 KV behavior, change sampling, or report a
short route-hit as a speed result.

## Source And Ownership

- Integration line: `onecat/main`
- Base SHA: `6fc9fe949264842bbb530fc19763231041527d3f`
- Branch:
  `agent/v100-dsv4-pp2tp4-nospec-100tps-20260824-165007`
- Worktree:
  `worktrees/v100-dsv4-pp2tp4-nospec-100tps-20260824-165007`

The earlier PP topology work and PR #260 established the `22+21` layer
partition and preserved the four mHC streams across the pipeline boundary.
This branch owns the no-spec decode performance follow-up. It does not reuse
the older branch's unmerged DSpark-only change.

## Hardware Topology

The eight GPUs form two natural four-GPU NUMA-local TP groups. PP rank 0 must
own physical GPUs 0-3 and PP rank 1 must own physical GPUs 4-7. Each TP4 group
must use its stage-local custom all-reduce route; the TP8 hierarchical route
must be disabled. The PP boundary crosses NUMA nodes, but the paired edges do
not fall back to `SYS`: the measured topology is GPU 0-to-4 and 1-to-5 over
`NV2`, and GPU 2-to-6 and 3-to-7 over `NV1`.

Runtime evidence must prove the actual rank-to-device mapping, TP communicator
membership, custom all-reduce dispatch, PP send/receive path, CUDA Graph route,
TurboMind FP8/MXFP4 routes, sparse MLA route, and FP8 KV route. Requested flags
alone are not route evidence.

## Initial Benchmark Contract

- Model: `/data/models/DeepSeek-V4-Flash-0731`
- GPUs: physical 0-7, ordered by PCI bus ID
- Parallelism: PP2 x TP4, default `22+21` layer partition
- Input/output: exactly 1,024 prompt tokens and 256 generated tokens
- Sampling: model-default temperature 1.0 and top-p 1.0, natural EOS
- Activations: FP16
- KV cache: FP8 E5M2, block size 256
- Maximum model length and batched-token limit: 4,096
- Maximum sequences: 1
- Prefix caching: disabled
- Speculative decoding and DSpark: absent
- CUDA Graph: enabled; eager execution disabled

The accepted speed number is the unprofiled steady decode rate over the 255
post-prefill token intervals. Report TTFT and prefill separately. A focused
Nsight run may use a shorter output only for composition and must not replace
the unprofiled absolute result.

The historical TP8 no-spec result at source `c4c9b840fe` was 66.014 tokens/s
with 15.148 ms mean TPOT. It is a directional reference, not a matched PP2 x
TP4 baseline and not an acceptance comparator for this branch.

The same source also reproduced the remembered DSpark 100+ endpoint at
`118.481/115.975/116.288 token/s`. That contract is TP8, probabilistic DSpark7,
120 prompt tokens, and 3,500 completion tokens; it emits about 5.1 tokens per
roughly 44 ms verification round. This proves that the 100+ result was real,
but it is neither no-speculation nor the same context/output/topology contract.
The current milestone still requires a true `<=10 ms` no-spec token path.

## Gates

1. Startup and route smoke: complete one natural-EOS request with healthy text
   and retain final worker logs proving the PP2 x TP4 routes.
2. Unprofiled baseline: three same-contract requests after warmup; report mean,
   p50, p90, and p99 TPOT and run-to-run variation.
3. Synchronized timing: split both PP stages, stage-local TP4 collectives, the
   PP boundary, LM head/sample, and host residual on the critical request.
4. Nsight Systems: capture graph nodes for a short steady decode and report
   critical-rank service separately from TP GPU sums and overlap.
5. Numerical and graph stability: exact-shape collective microbenchmarks and
   repeated CUDA Graph requests must complete without rank skew hangs.
6. Quality: paired no-regression checks use the existing V4 chat smoke first,
   then the pinned HumanEval, GSM8K, and LongBench subset before promotion.
7. Final performance: at least 100 steady decode tokens/s with no profiler and
   the same model, topology, sampling, lengths, route flags, and cache policy.

## Experiment Policy

Use only idle GPUs and record task ownership before launch. Do not terminate an
unrelated process. Queue through the event-based GPU gate rather than frequent
polling, and release every task-owned process and GPU immediately after each
run. Every failed experiment must either reject a concrete implementation
choice or be skipped.

## Current Status

The first current-main PP2 x TP4 no-spec baseline is complete. Three unprofiled
same-contract runs produced the following pure-decode results:

| run | tokens/s | mean TPOT | p50 | p90 | p99 | TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59.232 | 16.883 ms | 16.867 ms | 17.167 ms | 17.540 ms | 5.749 s |
| 2 | 59.256 | 16.876 ms | 16.852 ms | 16.959 ms | 17.330 ms | 2.794 s |
| 3 | 59.248 | 16.878 ms | 16.863 ms | 16.990 ms | 17.229 ms | 2.801 s |

The median is 59.248 tokens/s, so the 100 tokens/s target is not met. All three
runs generated the same output-token SHA-256 and the route smoke produced
coherent text. This is not a substitute for the pinned quality suites.

The topology and route audit established the following:

- ranks 0-3 form PP stage 0 and its TP4 group on GPUs 0-3; ranks 4-7 form PP
  stage 1 and its TP4 group on GPUs 4-7;
- every rank-aligned PP pair has a direct NVLink edge (`NV2`, `NV2`, `NV1`,
  `NV1` respectively), so the measured endpoint is not using a CPU-routed PP
  data path;
- both TP4 groups select the SM70 custom-all-reduce and PyNCCL backends;
- standard TP with expert parallelism disabled replicates all 256 experts and
  shards the expert intermediate dimension, giving local intermediate widths
  512 on TP4 and 256 on TP8;
- the retained compact MXFP4 path admitted only the TP8 shapes. TP4 therefore
  fell back to twelve per-expert GEMMs per MoE layer until this branch added the
  `(K=4096,N=1024)` and `(K=512,N=4096)` shapes;
- a PP stage returns the four mHC streams as one FP16 tensor with shape
  `[tokens,4,4096]`. At B1 the payload is 32 KiB per TP rank;
- a single autoregressive request cannot overlap the two PP stages without
  predicting a future token. Concurrent aggregate throughput must therefore be
  reported separately and cannot replace the single-request acceptance number.

The historical `TP AllReduce 4.192 ms` row is not one anomalously slow
collective and is not a PP2 x TP4 measurement. It is the TP8 rank-average sum
of 87 NCCL Ring-LL kernels per token: 48.194 us per call. That trace measured
63.740 us from all-ranks-input-ready to collective completion and showed that
rank/graph coordination, not the 8 KiB payload alone, dominated the critical
path. The current route uses two independent stage-local TP4 custom
all-reduces; its cost must come from the queued PP2 trace and must not inherit
the TP8 4.192 ms category value.

## Focused Measurements

The exact-shape TP4 MXFP4 microbenchmark was bitwise equal to the retained slow
path and passed repeated CUDA Graph replay:

| stage | slow per-expert | compact grouped | speedup |
|---|---:|---:|---:|
| W13 `(K=4096,N=1024)` | 0.1511 ms | 0.0406 ms | 3.7x |
| W2 `(K=512,N=4096)` | 0.0706 ms | 0.0242 ms | 2.9x |

Across 43 layers this removes approximately 6.7 ms/token relative to the
previous TP4 fallback. The full-model baseline above already includes this
repair.

The next exact MXFP4 candidate removes the remaining B1/top-6 activation
replication before W13. A first explicit-row-index implementation was rejected:
it changed TurboMind from the blocked loader to the indexed loader and differed
from the retained path in 6 of 6,144 FP16 values, with maximum absolute error
`1.5259e-5`. The replacement writes six 16-byte `StridedPtr` rows in the
existing prepare kernel, all pointing at the same physical input row. It keeps
the blocked loader and is bitwise equal through W13 and the complete
W13/SwiGLU/W2 pipeline. On one V100, 1,010 CUDA Graph replays measured the old
direct-top-6 pipeline at `0.116904 ms` and the broadcast-pointer pipeline at
`0.063707 ms`, a `0.053197 ms` (`45.5%`) per-layer reduction. The route remains
default-off until the same-contract full-model endpoint and output-hash gate
complete. Evidence is under
`/data/models/v100-dsv4-0731-pp2tp4-mxfp4-broadcast-ptrs-micro-20260825-r1/`;
the rejected indexed attempt is retained separately under
`/data/models/v100-dsv4-0731-pp2tp4-mxfp4-broadcast-micro-20260825-r1/`.

The retained mHC route is not presently a suspected quality regression. Its
FP32 staging implementation was bitwise equal to the prior fused decode path
across all 85 real checkpoint calls, and its matched official-sampling text
health gate passed. `VLLM_SM70_DSV4_MHC_FP32_STAGE=1` stays fixed on both sides
of this campaign; disabling it would change the numerical and performance
contract rather than isolate the PP2 x TP4 work.

The first synthetic TP4 B1 FP8 screen projected 6.654 ms/token across 279
calls, but a direct checkpoint-and-dispatch audit invalidated that projection.
`wo_a` is two local grouped-BMM calls per layer with shape
`(K=4096,N=1024)`, not one `(K=4096,N=2048)` call. The C4 indexer `wq_b` is
bypassed below the 2,048-token `index_topk` boundary; above that boundary its
shape is `(K=1024,N=8192)`, not `(K=1024,N=16384)`. The corrected seven-tensor
screen used separate real checkpoint weights for main and indexer `wq_b` even
though their TP-local shapes match. It covers 301 actual short-context GEMM
launches per token; crossing the indexer boundary adds 21 independently
screened calls. All selected winners use the existing fast-decoder production
dispatch, match the unrestricted winners, and pass repeated CUDA Graph replay.
Do not use the earlier 6.654 ms projection for an implementation or endpoint
claim.

| real operator | calls/token | TurboMind warm | QPN8 warm | split-K / nacc | projected saving |
|---|---:|---:|---:|---:|---:|
| `fused_wqa_wkv`, `4096x1536` | 43 | 61.932 us | 11.769 us | 32 / 2 | 2.157 ms |
| main `wq_b`, `1024x8192` | 43 | 16.855 us | 13.353 us | 8 / 2 | 0.151 ms |
| `wo_b`, `2048x4096` | 43 | 18.954 us | 13.182 us | 16 / 2 | 0.248 ms |
| two `wo_a` groups, `4096x1024` | 86 | 17.176 us | 10.056 us | 32 / 2 | 0.612 ms |
| fused shared gate/up, `4096x1024` | 43 | 17.705 us | 15.346 us | 16 / 2 | 0.101 ms |
| shared down, `512x4096` | 43 | 9.011 us | 5.072 us | 16 / 2 | 0.169 ms |

For the short route, the measured-kernel projection changes from 6.829 to
3.390 ms/token warm and from 7.722 to 4.455 ms/token cold, saving 3.439 and
3.267 ms/token respectively. The largest numerical difference from TurboMind
is the fused shared gate/up result: relative L2 `6.05e-4`, cosine
`0.9999997`, and maximum absolute difference `0.00390625`; it is closer to the
FP32 reference than the current TurboMind result in that sample. This is a
kernel screen, not a model-quality result. The machine-readable artifact is
`/data/models/dsv4-pp2tp4-nospec-build-staging-20260825-r1/tp4_fp8_qpn8_dense_microbench.json`.

The default-off production candidate deliberately excludes the replicated C4
indexer. During long prefill the main `wq_b` runs on the default stream while
the indexer `wq_b` can run concurrently on an auxiliary stream; directing both
large-M fallbacks through the same bounded 16 MiB dense workspace would race.
This leaves only 0.074 ms/token warm and 0.077 ms/token cold of additional
long-context savings on the table while retaining the full short-route
projection. A one-V100 source-integration gate then passed on real
`fused_wqa_wkv`, fused shared gate/up, and grouped `wo_a` weights: M=1 CUDA
Graph replay was stable, M=9 exercised the dense prefill fallback, all three
numerical gates passed, and exactly one 16 MiB workspace was allocated. Its
result is
`/data/models/dsv4-pp2tp4-qpn8-source-gate-20260825-r1/results/source_gate.json`.
The task released GPU 4 with no remaining compute application. Full-model
endpoint and dataset gates remain mandatory.

The first full-model QPN8 endpoint is now complete at clean source
`6e0f52aa7c`. Three same-contract runs measured `64.380`, `64.350`, and
`64.359` token/s, for a `64.359 token/s` median and `15.538 ms` median mean
TPOT. Relative to the `59.248 token/s`, `16.878 ms` baseline, this is an
`8.63%` throughput improvement and only about `1.340 ms/token` of endpoint
saving. The real endpoint therefore does not inherit the additive
`3.439 ms/token` kernel-screen projection; overlap and critical-path movement
absorb most of that service reduction. All three candidate runs have the same
token hash and the chat health check is coherent, but the candidate hash
`57da31a6...` differs from the baseline hash `86aa0403...`. This is a
performance candidate, not accepted output-quality evidence; the pinned paired
datasets remain mandatory. Evidence is under
`/data/models/v100-dsv4-0731-pp2tp4-qpn8-fullmodel-20260825-r3/`.

The first deterministic paired GSM8K-64 gate rejected the all-shape QPN8
candidate. Baseline and QPN8 both scored `63/64` with zero invalid answers and
identical dataset hashes and evaluation contracts, but only 62 normalized
predictions matched. On source row 12 the baseline correctly concluded that
`x > 12` requires 13 years, while QPN8 answered 12; row 54 moved in the other
direction from 42 to the correct 40. Equal aggregate accuracy therefore hid a
real baseline-correct-to-candidate-wrong regression. QPN8 remains default-off,
and its post-change Nsight/promotion gate is paused while projection suffixes
are bisected. The comparison is
`/data/models/v100-dsv4-0731-pp2tp4-quality-qpn8-gsm8k64-20260825-r1/results/paired_gsm8k_comparison.json`.

The first two suffix bisections also failed the targeted row-12 quality gate.
Enabling only `fused_wqa_wkv` measured `59.251 token/s` and `16.877 ms` mean
TPOT, indistinguishable from baseline, but changed the answer from the correct
13 to an arithmetically inconsistent 10. Enabling the other five projections
while retaining TurboMind for `fused_wqa_wkv` measured `62.029 token/s` and
`16.121 ms`, a real `4.69%` speedup over baseline, but produced the same wrong
answer 10. The latter result also shows that the full candidate's `64.359`
token/s depends on interaction between WQA and the remaining projections: WQA
alone is not an endpoint win, while removing it gives up roughly half of the
full candidate's TPOT saving. Both tasks exited normally and verified that no
owned GPU process remained. Neither subset is quality-acceptable; the next
bisection isolates `wq_b + wo_a + wo_b` from shared-expert gate/up and down.
Evidence is under
`/data/models/v100-dsv4-0731-pp2tp4-qpn8-bisect-wqa-gsm8k13-20260825-r2/`
and
`/data/models/v100-dsv4-0731-pp2tp4-qpn8-bisect-no-wqa-gsm8k13-20260825-r1/`.

The remaining bisections close the QPN8 route for this quality contract.
`wq_b + wo_a + wo_b` measured `62.203 token/s` but scored only `11/13` on the
targeted set, regressing rows 5 and 12. The narrower `wo_a + wo_b` subset
measured `61.805 token/s` and passed the targeted `13/13`, but its required
paired GSM8K-64 run scored `62/64`: it changed one normalized prediction and
regressed source row 37 from the baseline-correct answer 2 to 5. The successful
run is
`/data/models/v100-dsv4-0731-pp2tp4-qpn8-woab-quality-gsm8k64-20260825-r2/`;
an earlier startup collision with an unrelated TP4 owner is excluded from all
evidence. Every tested QPN8 subset therefore has a baseline-correct-to-candidate-
wrong regression. QPN8 remains an approximate, default-off experiment and does
not advance to Nsight, HumanEval, LongBench, or promotion unless a bitwise-exact
kernel is developed.

The shared-workspace concurrency audit follows the actual model call graph.
During the first attention fan-out, `fused_wqa_wkv` is the only QPN8 operator;
the three auxiliary-stream projections are FP16. Those streams join before
main `wq_b`. During the second fan-out the only block-FP8 peer is the excluded
TP1 indexer `wq_b`; compressor work is FP16. Grouped `wo_a` and `wo_b` then run
serially, while shared gate/up and down share one auxiliary stream and join
before the layer output is consumed. The runtime gate additionally rejects
DBO and explicit ubatching, so a second microbatch cannot concurrently reuse
the workspace. This serialization proof is part of the production contract;
enabling any excluded concurrency mode must retain TurboMind or allocate
separate workspaces.

A direct safetensors audit covered all 365 relevant dense scale tensors: eight
main-path sources per layer across 43 layers plus the 21 C4 indexer `wq_b`
tensors used beyond the short-context boundary. Every value is finite,
positive, and an exact power of two in the exponent range `[-13,-6]`; the
QPN8 scale transform
`FP16(scale * 256) / 256` has zero mismatches. QPN8 also retains every original
FP8 weight byte. This removes extra weight/scale requantization as a quality
risk, but it does not prove model equivalence because GEMM accumulation and
epilogue rounding still differ. The full paired quality gates therefore remain
mandatory. The machine-readable result is
`/data/models/dsv4-pp2tp4-nospec-build-staging-20260825-r1/dsv4_qpn8_scale_audit.json`.

The current generic PP transfer protocol sends one quarter of the replicated
hidden tensor across each PP pair, reconstructs it with a receiver-side TP4
all-gather, exchanges pickled Gloo metadata every step, allocates a temporary
receive tensor, and copies into the model runner's persistent graph input. An
eight-rank B1 protocol microbenchmark measured these rank-max medians:

| protocol | rank-max median |
|---|---:|
| current-like partial send + TP4 all-gather + metadata | 0.573 ms |
| full pairwise tensor + metadata, no all-gather | 0.408 ms |
| static full pairwise tensor, no metadata or all-gather | 0.116 ms |

The protocol microbenchmark projected approximately 0.46 ms/token, but the
combined full-model endpoint at clean source `b4bab1e452` measured `64.502`,
`64.479`, and `64.503` token/s. Its `64.502 token/s` median and `15.503 ms`
median mean TPOT improve on QPN8 alone by only `0.22%`, or about
`0.034 ms/token`. All three runs retained the QPN8 token hash
`57da31a6...`, the static route logged on every rank, and the supervisor exited
without an owned GPU process. The micro projection therefore did not survive
the PP dependency overlap in the endpoint. Static PP remains default-off and
is rejected as a material direction for this single-request target.
The earlier `12.98 ms/token`, `77 token/s` arithmetic from adding the static-PP
and QPN8 micro projections is rejected as an endpoint estimate: QPN8 alone
saved only `1.340 ms/token` in the full model and static PP then recovered only
`0.034 ms/token`. The combined evidence is under
`/data/models/v100-dsv4-0731-pp2tp4-qpn8-static-fullmodel-20260825-r1/`.

## External Evidence Triage

The long-context direction is the exact split-KV algorithm described by
[Flash-Decoding](https://princeton-nlp.github.io/flash-decoding/): split the KV
sequence, compute per-split attention and log-sum-exp, then reduce the partials.
This directly addresses B1 decode under-utilization without approximating the
attention result. The current DeepSeek V4 SM70 route already implements this
structure for SWA, C4, and C128 sparse MLA, including a separate QK-D split.
The next decision therefore requires measured context sweeps and trace-guided
split sizing rather than adding a second generic attention package.

The official
[FlashAttention CUDA support matrix](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support)
limits FlashAttention-2 to Ampere, Ada, and Hopper. It is not a drop-in Volta
backend. The algorithm remains relevant, but the V100 implementation must stay
in the owned SM70 kernels and must retain exact softmax reduction checks.

[Sarathi-Serve](https://arxiv.org/abs/2403.02310) and
[DeepSpeed-FastGen](https://arxiv.org/abs/2401.08671) reduce pipeline bubbles
through multi-request batching and prompt/decode composition. Their published
gains are serving-capacity or latency-throughput results, so they do not remove
the fundamental PP2 bubble for this one-request, no-prediction acceptance
contract. They may be evaluated later as a separately labelled aggregate
throughput route, never as evidence for 100 token/s single-request decode.

NCCL's
[environment guidance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
warns that forcing more CTAs can hurt small messages and that LL128 must not be
enabled on unsupported platforms. NVLink SHARP also requires Hopper-era
NVSwitch hardware. Consequently the 87 per-token 8 KiB TP collectives remain a
trace-first custom-all-reduce problem on V100; blind `NCCL_PROTO`, CTA, or NVLS
environment changes are excluded from accepted experiments.

The pinned GSM8K gate is the first 64 test rows with the first five training
rows as demonstrations, temperature 0, top-p 1, seed 42, and at most 256
completion tokens, issued strictly sequentially. The durable train and test
SHA-256 values are respectively
`17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`
and `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
`benchmarks/benchmark_dsv4_gsm8k_api.py` records the full request contract and
normalizes the final signed integer without treating a fractional answer as an
integer. `benchmarks/compare_dsv4_quality_results.py` requires matching dataset
hashes, few-shot and sampling contracts, and rejects every
reference-correct-to-candidate-wrong transition in addition to aggregate loss.
The HumanEval/LongBench gate likewise records and compares its fixed sampling,
selection, truncation, endpoint, and HumanEval sandbox contract; matching file
hashes alone are insufficient for a paired quality claim.

Two attempted Nsight captures are excluded from performance evidence. The
first collided with a different task that had begun CPU-side startup before
creating CUDA contexts. The GPU gate now detects pending launchers and claims
the GPUs before stabilization. The second compiled TileLang kernels under
Nsight child-process instrumentation, hit an NVCC temporary-output failure on
one rank, and left peers waiting in a collective. The follow-up reuses the
exact successful baseline caches and includes marker-based cleanup for child
processes that create their own sessions.

The bounded follow-up baseline capture completed. Nsight 2022.4 failed only
during automatic report import and the benchmark's optional request metrics
were absent; the retained 36.6 MB QDSTRM was recovered with the installed
`QdstrmImporter`, exported to SQLite, and parsed without another GPU run. The
stable window contains 30 replay steps per rank and drops two edges on each
side, leaving 26 composition steps. Nsight raises the mean/p50 replay interval
to `18.939/18.696 ms`, so those wall values are diagnostic and do not replace
the unprofiled `16.878 ms` endpoint.

The corrected stage-aware trace separates dependency residence from transfer
cost. The PP1 receive kernel resides for `9.505 ms` while it waits for PP0;
the reciprocal sampled-token broadcast resides for `9.024 ms` on PP0 while it
waits for PP1. Their source-side transfer launches are only about
`0.010-0.014 ms`; the 9 ms rows are pipeline dependencies, not slow NVLink
copies. True TP4 custom-all-reduce rank-max service is `0.781 ms` on PP0 and
`0.695 ms` on PP1, with about 43 launches per rank/token and a
`15.776 us` mean custom-reduce kernel. The largest non-wait service categories
summed across stage rank maxima are TurboMind FP8 dense GEMM `6.434 ms`, MXFP4
MoE `2.647 ms`, FP16 GEMV/GEMM `2.582 ms`, mHC `1.714 ms`, sparse MLA
`1.695 ms`, routing/activation `1.694 ms`, Q/KV preparation `1.688 ms`, and
TP all-reduce `1.476 ms`. These service sums overlap and are not an additive
wall-time decomposition. The report is
`/data/models/v100-dsv4-0731-pp2tp4-nospec-100tps-nsys-20260825-r3/results/pp2tp4_decode_breakdown.{json,md}`.
