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

The eight GPUs form two natural four-GPU NVLink/NUMA islands. PP rank 0 must
own physical GPUs 0-3 and PP rank 1 must own physical GPUs 4-7. Each TP4 group
must use its stage-local custom all-reduce route; the TP8 hierarchical route
must be disabled. The one PP boundary transfer crosses the two islands.

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

The exact TP4 B1 FP8 dense shapes account for a projected 6.654 ms/token across
279 calls. The largest individual component is the replicated
`fused_wqa_wkv` projection at 2.585 ms/token. This projection total is a
standalone kernel estimate, not an additive endpoint decomposition.

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

The static protocol can recover approximately 0.46 ms/token, but it cannot by
itself close the 6.88 ms gap to 10 ms TPOT. Its production candidate remains
default-off until full-model deadlock, output-token, and endpoint A/B gates pass.

Two attempted Nsight captures are excluded from performance evidence. The
first collided with a different task that had begun CPU-side startup before
creating CUDA contexts. The GPU gate now detects pending launchers and claims
the GPUs before stabilization. The second compiled TileLang kernels under
Nsight child-process instrumentation, hit an NVCC temporary-output failure on
one rank, and left peers waiting in a collective. The follow-up reuses the
exact successful baseline caches and includes marker-based cleanup for child
processes that create their own sessions.
