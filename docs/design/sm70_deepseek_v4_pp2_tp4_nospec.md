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

No current-main PP2 x TP4 no-spec performance claim exists yet. The first task
is to audit the retained PP2 x TP4 launch artifact, then run the frozen baseline
when all eight GPUs are free. Source changes begin only after that evidence
identifies the first critical-path bottleneck.
