# SM70 Qwen3.8 NVFP4 DFlash2 17 ms Campaign

## Scope and frozen baseline

This campaign reduces the complete batch-one DFlash2 speculative round for
Qwen3.8-27B NVFP4 on four V100 GPUs. It starts from
`onecat/main@34403018d917054dd7765d5e820ad29c8d342348`; it does not reuse a
historical worktree or credit route-hit logs as performance evidence.

The target is the local mixed NVFP4/channel-FP8 checkpoint with a BF16 LM head:

`/data/minimax-h3/task-cache/v100-dflash2-target-graph-20ms/models/Qwen3.8-27B-NVFP4-e5m2-unit-kv-bf16-lmhead`

The DFlash2 draft is
`/data/models/v100-dflash2-20260820/draft` at revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`. The frozen model contract is TP4
on V100 GPUs 0--3, probabilistic DFlash2 with seven draft tokens, selector
K=16, target top-k=20, FP8 E5M2 target KV, FP16 draft KV, Flash-V100 for target
and draft attention, and FULL target and draft CUDA Graphs. GPUs 4--7 are
outside this campaign because they host an unrelated service.

Two runtime contracts are deliberately kept separate:

1. The localization contract uses a 32K maximum model length, 512 maximum
   batched tokens, and disables prefix caching, Mamba alignment, and parsers.
   It is used only for short unprofiled repeats, graph-node traces, and
   exact-shape microbenchmarks.
2. The promotion contract uses a 256K maximum model length, 4096 maximum
   batched tokens, prefix caching, Mamba alignment, and the Qwen tool and
   reasoning parsers. A speedup is not accepted until it reproduces under this
   practical configuration.

The current accepted practical baseline on PR #288 is `18.465--18.537 ms` per
complete round for 512-token coding requests and `18.587--18.603 ms` for
1,024-token coding requests. The high-acceptance MBPP item 28 baseline is
`18.567 ms`, acceptance length `4.686`, and `251.60 token/s`, with a natural
EOS and EvalPlus base/plus `1/1`. A fresh current-source 16-prompt GSM8K run
measures request-mean acceptance `4.45732`, pooled acceptance `4.07740`, and
`19.3688 ms` per round with diagnostic counters enabled. Diagnostic timing is
not a production baseline.

## Acceptance gates

A short-context candidate is accepted only when all of the following hold:

- the no-diagnostic promotion contract measures a mean and median complete
  speculative round at or below `17.0 ms` across at least three independent
  steady-state requests; the p90 must be at or below `17.5 ms`;
- the same fixed prompt, seed, sampling parameters, output cap, CUDA Graph
  shapes, and GPU set are used for the baseline/candidate pair;
- paired request-mean acceptance does not fall by more than `0.05`, the
  per-position counters remain healthy, and no stale-buffer, graph-replay, or
  draft-KV mismatch appears;
- the existing GSM8K, MATH-500, HumanEval, corrected MBPP, and WikiText PPL
  gates do not regress versus target-only and the accepted DFlash2 baseline;
- every default-on optimization has an exact admission predicate, a rollback
  switch, focused numerical coverage, and an unprofiled endpoint win.

Long-context work is evaluated at 32K, 128K, and 256K. The candidate must not
make complete-round latency or pure-decode throughput worse by more than 1%
at any length, and it must preserve generated tokens and acceptance counters
within each deterministic route probe. The 17 ms target applies to the short
round; long-context results are reported as an explicit decay curve because
the verifier attention cost necessarily grows with resident context.

## Trace-first implementation sequence

1. Reproduce the current no-diagnostic NVFP4 baseline in this worktree and
   record source SHA, extension hashes, route logs, GPU clocks, and per-request
   round statistics.
2. Capture the smallest generation-only Nsight Systems trace that contains
   steady CUDA Graph replays. Split every round into draft, draft-to-target,
   target verification, and target-to-draft phases, then expand the target
   phase into exact kernel categories and launch counts.
3. Microbenchmark only the largest current-source bucket at its production
   M/N/K, TP, dtype, and graph shapes. A microbenchmark result is directional
   evidence, never an endpoint claim.
4. Implement the smallest exact change that removes the measured bottleneck.
   Rejected DDTree index reuse, skipped verification, generic auxiliary-stream
   overlap, variable K, and approximate reranking are not retried: they either
   changed output/acceptance or increased the complete round.
5. Promote a default only after focused CPU/GPU numerical tests, graph replay,
   three clean short repeats, the practical 256K configuration, the quality
   suite, and the 32K/128K/256K decay sweep pass.

The first trace should determine whether the remaining roughly `1.6 ms` is in
NVFP4 QPN2/QPN8 projections, draft non-causal KV work, target verification,
or scheduler/metadata boundaries. No saving is assigned to a hypothesis
before that trace and its exact-shape microbenchmark exist.

## Draft PR record

Purpose: reduce the complete Qwen3.8-27B NVFP4 DFlash2 round to 17 ms on SM70
without changing the sampling contract, acceptance, or task quality.

Test plan: unprofiled paired endpoint repeats; Nsight graph-node phase
breakdown; exact-shape microbenchmarks; focused numerical and CUDA Graph replay
tests; GSM8K/MATH-500/HumanEval/MBPP/PPL quality gates; 32K/128K/256K decay
sweep.

Test result: pending current-source baseline and trace.
