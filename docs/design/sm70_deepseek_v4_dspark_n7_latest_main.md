# SM70 DeepSeek V4 DSpark N7 Latest-Main Campaign

## Scope

Measure and optimize DeepSeek V4 Flash DSpark with seven draft tokens on the
current accepted SM70 performance stack. The old DSpark endpoint result was
measured before the grouped MXFP4, sparse MLA, TP8 collective, FP16 GEMV, and
mHC integrations, so it is not a valid speed baseline for this campaign.

## Fixed Contract

- Integration line: `onecat/main`
- Base SHA: `48e89751b4b98c18e1be6506dca15f015155d068`
- Model: `/home/fudanwl/Desktop/dir`
- Topology: TP8 on eight V100-SXM2-32GB GPUs
- Quantization: FP8 dense plus MXFP4 routed experts
- KV cache: `fp8_ds_mla`
- Workload: exactly 1024 input tokens and up to 256 output tokens
- Sampling: target `temperature=1.0`, `top_p=1.0`; greedy DSpark draft
- Speculation: `method=dspark`, `num_speculative_tokens=7`, verifier M=8
- Runtime: CUDA Graph enabled, `max_num_seqs=1`, no eager execution
- Quality: natural EOS, coherent chat/code output, no `ignore_eos`

## Promotion Gates

1. Prove DSpark, M=8 verifier, SM70 sparse draft attention, TurboMind
   FP8/MXFP4, FP8 MLA KV, TP8, and CUDA Graph route selection in worker logs.
2. Compare no-speculation and DSpark7 under the exact same source, flags,
   model, prompt tokens, sampling, and warmup state.
3. Report TTFT, steady TPOT, output throughput, accepted length, per-position
   acceptance, and rejection distribution.
4. Split verifier, target sample/state, three-layer draft forward, seven
   Markov/sample steps, KV work, collective work, and host residual before
   selecting an optimization.
5. Admit a change only after its exact-shape microbenchmark and numerical
   oracle pass. Promote only if the unprofiled endpoint and quality gates pass.

## Baseline Status

The implementation and N=7 microbenchmarks landed through PR #165. Its old
same-source endpoint improved 7.689 to 8.070 token/s with mean accepted length
1.555, but that source predates the accepted main performance stack. The
current no-speculation integration baseline is about 19.457 ms/token, or
51.40 token/s, and DSpark7 has not yet been measured on this exact tree.

## Experiment Log

| Date | Source | Test | Result | Decision |
|---|---|---|---|---|
| 2026-08-03 | `48e89751b4` | Campaign opened | Baseline pending | Run static and exact N7 gates first |

## Artifacts

Remote source, compiler caches, launch commands, server logs, endpoint JSON,
acceptance metrics, profiles, and rejected variants will be recorded here as
they are produced. Task-owned services must be stopped before handoff.
