# SM70 DFlash2 + ngram Hybrid

## Purpose

Add an optional prompt-ngram assistant to the MRV2 DFlash2 path. An ngram hit
must skip the DFlash2 query/selector while preserving DFlash2 context-KV state,
and a miss must fall back to the existing DFlash2 implementation unchanged.
The target verifier and the probabilistic rejection-sampling contract remain
authoritative, so the optimization cannot change the target distribution.

This work is based on `onecat/main` at
`05d5aa4e5713b376e0fcc057a0a623c7eae53708` and is isolated on
`agent/v100-dflash2-ngram-hybrid-20260825-050018`.

## Upstream audit

- vLLM prompt lookup: PRs #12193, #22437, #24986, and #29184 provide the CPU
  KMP and GPU-vectorized implementations. MRV2 still has no combined ngram +
  model-drafter route.
- SGLang: PRs #17260, #21243, and #22737 show that overlap scheduling requires
  complete request-token state and explicit accepted-token indexing. A stale
  host output list is not a valid lookup source.
- llama.cpp: its comma-separated speculative configuration gives draftless
  ngram proposers priority, falls back to DFlash, and still calls `process()` on
  every proposer so model-drafter state remains synchronized. This is the
  closest reference architecture for the first implementation here.
- Arctic Suffix Decoding and SAM-Decoding are useful follow-ups, but their
  confidence policy, external dependency, and longer trees are intentionally
  outside this first block-8 implementation.

## Initial contract

- The feature is opt-in and only valid for `method=dflash` with a DFlash2
  checkpoint. Standalone `ngram`, Eagle, MTP, DFlash1, and `dflash_ddtree`
  routing is unchanged.
- Lookup reads the authoritative MRV2 request-token state and supports normal
  synchronous and overlap scheduling without rebuilding history in Python.
- Structured-output requests bypass the ngram assistant until grammar-aware
  proposal masking is proved correct. Tool/reasoning parsers without a grammar
  are unaffected.
- A hit returns at most the configured DFlash2 draft width (seven tokens in the
  official block-8 setup). The normal DFlash2 proposer handles misses.
- Context K/V materialization always runs. Only the DFlash2 query, candidate
  projection, selector, and selector walk may be skipped.
- Greedy and probabilistic modes are supported. In probabilistic mode the
  ngram proposal is represented as a one-hot draft distribution inside the
  existing sparse target-rejection interface, preserving target sampling.
- Prefix-cache hits may rebuild missing DFlash draft K/V as before; this change
  must not corrupt either target or draft cache state.

## Test plan

1. Unit tests: configuration/routing isolation; KMP hit, miss, truncation, and
   overlap cases; per-request mixed-source state; one-hot sparse draft support;
   grammar bypass; prefix/state transitions.
2. CPU microbenchmarks: lookup at 1K, 32K, 128K, and 256K contexts.
3. V100 correctness: eager before CUDA graph; batch 1/2/4; hit/miss/mixed;
   compare proposed tokens, accepted trajectory, target output, and draft K/V.
4. V100 performance: report lookup, context-KV, query/selector, target verify,
   full round, emitted tokens per round, and pure-decode tokens/s for baseline
   DFlash2 and DFlash2+ngram at short and long contexts.
5. Quality: compare target-only, DFlash2, and hybrid on coding/general/tool
   datasets under the same sampling configuration. Report scores, completion
   counts, wall time, hit rate, conditional acceptance length, and failures.

## Promotion gates

- No statistically meaningful quality regression against target-only or the
  existing DFlash2 route.
- DFlash2-miss acceptance trajectory remains unchanged in deterministic tests.
- Ngram hits actually skip query/selector work in a trace.
- Hybrid pure-decode throughput is non-regressing at every measured context;
  otherwise the feature remains opt-in while the losing shape is investigated.

## Status

- 2026-08-25: isolated worktree created; upstream audit and initial contract
  recorded.
- 2026-08-25: implemented opt-in MRV2 host lookup over the UVA request-token
  state, full-hit query/selector bypass, mixed-batch row override, and one-hot
  dense/sparse rejection caches. Structured-output batches bypass the assist.
- CPU split-history KMP microbenchmark (median/P95): 1K `2.66/2.74 us`, 32K
  `50.05/53.22 us`, 128K `190.72/201.03 us`, and 256K `369.57/393.17 us`.
- Tests: 22 DFlash2-ngram tests pass on V100, including exact parity with the
  standalone KMP policy and dense-vs-compact probabilistic rejection at
  `top_p=1.0/0.95`; existing DFlash2 CPU suite passes 61 tests with 12 expected
  CUDA skips, and the MRV2 route suite passes 10 tests. End-to-end model and
  dataset validation remain in progress.
- The first practical TP4 CUDA Graph run used FP8 E5M2 target KV, FP16 draft
  KV, 256K maximum context, 4096 maximum batched tokens, prefix caching, Mamba
  alignment, and the Qwen tool/reasoning parsers. Repeated text reached 92 full
  hits in 93 eligible rounds and a warmed `328.32 tok/s`; lookup itself averaged
  about `0.018 ms` on TP0. A natural coding completion reached `148.89 tok/s`
  with about 15% of rounds skipping the DFlash2 query/selector and average
  accepted length near 3.9. These are route proofs, not the final paired speed
  result: the existing public server uses additional uncommitted production
  optimizations, so its `418.02 tok/s` repeated-text result is not a valid
  baseline for this clean branch.
- Existing same-sampling 16K natural-stop quality runs provide the target and
  DFlash2 control. DFlash2 completed HumanEval32 in `374.08 s`, MBPP32 in
  `363.19 s`, and LiveCodeBench16 in `885.02 s` (`1622.29 s`, or 27.04 minutes,
  for 80 cases). A hybrid-only full rerun is therefore budgeted at roughly
  25--28 minutes after the paired micro-suite passes.
