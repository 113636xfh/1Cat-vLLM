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
  recorded. Implementation and validation are in progress.
