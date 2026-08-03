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
initial 4K no-speculation measurement was about 19.34 ms/token, but this was a
CUDA Graph capture-capacity artifact rather than the best valid 1K decode
baseline. The corrected no-speculation baseline is 15.357 ms/token, or 65.12
token/s, on the fixed contract above. DSpark7 must be remeasured against this
corrected baseline.

## No-Speculation Baseline Correction

DeepSeek V4 bypasses Lightning Indexer scoring while every compressed key is
covered by `index_topk`. For the C4 layers, the boundary is
`index_topk=512 * compress_ratio=4 = 2048` tokens. Before this change, a graph
captured with `max_model_len=4096` permanently selected the full Indexer path
even when the real request had only 1024 tokens.

Nsight Systems node traces proved the route difference. Across 11 decode
tokens, 21 C4 layers, eight TP ranks, the 4K graph added exactly 1848 calls each
to the Indexer query, key dequantization, weighted-query, persistent top-k, and
associated GEMV/GEMM path. The 2K graph instead added exactly 1848 calls to
`_fill_short_context_topk_indices`. Node-trace service times are diagnostic;
the endpoint numbers below are unprofiled.

The runtime now captures one additional single-request graph at the threshold
derived from the model config. It selects that graph only while the real
attention context fits, then falls back to the original full graph. The
automatic path is limited to SM70, `DeepseekV4ForCausalLM`, and no speculative
decoding. `VLLM_SM70_DSV4_DECODE_CONTEXT_BUCKETS` remains an explicit override;
an empty value disables the automatic bucket.

| No-MTP configuration | Median TPOT | Decode throughput | Decision |
|---|---:|---:|---|
| 4K max length, old full graph | 19.342 ms | 51.70 token/s | Rejected baseline |
| 2K max length, old short graph | 15.353 ms | 65.13 token/s | Fast-path reference |
| 4K max length, automatic 2K graph | 15.357 ms | 65.12 token/s | Accepted baseline |

The automatic graph reduces TPOT by 3.985 ms, or 20.6%, and increases decode
throughput by 25.9% against the old 4K graph while matching the 2K reference.
The three final runs were 15.362, 15.346, and 15.357 ms/token.

One 1024-input, 1200-output boundary run forced decode across the threshold.
Tokens 2-1024 remained at 15.21-15.28 ms mean latency; tokens 1025-1200 used
the full graph at 18.99-19.18 ms. All 1200 tokens completed, and the following
short request returned to 15.284 ms/token with a correct natural-stop response.

The boundary test also exposed an independent SM70 prefill bug: the contiguous
FP8 Index-K dequantizer passed a `torch.float8_e4m3fn` pointer signature to
Triton, which rejects `fp8e4nv` on V100. Passing the same storage as `uint8`
allows the existing software E4M3 decoder to run without changing values or
accumulation. A V100 numerical test passes, and an exact 2304-token prompt now
completes with healthy output and 19.32-19.61 ms/token decode.

## Experiment Log

| Date | Source | Test | Result | Decision |
|---|---|---|---|---|
| 2026-08-03 | `48e89751b4` | Campaign opened | Baseline pending | Run static and exact N7 gates first |
| 2026-08-03 | candidate | 2K/4K no-MTP matrix | Only `max_model_len` controls the fixed 4 ms gap | Trace graph nodes |
| 2026-08-03 | candidate | Nsight graph-node A/B | Exact 21-layer C4 Indexer route difference | Add bounded graph |
| 2026-08-03 | candidate | 4K auto-bucket, 3 seeds | 15.357 ms median, 65.12 token/s | Accept speed gate |
| 2026-08-03 | candidate | 2048 crossing and 2304 prefill | Safe fallback, natural output, FP8 prefill fixed | Accept quality gate |

## Artifacts

Retained remote root:
`/home/fudanwl/v100-worktrees/runs/dsv4-dspark-n7-main-48e897-20260803`.

- Graph traces: `nospec-maxlen2k-maxbatch4k-i1k-o12-node.sqlite` and
  `nospec-maxlen4k-maxbatch4k-i1k-o12-node.sqlite`.
- Final endpoint runs: `nospec-4k-autobucket-seed4201.json` through
  `nospec-4k-autobucket-seed4203.json`.
- Boundary run: `nospec-4k-bucket2k-crossdecode-i1024-o1200.json`.
- Long prefill gate: `nospec-4k-autobucket-i2304-o32.json`.
- Natural-stop quality gate: `nospec-4k-autobucket-quality-natural-stop.json`.
- Final worker log: `server-nospec-4k-autobucket.log`.

Disabling prefix caching and reducing `max_num_batched_tokens` to 2048 did not
recover the gap; those paths were rejected. The task-owned API service must be
stopped before handoff.
