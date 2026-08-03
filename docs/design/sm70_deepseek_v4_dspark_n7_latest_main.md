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

## DSpark N7 Verifier Bottleneck

The corrected no-speculation baseline does not make the existing DSpark N7
route competitive. Three official-sampling runs measured 66.951-105.631 ms per
emitted token, with mean emitted lengths of 1.342-2.098 tokens per round. Draft
acceptance varies materially with the seed, but synchronized timing and Nsight
show a separate verifier implementation bottleneck that must be removed even
when acceptance improves.

One steady M=8 proposal round consists of approximately 124-134 ms of target
forward and 9.9-10.2 ms of draft GPU work. Target sampling, rejection, and
bookkeeping together remain below 0.5 ms. `draft_wall_cpu` overlaps the prior
asynchronous target forward and must not be added to those GPU times.

The target CUDA Graph on rank 0 contains 23,909 distinct nodes and takes
125.984 ms per replay in the node trace. TurboMind `gemm_kernel` service is
100.834 ms, or 80.04% of target service. It executes 22,274 GEMM kernels per
replay. Of these, exactly `43 layers * 256 experts * 2 MoE stages = 22,016`
come from launching W13 and W2 for every expert; the remaining 258 are the six
dense GEMMs in each layer. The verifier has at most 48 routed slots, so the
all-256-expert dispatch is the first target-forward optimization point.

The admitted design preserves all rows routed to the same expert. A graph-safe
SM70 Triton scan compacts the sorted 48 slots into unique expert IDs and
offsets entirely on device, then passes a fixed 48-entry active table to the
existing exact TurboMind stage. Empty tail entries retain graph shape without
host readback. B=1 continues to use the accepted direct top-6 implementation.
The active-token limit now defaults to eight when the existing active-expert
master switch is enabled; disabling that switch retains the dense fallback.

V100 gates completed so far:

- The compactor captures under CUDA Graph and replays a changed route with
  exact IDs and offsets.
- An exact M=8 full MoE microbenchmark, including permute, compaction, W13,
  SwiGLU, W2, and unpermute, is bitwise equal at every stage and after dynamic
  replay (`max_abs=0`). It has 34 unique experts in 48 routed slots.
- An exclusively owned repeat measured 5.321 ms for dense-256 and 4.384 ms for
  active-48, a 1.214x speedup that projects 40.31 ms saved over 43 layers. An
  earlier externally contended run projected a similar 42.9 ms saving.

The full-model synchronized profile confirms that the microbenchmark saving
reaches the production M=8 graph. With the same filtering rule, the steady
median target forward falls from 126.531 to 78.779 ms (-37.7%), draft GPU work
falls from 9.983 to 6.746 ms, and the serial GPU subtotal falls from 137.444 to
86.416 ms (-37.1%). A same-seed profiled endpoint improved from 9.028 to 15.233
token/s, but profiled throughput is diagnostic only; the unprofiled three-seed
gate remains required.

The unprofiled matched-seed gate passes:

| Seed | Old token/s | Active-48 token/s | Gain | Old emitted/round | Active-48 emitted/round |
|---:|---:|---:|---:|---:|---:|
| 4201 | 12.194 | 22.179 | +81.9% | 1.782 | 2.048 |
| 4202 | 9.467 | 15.024 | +58.7% | 1.342 | 1.364 |
| 4203 | 14.936 | 16.545 | +10.8% | 2.098 | 1.500 |

Median TPOT falls from 82.007 to 60.441 ms (-26.3%), and median throughput
rises from 12.194 to 16.545 token/s (+35.7%). Mean emitted length across the
three seeds falls from 1.741 to 1.637, so the endpoint improvement is not an
acceptance-rate artifact. The official-sampling natural-stop gate also passes:
the model emits exactly two coherent sentences and stops after 65 tokens.

The post-change Nsight node trace closes the verifier composition. Graph 557
contains 6,021 nodes per replay instead of 23,909. The exact reduction of
17,888 nodes equals `43 * (256 - 48) * 2`. Node tracing reports 89.022 ms of
rank-0 target service per replay; the lower-overhead synchronized target wall
remains 78.779 ms and is the accepted absolute latency.

| Rank-0 target category | Dense-256 trace | Active-48 trace | Active-48 share |
|---|---:|---:|---:|
| MXFP4 MoE W13 | 60.940 ms | 42.673 ms | 47.94% |
| MXFP4 MoE W2 | 33.682 ms | 14.975 ms | 16.82% |
| mHC kernels | 6.884 ms | 6.850 ms | 7.70% |
| TurboMind dense GEMM | 6.212 ms | 6.211 ms | 6.98% |
| Sparse MLA/indexer | 5.407 ms | 5.415 ms | 6.08% |
| CUTLASS/cuBLAS GEMM | 4.831 ms | 4.868 ms | 5.47% |
| NCCL all-reduce | 4.532 ms | 4.492 ms | 5.05% |
| MoE routing/activation | 1.987 ms | 2.118 ms | 2.38% |
| Generic elementwise | 1.051 ms | 1.047 ms | 1.18% |
| Other | 0.457 ms | 0.372 ms | 0.42% |

The active table reduces MoE W13+W2 service from 94.622 to 57.648 ms. The new
compaction kernel itself is 0.120 ms per round. The next verifier optimization
must therefore reduce the real active-expert W13/W2 work or its 4,128 remaining
per-expert launches; optimizing sampling or the compactor first would target
the wrong scale.

A same-contract no-speculation regression run measures 15.377 ms/token, or
65.03 token/s, versus the accepted 15.357 ms/token baseline. The 0.13%
difference is noise, so extending graph-safe buffers through M=8 does not
regress the B=1 path.

## Experiment Log

| Date | Source | Test | Result | Decision |
|---|---|---|---|---|
| 2026-08-03 | `48e89751b4` | Campaign opened | Baseline pending | Run static and exact N7 gates first |
| 2026-08-03 | candidate | 2K/4K no-MTP matrix | Only `max_model_len` controls the fixed 4 ms gap | Trace graph nodes |
| 2026-08-03 | candidate | Nsight graph-node A/B | Exact 21-layer C4 Indexer route difference | Add bounded graph |
| 2026-08-03 | candidate | 4K auto-bucket, 3 seeds | 15.357 ms median, 65.12 token/s | Accept speed gate |
| 2026-08-03 | candidate | 2048 crossing and 2304 prefill | Safe fallback, natural output, FP8 prefill fixed | Accept quality gate |
| 2026-08-03 | `3765c56a96` | DSpark N7 target graph-node trace | 80.04% target service in GEMM; 22,016 inactive-capable MoE launches | Compact M=8 routed experts on device |
| 2026-08-03 | candidate | M=8 active-48 CUDA Graph microbenchmark | Bitwise exact; first timing 1.30x but externally contended | Repeat exclusively, then run full model |
| 2026-08-03 | candidate | Exclusive M=8 active-48 microbenchmark | 5.321 -> 4.384 ms; bitwise exact; projected -40.31 ms/round | Admit to full model |
| 2026-08-03 | candidate | Same-contract synchronized profile | Target 126.531 -> 78.779 ms; GPU subtotal -37.1% | Run unprofiled three-seed gate |
| 2026-08-03 | candidate | Unprofiled matched seeds 4201-4203 | Median 12.194 -> 16.545 token/s; emitted length decreases | Accept endpoint speed gate |
| 2026-08-03 | candidate | Official-sampling natural stop | Coherent two-sentence response; `finish_reason=stop` | Accept text-health gate |
| 2026-08-03 | candidate | Post-change graph-node trace | 23,909 -> 6,021 nodes; current MoE is 64.8% | Target active W13/W2 next |
| 2026-08-03 | candidate | No-speculation B=1 regression | 15.377 ms/token, 65.03 token/s | Accept regression gate |

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
- DSpark target trace: `dspark7-head3765-i1k-o16-node.sqlite`.
- M=8 microbenchmark: `mxfp4-verifier-m8-dense256-vs-active48-seed4201.log`.
- Exclusive repeat: `mxfp4-verifier-m8-dense256-vs-active48-exclusive-seed4201.log`.
- Candidate profile: `server-dspark7-profile-active48.log` and
  `dspark7-profile-active48-i1024-o128.json`.
- Unprofiled comparison: `dspark7-active48-unprofiled-comparison.json` and
  `dspark7-active48-unprofiled-seed4201.json` through `seed4203.json`.
- Candidate quality: `dspark7-active48-quality-natural-stop.json`.
- Post-change trace: `dspark7-active48-i1k-o16-node.sqlite` and
  `dspark7-active48-node-trace-comparison.json`.
- No-speculation regression: `nospec-active48-regression-seed4201.json`.

Disabling prefix caching and reducing `max_num_batched_tokens` to 2048 did not
recover the gap; those paths were rejected. The task-owned API service must be
stopped before handoff.
