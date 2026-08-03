# SM70 DeepSeek V4 Prefill Optimization

## Scope

- Integration base: `48e89751b4b98c18e1be6506dca15f015155d068`
- Model: DeepSeek-V4-Flash, FP8 dense and MXFP4 routed experts
- Hardware: TP8 on eight V100-SXM2-32GB GPUs
- Quantization backend: TurboMind only; Marlin is out of scope
- KV cache: `fp8_ds_mla`
- Runtime: CUDA Graph enabled, no eager execution, no speculative decoding
- Initial workload: exactly 1024 input tokens and one naturally sampled output
- Sampling: official `temperature=1.0`, `top_p=1.0`

The first gate is a same-source, unprofiled single-request prefill baseline.
Report tokenized prompt length, request TTFT, worker prefill wall time, and
prompt throughput separately. Decode timing must not be included in the
prefill claim.

## Measurement Definitions

- Request TTFT is measured by the streaming OpenAI client from request start
  to the first non-empty text delta. Effective prompt throughput is
  `1024 / TTFT`; it includes frontend, scheduling, LM-head, sampling, and
  streaming overhead and is not pure GPU prefill throughput.
- The Nsight GPU envelope is the interval from the first to the last CUDA
  kernel on one rank during the captured request. The critical rank is the GPU
  with the longest envelope.
- GPU busy time is the interval union across streams. `envelope - busy` is the
  explicit GPU idle/dependency gap and closes the per-rank wall clock.
- Kernel and category service sums describe composition. They are kept
  separate from the envelope because concurrent streams can overlap.

## Measurement Order

1. Verify worker logs select DeepSeek V4, TurboMind FP8/MXFP4, SM70 sparse
   attention, FP8 MLA KV, TP8, and non-eager execution.
2. Run warm and cold unprofiled 1024-token prefill samples and retain raw JSON.
3. Capture one Nsight Systems request with CUDA Graph node and NVTX data.
4. Split prefill GPU service into FP8 dense, MXFP4 MoE, sparse/SWA attention,
   compressor and KV work, mHC, collectives, routing, and residual categories.
5. Use Nsight Compute only on a confirmed dominant kernel and preserve an
   explicit unattributed residual between GPU service and worker/request wall.
6. Admit an optimization only after an exact-shape microbenchmark and numerical
   oracle pass, then rerun the unprofiled endpoint and output-quality gates.

## Baseline

The task-owned endpoint used TP8, `fp8_ds_mla`, prefix caching disabled,
`max_model_len=2048`, `max_num_batched_tokens=2048`, no MTP, and CUDA Graph.
The first cold request took 11681.327 ms and compiled eight prefill Triton
kernels. It is retained as cold-start evidence and excluded from the warm
baseline. Two additional warmups measured 1652.083 and 1653.033 ms.

Five measured requests then produced:

| Seed | TTFT (ms) | Prompt tokens | Completion tokens |
| ---: | --------: | ------------: | ----------------: |
| 6101 | 1651.805 | 1024 | 1 |
| 6102 | 1651.730 | 1024 | 1 |
| 6103 | 1651.969 | 1024 | 1 |
| 6104 | 1654.750 | 1024 | 1 |
| 6105 | 1649.518 | 1024 | 1 |

The median is 1651.805 ms, the mean is 1651.954 ms, and the median effective
request-level prompt throughput is 619.93 token/s. All samples used official
`temperature=1.0` and `top_p=1.0` sampling. A separate 1024/64 smoke decoded
readable text but drifted into an HTML fragment near its token-limit stop, so
it is retained only as baseline text-health evidence, not as a full quality
pass.

Raw artifacts are retained under
`/home/fudanwl/v100-worktrees/runs/dsv4-prefill-trace-20260803/`.

## Nsight Systems Trace

After the cold JITs and two profiler-server warmups, one exact 1024/1 request
was captured with CUDA Graph node tracing. The profiled request had 1713.796
ms TTFT. Its critical rank was device 5:

| Same-request interval | Time (ms) | Share of request TTFT |
| --- | ---: | ---: |
| Critical-rank GPU envelope | 1686.097 | 98.38% |
| GPU busy interval union | 1541.559 | 89.95% |
| GPU idle/dependency gap | 144.538 | 8.43% |
| Request residual outside GPU envelope | 27.699 | 1.62% |

The second and third rows close the GPU envelope. The last row is only the
same-profile request residual; it must not be interpreted as a pure scheduler
measurement. Across all eight ranks, envelopes were tightly grouped from
1685.221 to 1686.097 ms.

Critical-rank kernel service was composed as follows. Percentages use total
kernel service as their denominator and therefore do not close the GPU wall
clock when streams overlap.

| Category | Service (ms) | Service share | Launches |
| --- | ---: | ---: | ---: |
| TurboMind MXFP4 MoE GEMM | 1128.102 | 72.69% | 22016 |
| SM70 sparse MLA/SWA attention | 212.538 | 13.69% | 43 |
| NCCL collectives | 64.355 | 4.15% | 88 |
| TurboMind FP8 dense GEMM | 48.321 | 3.11% | 258 |
| mHC | 43.237 | 2.79% | 259 |
| FP16/CUTLASS GEMM | 17.856 | 1.15% | 148 |
| KV compression/indexer/rope | 17.060 | 1.10% | 346 |
| MoE routing | 15.349 | 0.99% | 344 |
| All remaining categories | 5.147 | 0.33% | 574 |

The dominant MXFP4 service splits into exactly two repeated launch shapes:

| Stage | Service (ms) | Launches | Mean launch | Launch geometry |
| --- | ---: | ---: | ---: | --- |
| W13 gate/up, local N=512 | 886.889 | 11008 | 80.57 us | `grid=(49,4,2)`, 128 threads, 255 registers, 32784 B dynamic shared memory |
| W2 down, local N=4096 | 241.213 | 11008 | 21.91 us | `grid=(49,32,1)`, 128 threads, 255 registers, 32784 B dynamic shared memory |

The count is structural: 43 MoE layers times 256 experts times two stages is
22016 launches. `mxfp4_moe_dense_stage_sm70_out()` currently loops over every
expert and invokes TurboMind with `num_experts=1`. This launch decomposition,
not sparse attention, is the first prefill optimization target.

The first candidate uses four 64-expert TurboMind dispatches per stage instead
of 256 one-expert dispatches. PR #179 contained a default-off unvalidated sketch
of this direction. The sketch was ported to the current integration base, but
its unbounded 256-expert grouped call failed the numerical gate despite a large
speedup:

| Candidate | W13 legacy | W13 grouped | Speedup | Numerical result |
| --- | ---: | ---: | ---: | --- |
| 256 experts/launch | 29.66 ms | 1.299 ms | 22.83x | Rejected: max abs 0.00390625 |
| 256 experts/launch, legacy dispatch policy | 28.95 ms | 2.074 ms | 13.96x | Rejected: same max abs 0.00390625 |

The second failure used the exact same TurboMind kernel, split count, swizzle,
CTA shape, and stages as the legacy loop. The difference therefore comes from
the large grouped scheduler execution shape, not a quantization or dispatch
policy change. It affected 2645 of 3145728 W13 elements across all 256 experts,
with no sign flips or row-wise argmax changes. It is still rejected because
the full-model effect of repeating the difference across 43 layers is not
proven safe.

Sweeping the number of experts handled per grouped launch found a strict
bitwise boundary:

| Experts/launch | W13 grouped GPU median | Speedup | Cross-route result |
| ---: | ---: | ---: | --- |
| 1 | 28.273 ms | 1.06x | bitwise |
| 2 | 16.644 ms | 1.81x | bitwise |
| 4 | 9.767 ms | 2.92x | bitwise |
| 8 | 5.040 ms | 5.54x | bitwise |
| 16 | 2.557 ms | 11.00x | bitwise |
| 32 | 2.489 ms | 11.31x | bitwise |
| 64 | 2.464 ms | 11.51x | bitwise |
| 128 | 2.372 ms | 11.92x | rejected, max abs 0.00390625 |
| 256 | 2.081 ms | 13.56x | rejected, max abs 0.00390625 |

The candidate is therefore hard-limited to at most 64 experts per launch. At
that width the 1024-token stage benchmark passed cross-route and repeated-run
bitwise equality for W13 and W2, balanced routing and a half-active routing
stress, across seeds 29, 101, and 202. Balanced-route W13 speedup ranged from
10.14x to 12.40x; W2 ranged from 5.64x to 5.89x. The structural request count
becomes 43 layers times two stages times four launches, or 344 launches instead
of 22016.

The original projection put MXFP4 service near 123 ms. The matched endpoint and
post-candidate trace have now measured the result rather than relying on that
projection.

## Grouped-Prefill Candidate

The candidate keeps the route default-off, requires at least 6144 routed rows
(1024 tokens times top-k 6), and hard-clamps the group width to 64. This keeps
unvalidated high-concurrency decode shapes out of the prefill route. Worker
logs proved that a 1024-token API request reached the grouped C++ path with a
12288-row internal staging shape. Decode graph capture remained on the existing
dense-stage path with 12 rows and did not enter grouped prefill.

After a cold request and two warmups, five unprofiled requests produced:

| Seed | Candidate TTFT (ms) |
| ---: | ---: |
| 6601 | 567.607 |
| 6602 | 533.535 |
| 6603 | 531.873 |
| 6604 | 531.416 |
| 6605 | 568.867 |

The candidate median is 533.535 ms and the mean is 546.660 ms. Relative to the
same-contract 1651.805 ms baseline median, TTFT falls by 1118.270 ms or 67.70%,
a 3.096x speedup. Effective request-level prompt throughput rises from 619.93
to 1919.27 token/s. These are unprofiled endpoint measurements, not kernel
service projections.

The matched post-candidate Nsight request measured 596.478 ms TTFT. Device 5
was again the critical rank:

| Same-request interval | Time (ms) | Share of request TTFT |
| --- | ---: | ---: |
| Critical-rank GPU envelope | 572.791 | 96.03% |
| GPU busy interval union | 545.071 | 91.38% |
| GPU idle/dependency gap | 27.721 | 4.65% |
| Request residual outside GPU envelope | 23.687 | 3.97% |

The candidate busy interval and idle gap close the candidate GPU envelope.
Critical-rank envelopes ranged from 572.114 to 572.791 ms across all ranks.
Its service composition was:

| Category | Service (ms) | Service share | Launches |
| --- | ---: | ---: | ---: |
| SM70 sparse MLA/SWA attention | 228.628 | 41.18% | 43 |
| TurboMind MXFP4 MoE GEMM | 133.226 | 24.00% | 344 |
| TurboMind FP8 dense GEMM | 50.828 | 9.16% | 258 |
| mHC | 45.335 | 8.17% | 259 |
| NCCL collectives | 39.792 | 7.17% | 88 |
| KV compression/indexer/rope | 18.161 | 3.27% | 346 |
| FP16/CUTLASS GEMM | 17.989 | 3.24% | 148 |
| MoE routing | 15.729 | 2.83% | 344 |
| All remaining categories | 5.467 | 0.98% | 574 |

MXFP4 service therefore falls from 1128.102 to 133.226 ms, an 88.19%
reduction, while its launch count falls exactly 64x from 22016 to 344. W13
accounts for 86.319 ms and 172 launches; W2 accounts for 46.906 ms and 172
launches. The new first optimization target is the 228.628 ms sparse MLA/SWA
attention kernel, not further grouped-MoE launch reduction.

### Numerical And Text Gates

The operator oracle now covers the internal 2048-token execution shape seen in
the route log: 12288 routed rows, all 256 experts active, random non-uniform
expert counts, and the entire output buffer initialized with a sentinel. At
64 experts per launch, both W13 and W2 match the legacy path bit-for-bit,
including the full-buffer tail, and repeated grouped runs are bitwise stable.

The candidate also completed repeated official-sampling 1024/64 requests with
stable, readable text, no repeated-token collapse, and unchanged steady decode
latency. A separate greedy cross-process comparison shared its first 25 output
tokens with the default-off run and then selected a different valid
continuation. Because that comparison crossed server restarts, it is not a
valid operator-level numerical oracle; a second default-off restart intended
to measure restart variance was blocked when an unrelated TP8 service acquired
the GPUs. The feature therefore remains default-off until same-process model
output or logit equality is closed. No quality-pass claim is made from the
semantic smoke alone.

## Exact 8K Chunk Comparison

The next workload uses exactly 8192 prompt tokens, one officially sampled
output token, TP8, `fp8_ds_mla`, no prefix cache, no MTP, and non-eager
breakable CUDA Graph execution. The benchmark prompt builder preserves its
existing token prefix and repeats the already-tokenized context only when the
requested length exceeds the original 80-paragraph fixture.

The first 4096-token chunk request exposed a correctness bug before timing:
the SM70 indexer dequantization kernel has a `uint8` AOT signature and performs
software E4M3 decoding, but its runtime call passed the same storage with a
native `float8_e4m3fn` type. Triton rejects that native FP8 type on SM70 before
entering the kernel. Passing a zero-copy `uint8` view at the call boundary
preserves every value and scale. A direct GPU oracle covering weighted-Q,
software FP8 K dequantization, and FP16 HMMA was bitwise equal to the reference
with zero mismatched elements.

The endpoint comparison measured:

| Chunk | Grouped prefill | Median TTFT | Mean TTFT | Prompt throughput |
| ---: | :---: | ---: | ---: | ---: |
| 4096 | off | 7458.383 ms | 7457.075 ms | 1098.36 token/s |
| 4096 | group-64 | 3179.100 ms | 3178.883 ms | 2576.83 token/s |
| 8192 | group-64 | 2997.086 ms | 2997.686 ms | 2733.32 token/s |

Each row uses five measured requests after a cold request and two warmups. The
4096 rows used `gpu_memory_utilization=0.90`. An 8192-token profile run leaves
only 2.92 GiB for KV at that setting, below the 3.32 GiB startup admission
requirement for `max_model_len=10240`, so the 8192 row used 0.95 and retained
15,667 KV tokens. GPU memory utilization changes the allocated KV pool rather
than the active 8192-token execution, but a strict 0.95-versus-0.95 endpoint
repeat remains outstanding. Subject to that caveat, one 8192-token chunk is
5.73% faster than two 4096-token chunks.

The exact 8192-token, 49,152-routed-row operator gate passed before the model
run. Group-64 W13 measured 5.124 ms versus 54.956 ms legacy (10.72x); W2
measured 3.237 ms versus 14.504 ms (4.48x). Both stages were cross-route and
repeated-run bitwise equal over the complete output buffer.

An official-sampling 8192/64 request completed all 64 tokens with 20.330 ms
mean decode interval, but its continuation ended in an unrelated TypeScript
fragment. It is not a text-quality pass. The speed route remains experimental
until the same prompt and seed are compared with the 4096-token chunk path and
the broader model-quality investigation is closed.

Raw artifacts include:

- `baseline-c4096-seed820{1..5}-fixed-i8192-o1.json`
- `group64-c4096-seed840{1..5}-i8192-o1.json`
- `group64-c8192-u095-seed860{1..5}-i8192-o1.json`
- `microbench-grouped-prefill-chunk8192-random-b64.log`
- `group64-c8192-u095-quality-i8192-o64.json`

They are retained in
`/home/fudanwl/v100-worktrees/runs/dsv4-prefill-trace-20260803/`.

As a same-source preliminary reference, an earlier no-MTP run used runtime
source `6f946b603a`, the same TP8 and KV-cache configuration, and 256 output
tokens. Its exact 1024-token prompt had these request TTFT values:

| Seed | TTFT (ms) |
| ---: | --------: |
| 4201 | 1818.623 |
| 4202 | 1826.996 |
| 4203 | 1817.802 |

The median is 1818.623 ms, or 563.06 prompt tok/s at request level. This is a
reference rather than the final task-owned result because the requested
prefill contract uses one output token.

## Sparse-Prefill HMMA Candidate

The exact TP8 8192-token Nsight Systems trace identified sparse gathered
attention as the next dominant category: 1600.504 ms over 43 launches, or
53.33% of critical-rank kernel service. The layer split was 25.593 ms for two
SWA layers, 1142.831 ms for 21 C4 layers, and 432.080 ms for 20 C128 layers.
The C4 mean of 54.421 ms was reproduced by the standalone exact-shape
benchmark at about 54.2 ms, so the microbenchmark is representative of the
model path.

Nsight Compute rejected further Triton launch tuning as the primary route.
The C4 kernel used 255 registers per thread and 49.15 KiB dynamic shared
memory, reached 12.49% achieved occupancy, and had no eligible warp in 91.69%
of scheduler cycles. DRAM throughput was only 5.09%, while L1/TEX throughput
was 96.12%. It accumulated 4.868 billion shared bank conflicts and 6.235
billion shared wavefronts; MIO throttle was the dominant issue stall. SASS
contained no HMMA instructions and lowered the two `tl.dot` operations to
shared-memory FFMA sequences.

Shallow shape tuning did not repair the lowering. `BLOCK_H=4/2/1` and eight
warps were slower. Padding the eight valid heads to `BLOCK_H=16` remained
bitwise equal but measured 128.010 ms with eight warps and 1313.295 ms with
four warps, versus about 63.6 ms for the control in that clock state. These
variants are rejected.

The candidate is a fused SM70 CUDA kernel with one CTA per query and eight
warps:

1. Eight warps split the 512-dimensional QK reduction into 64 dimensions each
   and use Volta WMMA/HMMA. Each warp also owns 64 PV output dimensions.
2. The CTA gathers each 16-key KV tile once with aligned `half2` loads and
   reuses it for both QK and PV.
3. Q is staged once per CTA instead of being copied again for every key tile.
4. QK and probability rows use padded shared-memory strides. QK partial and PV
   scratch storage overlap because their lifetimes do not.
5. Online softmax keeps the original 16-key update order and converts
   probabilities to FP16 before PV. The changed HMMA reduction tree is the
   only intended numerical difference from the Triton path.

The final object contains 128 static HMMA instructions, uses 110 registers per
thread and 46.27 KiB dynamic shared memory, and has no local-memory spills. It
supports all three gathered layouts used by the model. Same-process 8192-query
microbenchmarks measured:

| Layer pattern | Triton median | HMMA median | Speedup | Max abs error |
| :--- | ---: | ---: | ---: | ---: |
| C4, width 640 | 54.211 ms | 20.337 ms | 2.666x | 0.0009765625 |
| C128, width 256 | 21.636 ms | 7.264 ms | 2.979x | 0.0009765625 |
| SWA, width 128 | 10.918 ms | 4.401 ms | 2.481x | 0.0009765625 |

All outputs were finite. GPU tests cover widths 128, 256, and 640 and accept at
most one FP16 ULP. The candidate is not bitwise equal to the scalar-FFMA Triton
route.

The final C4 Nsight Compute profile measured 23.31 ms, 24.92% achieved
occupancy, 98.16 GB/s memory throughput, 73.29% scheduler cycles with no
eligible warp, 5.58 long-scoreboard cycles per issued instruction, and 0.99
barrier cycles per issued instruction. The final warp-softmax implementation
executes 2.549 billion instructions; its lower barrier cost is partly offset by
shuffle instructions, so further softmax-only work is at marginal return.

The source-level profile attributed 95.6% of the intermediate eight-warp
kernel's long-scoreboard samples to the KV global-load-to-shared-store chain.
Staging Q once and vectorizing KV movement removed 1.205 billion executed
instructions from that intermediate version. The remaining 293.6 million
excessive shared wavefronts are predominantly inside Volta WMMA fragment
loads and stores. Bypassing those stores requires fixed SM70 fragment or SASS
mapping and remains a separate high-risk experiment.

The same-binary, default-off endpoint control measured 8K TTFT values of
3030.361, 2993.855, 2997.582, 2991.335, and 2990.318 ms after cold JIT and two
warmups. Its median is 2993.855 ms, or 2736.27 prompt token/s. The final HMMA
candidate measured 1900.654, 1901.288, 1889.743, 1889.015, and 1891.466 ms.
Its median is 1891.466 ms, or 4331.03 prompt token/s. TTFT falls by 1102.389 ms
or 36.82%; request-level prompt throughput rises by 58.28%. These are
unprofiled endpoint measurements rather than a service-time projection.

### HMMA Quality Blocker

The performance path remains default-off. Official-sampling tests use the
model's `temperature=1.0`, `top_p=1.0`, and natural stopping. Both paths were
coherent for the clean PagedAttention material at seed 9601, but the candidate
leaked source text at seed 9602 while the repeated baseline stayed on topic.
At the synthetic 8192/256 seed 9202 workload, two independent baseline starts
produced coherent Chinese analyses; two candidate starts produced an English
paper excerpt or prompt-template leakage. A one-ULP operator bound therefore
does not close the model-quality gate.

Raw artifacts are retained as:

- `microbench-sparse-prefill-hmma-8warp-vectored-{c4,c128,swa}-q8192.json`
- `ncu-hmma-vectored-c4-q8192.ncu-rep`
- `hmma-vectored-candidate-seed970{3..7}-i8192-o1.json`
- `hmma-{baseline-repeat,vectored-candidate}-quality-*.json`

The release-guarded binary hash for these results is
`ac8fd1f20d289e947517636310f2b5c10c72b11d3f6d31a204035121e38d7af2`.

## mHC Prenorm Weight Reuse

After the sparse-attention reduction, the exact 8K trace exposed mHC as the
next target: 302.284 ms of critical-rank service, including 196.032 ms over 86
launches of `hc_prenorm_gemm_block_m_tilelang_kernel`. Its baseline NCU report
showed 85.75% L2 throughput, 91.43% L2 hit rate, only 22.14% DRAM throughput,
and 18.26 long-scoreboard cycles per issued instruction. The repeated 1.5 MiB
FP32 `fn` weight was being fetched from L2 once per two-token CTA block.

The SM70 candidate changes the prenorm tile from `(block_m, tile_n)=(2,12)` to
`(4,6)`. Both shapes keep 24 FP32 accumulators per thread and the same 8192 CTA
count at M=8192, but the candidate reuses each weight load across four tokens.
It is guarded by `VLLM_SM70_DSV4_MHC_PREFILL_WEIGHT_REUSE=1`, requires the
DeepSeek V4 FP16 shape, and remains default-off until endpoint validation.

Clean CUDA Graph microbenchmarks measured:

| Tokens | Baseline `(2,12)` | Candidate `(4,6)` | Reduction | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.3359 ms | 0.2422 ms | 27.91% | 1.387x |
| 4096 | 1.1716 ms | 0.9071 ms | 22.57% | 1.292x |
| 8192 | 2.3161 ms | 1.7876 ms | 22.82% | 1.296x |

All candidate outputs were bitwise equal to both the runtime oracle and eager
execution. The non-divisible M=1025 tail also passed bitwise. At M=8192, the
86-launch model-forward projection falls from 199.186 to 153.732 ms, a 45.454
ms saving; this is a projection, not an endpoint result.

NCU confirms that the candidate reduces L2 work without sacrificing residency:

| M=8192 metric | Baseline `(2,12)` | Candidate `(4,6)` |
| :--- | ---: | ---: |
| NCU duration | 2.85 ms | 1.89 ms |
| Registers/thread | 55 | 56 |
| Achieved occupancy | 49.35% | 48.71% |
| L2 throughput | 85.75% | 86.36% |
| L2 hit rate | 91.43% | 73.57% |
| DRAM throughput | 22.14% | 63.93% |
| Compute throughput | 26.68% | 39.44% |
| Scheduler cycles with no eligible warp | 73.76% | 60.87% |

Larger reuse blocks were rejected after clean reruns: `(5,5)` measured 2.009
ms, `(6,4)` 2.204 ms, and `(8,3)` 3.109 ms. The first two already require 64
registers per thread; added x rereads and dependency stalls outweigh further
weight reuse even though they retain two CTAs per SM.

### Remaining mHC Kernels

The other two repeated mHC kernels account for another 104.653 ms in the 8K
trace. `mhc_post_tilelang_kernel` reproduces at 0.7437 ms per call and reaches
805.47 GB/s, or 89.71% DRAM throughput. Its mandatory 40 KiB input and 32 KiB
output per token make standalone tuning a sub-10% opportunity, so it is not
the next implementation target.

`mhc_pre_big_fuse_with_norm_tilelang_kernel` accounts for 40.669 ms over 85
calls. NCU measures 79.22% DRAM throughput, 103 registers per thread, 29.04
KiB dynamic shared memory, and only 12.32% achieved occupancy. Its 1024-wide
software-pipelined tile double-buffers residual and norm-weight staging.
Temporary 512- and 256-wide benchmark variants measured only 0.36% and 1.69%
faster, respectively, while both changed `layer_input` by up to 0.00012207.
They are rejected and the production tile remains 1024.

Raw artifacts are retained as:

- `microbench-mhc-prenorm-m{1024,4096,8192}-idle-repeat-*.json`
- `microbench-mhc-prenorm-m1025-tail-blockm4-tilen6.json`
- `microbench-mhc-prenorm-m4096-production-dispatch-weight-reuse.json`
- `ncu-mhc-prenorm-m8192-weight-reuse.ncu-rep`
- `ncu-mhc-prenorm-m8192-blockm{5,6}-*-resources.ncu-rep`
- `ncu-mhc-{post,pre-with-norm}-m8192-baseline.ncu-rep`

## Indexed MXFP4 W13 Prefill

The final 8K trace attributes 61.950 ms over 43 launches to
`expandInputRowsKernel`, or 1.441 ms per MoE layer. It materializes a
`[8192 * 6, 4096]` FP16 matrix before W13 even though every destination row is
an unchanged source-token row. This is a routing algorithm cost rather than a
GEMM tile-selection problem.

TurboMind's existing SM70 grouped MXFP4 kernels already instantiate an
indexed-A iterator. The candidate therefore keeps the same W13 kernel and
arithmetic but supplies the sorted source-token row map directly:

1. CUB still sorts the same expanded `(token, expert-slot)` IDs and produces
   the same expert offsets.
2. A 256-thread metadata kernel builds the existing forward and inverse
   permutation maps and converts each sorted expanded ID to its source-token
   row. It does not copy activation data.
3. Grouped W13 reads `x[source_row, :]` through TurboMind's indexed-A iterator.
4. SwiGLU, W2, weighted unpermute, FP16 accumulation order, and router weights
   are unchanged.

The route is guarded by `VLLM_SM70_MXFP4_MOE_INDEXED_PREFILL=1`, also requires
grouped prefill, at least 1024 prompt tokens, top-k 6, and fully replicated 256
experts. Expert-parallel and short/decode shapes remain on the materialized
path. The switch remains default-off pending an unprofiled endpoint run.

Clean full `permute + W13` microbenchmarks measured:

| Prompt tokens | Materialized | Indexed | Speedup | 43-layer projected saving |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 3.258 ms | 3.065 ms | 1.063x | 8.319 ms |
| 1025 tail | 3.213 ms | 3.052 ms | 1.053x | 6.921 ms |
| 4096 | 3.817 ms | 3.119 ms | 1.224x | 30.000 ms |
| 8192 | 7.395 ms | 6.001 ms | 1.232x | 59.928 ms |
| 8192, second seed | 7.390 ms | 5.977 ms | 1.237x | 60.771 ms |

All rows passed bitwise output equality, bitwise expert offsets, forward and
inverse permutation equality, source-row-map equality, and bitwise CUDA Graph
replay. The clean M=8192 result recovers about 96.7% of the 61.950 ms trace
cost; indexed W13 adds only about 0.05 ms per layer relative to removing the
copy in isolation.

The validated extension hashes are:

- `_C.abi3.so`: `fe03604ac5f681a9c4ca5be4443d271749c79dfcd67a8abd64bb3f363caafa77`
- `_moe_C.abi3.so`: `45c69eba91875db122a61c371c35b6a840c566a4d170ffcaa1fd74ce5f42ee61`

Raw artifacts are retained as
`microbench-mxfp4-indexed-prefill-m*.json`,
`indexed-prefill-*-peer-util.log`, and `build-indexed-prefill.log` under the
task run directory. The next gate is a matched 8K TP8 endpoint comparison with
the sparse-HMMA and mHC switches held fixed.

## Exact MXFP4 W2 Prefill Selector

The grouped MXFP4 stage was retuned only after the indexed-W13 change. Generic
autotuning selected CTA 16x128x32 for W13, but grouped W13 regressed from 5.045
to 5.261 ms, so that result is rejected. For W2, changing only the swizzle of
the existing CTA 128x128x16, split-1 kernel from 0 to 4 is consistently faster.

`VLLM_SM70_MXFP4_MOE_PREFILL_FAST_SELECTOR=1` enables the exact
`49152x4096x256` TP8 descriptor only when group-64 prefill is also enabled.
This combined gate matters because swizzle 4 improves grouped execution but
regresses the per-expert route. All other shapes retain normal dispatch.

Same-binary A/B/A grouped-only measurements produced:

| Routing | Baseline mean | Swizzle 4 | Reduction | 43-layer saving |
| :--- | ---: | ---: | ---: | ---: |
| balanced | 3.578 ms | 3.138 ms | 12.29% | 18.91 ms |
| random | 3.574 ms | 3.134 ms | 12.32% | 18.94 ms |
| half-active | 2.974 ms | 2.469 ms | 17.00% | 21.74 ms |

Every route matches the original tactic's cross-process output hash and is
bitwise stable across direct repeats and CUDA Graph replay. The validated
binary hash is
`dba88d00245f6ace0d56fac66e1b8c54ac309512dd6649f812ada68390759c88`.
Raw artifacts are retained as
`microbench-mxfp4-w2-m8192-{grouped-only-baseline,grouped-only-baseline-repeat,grouped-only-fast-selector-graph}.json`.

## Exact FP8 Dense Prefill Selector

The 8K trace attributes 315.809 ms to dense FP8 GEMMs. A clean exhaustive
tactic search found faster, bitwise-identical choices for five exact DeepSeek
V4 TP8 shapes at M=8192. `VLLM_SM70_FP8_PREFILL_FAST_SELECTOR=1` selects only
those descriptors; all other token counts and shapes retain the normal
selector. The switch remains default-off until endpoint validation.

| Projection | Exact tactic |
| :--- | :--- |
| fused WQA/WKV, 8192x1536x4096 | CTA 128x128x16, split 1, swizzle 4 |
| WQ-B/WO-B, 8192x4096x1024 | CTA 128x128x16, split 1, swizzle 4 |
| WO-A group, 8192x1024x4096 | CTA 128x128x16, split 1, swizzle 3 |
| shared down, 8192x4096x256 | CTA 128x128x16, split 1, swizzle 4 |
| C4 indexer WQ-B, 8192x8192x1024 | CTA 64x256x16, split 1, swizzle 2 |

The shared gate/up shape is deliberately excluded. Its faster measured
tactics changed the FP16 output hash, including CTA 64x256x16 and several
128x128/96x128/64x128 alternatives. It therefore stays on the baseline CTA
128x256x16, split-8 route.

A same-binary A/B/A CUDA Graph run measured a 328.108 ms selector-off
projection and 311.574/312.763 ms selector-on projections. The candidate mean
is 312.168 ms, saving 15.939 ms or 4.86% of this dense substage. All six output
hashes match the paired baseline, direct versus graph output is bitwise equal,
and the other seven GPUs stayed idle during measurement. The final independent
selector check used binary
`dba88d00245f6ace0d56fac66e1b8c54ac309512dd6649f812ada68390759c88`.

Raw artifacts are retained as
`microbench-fp8-dense-m8192-{paired-baseline,fast-selector,fast-selector-repeat}.json`.

## Bitwise-Safe Combined 8K Endpoint

The final endpoint comparison keeps sparse HMMA disabled because its text gate
is unresolved. Both routes use TP8, one 8192-token chunk, `fp8_ds_mla`, no
prefix cache, no MTP, breakable CUDA Graph, group-64 MXFP4 prefill, official
`temperature=1.0`/`top_p=1.0`, and one sampled output token. The candidate
adds only mHC weight reuse, indexed W13, the exact W2 selector, and the exact
FP8 dense selector.

One cold request and two warmups precede five measured requests in each run:

| Route | Median TTFT | Prompt throughput |
| :--- | ---: | ---: |
| selector-off control A | 2993.893 ms | 2736.24 token/s |
| four-way candidate | 2858.855 ms | 2865.48 token/s |
| selector-off control A repeat | 2996.340 ms | 2734.00 token/s |

The candidate saves 135.038-137.485 ms against the two controls. TTFT falls by
4.51-4.59%, and request-level prompt throughput rises by 4.72-4.81%. This is a
measured endpoint result rather than a sum of microbenchmark projections. All
eight paired one-token seeds returned the same token in control and candidate.

The longer text gate does not support a stronger quality claim. At 8192/64,
the candidate and control chose different continuations after the same first
token. More importantly, the selector-off control itself produced different
continuations for two repeated requests with the same seed. It remained
non-reproducible with `temperature=0` and the same seed, and both control and
candidate samples could leak prompt-like instructions or English text. The
operator and first-token evidence shows no detected regression from these
four changes, but the model's endpoint text-quality baseline is not closed.
All four switches therefore remain default-off.

Raw endpoint artifacts are retained as
`final-safe-{baseline,baseline-repeat,combined}-seed*-i8192-o1.json` and
`final-safe-{baseline-repeat,combined}-quality-seed8701-i8192-o64.json`.

## Rejected Evidence

The earlier report
`dsv4-combined-latest-graphtrace-20260803/graph_node_combined_i1024_o64`
cannot serve as the full-prefill baseline. That server had prefix caching
enabled, and the captured 1024-token request repeated the warmup prompt. The
trace therefore measured a partial prefix hit: request TTFT was 1039.958 ms
and the critical-rank GPU envelope was 1019.709 ms. Its kernel data remains
useful only for validating the SQLite parser and category rules.

The prefix-cache diagnosis is independently visible in same-source request
results: a repeated 1024-token prompt with cache enabled measured about
932.83-933.25 ms TTFT, while the no-prefix request measured 1872.07 ms. The
new trace must therefore keep prefix caching disabled.

## Runtime Setup Notes

- A task-private empty TileLang cache exposed two startup prerequisites before
  performance measurement: set `TILELANG_TARGET=cuda` (resolved as `sm_70`)
  and point `CUDA_HOME` at the Conda CUDA 12.8 toolkit containing `nvcc`.
- Two attempted launches correctly aborted when another registered TP8 task
  acquired the GPUs first. These are ownership failures, not model startup or
  performance regressions, and their logs are retained with the raw artifacts.
