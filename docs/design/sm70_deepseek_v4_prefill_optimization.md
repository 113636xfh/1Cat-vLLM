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

The task-owned `1024/1` baseline and no-prefix trace are pending. Raw artifacts
will be retained under
`/home/fudanwl/v100-worktrees/runs/dsv4-prefill-trace-20260803/`.

As a same-source preliminary reference, the pre-existing no-MTP run used
runtime source `6f946b603a`, TP8, `fp8_ds_mla`, prefix caching disabled,
`max_model_len=2048`, and `max_num_batched_tokens=2048`. Its exact 1024-token
prompt had these request TTFT values with 256 output tokens:

| Seed | TTFT (ms) |
| ---: | --------: |
| 4201 | 1818.623 |
| 4202 | 1826.996 |
| 4203 | 1817.802 |

The median is 1818.623 ms, or 563.06 prompt tok/s at request level. This is a
reference rather than the final task-owned result because the requested
prefill contract uses one output token.

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
