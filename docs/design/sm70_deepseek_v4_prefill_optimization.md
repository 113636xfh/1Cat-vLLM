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

Pending. Raw artifacts will be retained under
`/home/fudanwl/v100-worktrees/runs/dsv4-prefill-trace-20260803/`.
