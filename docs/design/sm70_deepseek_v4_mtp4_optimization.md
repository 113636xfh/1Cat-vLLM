# DeepSeek V4 Flash MTP4 on SM70

## Scope

Optimize single-request MTP4 decode for `deepseek-ai/DeepSeek-V4-Flash` on
eight V100-SXM2-32GB GPUs. Correct output and a measured end-to-end gain over
the same-contract no-MTP route are mandatory. This work does not use eager
execution, Marlin, altered model weights, or reduced-precision shortcuts.

## Source And Runtime Contract

- Integration base: `agent/v100-dsv4-quality-rootcause-20260802`
- Base SHA: `a089aa6c22b9421f529dcaa27a2b59c769f9465f`
- Branch: `agent/v100-dsv4-mtp4-20260802-132909`
- Worktree: `worktrees/v100-dsv4-mtp4-20260802-132909`
- Model: `/home/fudanwl/Desktop/dir`, TP8 on eight V100-SXM2-32GB GPUs
- Weights: MXFP4 routed experts and FP8 dense layers
- KV cache: `fp8_ds_mla`
- Decode: CUDA Graph enabled, `max_num_seqs=1`, no prefix cache, no eager
- MTP: native one-layer checkpoint, four serial draft steps, verifier width 5
- Drafter attention: `FLASH_ATTN_V100`
- Initial workload: exactly 1024 prompt tokens and at most 256 output tokens
- Sampling: official `temperature=1.0`, `top_p=1.0`; natural EOS is preserved
- Initial context limit and token budget: 4096
- Active-expert candidate: disabled during the first MTP/no-MTP comparison

The checkpoint index contains 4,705 `mtp.0.*` tensors. Its single MTP layer is
reused autoregressively for four dependent draft positions; MTP4 does not
require four independent checkpoint layers.

## Acceptance Gates

1. Worker logs must prove native MTP, `FLASH_ATTN_V100`, FP8 DS MLA KV,
   TurboMind dense/MXFP4 dispatch, TP8, and CUDA Graph M=5 capture.
2. A deterministic request must be repeatable after the KV RoPE race fix.
3. Official-sampling output must be coherent and stop naturally. Report mean
   acceptance length, per-position acceptance, and rejection distribution.
4. Report TTFT, steady TPOT, and output throughput separately for no-MTP and
   MTP4 under the exact same contract.
5. A candidate is accepted only when output quality remains valid and its
   unprofiled end-to-end TPOT improves. Profile-only service-time reductions
   are not performance claims.

## Measurement Order

1. Establish unprofiled no-MTP and MTP4 quality and speed baselines.
2. Enable the existing synchronized MTP instrumentation to split target
   forward, target logits/rejection sampling, state updates, four drafter
   forward/sample steps, bookkeeping, and host wall time.
3. Capture a focused Nsight Systems CUDA Graph node trace for the critical TP
   rank and keep an explicit unattributed residual.
4. Use Nsight Compute only on a confirmed hot kernel. Record its exact shape,
   duration, occupancy, registers, shared memory, memory/SM throughput, and
   dominant stalls.
5. Reject or admit each optimization with the smallest exact-shape
   microbenchmark, then rerun the full quality and endpoint benchmark.

## Baseline

| Route | TTFT | Steady TPOT | Throughput | Acceptance length | Quality |
| --- | ---: | ---: | ---: | ---: | --- |
| TP8 no-MTP, current source | Pending | Pending | Pending | N/A | Pending |
| TP8 MTP4, current source | Pending | Pending | Pending | Pending | Pending |

The older pre-fix no-MTP reference was 134.143 ms TPOT (7.455 tokens/s), but
it is not a valid MTP speed comparison until both routes are rerun from this
source with the fixed contract above.

## Experiment Log

| Date | Change or test | Result | Decision |
| --- | --- | --- | --- |
| 2026-08-02 | Audit checkpoint and local MTP route | One complete MTP predictor layer is present; MTP4 serially reuses it. Existing instrumentation covers verifier, drafter steps, sampling, state, and bookkeeping. | Establish the exact unprofiled baseline before changing kernels. |
| 2026-08-02 | First TP8 MTP4 startup | Startup incorrectly requested the Qwen3.6-27B TP8 dynamic-vocabulary asset before loading weights. Only Qwen dense TP2/TP4 assets exist, and their ranking is not valid for DeepSeek V4. | Scope the automatic reduced-vocabulary route to its validated Qwen architecture and TP sizes. DeepSeek V4 starts with the full draft vocabulary. |

## Artifacts And Handoff

- Remote source: `/home/fudanwl/v100-worktrees/deepseek-v4-mtp4-a089aa6c22`
- Compiler caches: `/home/fudanwl/v100-worktrees/cache/dsv4-mtp4-a089`
- Run artifacts: `/home/fudanwl/v100-worktrees/runs/dsv4-mtp4-baseline-20260802`
- API port: `18082`

Record every launch command, source hash, output token IDs, metrics snapshot,
profile path, failed experiment, and active process in the run artifact tree.
