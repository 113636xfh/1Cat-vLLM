# SM70 DeepSeek V4 MXFP4 Active-Expert Optimization

## Scope

- Base: `dd462e37f2552f3e038f1ed7128e62bd7b4ab0d7`
- Dependency: DeepSeek V4 SM70 bring-up PR #159
- Model: `deepseek-ai/DeepSeek-V4-Flash`, MXFP4 experts
- Runtime: 8 x V100-SXM2-32GB, TP8, FP16 activations, `fp8_ds_mla` KV
- Quantization backend: TurboMind only; Marlin is out of scope
- Decode mode: CUDA Graph enabled, no MTP, no eager execution

The first optimization target is the graph-safe batch-one MXFP4 MoE path. The
change must preserve expert routing, accumulation order, output dtype, graph
replay stability, and official-sampling output quality.

## Baseline Evidence

The accepted trace request used exactly 1024 prompt tokens and 256 generated
tokens with `temperature=1.0`, `top_p=1.0`, and natural EOS behavior. It
completed all 256 tokens without malformed output.

Raw artifacts are retained on the profiling host at:

```text
/home/fudanwl/v100-worktrees/runs/
  dsv4-tp8-nsys-i1024-o256-retry1-20260802/
```

The Nsight Systems report contains 255 decode replays per rank: the first
emitted token comes from prefill, followed by 255 decode forwards. The parser
drops four fill/drain steps at each edge and aggregates 247 steady steps.

| Item | Mean per token | Notes |
|---|---:|---|
| Node-trace TPOT | 149.687 ms | Composition only; CUPTI adds overhead |
| TP rank interval max | 150.183 ms | Replay-to-next-replay |
| Rank-average GPU service | 130.080 ms | Categories sum exactly to this value |
| TurboMind MXFP4 MoE | 54.594 ms | 22,016 launches/rank/token |
| SM70 sparse MLA attention | 46.880 ms | 43 launches/rank/token |
| TP all-reduce | 10.154 ms | Overlaps other streams; not additive wall |
| TurboMind FP8 dense | 6.064 ms | 279 launches/rank/token |

The prior unprofiled artifact was labeled 1024 tokens but the serving
tokenizer reports 1020. Its 134.143 ms TPOT remains useful as a provisional
speed reference only. An exact-1024 unprofiled baseline is required before an
end-to-end speed claim.

## Root Cause

The graph-safe path launches both MXFP4 stages for every local expert:

```text
43 layers * 256 experts * 2 stages = 22,016 launches/token/rank
```

On rank 0, 5,476,780 of 5,614,080 MXFP4 decode launches (97.55%) finish in
less than 2.5 microseconds. Those empty or near-empty launches consume
42.738 ms/token of traced GPU service. Approximately 538 launches/token do
material work, close to the `43 * top_k(6) * 2 = 516` active-expert bound.

This makes active-expert device-side dispatch or a persistent grouped stage a
higher-value target than tuning the existing per-expert GEMM tile.

## Optimization Gate

1. Reproduce the exact DeepSeek V4 shapes in an operator microbenchmark.
2. Compare against the current graph-safe dense-expert path with fixed routing.
3. Verify numerical output against the current FP16 output, including repeated
   expert IDs and all six routed slots; do not change quantization or precision.
4. Prove CUDA Graph capture/replay with routing IDs changed between replays.
5. Require fewer expert-stage launches and lower CUDA-event wall time.
6. Run the full-model 1024/256 official-sampling quality gate.
7. Accept an end-to-end claim only from an unprofiled same-contract A/B.

If active-expert dispatch cannot remain graph-safe or its numerical result
changes, reject it and move to the next trace hotspot rather than weakening the
quality contract.

## Rejected Profiling Paths

- The old token parser grouped TP ranks by an 8 ms time window and recognized
  only `cudagraph.FULL.replay`; that is invalid for this TP8 Breakable Graph
  trace. The profiling skill now aligns each worker's Nth replay by ordinal.
- `nsys stats cuda_gpu_kern_sum` reached 77 GB RSS on the 49-million-kernel
  trace and was stopped before OOM. SQLite streaming aggregation produced the
  accepted table with bounded memory.
