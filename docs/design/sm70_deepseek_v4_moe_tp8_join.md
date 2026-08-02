# SM70 DeepSeek V4 MoE TP8 Join

## Scope

This route targets batch-one DeepSeek-V4-Flash decode on eight V100 GPUs. It
keeps TurboMind MXFP4, FP8 shared experts, CUDA Graph execution, top-k=6, and
the existing hierarchical TP8 all-reduce. It is stacked on the active-expert,
FP8 dense, hierarchical all-reduce, and sparse-MLA QK-D worktrees.

## Trace Diagnosis

The 64-token graph-node trace contains 87 all-reduces per token. The 44
elementwise-preceded calls have 44.333 us mean input-ready skew, versus 18.527
us for the 43 FP8-GEMM-preceded calls. The former are the shared+routed MoE
boundary. Their 43 steady MoE segments average 167.386 us/layer, including
43.011 us of routed MXFP4 GEMM and 28.022 us of shared FP8 GEMM.

The generic shared-expert auxiliary stream is active. An exact-shape graph
screen measures 110.653 us/layer when serialized and 98.318 us/layer with the
existing overlap, so disabling or replacing that overlap is not useful.

## Candidate

The direct top-6 MXFP4 path now reuses the exact single-token weighted-reduce
kernel. When `VLLM_SM70_MXFP4_MOE_FUSED_SHARED_REDUCE=1`, the same kernel also
performs the following FP16 shared+routed add before writing the all-reduce
input. W13, SwiGLU, W2, route order, FP32 FMA order, FP16 downcast point, and
the hierarchical cross-rank reduction order remain unchanged.

The generic `FusedMoEMethodBase` finalization hook is a no-op for every other
quantization method. The MXFP4 hook is restricted to batch one, direct top-6,
fully replicated experts, a present shared expert, and the new operator.

## Microbenchmarks

Single-GPU exact-shape CUDA Graph:

| Route | Median per layer | Change from current overlap |
|---|---:|---:|
| Current unpermute + add | 97.861 us | - |
| Exact weighted-reduce + add | 94.537 us | -3.323 us |
| Fused weighted-reduce-add | 92.620 us | -5.240 us |

The fused route projects to 0.225 ms/token over 43 layers. Its output is
bitwise equal to generic `moe_unpermute` followed by FP16 add.

The eight-rank joined graph includes the MoE tail and hierarchical all-reduce:

| Route | Rank-max median per layer | 43-layer projection |
|---|---:|---:|
| Unpermute + add + hierarchical AR | 118.437 us | - |
| Fused reduce-add + hierarchical AR | 113.899 us | -0.195 ms/token |

Initial and changed-input graph replays are bitwise equal to the control on
all eight ranks. The focused CUDA test also passes with zero tolerance.

## Rejected Paths

| Path | Result | Decision |
|---|---:|---|
| Main stream priority -1/-2 | At most 0.004 ms/token projected | Reject |
| One-CTA hierarchical sum2 | 18.096 to 20.150 us/call | Remove; local add and fence dominate |
| Fused reduce-add, 128 threads | Lower than 256-thread benefit | Reject |
| Fused reduce-add, 64 threads | 0.191 ms/token in TP8 join | Reject; keep 256 |

## Artifacts

```text
/home/fudanwl/v100-worktrees/runs/dsv4-tp8-latest-graphtrace-20260803/
/home/fudanwl/v100-worktrees/runs/dsv4-shared-moe-priority-micro-20260803/
  screen_v3_fused_reduce_add.json
  moe_tp8_join_v1.json
  moe_tp8_join_t64.json
  tp8_hierarchical_sum2_v1.json
```

## Remaining Gates

1. Prove the fused route is selected inside the real model CUDA Graph.
2. Run matching 1K/64 low-overhead endpoint timing.
3. Run the model-level official-sampling quality gate before defaulting it.
