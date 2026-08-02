# SM70 DeepSeek V4 TP8 Two-CTA All-Reduce

## Scope

This experiment targets the exact DeepSeek-V4-Flash TP8 decode collective:
4096 FP16 elements, eight V100 GPUs, CUDA Graph execution, and the existing
two-clique hierarchical reduction order. It is gated by
`VLLM_SM70_TP8_HIERARCHICAL_TWO_CTA=1` and is default-off.

## Trace Rationale

The latest trace executes 87 hierarchical collectives per token. Their summed
rank service is about 4.50 ms/token, while MoE-preceded calls also expose about
1.95 ms/token of rank-arrival skew. The accepted one-CTA kernel assigns all 512
packed values to one SM. This experiment tests whether two independent CTAs
can hide peer-memory latency without changing arithmetic.

## Protocol

Each 256-thread CTA owns one half of the packed tensor and one independent
double-buffered signal counter. Both CTAs preserve global rank order when
forming the FP32 four-rank clique partial, exchange the FP32 partial with the
same paired rank, then perform the same final FP16 downcast. Signal and data
slots are not reused until the paired consumer acknowledges the matching CTA.

## Gates

1. Eight-rank initial and changed-input outputs must be bitwise equal to the
   one-CTA route.
2. The 87-call CUDA Graph stress must complete repeatedly without signal
   overtake or a GPU synchronization spin.
3. The joined projection/collective microbenchmark must reduce rank-max wall
   time before any full-model launch.
4. A candidate that only moves isolated service but not joined wall time is
   rejected.
