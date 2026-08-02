# SM70 DeepSeek V4 TP8 Two-CTA All-Reduce

## Scope

This rejected experiment targeted the exact DeepSeek-V4-Flash TP8 decode collective:
4096 FP16 elements, eight V100 GPUs, CUDA Graph execution, and the existing
two-clique hierarchical reduction order. It is gated by
`VLLM_SM70_TP8_HIERARCHICAL_TWO_CTA=1`. The implementation and environment
gate were removed after the microbenchmark regression.

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

## Result

Both routes produced the same SHA-256 output on all eight ranks. The two-CTA
graph completed the short 87-call stress without signal overtake.

| Test | One CTA | Two CTA | Change |
|---|---:|---:|---:|
| Pure 87-call graph | 18.710 us/call | 19.879 us/call | +6.25% |
| Projection join + collective | 32.498 us/call | 35.035 us/call | +7.80% |

The joined regression projects to `+0.221 ms/token` over 87 calls. At this
8 KiB shape, duplicating clique and pair handshakes costs more than using a
second SM saves in peer-memory latency. The candidate was removed without a
full-model launch.

Artifacts:

```text
/home/fudanwl/v100-worktrees/runs/dsv4-tp8-two-cta-ar-micro-20260803/
  one_cta_short_pure.json
  two_cta_short_pure.json
  join_two_cta_0.json
  join_two_cta_1.json
```
