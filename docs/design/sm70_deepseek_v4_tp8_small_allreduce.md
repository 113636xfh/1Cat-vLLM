# SM70 DeepSeek V4 TP8 Small All-Reduce

## Scope

DeepSeek-V4-Flash TP8 batch-one decode performs 87 FP16 all-reduces per token.
The traced payload is 4096 elements, or 8 KiB, and NCCL selects `RING_LL`.
The latest graph-node trace attributes 4.176 ms/token of GPU service to these
calls, with a mean of 47.996 us per launch.

The target host has a DGX-1-style hybrid topology. It is not fully connected,
but every rank has NVLink peers and CUDA P2P validation decides whether all
rank pairs are accessible. vLLM's generic custom all-reduce rejects this
topology before running the P2P test.

## Candidate

`VLLM_SM70_TP8_NONFULL_CUSTOM_AR=1` is an explicit, default-off experiment. It
only relaxes the topology gate for SM70, TP8, non-fully-connected groups. The
normal all-pairs P2P test remains mandatory. The existing one-stage custom
kernel and CUDA Graph buffer registration are reused unchanged.

The acceptance gate is the exact 4096-element FP16 graph benchmark:

1. output must equal the sum of ranks;
2. rank-max latency must beat NCCL across repeated runs;
3. projected saving across 87 calls must exceed 0.2 ms/token;
4. a later full-model quality gate must pass before enabling the route.

The experiment remains default-off until the microbenchmark and accumulated
endpoint gate are complete.
