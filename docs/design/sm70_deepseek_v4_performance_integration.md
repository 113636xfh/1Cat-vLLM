# SM70 DeepSeek V4 Performance Integration

## Source Composition

This Draft provides one reproducible Git tree for the current DeepSeek V4
Flash SM70 work. It starts from `onecat/main` at
`270a468d112a628182c748c171ad002f44d79b21`, which already contains PRs
#159, #160, and #162, then integrates:

| PR | Scope |
|---|---|
| #165 | DeepSeek V4 DSpark N7 support |
| #170 | Compact MXFP4 decode and skew-safe TP8 graph all-reduce |
| #171 | Sparse MLA split-K and QK dimension split |
| #175 | Exact SM70 FP16 GEMV and mHC FP32 staging |

The component PRs remain the review and rollback boundaries. This branch does
not replace them and must not be merged before their required quality gates.

## Endpoint Evidence

The matched TP8, 8 x V100, 1024-input/256-output, FP8 MLA KV, no-MTP, CUDA
Graph endpoint A/B recorded by PR #175 measured:

| Route | Median TPOT | Decode throughput |
|---|---:|---:|
| Combined stack, mHC route off | 20.764 ms/token | 48.16 token/s |
| Combined stack, mHC route on | 19.366 ms/token | 51.64 token/s |

This merge-only integration tree has not been rebuilt and rerun as one SHA.
That exact-tree endpoint and official-sampling quality run remain mandatory
before promotion.

## Excluded Drafts

PR #178 compressor state-save fusion and PR #179 MXFP4 grouped prefill remain
default-off WIP screens and are intentionally excluded. Negative benchmark
PRs remain separate documentation and do not alter this runtime tree.
