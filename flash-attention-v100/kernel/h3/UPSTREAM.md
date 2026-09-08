# H3 D128 non-causal FlashAttention-V100

This inference path uses SM70 CUTLASS TensorOp QK/PV with FP32 accumulation,
online max/sum and output state. Each MHA head is a separate grouped problem;
there is no GQA head packing, D256 padding or global square probability matrix.
The QK tile is 64 x 64 and the PV tile is 64 x 128. The original D128 scale
is passed in FP32. This is an independent FlashAttention-V100 operator and
does not call the FlashInfer backend.

`default_fmha.h` and `fmha.h` derive from NVIDIA CUTLASS
`examples/41_fused_multi_head_attention/{default_fmha_grouped.h,fmha_grouped.h}`
at revision `da5e086dab31d63815acafdac9a9c5893b1c69e2` (v4.4.2). Both retain
their BSD-3-Clause notices. Unmodified helper headers are build dependencies
from the same pinned CUTLASS source already fetched by 1Cat CMake.

`prefetch_mma.h` adapts the SM70 PV main loop from that revision's
`gemm/mma_from_smem.h`, retaining its BSD-3-Clause notice. It accepts the
first V fragment loaded before softmax and keeps the remaining CUTLASS
software pipeline and accumulation order. The first fragment uses the same
residual mask as the original prologue, including 32-key V tile boundaries.
Register allocation remains 234/thread without spills; shared memory stays
26,128 bytes/block. There is no extra persistent device allocation.

The adaptation separates the QK and PV tile widths. All 128 output channels
remain in FP32 registers while each iteration consumes 64 keys. Rescaling the
output uses the PV accumulator's lane/row mapping; reusing the QK mapping
would corrupt the second half of the output. Query/key tails are predicated.
The wrapper creates batch/head pointers on the active stream, supports graph
replay, and copies an unaligned or strided input only when necessary.

The existing v37 D256/GQA implementation inspired the large TensorOp route
investigation. This fused operator is derived from CUTLASS FMHA, not a direct
port of the v37 prefix/causal-tail dispatcher. Historical D256 throughput
does not establish H3 performance. The separate v37-derived D128 experiment
and its result are retained in the H3 migration control record.
