# SM70 DeepSeek V4 FP16 GEMV

## Trace Evidence

The 1024-input TP8 decode trace attributes 3.235 ms/token to dense GEMV,
split-K reducers, and compressor work. The fixed FP16 GEMV shapes are:

| role | shape N x K | calls/token | traced main-kernel mean |
|---|---:|---:|---:|
| MoE router | 256 x 4096 | 43 | 9.73 us plus reducer/cast |
| Indexer weights | 64 x 4096 | 21 | 20.07 us plus reducer |
| C4 indexer compressor | 512 x 4096 | 21 | 16.02 us plus reducer |
| C4 main compressor | 2048 x 4096 | 21 | 38.90 us |
| C128 main compressor | 1024 x 4096 | 20 | 22.15 us plus reducer |

The C4 input projections run on separate CUDA streams, so summed service-time
savings are not an endpoint projection. A candidate must also shorten their
joined graph envelope.

## Candidate

The screening kernel assigns one program to one output row, accumulates FP16
products in FP32, and performs no cross-program split-K reduction. It sweeps
the K tile and warp count using real checkpoint weights.

Acceptance requires:

1. lower graph replay latency for each material shape;
2. lower joined C4 multi-stream envelope before production integration;
3. stable router top-6 IDs across real-weight seeds;
4. bounded compressor error and a later full-model quality gate.

No production dispatch is changed until these gates pass.
