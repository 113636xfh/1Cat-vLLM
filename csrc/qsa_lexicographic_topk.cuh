// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_
#define VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cstdint>
#include <limits>

namespace vllm::qsa {

constexpr int kLexicographicTopKThreads = 1024;
constexpr int kLexicographicTopKBins = 256;

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  // IEEE -0.0 and +0.0 compare equal, so keep them in the same score bucket
  // and let the block index provide the deterministic tie break.
  if (value == 0.0f) value = 0.0f;
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint64_t lexicographic_key(float score,
                                                      uint32_t index) {
  // Larger keys win. Scores are ordered first; equal scores prefer the lower
  // block index. The resulting key is unique for every element in a row.
  return (static_cast<uint64_t>(ordered_float_bits(score)) << 32) |
         (std::numeric_limits<uint32_t>::max() - index);
}

template <int TopK>
struct LexicographicTopKShared {
  using BlockScan = cub::BlockScan<uint32_t, kLexicographicTopKThreads>;

  uint32_t histogram[kLexicographicTopKBins];
  typename BlockScan::TempStorage scan;
  uint64_t prefix;
  uint64_t pivot;
  uint32_t remaining;
  uint32_t output_base;
  uint32_t chunk_base;
};

template <int TopK>
__global__
__launch_bounds__(kLexicographicTopKThreads) void qsa_lexicographic_topk_kernel(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ output, uint32_t num_rows, uint32_t columns,
    uint32_t stride) {
  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t raw_length = lengths[row];
  const uint32_t length =
      raw_length > 0 ? min(static_cast<uint32_t>(raw_length), columns) : 0;
  const float* row_logits = logits + static_cast<uint64_t>(row) * stride;
  int32_t* row_output = output + static_cast<uint64_t>(row) * TopK;

  if (length <= TopK) {
    for (uint32_t index = tx; index < TopK;
         index += kLexicographicTopKThreads) {
      row_output[index] = index < length ? static_cast<int32_t>(index) : -1;
    }
    return;
  }

  __shared__ LexicographicTopKShared<TopK> shared;
  if (tx == 0) {
    shared.prefix = 0;
    shared.remaining = TopK;
  }
  __syncthreads();

  // Select the exact k-th (score descending, index ascending) key. Eight
  // byte-wide radix passes cover the full 64-bit composite key without a
  // bounded candidate buffer, so dense ties cannot overflow or race.
#pragma unroll
  for (int pass = 0; pass < 8; ++pass) {
    for (uint32_t bin = tx; bin < kLexicographicTopKBins;
         bin += kLexicographicTopKThreads) {
      shared.histogram[bin] = 0;
    }
    __syncthreads();

    const int shift = 56 - pass * 8;
    const uint64_t prefix = shared.prefix;
    const uint64_t prefix_mask = pass == 0 ? 0 : (~uint64_t{0} << (shift + 8));
    for (uint32_t index = tx; index < length;
         index += kLexicographicTopKThreads) {
      const uint64_t key = lexicographic_key(row_logits[index], index);
      if ((key & prefix_mask) == prefix) {
        atomicAdd(&shared.histogram[(key >> shift) & 0xffu], 1u);
      }
    }
    __syncthreads();

    if (tx == 0) {
      uint32_t remaining = shared.remaining;
      for (int bin = kLexicographicTopKBins - 1; bin >= 0; --bin) {
        const uint32_t count = shared.histogram[bin];
        if (remaining > count) {
          remaining -= count;
        } else {
          shared.prefix |= static_cast<uint64_t>(bin) << shift;
          shared.remaining = remaining;
          break;
        }
      }
    }
    __syncthreads();
  }

  if (tx == 0) {
    shared.pivot = shared.prefix;
    shared.output_base = 0;
  }
  __syncthreads();

  // Compact in increasing index order. This is both the deterministic tie
  // contract and QSA's canonical accumulation order, so no repair or second
  // sorting pass is needed.
  using BlockScan = typename LexicographicTopKShared<TopK>::BlockScan;
  for (uint32_t base = 0; base < length; base += kLexicographicTopKThreads) {
    const uint32_t index = base + tx;
    const uint32_t selected =
        index < length &&
                lexicographic_key(row_logits[index], index) >= shared.pivot
            ? 1u
            : 0u;
    uint32_t offset = 0;
    uint32_t aggregate = 0;
    BlockScan(shared.scan).ExclusiveSum(selected, offset, aggregate);
    __syncthreads();
    if (tx == 0) {
      shared.chunk_base = shared.output_base;
      shared.output_base += aggregate;
    }
    __syncthreads();
    if (selected) {
      row_output[shared.chunk_base + offset] = static_cast<int32_t>(index);
    }
    __syncthreads();
  }
}

template <int TopK>
void launch_qsa_lexicographic_topk(const float* logits, const int32_t* lengths,
                                   int32_t* output, uint32_t num_rows,
                                   uint32_t columns, uint32_t stride,
                                   cudaStream_t stream) {
  qsa_lexicographic_topk_kernel<TopK>
      <<<num_rows, kLexicographicTopKThreads, 0, stream>>>(
          logits, lengths, output, num_rows, columns, stride);
}

}  // namespace vllm::qsa

#endif  // VLLM_CSRC_QSA_LEXICOGRAPHIC_TOPK_CUH_
