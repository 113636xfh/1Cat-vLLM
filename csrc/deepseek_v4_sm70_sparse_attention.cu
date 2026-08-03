/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/all.h>

#include <limits>

namespace vllm::deepseek_v4_sm70_sparse {

namespace {

constexpr int kHeads = 8;
constexpr int kHeadDim = 512;
constexpr int kMaxIndexWidth = 640;
constexpr int kKeysPerTile = 16;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kDimsPerWarp = kHeadDim / kWarps;
constexpr int kQTilesPerWarp = kDimsPerWarp / 16;
constexpr int kKvStride = kHeadDim + 8;
constexpr int kQStageStride = kDimsPerWarp + 8;
constexpr int kScoreStride = 20;
constexpr int kProbStride = 24;

using half = __half;
using half2 = __half2;
namespace wmma = nvcuda::wmma;

struct alignas(16) SharedStorage {
  half kv[kKeysPerTile][kKvStride];
  half q_stage[kWarps][16][kQStageStride];
  union {
    float qk[kWarps][16][kScoreStride];
    float pv[kWarps][16][16];
  } mma;
  half probs[16][kProbStride];
  float alpha[kHeads];
  float final_scale[kHeads];
  int slots[kKeysPerTile];
  int valid[kKeysPerTile];
};

__device__ __forceinline__ float fast_exp(float value) { return __expf(value); }

__global__ __launch_bounds__(kThreads, 2) void sparse_attention_hmma_kernel(
    half const* __restrict__ q, half const* __restrict__ kv,
    int const* __restrict__ indices, int const* __restrict__ lengths,
    float const* __restrict__ sink, half* __restrict__ out,
    int64_t q_stride_t, int64_t q_stride_h, int64_t kv_stride_n,
    int64_t indices_stride_t, int64_t out_stride_t, int64_t out_stride_h,
    int index_width, int num_kv, float scale) {
  int const query_idx = blockIdx.x;
  int const tid = threadIdx.x;
  int const warp = tid / 32;
  int const lane = tid % 32;
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto& shared = *reinterpret_cast<SharedStorage*>(shared_bytes);

  using MatrixA = wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                                 wmma::row_major>;
  using MatrixBCol = wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                                    wmma::col_major>;
  using MatrixBRow = wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                                    wmma::row_major>;
  using Accumulator = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

  float output_acc[kQTilesPerWarp * 4];
#pragma unroll
  for (float& value : output_acc) {
    value = 0.0f;
  }
  float running_max = -std::numeric_limits<float>::infinity();
  float running_sum = 0.0f;
  int const valid_len = lengths[query_idx];

#pragma unroll
  for (int pair = lane; pair < 16 * (kDimsPerWarp / 2); pair += 32) {
    int const row = pair / (kDimsPerWarp / 2);
    int const column_pair = pair % (kDimsPerWarp / 2);
    int const dim = warp * kDimsPerWarp + column_pair * 2;
    half2 value = __float2half2_rn(0.0f);
    if (row < kHeads) {
      auto const* q_row = reinterpret_cast<half2 const*>(
          q + query_idx * q_stride_t + row * q_stride_h);
      value = q_row[dim / 2];
    }
    auto* q_stage_row =
        reinterpret_cast<half2*>(&shared.q_stage[warp][row][0]);
    q_stage_row[column_pair] = value;
  }
  __syncwarp();

  for (int key_start = 0; key_start < index_width;
       key_start += kKeysPerTile) {
    if (tid < kKeysPerTile) {
      int const slot = indices[query_idx * indices_stride_t + key_start + tid];
      bool const is_valid = key_start + tid < valid_len && slot >= 0 &&
                            slot < num_kv;
      shared.slots[tid] = is_valid ? slot : 0;
      shared.valid[tid] = is_valid;
    }
    __syncthreads();

#pragma unroll 1
    for (int pair = tid; pair < kKeysPerTile * (kHeadDim / 2);
         pair += kThreads) {
      int const key = pair / (kHeadDim / 2);
      int const dim_pair = pair % (kHeadDim / 2);
      half2 value = __float2half2_rn(0.0f);
      if (shared.valid[key]) {
        auto const* kv_row = reinterpret_cast<half2 const*>(
            kv + static_cast<int64_t>(shared.slots[key]) * kv_stride_n);
        value = kv_row[dim_pair];
      }
      auto* shared_kv_row = reinterpret_cast<half2*>(&shared.kv[key][0]);
      shared_kv_row[dim_pair] = value;
    }
    __syncthreads();

    Accumulator score_fragment;
    wmma::fill_fragment(score_fragment, 0.0f);
#pragma unroll
    for (int tile = 0; tile < kQTilesPerWarp; ++tile) {
      MatrixA query_fragment;
      MatrixBCol key_fragment;
      wmma::load_matrix_sync(query_fragment,
                             &shared.q_stage[warp][0][tile * 16],
                             kQStageStride);
      wmma::load_matrix_sync(
          key_fragment,
          &shared.kv[0][warp * kDimsPerWarp + tile * 16], kKvStride);
      wmma::mma_sync(score_fragment, query_fragment, key_fragment,
                     score_fragment);
    }
    wmma::store_matrix_sync(&shared.mma.qk[warp][0][0], score_fragment,
                            kScoreStride, wmma::mem_row_major);
    __syncthreads();

    float score = -std::numeric_limits<float>::infinity();
    if (lane < kKeysPerTile) {
      score = shared.mma.qk[0][warp][lane];
#pragma unroll
      for (int partial = 1; partial < kWarps; ++partial) {
        score += shared.mma.qk[partial][warp][lane];
      }
      score *= scale;
    }

    float block_max = -std::numeric_limits<float>::infinity();
#pragma unroll
    for (int key = 0; key < kKeysPerTile; ++key) {
      float const key_score = __shfl_sync(0xffffffff, score, key);
      if (lane == 0 && shared.valid[key]) {
        block_max = fmaxf(block_max, key_score);
      }
    }

    float new_max = 0.0f;
    if (lane == 0) {
      new_max = fmaxf(running_max, block_max);
    }
    new_max = __shfl_sync(0xffffffff, new_max, 0);
    float probability = 0.0f;
    if (lane < kKeysPerTile && shared.valid[lane]) {
      probability = fast_exp(score - new_max);
    }

    float block_sum = 0.0f;
#pragma unroll
    for (int key = 0; key < kKeysPerTile; ++key) {
      float const key_probability =
          __shfl_sync(0xffffffff, probability, key);
      if (lane == 0) {
        block_sum += key_probability;
      }
    }

    if (lane == 0) {
      float const alpha = fast_exp(running_max - new_max);
      running_sum = running_sum * alpha + block_sum;
      running_max = new_max;
      shared.alpha[warp] = alpha;
    }
    if (lane < kKeysPerTile) {
      shared.probs[warp][lane] = __float2half_rn(probability);
      shared.probs[kHeads + warp][lane] = __float2half(0.0f);
    }
    __syncthreads();

    MatrixA probability_fragment;
    wmma::load_matrix_sync(probability_fragment, &shared.probs[0][0],
                           kProbStride);
#pragma unroll
    for (int tile = 0; tile < kQTilesPerWarp; ++tile) {
      MatrixBRow value_fragment;
      Accumulator output_fragment;
      wmma::load_matrix_sync(
          value_fragment,
          &shared.kv[0][warp * kDimsPerWarp + tile * 16], kKvStride);
      wmma::fill_fragment(output_fragment, 0.0f);
      wmma::mma_sync(output_fragment, probability_fragment, value_fragment,
                     output_fragment);
      wmma::store_matrix_sync(&shared.mma.pv[warp][0][0], output_fragment, 16,
                              wmma::mem_row_major);
      __syncwarp();
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        int const linear = lane + element * 32;
        int const head = linear / 16;
        output_acc[tile * 4 + element] =
            output_acc[tile * 4 + element] * shared.alpha[head] +
            shared.mma.pv[warp][head][linear % 16];
      }
      __syncwarp();
    }
  }

  if (lane == 0) {
    float const final_max = fmaxf(running_max, sink[warp]);
    float const alpha = fast_exp(running_max - final_max);
    float const final_sum =
        running_sum * alpha + fast_exp(sink[warp] - final_max);
    shared.final_scale[warp] = alpha / fmaxf(final_sum, 1.0e-30f);
  }
  __syncthreads();

#pragma unroll
  for (int tile = 0; tile < kQTilesPerWarp; ++tile) {
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      int const linear = lane + element * 32;
      int const head = linear / 16;
      int const dim = warp * kDimsPerWarp + tile * 16 + linear % 16;
      out[query_idx * out_stride_t + head * out_stride_h + dim] =
          __float2half_rn(output_acc[tile * 4 + element] *
                          shared.final_scale[head]);
    }
  }
}

}  // namespace

void launch(torch::Tensor const& q, torch::Tensor const& kv,
            torch::Tensor const& indices, torch::Tensor const& lengths,
            torch::Tensor const& sink, torch::Tensor& out, double scale) {
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda() &&
                  lengths.is_cuda() && sink.is_cuda() && out.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(q.device() == kv.device() && q.device() == indices.device() &&
                  q.device() == lengths.device() && q.device() == sink.device() &&
                  q.device() == out.device(),
              "all tensors must be on the same CUDA device");
  at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 7 &&
                  at::cuda::getCurrentDeviceProperties()->minor == 0,
              "sm70_deepseek_v4_sparse_attention_hmma requires SM70");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 &&
                  kv.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "q, kv, and out must be float16");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 &&
                  lengths.scalar_type() == torch::kInt32 &&
                  sink.scalar_type() == torch::kFloat32,
              "indices and lengths must be int32; sink must be float32");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeads &&
                  q.size(2) == kHeadDim,
              "q must have shape [tokens, 8, 512]");
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == kHeadDim,
              "kv must have shape [tokens, 512]");
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) == q.size(0),
              "indices must have one row per query token");
  TORCH_CHECK(indices.size(1) == 128 || indices.size(1) == 256 ||
                  indices.size(1) == kMaxIndexWidth,
              "indices width must be 128, 256, or 640");
  TORCH_CHECK(lengths.numel() == q.size(0),
              "lengths must contain one entry per query token");
  TORCH_CHECK(sink.numel() >= kHeads, "sink must contain at least 8 entries");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
  TORCH_CHECK(q.stride(2) == 1 && kv.stride(1) == 1 &&
                  indices.stride(1) == 1 && out.stride(2) == 1,
              "innermost tensor dimensions must be contiguous");
  TORCH_CHECK(lengths.is_contiguous() && sink.is_contiguous(),
              "lengths and sink must be contiguous");
  TORCH_CHECK(q.stride(0) % 2 == 0 && q.stride(1) % 2 == 0 &&
                  kv.stride(0) % 2 == 0,
              "q and kv row strides must be aligned for half2 loads");
  if (q.size(0) == 0) {
    return;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  sparse_attention_hmma_kernel<<<q.size(0), kThreads, sizeof(SharedStorage),
                                 stream>>>(
      reinterpret_cast<half const*>(q.data_ptr()),
      reinterpret_cast<half const*>(kv.data_ptr()), indices.data_ptr<int>(),
      lengths.data_ptr<int>(), sink.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr()), q.stride(0), q.stride(1),
      kv.stride(0), indices.stride(0), out.stride(0), out.stride(1),
      static_cast<int>(indices.size(1)), static_cast<int>(kv.size(0)),
      static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vllm::deepseek_v4_sm70_sparse

void sm70_deepseek_v4_sparse_attention_hmma(
    torch::Tensor const& q, torch::Tensor const& kv,
    torch::Tensor const& indices, torch::Tensor const& lengths,
    torch::Tensor const& sink, torch::Tensor& out, double scale) {
  vllm::deepseek_v4_sm70_sparse::launch(q, kv, indices, lengths, sink, out,
                                        scale);
}
