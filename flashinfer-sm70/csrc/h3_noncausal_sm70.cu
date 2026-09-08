// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <flashinfer/attention/sm70/volta_mma.cuh>

namespace fi = flashinfer::attention::sm70;
namespace {
constexpr int D = 128;
template <int BQ, int BK, bool Padded = false>
constexpr int shared_bytes() {
  constexpr int QLD = D + (Padded ? 8 : 0), PLD = BK + (Padded ? 8 : 0);
  constexpr int OLD = D + (Padded ? 4 : 0);
  return (BQ * QLD + 2 * BK * QLD + BQ * PLD) * 2 +
         (BQ * PLD + BQ * OLD + BQ * 3) * 4;
}

template <int BQ, int BK, bool Padded = false>
__global__ void h3_noncausal(const half* q, const half* k, const half* v,
                             half* output, int length, int heads, float scale) {
  constexpr int QLD = D + (Padded ? 8 : 0), PLD = BK + (Padded ? 8 : 0);
  constexpr int OLD = D + (Padded ? 4 : 0);
  extern __shared__ __align__(32) unsigned char raw[];
  half* qs = reinterpret_cast<half*>(raw);
  half* ks = qs + BQ * QLD;
  half* vs = ks + BK * QLD;
  half* probabilities = vs + BK * QLD;
  float* scores = reinterpret_cast<float*>(probabilities + BQ * PLD);
  float* os = scores + BQ * PLD;
  float* maximum = os + BQ * OLD;
  float* denominator = maximum + BQ;
  float* alpha = denominator + BQ;
  const int tid = threadIdx.x, warp = tid / 32;
  const int warp_q = warp / (BK / 16), warp_k = warp % (BK / 16);
  const int q_start = blockIdx.x * BQ;
  const int head = blockIdx.y % heads;
  const int batch = blockIdx.y / heads;
  const int64_t base = int64_t(batch) * length * heads * D + head * D;
  for (int i = tid; i < BQ * D; i += blockDim.x) {
    const int row = q_start + i / D;
    qs[(i / D) * QLD + i % D] = row < length
                                    ? q[base + int64_t(row) * heads * D + i % D]
                                    : __float2half(0.f);
    os[(i / D) * OLD + i % D] = 0.f;
  }
  if (tid < BQ) {
    maximum[tid] = -INFINITY;
    denominator[tid] = 0.f;
  }
  __syncthreads();
  for (int start = 0; start < length; start += BK) {
    for (int i = tid; i < BK * D; i += blockDim.x) {
      const int row = start + i / D;
      const int64_t position = base + int64_t(row) * heads * D + i % D;
      ks[(i / D) * QLD + i % D] =
          row < length ? k[position] : __float2half(0.f);
      vs[(i / D) * QLD + i % D] =
          row < length ? v[position] : __float2half(0.f);
    }
    __syncthreads();
    fi::AccumulatorFragment qk;
    fi::init_accumulator_fragment(qk);
#pragma unroll
    for (int dim = 0; dim < D; dim += 16) {
      fi::AFragment qa;
      fi::QKBFragment kb;
      fi::load_a_fragment(qa, qs + warp_q * 16 * QLD + dim, QLD);
      fi::load_qk_b_fragment(kb, ks + warp_k * 16 * QLD + dim, QLD);
      fi::mma_sync_m16n16k16_row_col_f16f16f32(qk, qa, kb);
    }
    fi::store_accumulator_fragment(scores + warp_q * 16 * PLD + warp_k * 16, qk,
                                   PLD);
    __syncthreads();
    // Each warp reduces one query row. Serial per-thread expf over BK was
    // the dominant cost in the initial correctness kernel.
    const int lane = tid % 32;
    for (int row = warp; row < BQ; row += blockDim.x / 32) {
      float values[BK / 32];
      float row_max = -INFINITY;
#pragma unroll
      for (int part = 0; part < BK / 32; ++part) {
        const int col = lane + part * 32;
        values[part] =
            start + col < length ? scores[row * PLD + col] * scale : -INFINITY;
        row_max = fmaxf(row_max, values[part]);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, offset));
      const float m = fmaxf(maximum[row], row_max);
      const float a = __expf(maximum[row] - m);
      float sum = 0.f;
#pragma unroll
      for (int part = 0; part < BK / 32; ++part) {
        const float p = __expf(values[part] - m);
        probabilities[row * PLD + lane + part * 32] = __float2half_rn(p);
        sum += p;
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
      if (lane == 0) {
        denominator[row] = denominator[row] * a + sum;
        maximum[row] = m;
        alpha[row] = a;
      }
    }
    __syncthreads();
    for (int i = tid; i < BQ * D; i += blockDim.x)
      os[(i / D) * OLD + i % D] *= alpha[i / D];
    __syncthreads();
#pragma unroll
    for (int part = 0; part < D / BK; ++part) {
      const int col = warp_k * (D * 16 / BK) + part * 16;
      const int row = warp_q * 16;
      fi::AccumulatorFragment pv;
      fi::load_accumulator_fragment(pv, os + row * OLD + col, OLD);
#pragma unroll
      for (int kv = 0; kv < BK; kv += 16) {
        fi::AFragment pa;
        fi::PVBFragment vb;
        fi::load_a_fragment(pa, probabilities + row * PLD + kv, PLD);
        fi::load_pv_b_fragment(vb, vs + kv * QLD + col, QLD);
        fi::mma_sync_m16n16k16_row_row_f16f16f32(pv, pa, vb);
      }
      fi::store_accumulator_fragment(os + row * OLD + col, pv, OLD);
    }
    __syncthreads();
  }
  for (int i = tid; i < BQ * D; i += blockDim.x) {
    const int row = q_start + i / D;
    if (row < length)
      output[base + int64_t(row) * heads * D + i % D] =
          __float2half_rn(os[(i / D) * OLD + i % D] / denominator[i / D]);
  }
}
}  // namespace

template <int BQ, int BK, bool Padded = false>
torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                      double scale) {
  TORCH_CHECK(
      q.is_cuda() && k.device() == q.device() && v.device() == q.device(),
      "H3 attention requires same-device CUDA inputs");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 &&
                  k.scalar_type() == q.scalar_type() &&
                  v.scalar_type() == q.scalar_type(),
              "H3 FlashInfer-SM70 requires FP16");
  TORCH_CHECK(
      q.dim() == 4 && q.size(3) == D && q.size(1) > 0 &&
          q.sizes() == k.sizes() && q.sizes() == v.sizes(),
      "H3 FlashInfer-SM70 requires matching nonempty BSHD D128 MHA inputs");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "H3 FlashInfer-SM70 inputs must be contiguous");
  const c10::cuda::CUDAGuard guard(q.device());
  const auto* properties = at::cuda::getDeviceProperties(q.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "H3 FlashInfer kernel requires SM70");
  TORCH_CHECK(q.size(1) <= INT_MAX && q.size(0) * q.size(2) <= 65535,
              "H3 FlashInfer grid overflow");
  auto output = torch::empty_like(q);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      h3_noncausal<BQ, BK, Padded>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes<BQ, BK, Padded>()));
  dim3 grid((q.size(1) + BQ - 1) / BQ, q.size(0) * q.size(2));
  h3_noncausal<BQ, BK, Padded>
      <<<grid, (BQ / 16) * (BK / 16) * 32, shared_bytes<BQ, BK, Padded>(),
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()), q.size(1),
          q.size(2), float(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward<64, 32>);
  m.def("forward_bq16", &forward<16, 64>);
  m.def("forward_padded", &forward<64, 32, true>);
}
