// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Copyright contributors to the 1Cat-vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>

#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/kernel/default_gemm.h>

#include "default_fmha.h"

namespace {
using Half = cutlass::half_t;
constexpr int kHeadDim = 128;
constexpr int kQueries = 64;
using Kernel =
    typename cutlass::gemm::kernel::H3FMHA<Half, cutlass::arch::Sm70, true,
                                           kQueries, 64, kHeadDim>::FMHAKernel;

// Keep pointer and shape preparation on the caller's CUDA stream, including
// graph replay. Each independent MHA head is a separate grouped GEMM problem.
__global__ void prepare_metadata(int64_t* meta, Half* q, Half* k, Half* v,
                                 Half* out, int queries, int keys, int heads,
                                 int groups) {
  int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= groups) return;
  int batch = group / heads;
  int head = group % heads;
  int64_t q_offset = (int64_t(batch) * queries * heads + head) * kHeadDim;
  int64_t kv_offset = (int64_t(batch) * keys * heads + head) * kHeadDim;
  meta[group] = reinterpret_cast<int64_t>(q + q_offset);
  meta[groups + group] = reinterpret_cast<int64_t>(k + kv_offset);
  meta[2 * groups + group] = 0;  // No global probability matrix.
  meta[3 * groups + group] = reinterpret_cast<int64_t>(v + kv_offset);
  meta[4 * groups + group] = reinterpret_cast<int64_t>(out + q_offset);
  meta[5 * groups + group] = 0;  // FP32 output state stays in registers.
  for (int slot = 6; slot < 10; ++slot)
    meta[slot * groups + group] = int64_t(heads) * kHeadDim;
  auto sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(meta + 10 * groups);
  sizes[group] = cutlass::gemm::GemmCoord(queries, keys, kHeadDim);
  sizes[groups + group] = cutlass::gemm::GemmCoord(queries, kHeadDim, keys);
}

__global__ __launch_bounds__(Kernel::kThreadCount,
                             1) void h3_flash_v100_d128(Kernel::Params params) {
  extern __shared__ __align__(16) unsigned char storage[];
  Kernel kernel;
  kernel(params, *reinterpret_cast<Kernel::SharedStorage*>(storage));
}

at::Tensor aligned_contiguous(const at::Tensor& tensor) {
  auto result = tensor.contiguous();
  // contiguous() may preserve a contiguous view with an unaligned offset.
  if (reinterpret_cast<uintptr_t>(result.data_ptr()) % 16 != 0)
    result = result.clone();
  return result;
}
}  // namespace

at::Tensor h3_flash_attention_forward(at::Tensor q, at::Tensor k, at::Tensor v,
                                      double scale) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.scalar_type() == at::kHalf,
              "H3 FlashAttention-V100 requires CUDA FP16 BSND tensors");
  TORCH_CHECK(k.device() == q.device() && v.device() == q.device() &&
                  k.scalar_type() == q.scalar_type() &&
                  v.scalar_type() == q.scalar_type(),
              "H3 Q/K/V must share device and FP16 dtype");
  TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 &&
                  q.size(3) == kHeadDim && k.dim() == 4 && k.size(1) > 0 &&
                  k.size(0) == q.size(0) && k.size(2) == q.size(2) &&
                  k.size(3) == kHeadDim && k.sizes() == v.sizes(),
              "H3 FlashAttention-V100 requires non-empty D128 MHA");
  TORCH_CHECK(std::isfinite(scale) && scale > 0 &&
                  scale <= std::numeric_limits<float>::max(),
              "H3 attention scale must be finite and positive");
  TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
              "H3 FlashAttention-V100 is an inference-only operator");
  const c10::cuda::CUDAGuard guard(q.device());
  auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "H3 FlashAttention-V100 requires SM70");
  int64_t groups64 = q.size(0) * q.size(2);
  int64_t blocks64 = ((q.size(1) + kQueries - 1) / kQueries) * groups64;
  TORCH_CHECK(q.size(1) <= INT_MAX && k.size(1) <= INT_MAX &&
                  groups64 <= INT_MAX / 16 && blocks64 <= INT_MAX,
              "H3 attention shape exceeds kernel index limits");
  q = aligned_contiguous(q);
  k = aligned_contiguous(k);
  v = aligned_contiguous(v);
  auto output = at::empty_like(q);
  int groups = groups64;
  auto metadata =
      at::empty({int64_t(groups) * 16}, q.options().dtype(at::kLong));
  auto* ptr = metadata.data_ptr<int64_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  prepare_metadata<<<(groups + 127) / 128, 128, 0, stream>>>(
      ptr, reinterpret_cast<Half*>(q.data_ptr()),
      reinterpret_cast<Half*>(k.data_ptr()),
      reinterpret_cast<Half*>(v.data_ptr()),
      reinterpret_cast<Half*>(output.data_ptr()), q.size(1), k.size(1),
      q.size(2), groups);
  Kernel::Arguments args;
  args.problem_sizes0 =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(ptr + 10 * groups);
  args.problem_sizes1 = args.problem_sizes0 + groups;
  args.problem_count = groups;
  args.threadblock_count = blocks64;
  args.ptr_Q = reinterpret_cast<Half**>(ptr);
  args.ptr_K = reinterpret_cast<Half**>(ptr + groups);
  args.ptr_P = reinterpret_cast<float**>(ptr + 2 * groups);
  args.ptr_V = reinterpret_cast<Half**>(ptr + 3 * groups);
  args.ptr_O = reinterpret_cast<Half**>(ptr + 4 * groups);
  args.ptr_O_accum = reinterpret_cast<float**>(ptr + 5 * groups);
  args.ldq = ptr + 6 * groups;
  args.ldk = ptr + 7 * groups;
  args.ldv = ptr + 8 * groups;
  args.ldo = ptr + 9 * groups;
  args.causal = false;
  args.scale = static_cast<float>(scale);
  Kernel::Params params(args, nullptr, args.threadblock_count);
  // The D128 tile fits the default shared-memory limit (26,128 bytes).
  // Only larger future specializations need the dynamic-memory opt-in.
  if constexpr (sizeof(Kernel::SharedStorage) > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        h3_flash_v100_d128, cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(Kernel::SharedStorage)));
  }
  h3_flash_v100_d128<<<args.threadblock_count, Kernel::kThreadCount,
                       sizeof(Kernel::SharedStorage), stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &h3_flash_attention_forward);
}
