// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace {
__global__ void dequantize_rows(const int8_t* weights, const float* scales,
                                half* output, int64_t count, int64_t width) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(gridDim.x) * blockDim.x) {
    output[i] = __float2half_rn(float(weights[i]) * scales[i / width]);
  }
}

__global__ void convrot256(const half* input, half* output, int64_t groups) {
  __shared__ float values[256];
  const int lane = threadIdx.x;
  for (int64_t group = blockIdx.x; group < groups; group += gridDim.x) {
    values[lane] = __half2float(input[group * 256 + lane]);
    __syncthreads();
#pragma unroll
    for (int stride = 1; stride <= 64; stride *= 4) {
      const int digit = (lane / stride) % 4;
      const int base = lane - digit * stride;
      float a = values[base], b = values[base + stride];
      float c = values[base + 2 * stride], d = values[base + 3 * stride];
      float value = digit == 0   ? a + b + c - d
                    : digit == 1 ? a + b - c + d
                    : digit == 2 ? a - b + c + d
                                 : -a + b + c + d;
      __syncthreads();
      values[lane] = value;
      __syncthreads();
    }
    output[group * 256 + lane] = __float2half_rn(values[lane] * (1.f / 16.f));
    __syncthreads();
  }
}

void validate_cuda(const torch::Tensor& value) {
  TORCH_CHECK(value.is_cuda() && value.is_contiguous(),
              "H3 requires contiguous CUDA tensors");
  const auto* p = at::cuda::getDeviceProperties(value.get_device());
  TORCH_CHECK(p->major == 7 && p->minor == 0,
              "H3 SM70 operator requires Volta");
}
}  // namespace

torch::Tensor h3_dequantize(torch::Tensor weight, torch::Tensor scale) {
  validate_cuda(weight);
  validate_cuda(scale);
  TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == torch::kInt8,
              "H3 weight must be a signed INT8 matrix");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32 &&
                  scale.numel() == weight.size(0) &&
                  scale.device() == weight.device(),
              "H3 requires same-device FP32 row scales");
  const c10::cuda::CUDAGuard guard(weight.device());
  auto output =
      torch::empty(weight.sizes(), weight.options().dtype(torch::kFloat16));
  if (weight.numel()) {
    const int grid = std::min<int64_t>((weight.numel() + 255) / 256, 65535);
    dequantize_rows<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        weight.data_ptr<int8_t>(), scale.data_ptr<float>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()), weight.numel(),
        weight.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

torch::Tensor h3_rotate(torch::Tensor input) {
  validate_cuda(input);
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 && input.dim() >= 1 &&
                  input.size(-1) % 256 == 0,
              "H3 rotation requires FP16 groups of 256");
  const c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty_like(input);
  const int64_t groups = input.numel() / 256;
  if (groups) {
    convrot256<<<std::min<int64_t>(groups, 65535), 256, 0,
                 at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()), groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

torch::Tensor h3_fp16_gemm(torch::Tensor input, torch::Tensor weight,
                           bool output_fp32) {
  validate_cuda(input);
  validate_cuda(weight);
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 &&
                  input.size(1) == weight.size(1) &&
                  input.device() == weight.device() &&
                  input.scalar_type() == torch::kFloat16 &&
                  weight.scalar_type() == torch::kFloat16,
              "H3 GEMM requires matching FP16 [M,K] and [N,K] matrices");
  const c10::cuda::CUDAGuard guard(input.device());
  const int64_t m = input.size(0), n = weight.size(0), k = input.size(1);
  TORCH_CHECK(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
              "GEMM dimension overflow");
  auto output = torch::empty(
      {m, n},
      input.options().dtype(output_fp32 ? torch::kFloat32 : torch::kFloat16));
  if (!m || !n) return output;
  // cuBLAS is an existing TurboMind SM70 dispatch option. Explicit 32F compute
  // prevents reduced-precision accumulation; W8 decode is outside the GEMM.
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  cublasMath_t saved_math;
  TORCH_CHECK(cublasGetMathMode(handle, &saved_math) == CUBLAS_STATUS_SUCCESS,
              "cannot read cuBLAS math mode");
  TORCH_CHECK(
      cublasSetMathMode(
          handle, static_cast<cublasMath_t>(
                      CUBLAS_TENSOR_OP_MATH |
                      CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)) ==
          CUBLAS_STATUS_SUCCESS,
      "cannot enable FP32 GEMM reductions");
  float alpha = 1.f, beta = 0.f;
  auto status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                             weight.data_ptr(), CUDA_R_16F, k, input.data_ptr(),
                             CUDA_R_16F, k, &beta, output.data_ptr(),
                             output_fp32 ? CUDA_R_32F : CUDA_R_16F, n,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  auto restored = cublasSetMathMode(handle, saved_math);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "H3 FP16 GEMM failed: ", int(status));
  TORCH_CHECK(restored == CUBLAS_STATUS_SUCCESS,
              "cannot restore cuBLAS math mode");
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequantize", &h3_dequantize);
  m.def("rotate", &h3_rotate);
  m.def("gemm", &h3_fp16_gemm, pybind11::arg("input"), pybind11::arg("weight"),
        pybind11::arg("output_fp32") = false);
}
