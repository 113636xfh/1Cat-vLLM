// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>

namespace h3 {
inline void check_lt(cublasStatus_t status) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "H3 cuBLASLt status ",
              int(status));
}

// A member owns partially initialized resources too, including when plan
// construction throws. Plans have no device workspace or weight ownership.
struct LtResources {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t desc{};
  cublasLtMatrixLayout_t a{}, b{}, c{};
  cublasLtMatmulPreference_t preference{};
  ~LtResources() {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (desc) cublasLtMatmulDescDestroy(desc);
    if (handle) cublasLtDestroy(handle);
  }
};

class FP16GemmPlan {
 public:
  FP16GemmPlan(int64_t m, int64_t n, int64_t k, bool fp32, int device)
      : m_(m), n_(n), k_(k), fp32_(fp32), device_(device) {
    TORCH_CHECK(
        m > 0 && n > 0 && k > 0 && m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
        "H3 cached GEMM requires positive, bounded dimensions");
    const c10::cuda::CUDAGuard guard(device_);
    auto* properties = at::cuda::getDeviceProperties(device_);
    TORCH_CHECK(properties->major == 7 && properties->minor == 0,
                "H3 cached GEMM requires SM70");
    check_lt(cublasLtCreate(&r_.handle));
    check_lt(
        cublasLtMatmulDescCreate(&r_.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    // Physical weight [K,N] is column-major [N,K]. Input [M,K] is
    // column-major [K,M], so both operands use OP_N and C is [N,M].
    check_lt(cublasLtMatrixLayoutCreate(&r_.a, CUDA_R_16F, n, k, n));
    check_lt(cublasLtMatrixLayoutCreate(&r_.b, CUDA_R_16F, k, m, k));
    check_lt(cublasLtMatrixLayoutCreate(&r_.c, fp32 ? CUDA_R_32F : CUDA_R_16F,
                                        n, m, n));
    check_lt(cublasLtMatmulPreferenceCreate(&r_.preference));
    size_t workspace_bytes = 0;
    uint32_t reductions = CUBLASLT_REDUCTION_SCHEME_NONE;
    check_lt(cublasLtMatmulPreferenceSetAttribute(
        r_.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes, sizeof(workspace_bytes)));
    check_lt(cublasLtMatmulPreferenceSetAttribute(
        r_.preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reductions,
        sizeof(reductions)));
    cublasLtMatmulHeuristicResult_t choices[32];
    int count = 0;
    check_lt(cublasLtMatmulAlgoGetHeuristic(r_.handle, r_.desc, r_.a, r_.b,
                                            r_.c, r_.c, r_.preference, 32,
                                            choices, &count));
    bool found = false;
    for (int i = 0; i < count; ++i) {
      auto& choice = choices[i];
      if (choice.state != CUBLAS_STATUS_SUCCESS || choice.workspaceSize)
        continue;
      int id = 0, split = 0;
      uint32_t tile = 0, reduction = 0;
      size_t written = 0;
      check_lt(cublasLtMatmulAlgoConfigGetAttribute(
          &choice.algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written));
      check_lt(cublasLtMatmulAlgoConfigGetAttribute(
          &choice.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile),
          &written));
      check_lt(cublasLtMatmulAlgoConfigGetAttribute(
          &choice.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof(split),
          &written));
      check_lt(cublasLtMatmulAlgoConfigGetAttribute(
          &choice.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction,
          sizeof(reduction), &written));
      if (split > 1 || reduction != CUBLASLT_REDUCTION_SCHEME_NONE) continue;
      // Prefer the measured Volta tile if this library advertises support for
      // the exact shape. Never force an algorithm from another CUDA version.
      const bool preferred = id == 21 && tile == 24;
      if (!found || preferred) {
        algo_ = choice.algo;
        info_ = {id, tile, split, reduction};
        found = true;
      }
      if (preferred) break;
    }
    TORCH_CHECK(found,
                "no workspace-free FP32 H3 GEMM algorithm without split K");
  }

  FP16GemmPlan(const FP16GemmPlan&) = delete;
  FP16GemmPlan& operator=(const FP16GemmPlan&) = delete;

  std::vector<int64_t> algorithm_info() const { return info_; }

  torch::Tensor run(torch::Tensor input, torch::Tensor weight_t) const {
    TORCH_CHECK(
        input.is_cuda() && weight_t.device() == input.device() &&
            input.get_device() == device_ && input.dim() == 2 &&
            weight_t.dim() == 2 && input.is_contiguous() &&
            weight_t.is_contiguous() &&
            input.scalar_type() == torch::kFloat16 &&
            weight_t.scalar_type() == torch::kFloat16 && input.size(0) == m_ &&
            input.size(1) == k_ && weight_t.size(0) == k_ &&
            weight_t.size(1) == n_,
        "H3 cached GEMM requires this plan's FP16 [M,K] and [K,N] matrices");
    // cuBLASLt heuristics default to 256-byte operand alignment. Native H3
    // preparation/cache allocations satisfy this; reject offset views before
    // dispatch rather than executing an incompatible vector-load kernel.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 256 == 0 &&
                    reinterpret_cast<uintptr_t>(weight_t.data_ptr()) % 256 == 0,
                "H3 cached GEMM requires 256-byte aligned operands");
    const c10::cuda::CUDAGuard guard(device_);
    auto output = torch::empty(
        {m_, n_},
        input.options().dtype(fp32_ ? torch::kFloat32 : torch::kFloat16));
    float alpha = 1.f, beta = 0.f;
    check_lt(cublasLtMatmul(
        r_.handle, r_.desc, &alpha, weight_t.data_ptr(), r_.a, input.data_ptr(),
        r_.b, &beta, output.data_ptr(), r_.c, output.data_ptr(), r_.c, &algo_,
        nullptr, 0, at::cuda::getCurrentCUDAStream()));
    return output;
  }

 private:
  LtResources r_;
  cublasLtMatmulAlgo_t algo_{};
  std::vector<int64_t> info_;
  int64_t m_, n_, k_;
  bool fp32_;
  int device_;
};
}  // namespace h3
