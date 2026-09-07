# FlashAttention-V100 v37 prefill integration

## Scope

We integrate the v37 FP32-accumulating long-prefill route and an explicit
E4M3-to-FP16 bridge into the SM70 FA2 extension. We retain ordinary decode,
model weights, sampling, SSM arithmetic, and speculative-decoding behavior.
No speculative configuration is used in the model validation below.

The implementation and admission contract are described in
[the kernel README](../../csrc/attention/sm70_v37/README.md). This change
does not rename the global `fp8` encoding or turn an E5M2 byte cache into E4M3.
The new E4M3 bridge is selected only for explicit `fp8_e4m3` storage.

## Why this is more than copying the private kernel

The private endpoint accepted Q8000. The current chunked-prefill scheduler
uses a different family of query lengths, so installing that endpoint alone
does not prove that a model benefits. We replace fixed packed-row strides
with the actual query/head extent and test the admitted alignment family.
The persistent score allocation is bounded by the maximum admitted query
extent. We retain the causal offset when prepending query padding.

The v37 symbol is separate from the legacy GQA symbol. Missing new native
code triggers a warning and an exact fallback, never a false v37 route hit.
We also join the private tail on exception before releasing its inputs and
temporary state. The legacy architecture remains available for rollback.

Set `VLLM_FLASH_V100_PREFILL_D256_GQA_V37=0` before starting workers to use
the old architecture loader and disable the new E4M3 bridge. Runtime
environment mutation in an already initialized engine is not a rollback
mechanism.

## Operator evidence

The fixed-shape port matches the private v37 output bitwise on 12 complete
real-operand replays: three attention layers and KV lengths 16K, 64K, 128K,
and 256K. After replacing fixed row strides, the layer-63 Q8000 replay still
matches bitwise at all four lengths. Cropped operands are operator tests,
not fresh model executions.

On one physical V100, 20 warmups and 100 ABBA pairs compare the dynamic port
with the retained private v37 at Q8000/Hq6/Hkv1/D256/FP16/causal:

| KV tokens | Private v37 median ms | Port median ms | Median paired speedup |
| ---: | ---: | ---: | ---: |
| 128000 | 99.460 | 99.601 | 0.99877 |
| 256000 | 201.507 | 201.743 | 0.99948 |

These are complete dense attention endpoints, including prefix, causal
tail, state merging and synchronization. They exclude paged gathering and
FP8 conversion. The speedup is the median of paired ratios, not the ratio
of the two reported marginal medians. This is a port-regression comparison,
not a new speedup over an old 18-TFLOP/s or 60-TFLOP/s implementation.

A separate retained all-row FP64 audit on identical E4M3-representable KV
values found the following relative L2 errors for layer-63 Q8000 operands:

| KV tokens | Native paged error | Private v37 error |
| ---: | ---: | ---: |
| 64000 | 0.15823% | 0.02788% |
| 128000 | 0.26685% | 0.02679% |
| 256000 | 0.44761% | 0.02581% |

This isolates attention arithmetic from KV quantization. It does not show
that E4M3 has no quantization loss, nor that all model tokens must match an
unquantized reference. A constant-V result of exactly one on one layer is
also not a universal guarantee: other retained layers differ by one FP16
rounding step.

## Model comparison and interpretation

The initial source-overlay comparison fixes Qwen3.8-27B-FP8, TP4 on four
V100s, FP8 weights, explicit E4M3 KV, FP16 activations/conv state, FP32 SSM
state, no MTP, chunk budget 8192, max length 262144, prefix caching off and
CUDA graphs on. Every worker reports the actual KV page size as 1568.
The control uses the old main E4M3 direct-paged route; the candidate adds
the E4M3 bridge and v37. All other native libraries and settings are shared.

The initial post-warmup results are single observations per context, not a
publication-grade repeated timing study:

| Prompt tokens | Control prefill s | Candidate prefill s | Control decode tok/s | Candidate decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 1.709 | 1.700 | 59.433 | 59.268 |
| 65536 | 76.328 | 16.825 | 37.650 | 37.448 |
| 128000 | 272.040 | 39.775 | 26.688 | 26.508 |
| 256000 | 1049.836 | 106.891 | 16.790 | 16.763 |

Prefill uses the engine's scheduled-to-first-token interval. Decode excludes
the first token and uses the interval between the first and last generated
tokens. The timing request forces 64 tokens and requests five logprobs;
retrieval quality is scored only before the first EOS. Natural-EOS text
checks are recorded separately. The initial retrieval answers match at all
four lengths, and every rank records 848 v37/E4M3-bridge prefill executions.

The large model prefill ratios combine two changes: enabling a previously
missing E4M3 bridge and selecting the v37 kernel. They must not be described
as v37-only kernel gains, reused as PR122/E5M2 baselines, or substituted into
the paper's earlier 60-TFLOP/s comparison without a matching contract.
Both initial runs omitted the existing E4M3 B1 long-context auto/wave
partition flags. Their similar decode rates therefore establish neither
historical performance parity nor decode-speed admission. The earlier
PR285 result of 50.376 tok/s at final context 262144 used this long-wave
route, with NVFP4 rather than the FP8 model weights used here. Promotion is
paused while a same-contract FP8 comparison isolates the missing dispatch.

## Rejected paths and remaining admission work

An initial variable-shape build retained a fixed 48000-row PV-statistic
stride and failed with an illegal access. The stride is now dynamic.
Q1664/KV4848, which has only 16-token KV alignment, failed the FP64 test with
relative L2 0.151814. We reject that shape before launch and retain the
32-token KV requirement; it is not a passed numerical result.

The source overlay is not a complete rebuilt vLLM wheel. The final compiled
FA2 target, clean native Flash dependency, stream/memory-safety checks,
short-query latency screening, and natural long-output comparison must be
qualified before promotion. No result above by itself establishes universal
model-quality equivalence or authorizes an unrelated MTP change.
