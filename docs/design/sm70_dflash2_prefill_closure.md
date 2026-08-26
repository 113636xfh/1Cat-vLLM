# SM70 DFlash2 Prefill Closure

## Scope and integration base

This private campaign is stacked on the quality-audited DFlash2 branch at
`ee4ac48a479c3dbd458d5f7c09a59f39fd271d82`. It keeps the accepted NVFP4
target, official BF16 DFlash2 draft, FP8 E5M2 target KV, FP16 draft KV,
prefix caching, Mamba alignment, and CUDA Graph decode contract. Draft-MLP
QPN8 remains disabled.

The first objective is to restore already accepted SM70 prefill operators in
source-overlay deployments. Kernel arithmetic is not changed by that repair.
Any later shape expansion is a separate gate.

## Project PR audit

The historical short- and long-prefill numbers use different contracts:

| Evidence | Contract | Accepted result | Boundary |
|---|---|---:|---|
| Public PR #271 | Qwen3.8-27B-FP8, TP4, exact input 8000, target-only | 5121.44 request-wall and 5170.96 pure-prefill tok/s | Exact-8K only |
| Public PR #324 | Same exact-8K FP8 contract | 5500 tok/s is a campaign target, not a measured implementation result | Documentation-only PR |
| Private PR #8/#13 | Qwen3.8-27B-FP8, TP4, input 261888, chunk 8192 with Q8000 aligned chunks, target-only | 2438.89 prompt tok/s | Stable max-aware D256 architecture |
| Rejected public PR #315 lane | Same 256K FP8 contract | 2971.51 prompt tok/s | Rejected: 32 output token IDs were zero |

The 2438.89 tok/s route uses max-shifted exponentiation and max-aware online
softmax merging. Its output hash exactly matched the exact control. The removed
raw-logit half2 polynomial must not be restored.

## Current DFlash2 baseline

The same-card cold benchmark resets prefix cache before every warmup and
measurement. It uses the practical 256K API contract, including chunk 4096.

| Input | Mean pure prefill | Prompt throughput |
|---:|---:|---:|
| 32768 | 10.485347 s | 3125.12 tok/s |
| 65536 | 25.317796 s | 2588.54 tok/s |

All three repeats at each length emitted the same first-token hash. Artifact:
`/data/minimax-h3/task-cache/v100-dflash2-prefill-32k64k-20260827/current-dflash2-cold-prefill-v1/`.

## Confirmed root cause

Every retained practical DFlash2 long-prefill log reports that
`_vllm_fa2_C` cannot be imported. The source checkout contains the D256
dispatch and quality-safe architecture, but the source overlay shadows the
installed package containing the native extension. Therefore the merged D256
path is not merely underperforming; it has never executed in these runs.

The accepted stable binary is retained at
`/data/minimax-h3/task-cache/qwen38-d256-attn-80tflops-20260825/build/exact-stat-256k-py312-v2/_vllm_fa2_C.abi3.so`
with SHA256 `f9f9acbc610c87fce9984e8fbd93fe0c8fa59887542123a74b3eaef6d3b8abf9`.
It loads against the active Torch 2.10/CUDA 12.8 environment and registers the
required dense, paged, split-KV3, and stable GQA architecture operators.

This branch adds an explicit `VLLM_SM70_FA2_D256_LIBRARY` source-overlay
sidecar. It is opt-in and follows the existing SM70 native-sidecar convention.
Bundled-wheel behavior remains unchanged, and missing or incompatible
operators still fail closed to the existing fallback with a warning.

## Shape boundary and next measurements

The practical chunk-4096 contract can use the exact D256 Split-D operators,
but it cannot enter the stable long GQA architecture, whose validated kernel
contract is Q8000 with KV16K..256K in 8K steps. The next paired measurements
are therefore deliberately separated:

1. chunk 4096 plus the stable sidecar, to measure the dependency-closure gain;
2. chunk 8192 plus the same sidecar, to test the already validated Q8000
   architecture with DFlash2 and NVFP4;
3. only if chunk 4096 remains materially slower, profile a quality-exact Q4K
   architecture generalization instead of weakening attention math.

Each candidate must prove operator-route hits, preserve output validity, fit
the 256K DFlash2 memory contract, and retain the quality-audit PPL and scored
coding gates. Prefix-hit time is reported separately and never counted as cold
prefill throughput.

