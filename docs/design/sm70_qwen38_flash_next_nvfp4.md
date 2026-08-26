# Qwen3.8 Flash Next NVFP4 on SM70

## Status and ownership

- Status: bring-up in progress; no route, quality, memory, or speed claim yet.
- Integration line: `private/main`.
- Base SHA: `d63e9490f65f9e01f6649053c1ab72922034b931`.
- Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4` at revision
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- Model download: `/data/models/RadixArk/Qwen3.8-Flash-Next-NVFP4`.
- Upstream references:
  [vLLM PR 53896](https://github.com/vllm-project/vllm/pull/53896) and
  [SGLang PR 36497](https://github.com/sgl-project/sglang/pull/36497).

The upstream PRs are implementation references, not acceptance evidence. Both
were still open when this work started, so imports must be narrowed to the
Qwen4Exp route and validated against this branch.

## Frozen first-pass contract

The first correctness route deliberately excludes speculative decoding.

- Hardware: four NVIDIA V100-SXM2-32GB GPUs (SM70).
- Parallelism: TP4, PP1, no expert parallelism.
- Compute dtype: FP16; no BF16 or native FP8/NVFP4 tensor-core assumptions.
- Checkpoint: ModelOpt NVFP4 routed-expert weights, consumed as an SM70
  weight-only W4A16 route. Ignored dense, attention, GDN, shared-expert, GR,
  PLE, and LM-head modules stay in their checkpoint dtypes and execute in
  FP16 where required.
- PLE/N-gram table: allocate and load each TP shard directly in pinned host
  memory. It must never be materialized on a GPU before being moved to the
  host. Gathered rows are transferred asynchronously and converted to FP16.
- Initial KV cache: FP16. FP8 KV cache is a separate, quality-gated follow-up.
- Initial decoding: MTP disabled. MTP may be enabled only after the no-MTP
  route is correct and its emitted-token baseline is recorded.

## Architecture facts that affect the port

The text stack has 48 layers: 36 gated-delta-net layers and 12 QSA layers in a
3:1 pattern. Hidden size is 2560. QSA uses 24 query heads, two KV heads,
head-dimension 256, index dimension 128, compression ratio four, and a 2048
token sparse budget. The MoE has 512 routed experts, top-10 routing, a 640-wide
routed expert, and one 640-wide shared expert. General residual connections
use four streams and rank 320.

PLE is a learned trigram embedding, not prompt-ngram speculative decoding. It
uses 16 heads (`ngram_size=3`, eight heads per n-gram order), embedding width
2560, and FP8 E4M3 storage. Native speculative decoding is the separate
one-layer MTP head.

## Memory budget hypothesis

Safetensor payloads total about 125.910 GiB. The sharded PLE payload is about
47.684 GiB, or about 11.921 GiB of pinned host memory per TP rank. Removing PLE
from device residency leaves an idealized 19.556 GiB of checkpoint payload per
GPU before replicated tensors, KV/index caches, CUDA graphs, and workspaces.

This is a planning bound, not a measured peak. Startup must record host RSS,
pinned memory, per-rank device peak, post-load device residency, and whether a
loader creates duplicate staging buffers. A 262144-token context is admitted
only after the measured peak leaves a safe margin on every 32GB GPU.

## Acceptance gates

1. Static route: Transformers config, model registry, multimodal processor,
   QSA/GDN/GR/PLE modules, and ModelOpt NVFP4 mapping load without importing an
   Ampere-only backend.
2. Loader route: TP4 expert shards select TurboMind SM70 W4A16; PLE shards are
   born on pinned CPU memory and do not consume persistent device memory.
3. Numerical route: focused operator comparisons against FP32/FP16 references,
   followed by deterministic token-ID and output checks on the full model.
4. Memory route: 32K bring-up first, then 128K and the exact 262144 boundary;
   report controlled OOM separately from corrupted output.
5. Performance route: one request, TP4, PP1, MTP off, FP16 activations, 8192
   input tokens and 512 output tokens. Report TTFT/prefill separately and
   calculate steady pure decode from emitted tokens 33-512. The target is at
   least 100 emitted tokens/s (at most 10 ms/token) with CUDA graphs enabled.
   Record an otherwise identical eager control.
6. MTP follow-up: report accepted length, target passes, emitted tokens/s, and
   output quality separately; do not compare accepted candidates with emitted
   tokens.

## Initial implementation boundary

Reuse the upstream Qwen4Exp Python structure and tests where they match this
tree. Do not import unrelated AMD, SM90, build-system, or broad engine changes.
The first SM70-specific changes are limited to genericizing the existing
TurboMind NVFP4 MoE shape contract, adding the QSA/indexer route, and adding a
pinned-host PLE loader/gather path. Optimize GDN, GR, sparse attention, and MTP
only after profiles identify them as measured decode bottlenecks.
