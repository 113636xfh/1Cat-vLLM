# Qwen3.8 Flash Next NVFP4 on SM70

## Status and ownership

- Status: source bring-up, native Qwen4Exp MTP, and prefix-cache configuration
  are implemented; focused CPU/configuration gates pass and the local
  ModelScope snapshot is fully verified. Native MTP4 now completes TP4 model
  load, graph capture, warmup, and two 1024x256 requests. Acceptance, repeated
  token equality, memory, and steady decode are measured below; matched no-MTP
  token equality and verifier cost remain pending.
- Integration line: `private/main`.
- Base SHA: `d63e9490f65f9e01f6649053c1ab72922034b931`.
- Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4` at revision
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- Model download: `/data/models/RadixArk/Qwen3.8-Flash-Next-NVFP4`.
- Download source: ModelScope `master`, verified against the fixed Hugging Face
  revision above: all 419 file sizes match and all 208 comparable LFS SHA-256
  values match. After download, all 419 local files were independently hashed
  against the ModelScope manifest with zero missing, size-mismatched, or
  SHA-256-mismatched files.
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
- Model mode: `--language-model-only`. The initial SM70 route deliberately
  excludes the vision tower from its memory, quality, and performance gates;
  omitting the flag fails during configuration instead of reaching a private
  Qwen3.5 multimodal API mismatch at model construction.
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
- Model runner: V2 is the default initial route. Its Qwen4Exp model state keeps
  raw token IDs and builds the PLE context from committed tokens, so rejected
  speculative candidates cannot leak into the next trigram. V1 remains a
  correctness control, not the primary performance route.
- Prefix caching: enabled for the MTP validation route. Hybrid recurrent state
  uses `mamba_cache_mode=align` with chunked prefill. The fixed QSA compressor
  ring is explicitly non-cacheable and is excluded from prefix-hit
  reconciliation; a clean ring block is allocated after a hit. Main QSA KV,
  compressed QSA KV, and aligned GDN/Mamba state remain cacheable.

The native-MTP validation route keeps the same TP4/PP1, FP16 activation,
ModelOpt NVFP4, language-model-only, and FlashAttention-V100 contract. It uses
four speculative tokens, V2, CUDA graphs, prefix caching, a deterministic
greedy prompt, and two identical requests. The second request is both the hot
speed sample and the prefix-cache reuse check. A matched no-MTP run is still
required for exact token-ID quality comparison and incremental verifier cost.

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

The SM70 TurboMind repack changes routed-expert FP4 scales from FP8 to FP16.
For TP4, routed experts are estimated at about 15.82 GiB/rank in the source
checkpoint and 17.57 GiB/rank after repack. This puts the idealized final
device weights near 21.3 GiB/rank. Because layers are repacked sequentially,
the estimated transient weight peak is about 22.1 GiB/rank before runtime
buffers, KV/index caches, NCCL, and CUDA graphs. These are storage calculations,
not `torch.cuda.max_memory_allocated` measurements.

The loader marks the PLE parameter as permanently host-resident so generic
quantization post-processing cannot stage the entire 11.921 GiB TP shard on a
GPU. Only its small scale parameter resides on device; lookup reads selected
FP8 rows through a stable UVA view and converts the gathered output to FP16.

With the real SM70 platform alignment and the QSA attention backend selected,
the V2 scheduler block is 784 tokens, the recurrent-state block is 32768
tokens at a 32768-token initial maximum length, and each padded recurrent page
is 802816 bytes. The exact synthetic model layout has one 24-layer uniform QSA
main/compressed group, one 12-layer fixed circular QSA ring group, three
12-layer GDN state groups, and one PLE short-convolution state group. It
allocates 24 physical cache tensors. The aligned pool cost is 10235904 bytes
(9.762 MiB) per shared block per TP rank.

For one request, the resulting cache-pool planning estimates are about 0.448
GiB/rank at 32K (47 shared blocks), 1.649 GiB/rank at 128K (173 blocks), and
3.241 GiB/rank at 262144 tokens (340 blocks). Combining the last figure with
the estimated 21.3 GiB final weights gives about 24.54 GiB/rank before CUDA
graphs, workspaces, NCCL, allocator fragmentation, and loader transients. This
explains why TP4 is plausible, but it is not evidence that the maximum context
will load safely.

## Native MTP4 TP4 validation snapshot

The first complete native-MTP run uses V2, TP4, FP16 activations, ModelOpt
NVFP4 weights, `FLASH_ATTN_V100` for target and draft attention, four draft
tokens, `mamba_cache_mode=align`, prefix caching, chunked prefill, and
FULL+PIECEWISE CUDA graphs. It runs two identical deterministic 1024-token
prompts with 256 forced output tokens each. The artifact is
`.artifacts/qwen4_exp_mtp_tp4_20260827/mtp4_prefix_graph_i1024_o256_r2_hetero_v2.json`.

- Source HEAD is `d9a39ea434` on the Qwen3.8 worktree branch. The measurement
  also sees the worktree's separately owned, uncommitted SM70 kernel changes;
  it is bring-up evidence and must be repeated from a clean, pinned source
  before becoming release-baseline evidence.
- Target plus MTP weights use 23.16 GiB/rank. The aligned MTP attention block
  is 816 tokens with 1.62% recurrent-page padding. Available KV-cache memory
  is 4.77 GiB/rank, or 219,942 tokens. Observed peak device memory is
  32,330 MiB on every 32,768-MiB V100; the run therefore has only 438 MiB of
  peak device headroom at `gpu_memory_utilization=0.90`.
- PLE remains host-resident at 11.92 GiB/rank. The sampled minimum host
  `MemAvailable` is 48.107 GiB and minimum free swap is 247.994 GiB.
- Both repeats emit 256 tokens and are exactly equal token-for-token. The first
  request has 9.750-second TTFT and 53.125 steady decode tokens/s. The repeated
  prefix has 2.667-second TTFT and 52.187 steady decode tokens/s. Mean steady
  decode is 52.656 tokens/s; the first end-to-end request includes prefill and
  JIT and is not a decode baseline.
- Across 256 speculative steps, the MTP head proposes 1,024 draft tokens and
  254 are accepted. Mean acceptance length including the target bonus is
  1.9921875. Draft acceptance is 24.8047%; per-position acceptance is
  54.6875%, 26.5625%, 13.28125%, and 4.6875%.
- A strictly matched no-MTP run is required before claiming target-token
  equality or a verifier-cost ratio. Older 1024x256 artifacts use the distinct
  Qwen3.8-27B checkpoint and are not valid controls for Flash Next.

## Acceptance gates

1. Static route: Transformers config, model registry/processor registration,
   QSA/GDN/GR/PLE modules, and ModelOpt NVFP4 mapping load without importing an
   Ampere-only backend. Multimodal execution is outside this first route.
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

## Source validation snapshot

- The ModelScope download completed successfully at the path above. A full
  post-download verification checked 419 files and about 125.910 GiB of
  safetensor payload: zero files were missing and zero size or SHA-256 values
  differed from the remote manifest.
- The real downloaded `config.json` resolves without remote model code as
  `Qwen4ExpConfig` / `Qwen4ExpTextConfig`: 48 layers, 36 GDN, 12 QSA, 512
  experts, top-10, HC count four/rank 320, and one trigram PLE layer.
- Exact-SM70 configuration construction with FP16, TP4, prefix caching off,
  language-model-only mode, and V2 selects `ModelOptNvFp4Config`, the
  pinned-host PLE default, and the Qwen4Exp PLE/QSA compilation split
  operators. The same real configuration rejects the unvalidated multimodal
  route with an actionable `--language-model-only` error.
- Exact real-checkpoint construction with native MTP4 and prefix caching on
  resolves the target as `Qwen4ExpForConditionalGeneration`, the draft as
  `Qwen4ExpMTP`, target and draft attention as `FLASH_ATTN_V100`, and recurrent
  caching as `align` with 16-token configured blocks and chunked prefill.
  Focused prefix-cache/QSA coordinator and model-config tests pass. The
  non-cacheable QSA compressor ring is skipped during prefix-hit matching,
  matching the current upstream Qwen4Exp contract.
- TP4 loaded all 206 checkpoint shards in 109.27 seconds and then correctly
  rejected an incomplete runtime extension set: `_C` and
  `_C_stable_libtorch` were present, but `_moe_C` was omitted. A complete
  runtime must carry all three. The matching `_moe_C` artifact has SHA-256
  `a14eeb4fa06947e335cf69ee188e23509fc294da61cada786df09888ca5b4469`;
  its graph-safe permute, workspace-size, unpermute schemas, and SM70 support
  probe all pass before the next full-model attempt.
- Full 48-layer meta construction from the real checkpoint config succeeds in
  language-model-only mode. It instantiates QSA, GDN, HC, PLE, and all 512
  experts without materializing weights; the routed experts select
  `ModelOptNvFp4SM70MoEMethod(use_a16=True)` and the PLE table has shape
  `(320001536, 160)` with FP8 E4M3 storage. This constructor probe used TP1;
  TP4 selection and expert geometry are covered separately. Full TP4 target
  plus native-MTP loading now completes with 23.16 GiB/rank of loaded model
  state before cache and graph allocation.
- Focused CPU tests cover PLE shard loading and hashing, `seed=None`, permanent
  host residency during post-load processing, QSA cache grouping, V1 and V2
  n-gram inputs, V2 circular block-table sizing, scheduler-manager conversion,
  official checkpoint weight mappings, and Qwen3.6/Qwen3.8 NVFP4 route
  selection. The current CPU-only focused run is 76 passed and 7 CUDA skips;
  all 55 changed Python files pass Ruff, format, and compileall checks.
- In the pre-final real V100-SXM2-32GB snapshot, 63 focused tests pass. They
  include the Triton V2 slot-mapping kernel with its QSA circular group
  disabled, pinned-host FP8 lookup through a CUDA UVA view, the compressed QSA
  storage-page reshape, V2 committed-token PLE state, and the SM70 ModelOpt
  NVFP4 selection gates. A final V100 rerun is still required after the current
  GPU owners release a device.
- The upstream QSA fused pre-indexer executes on SM70 for both ordinary RoPE
  and MRoPE inputs and matches a PyTorch normalization reference. This also
  exposed and fixed two private-tree API differences: QKV projection is local
  because the branch's `Qwen3NextAttention` has no `_project_qkv_gate`, and its
  `triton_mrope` accepts eight rather than nine arguments.
- Actual SM70 platform alignment produces a 784-token attention block and an
  802816-byte padded recurrent page; the exact synthetic 48-layer cache layout
  validates successfully after that alignment.
- The HC grouped norm/gate/combine kernels pass FP16 reference checks on a real
  V100. A captured pinned-host FP8 embedding probe (228.9 MiB synthetic table,
  16 rows by 160 elements per replay) measured 95.81 microseconds/replay,
  including the input-ID copy. This only demonstrates that the isolated UVA
  lookup can be captured and is not by itself an end-to-end throughput result
  or a measurement of the full 11.921 GiB TP shard.
- The existing general KV-cache utility/manager suites pass 69 tests; one
  unrelated DeepSeek-v4 fixture failure is unchanged from the integration
  base because its `SimpleNamespace` omits `max_in_flight_tokens`.
