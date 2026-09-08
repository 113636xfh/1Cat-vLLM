# Native MiniMax H3 migration control

Status: implementation in progress; no video quality or 80 TFLOPS acceptance yet.

Workflow/LoRA expansion is tracked in the 2026-09-08 entry at the end of this
file and [WORKFLOWS.md](WORKFLOWS.md). Its distilled schedules are separate
from the fixed no-LoRA kernel-performance acceptance contract below.

Current decision: the user selects FlashInfer-SM70 as the H3 denoiser mainline.
Optimize complete denoise toward >80 useful TFLOPS on every participating GPU;
keep Flash-V100 as an explicit control. Development uses 39-frame clips and
20 actual updates for quality. The historical 60-TFLOPS D256/GQA Flash-V100
architecture was audited, but its H3 adaptation is not the selected mainline.

## Fixed scope

- Base: onecat/main 56f534e672657a6c7599afd6c0dcb2e2c211b2e3.
- Native model/pipeline/service; no vllm-omni runtime dependency.
- Model sources, licenses and revisions: see
  `vllm/model_executor/models/minimax_h3/UPSTREAM.md`.
- FL2VA and Ref2VA, original BF16 checkpoint and Comfy INT8 ConvRot AdaLN-pruned checkpoint.
- Four V100-SXM2-32GB, TP4 DiT/text encoder, native four-rank VAE tile parallel.
- FP16 compute with FP32 latent projections, timestep computation, pruned AdaLN and output heads.
- Acceptance: 1344x768, 243 frames, 24 FPS, seed 42, 50 sigma positions,
  no LoRA/step skipping/approximate cache; record actual DiT forwards.
- Each rank median useful model FLOPs / complete-denoise wall time >80 TFLOPS;
  warmup once, then three unprofiled measurements; denoise CV <=5%.
- Two attention backends must pass quality. Fastest qualified backend carries
  performance acceptance. BF16 checkpoint performance is reported separately.

## Implementation evidence

- Created isolated owned worktree and branch; canonical checkout left untouched.
- Native H3 packed layouts, reference processing, scheduler, transformer, Qwen3VL
  encoder and VAE adapter ports in progress. Omni framework, sequence-parallel,
  LoRA and approximate cache dependencies removed from native execution path.
- Signed INT8/FP32 scales, runtime QKV order, fused FFN shard loading and
  pruned AdaLN interpolation ported from open PR 6894.
- W8A16 currently has an explicit PyTorch numerical reference. This is not
  TurboMind kernel completion or a performance result.
- Flash-V100 path connected; FlashInfer D128 non-causal path must be implemented
  before backend support or qualification is claimed.

## Environment and failed paths

- Owned Python 3.12 environment uses Torch 2.10.0+cu128, Transformers 5.15.1,
  Diffusers 0.40.0. System nvcc is 12.0; owned CUDA 12.8 toolkit pending.
- Initial dependency resolution tried to upgrade Torch/CUDA; interrupted before
  installation and used pinned packages without replacing the shared environment.
- Bootstrap vLLM binaries are read-only links from the local 1Cat 1.5.0 environment;
  hashes are recorded in task artifacts. They are not a new source-build result.
- Upstream modulation and QK/RoPE kernels contain explicit BF16 roundings. These
  require FP16 adaptation and numerical tests on SM70 before performance use.

## Remaining gates

1. Targeted format, packing, TP/shard, FP16 numeric, padding and reference tests.
2. Native serial CLI/API, dependency installation, media export and job lifecycle.
3. TurboMind signed W8A16, bounded cache and real FlashInfer-SM70 non-causal D128.
4. Actual checkpoint loading and both-partition GPU functional generation.
5. Full video quality, fixed-shape performance measurements and profiler evidence.
6. Three review scopes/Draft PRs with signed commits and reproducible artifacts.

Every failed GPU experiment must record configuration, result and the resulting
implementation decision here or in linked benchmark evidence. Do not repeat an
unchanged experiment. No acceptance result may be inferred from route imports,
GPU utilization, synthetic operator peaks, or estimated hardware capabilities.

## 2026-09-08 native and operator checkpoint

- `tests/video`: 14 passed, including signed INT8/row-scale restoration,
  Kronecker ConvRot256/inverse, original QKV reorder, curve interpolation,
  partition metadata, 243-frame/49-forward schedule, poisoned padding,
  both non-causal attention backends on V100, W8A16 FP32 reduction, encoder
  causal GQA, and serial HTTP job lifecycle.
- Synthetic two-block INT8 curve DiT: TP1 versus TP4 on GPU 0–3 passed on every
  rank. Repeated after replacing reference linear execution with native W8A16;
  all four ranks passed (atol 0.0005, rtol 0.02). This is a correctness test,
  not full-model or performance acceptance.
- Native W8A16 decodes signed INT8 with original FP32 per-row scales, rotates
  activations in FP32 with FP16 output and calls cuBLAS FP16 GEMM from the
  TurboMind SM70 operator module. Uses explicit FP32 compute and disables
  reduced-precision reductions. Initial cuBLAS math-mode comparison failed;
  explicitly disabling reduced-precision reductions fixed the mismatch.
- FlashInfer-SM70 now owns a D128 non-causal online-softmax WMMA operator using
  the repository's Volta QK/PV primitives. Lengths 1,17,63,64,65,243,1025 passed
  the FP32 reference (largest observed absolute error 0.0009765625). No
  Flash-V100 delegation occurs in this route. Large-sequence optimization and
  full-video quality remain pending.
- CLI `vllm video generate/serve`, serial worker engine, HTTP jobs, optional
  video dependencies, fixed-list FP16 cache, useful-FLOP hooks, NVML sampling,
  media export and automated video checks are implemented. Runtime validation
  with actual full checkpoints remains pending while weights download.
- CUDA Toolkit 12.8.93 nvcc installed in the task artifact directory; independent
  extension builds passed against Torch 2.10.0+cu128. Wheel targets added, but
  complete wheel build has not yet been validated.
- Nsight Compute on GPU 0 returned `ERR_NVGPUCTRPERM`. No counters were collected.
  Do not claim Tensor Core activity/occupancy from this run. Nsight Systems
  tracing is being checked separately. No global driver settings were changed.

Raw evidence (not committed): task artifact directory
`/data/minimax-h3/native-h3-20260908/`, including `native-tests.log`,
`tp1-smoke.log`, `tp4-smoke.log`, `tp4-w8a16-smoke.log`,
`build-owned-extensions.log`, `flashinfer-numerics.log`, and `profiles/`.

### Actual components and loader recovery

The first real TP4 Qwen3VL encode exposed a missing optional residual argument
in the native RMSNorm adapter. Restored the reference residual-add/FP16 rounding
contract and added a focused regression; full TP4 encode is being repeated to
validate that fix. The VAE component checks produced finite 22-frame 64x64 video
(15,278,413,824 peak allocated bytes on GPU 0) and finite 32-kHz mono audio. T=2
was rejected by the VAE streaming decoder; the valid component smoke uses T=7.
The original text-encoder shard 10 stalled in Xet reconstruction. Recovered it
through HTTP, verified SHA256 `aded5a4d1d5e22dbd8b6f79266b6eb88c840411b09527c53917a1419ace22e2f`,
and stopped only the owned stalled downloader. INT8 downloads continue separately.
Nsight Systems 2025.1 CLI produced a usable trace; the W8A16 projection launched
`cutlass_70_tensorop_f16_s884gemm_relu_f16_128x128_tn_align8`. This confirms the
Tensor Core kernel route, not measured Tensor Core activity or full-denoise speed.

The next real-encoder check exposed an integration mismatch: Omni's encoder
ignored the return value of `group.all_reduce`, while native GroupCoordinator
can return a new tensor. Fixed both embedding and row-parallel reductions and
added a functional-collective regression. All four real TP4 encoder ranks now
have finite layer-50 output with identical amax 16088. The diagnostic comparison
initially attempted NCCL broadcast on the encoder's CPU result; the test is
being corrected to broadcast a CUDA copy before checking all-rank identity.

### Real INT8 FP16-range fixes (2026-09-08)

- The corrected text-encoder comparison passed bitwise identity across all
  four TP ranks, with finite layer-50 FP16 output. The full FL2VA INT8 file and
  Ref2VA INT8 file are downloaded. Original BF16 DiT shards are still downloading.
- Native explicit width/height requests previously failed Omni's separate
  aspect-ratio requirement. The native canvas now supplies that ratio; two
  canvas regressions pass, including the exact 1344x768 primary shape.
- The first real single-step pipeline reached DiT, then rejected non-finite
  velocity. Layer diagnostics found condition projection values up to 76615.5,
  block residuals above four million, and overflowing row projections and
  gated MLP products. FP16 accumulation settings alone cannot represent these
  values. Preserve condition projection, residuals and gated products in FP32.
  Normalize and convert attention/MLP GEMM inputs to FP16; use exact power-of-two
  row scaling for wide MLP activations, restoring scale in FP32 GEMM output.
  INT8 weights, FP32 checkpoint scales and ConvRot256 remain intact.
- A real 50-block INT8 DiT forward at the 256x256 diagnostic shape now has finite
  output on all four ranks (video amax 10.8243, audio amax 4.36589). This is one
  forward, not a complete schedule, quality pass or performance result.
- The current native suite passes 23 tests, including actual CUDA checks for
  FP32 residuals, above-FP16-range Tensor Core outputs, restored activation
  scales, signed INT8, padding and alias-preserving CPU/GPU staging.
- Pinned staging now replaces one host allocation at a time. The old snapshot
  retained all pageable weights while building the entire pinned copy, raising
  transient TP4 host memory and swap pressure.
- Nsight Compute succeeded with the user's sudo authorization. The earlier
  ERR_NVGPUCTRPERM result is superseded: the isolated FP16-output GEMM recorded
  462422016 Tensor Core instructions and 86.177% active tensor-pipe cycles.
  The new FP32-output variant needs its own profile. Complete-denoise >80 TFLOPS,
  main-shape memory, full-video quality and both-backend quality remain pending.
- Review scopes: native #557, kernels #558 (stacked). The next end-to-end run
  initially found both GPU groups occupied by other vLLM workers; the owned
  launcher waits for a free group without changing those processes.

Evidence: `encoder-real-tp4-final.log`, `pipeline-route-canvas.log`,
`dit-nan-diagnostic.log`, `dit-nan-fp32-condition.log`,
`dit-nan-fp32-residual.log`, `dit-nan-fp32-output.log`,
`dit-nan-fp32-gated.log`, `native-tests-fp32-islands.log`,
`kernel-build-fp32-output.log`, and `profiles/w8a16-hmma-root.ncu-rep`
under the task artifact directory. Do not repeat superseded failing routes
unless a new change requires them.

### Short development workloads and attention evidence (2026-09-08)

- User requested 1–2 second development clips and no repeated full-video runs.
  Native requests now accept 22 and 39 aligned frames; 39 frames at 24 FPS is
  1.625 seconds. The primary 243-frame/49-forward acceptance contract is unchanged.
  Ten targeted shape/schedule tests pass, including short audio latent alignment
  and rejection below the streaming VAE's minimum temporal chunk.
- The earlier primary run was interrupted before completion; the host rebooted.
  Its log reaches 22/49 DiT calls. Preserve it as partial diagnostic evidence,
  never as a completed run or acceptance measurement. Initial stable calls took
  about 129.5 seconds; late slow calls are not a reproducible speed baseline.
- NVML medians were 100% GPU utilization. Nsight Compute separately measured
  13.886% tensor-pipe activity in main-shape Flash-V100 attention and 78.150% in
  the FP16-input/FP32-output Tensor Core GEMM. Utilization is not useful TFLOPS.
  Attention at 73483 tokens took about 2.262 seconds without profiler, explaining
  most of the observed DiT time. The performance gate remains unqualified.
- Native TP4 video VAE decoded 39 frames at 1344x768 using 28 tiles. All four
  outputs were finite with the expected shape; each peak allocation was
  17805820416 bytes (16.58 GiB). Decode took 4.62–5.50 seconds per rank, without
  a synchronized performance protocol; this is a functional result only.
- CMake configure/build/install succeeded for both H3 extension targets. Five
  targeted CUDA numerical tests passed against those installed modules. This
  does not yet establish a complete release-wheel build.
- All fixed-revision original and Comfy weights have downloaded and checksum
  manifests are retained. Host memory is shared with another service, so current
  short denoise diagnostics reuse the previously validated TP4 text embeddings
  and load the VAE after releasing DiT, rather than retaining every component.
  Such runs must explicitly report cached text and separate loading costs.

Evidence under `/data/minimax-h3/native-h3-20260908/`:
`short-clip-contract-tests.log`, `vae-tp4-short.log`,
`cmake-numerics-tests.log`, `interrupted-baseline-summary.json`,
`original-checkpoint-manifest.json`, `comfy-checkpoint-manifest.json`,
`profiles/flashv100-attention.summary.json`, and
`profiles/w8a16-fp32-output.summary.json`.

### Short-video export and GPU ownership diagnostics

The register-softmax FlashInfer implementation in kernel draft #558 passes 38
native numerical/service tests. Its cached-text INT8 TP4 development run exported
a 1344x768, 39-frame, 24-FPS video with 1.632 seconds of decoded audio. Automatic
frame/dimension/audio/finite/black/static checks pass. Visual inspection of the
first screenshot shows pronounced ghosting and grid artifacts; the two-forward
schedule is an execution check and is not a quality pass.

The first export retry encountered other workers taking 27 GiB per GPU and ran
out of memory. A subsequent diagnostic shell performed a preflight but continued
after Python returned an error; it ran beside other workers using about 13 GiB
each. That run's timing is explicitly invalidated in its JSON. The native engine
already propagates selection errors; the separate diagnostic launcher now also
propagates them and checks external GPU processes before and after denoise.
NVML records now include compute PIDs and their memory to expose interference.
A 12-sample read-only telemetry check passed on the live GPU inventory.

Retained evidence: `outputs/dev39-int8-final/FLASHINFER_SM70/` (video, original
audio, latents, frames and invalidated timing), `denoise-short-int8-final.log`
(OOM), `denoise-short-int8-final-retry.log` (export passed, timing excluded),
`run_short_guarded.py`, and `nvml-process-record-smoke.jsonl` in the artifact
directory. As of this checkpoint GPU groups 0–3 and 4–7 have other vLLM workers.
No owned long-video or waiting-GPU process is being retained. BF16 and reference
generation checks still need an available four-GPU group.

### Twenty-step short-clip quality check requested

The user prioritizes normal output quality over further throughput tuning and
requested a 20-step check. The diagnostic now fixes 39 frames at 1344x768,
24 FPS, seed 42, the same INT8 checkpoint and FlashInfer implementation, and
20 actual DiT calls (`num_inference_steps=21` under the reference sigma-point
convention). Video/audio shifts remain 12/3, Turbo LoRA is off, and FP16 weight
cache is off. Reuse the verified text conditioning for the same prompt to keep
the comparison focused on the changed step count. Assert the completed call
count and retain latents before export.

At preparation time both GPU groups were occupied. A later idle GPU snapshot
was still covered by another task's group lease. Capacity-only preflight can
race such a task during service replacement. H3 now acquires the shared 1Cat
per-card/group locks, checks capacity again, and retains the lease until its
workers have exited. Failed acquisition/startup releases its own partial locks.
The quality diagnostic uses the same lease and propagates launch errors. Ten
CPU lease/service tests pass, including partial-lock rollback, whole-group
fallback and worker-startup failure. No unrelated process was stopped.

The 20-step GPU/video result is pending; do not infer that low step count is
the sole cause of the two-step clip's artifacts. Prepared contract and commands
are retained as `quality39-20steps-contract.json`, `denoise_quality.py`,
`run_quality_guarded.py`, and `quality39-20steps-launch.log` in the task artifact
directory. `gpu-lease-tests.log` records the focused regression result.

### GPU 0–3 priority authorization

The user explicitly prioritizes this H3 task on GPU 0–3 and authorizes stopping
conflicting jobs there. This authorization persists across turns; do not ask
again for the same GPU allocation. GPU 4–7 services remain outside that scope.
After checking PID/command identity and preserving its active-job metadata,
stopped the conflicting quasar lease scheduler PID 17903 with SIGTERM. Its
GPU child had already exited; no model/log/queue files were deleted. Acquired
the shared 0–3 lease for H3 and started the fixed 20-forward quality check.
The CPU text conditioning, INT8 weight encoding, shift rules and FP16 cache
settings match the earlier short clip. Keep quality as pending until the video
has decoded and been reviewed.

The interruption record is `gpu03-priority-handoff.json`; the active test log is
`quality39-20steps-run.log` in the retained artifact directory.

### Completed twenty-forward INT8 short quality comparison

Both backends completed the fixed 1344x768, 39-frame, 24-FPS, seed-42 request
with 20 actual denoise calls, after one single-call warmup. Original INT8/FP32
scale and ConvRot data, cached verified text conditioning, sigma shifts 12/3
and cache-off settings were retained. Each GPU had only its corresponding
H3 rank in the recorded NVML compute-process samples.

| Backend | Complete denoise | Seconds/call | Useful TFLOPS/rank |
| --- | ---: | ---: | ---: |
| FLASH_ATTN_V100 | 113.459387 s | 5.672969 | 30.5286 |
| FLASHINFER_SM70 | 105.642574 s | 5.282129 | 32.7875 |

All video/audio latents are finite and bitwise identical between TP ranks within
each backend. Both exported videos pass full decoding, 39-frame dimensions/FPS,
valid finite audio duration, no-black and no-prolonged-static checks. Inspecting
frames 0/19/38 for FlashInfer and 0/38 for Flash-V100 shows a clear red paper
boat, yellow duck, reflections and stable foliage; the broad grid and ghosting
from the two-call sample are absent. The controlled step-count comparison
supports undersampling as the main cause of those severe artifacts. It does
not prove universal quality or complete the primary-video acceptance gate.

Backend outputs are similar, not identical: decoded RGB PSNR over all 39 frames
is 30.888 dB; final video/audio latent relative L2 differences are 0.0777004 and
0.0153296. These numbers are auxiliary, not quality acceptance criteria. Audio
is valid 32-kHz stereo; semantic listening review is still pending, as is the
user's five-axis final review. No 80 TFLOPS qualification is claimed.

Evidence: `outputs/quality39-int8-20steps/` contains both videos, original audio,
latents, NVML samples/curves, rank/phase CSV, automated checks, visual-review
notes and backend comparison JSON. `quality-two-vs-twenty.png` compares the
same-seed two- and twenty-call outputs. Raw logs are
`quality39-20steps-run.log` and `quality39-20steps-flashv100.log`.

### Original-checkpoint short quality and FlashInfer mainline profiling

The original BF16 checkpoint also completed FL2VA text-to-video at 1344x768,
39 frames, seed 42 and 20 updates through Flash-V100, using FP16 matrix inputs
and FP32 sensitive intermediates on V100. Complete denoise took 110.569167 s
(5.528458 s/update), with 31.328872 useful TFLOPS per rank. DiT-only peak Torch
allocation was 17.186255 GiB; this does not include a whole-pipeline memory peak.
All automatic media checks pass and inspected first/last frames show a clear
boat and duck without the severe two-update artifacts. Human audio review,
Ref2VA/reference generation and primary acceptance remain pending. Evidence is
`outputs/quality39-original-20steps/FLASH_ATTN_V100/` in the artifact directory.

The user subsequently fixed FlashInfer-SM70 as the optimization mainline and
reaffirmed >80 TFLOPS/card. The CLI/config default now follows that choice.
A four-rank Nsight Systems trace captures the first two updates from the
unchanged 20-update schedule, with cached verified text, 39 frames and cache
off. It truncates the schedule for profiling and makes no quality or acceptance
claim. The rank-0 synchronized denoise span is 10.600430 s:

| Exclusive wall category | Two-update seconds |
| --- | ---: |
| FlashInfer attention | 5.466013 |
| Model GEMM | 2.692784 |
| TP communication | 1.088080 |
| Other GPU kernels | 1.011361 |
| ConvRot | 0.218711 |
| Weight dequantization | 0.061064 |
| Copies | 0.012464 |
| No recorded GPU activity | 0.049953 |

The parser keeps kernel service and exclusive wall coverage separate, including
an overlap category if present. This trace prioritizes attention and then TP
communication over weight caching or launch-overhead tuning. It must not be
substituted for the unprofiled 20-update result or primary three-run gate.

An initial warp-owned-query prototype is rejected: all sampled FP32 references
pass, but its Q64/K32, Q64/K64 and Q128/K32 variants take about 89.36, 87.88 and
59.46 ms at 12323 tokens, versus 53.27 ms for the retained kernel. Registers rise
to 198/230 per thread. Do not repeat those unchanged variants. A transposed V
layout and software-prefetch follow-up are being evaluated separately.

Evidence: `profile_h3_steps.py`, `run_profile_h3_steps.py`,
`profiles/h3-flashinfer-mainline-steps.nsys-rep`, the matching SQLite,
`profiles/h3-flashinfer-step-breakdown.json`, `query-owned-results.json`, and
`flashv100-60t-route-audit.md`. GPU 0–3 priority preemption is authorized; the
active development lease is recorded in `flashinfer-development-lease.json`.

### Official workflows and LightX2V Turbo expansion (2026-09-08)

Owned branch: `codex/v100-h3-workflows-lora-20260908-091941`.
Stack base: native PR #557 at `1d201f41344f1a9a50d91197a8ad3a5525e51190`;
integration remains `onecat/main` (observed `e5d63c51f0fcc1ddf75d229e3df06bf52df206f5`).
This scope adds workflow/LoRA execution and does not change the other tasks'
attention kernels. [Omni coverage](../omni_workflow_coverage.md) records the
90-row upstream support inventory, relevant task history and remaining families.

- Added complete-layout LightX2V Diffusers Turbo loading for FL2V and Ref2V,
  four/eight updates, 544p/768p training variants, metadata alpha and per-artifact
  modality shifts. Every A/B tensor must be consumed; ambiguous directories,
  wrong partitions, other export layouts and malformed shapes are rejected.
- TP-local Q/K/V delta slices and reordered MLP gate/value rows use the existing
  FP16-input/FP32-output GEMM with range scaling. INT8 ConvRot base weights stay
  unchanged; LoRA consumes unrotated activations. Buffers join pinned staging.
- CLI and HTTP expose task, flow shifts, reference-video offsets and request
  LoRA scale. Omitted sampling values come from the adapter; explicit mismatches
  fail before dispatch. Scale zero restores base defaults and bypasses deltas.
- Reference audio/video metadata is validated before queueing using the same
  source checks as preprocessing. The service remains healthy after rejected
  media. LoRA GEMM work is counted separately from base work.

Validation environment: Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8,
Transformers 5.15.1, Diffusers 0.40.0, V100-SXM2-32GB TP4, GPU0-3.
Signed Comfy INT8 and FP32 scales, AdaLN pruning and ConvRot256 retained;
FP16 weight cache and approximate caches off. Denoising uses the copied,
hash-recorded #558 FlashInfer-SM70 development binary; text encoder keeps its
causal Flash-V100 path. This is not a rebuilt release wheel. All generation
runs below encode their actual prompt/media and decode/export fresh audio/video.

| Workflow | Adapter | Frames | Calls/rank | Encode | Complete denoise | VAE | Generation total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T2VA | FL2V 4-step v1.2 768p, alpha 8, shifts 6/3 | 39 | 4 | 5.76 s | 27.57 s | 6.65 s max rank | 43.94 s |
| First+last FL2VA | same | 39 | 4 | 12.97 s | 34.36 s | 6.11 s max rank | 57.40 s |

Both outputs are 1344x768, 24 FPS, seed 42, with native finite 32-kHz audio.
All automatic checks pass. Inspected first/last screenshots show the red paper
boat and yellow duck; the FL2VA endpoints follow the supplied frames. Human
listening/temporal review is pending. The peak per-rank allocation is 16.59 GiB
for T2VA and 16.61 GiB for FL2VA. Cold worker startup is separate: maximum
94.26 s and 106.30 s, respectively. These are single functional runs with no
warmup/three-repeat performance protocol, not formal speed or quality gates.
The T2VA run preceded the separate LoRA FLOP counter addition; its recorded
numerator excludes adapter work and must not be used for TFLOPS claims.

Commands (from the owned worktree, with task-owned caches and media tools):

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest --confcutdir=tests/video tests/video -q
.venv/bin/ruff check \
  vllm/entrypoints/cli/video.py \
  vllm/model_executor/models/minimax_h3/{config,lora,pipeline,reference_video,validation}.py \
  vllm/video/{server,metrics}.py \
  tests/video/{test_h3_lora,test_h3_workflows}.py
```

Result: 84 passed, 8 GPU tests skipped by the CPU-only environment. Real TP4
CLI generation above provides GPU validation of both active adapter loading
and full pipeline execution. The new tests cover all eight official media
combinations, malformed media before dispatch, all eight Turbo filename
contracts, metadata alpha, TP1/2/4 algebra, ConvRot basis and exact zero-scale
bypass. The first fixture attempt lacked FFmpeg on PATH; rerunning with the
existing task-local FFmpeg/FFprobe 7.0.2 passed. Do not diagnose that setup error
as a model failure.

Ref2VA four-step v0.1 weights were downloaded and validated (624 tensors,
alpha 8, shifts 12/3). The mixed-reference real run reached four-rank adapter
binding and the 10337-token Qwen presentation for one image, one video and
standalone audio, then received SIGTERM (exit 143) before denoise. It produced
no completed video and is not counted as passing. Another H3 task held GPU0-3
immediately afterwards. A prior launch was correctly rejected while the group
was leased. No other task's process was stopped by this scope.

For short mixed-reference development use 56 frames: the existing post-encode
reference-audio length check rejects a 39-frame embedded soundtrack after it
is truncated below two seconds. This does not affect the upstream 4–15-second
output contract; it remains an explicit short-development limitation.

Retained artifacts: `/data/minimax-h3/workflows-lora-20260908/`:
`ownership.txt`, `worktree-create.log`, `binary-hashes.txt`, `lora-sha256.txt`,
`omni-supported-models.md`, `omni-h3-recipe.md`, `omni-lora.py`,
`cpu-final.log`, `workflow-tests.log`, `t2va-turbo4.log`, `fl2va-turbo4.log`,
`ref2va-turbo4-wait.log`, `run_reference.py`, and `outputs/*-turbo4/`.
Model and adapter weights remain outside Git. Original/Comfy and Turbo source
revisions are in `UPSTREAM.md`.

Remaining gates: completed mixed Ref2VA, original BF16-base + LoRA generation,
eight-step real outputs and other artifact versions, additional seeds, full
quality/audio review and release-wheel validation. FlashGen, FastH3, combined
partition serving and upstream multipart upload semantics remain separate
work. Keep the change Draft until its required review/quality gates pass.

### Direct frontend API expansion (2026-09-08)

The user authorized all official workflow capabilities and clarified that the
application frontend calls native vLLM APIs directly. ComfyUI integration is
not required. No ComfyUI source or runtime has been added. The full remaining
scope is retained in [ADAPTATION.md](ADAPTATION.md).

Continuing the owned PR #565 branch from `92e8c18e2beda42303268979b89519908740fd1c`:

- Added JSON and multipart request normalization, typed HTTP(S)/data URL
  references and request-owned temporary media. File names cannot choose staging
  destinations; media size/type/metadata checks run before worker dispatch.
- Added synchronous MP4 return, multiple outputs with seed offsets, async job
  listing/deletion, indexed downloads and model discovery. The frontend contract
  and examples are in [API.md](API.md), and `/openapi.json` includes request schemas.
- Preserved native adapter defaults consistently across transports: a startup
  adapter is active unless `lora_scale=0`. A request `lora` object must select
  that loaded file. This differs from upstream's preload-only PEFT default and
  is documented explicitly; adding `model` does not toggle activation.
- Tests cover 11 input combinations, source-file preservation, staged-file
  cleanup, malformed requests, sync/async results, output indexing and schemas.

CPU command and result: the existing `PATH="$PWD/.venv/bin:$PATH"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest --confcutdir=tests/video
tests/video -q` passed **110 tests**, with 8 GPU-only tests skipped. The complete
pre-commit checks on changed files passed. Raw records are
`/data/minimax-h3/workflows-lora-20260908/api-cpu-all.log` and
`api-precommit.log`; changed API source hashes are in `api-source-hashes.json`.

The new ASGI-to-native-engine mixed Ref2VA test was attempted once using
`run_api_reference.py`, with actual image/video/audio uploads, INT8 ConvRot TP4,
Ref2V four-step v0.1, 1344x768, 4.4 requested seconds, seed 42 and shifts 12/3.
It exited with status 75 before model loading because neither GPU group was
free and unleased. Record: `api-ref2va.log`. No GPU generation result is claimed
for this API revision; no owned workers, service ports or GPU leases remain.
Retry only after resource ownership changes. The other tasks were not stopped.

### FlashGen native four-step adapter and AdaLN restoration (2026-09-08)

Continues the owned PR #565 branch from
`79398b0b5a08e0ea6ed7cac37358f55624b651a7`, with the same base, Python/Torch
environment and direct native API scope.

- Added the official FlashGen rank-64/alpha-64 T2VA artifact as a separate
  native layout. The loader validates all 518 tensors / 259 pairs, including
  grouped QKV, native gate/up FFN and dense AdaLN targets. Runtime deltas use
  staged FP16 A/B buffers, FP32 intermediates and the unrotated input basis.
- Metadata supplies `[1, 0.7, 0.4, 0.15, 0]`; shifts are 12/3. CLI/HTTP defaults
  select four intervals (`num_inference_steps=4`). LightX2V's five/nine-point
  convention is unchanged. Wrong active tasks, layouts and schedules fail.
- A pruned INT8 base restores 106 original AdaLN/time tensors; all backbone
  signed INT8 weights and FP32 scales remain unchanged. Original transformer
  shards are required. Restored weights add about 6.1 GiB per TP4 rank before
  adapter/activation costs. This is a weight-size estimate; actual GPU peak
  remains pending. Scale zero retains restored AdaLN/time components; omit the
  adapter and restart to recover the exact earlier pruned deployment.
- The downloaded official artifact's revision, byte size and verified SHA256
  are recorded in `UPSTREAM.md`. Production TP4 meta-model binding consumed all
  259 targets with 518 actual CPU buffers, totaling 435,126,272 bytes per rank.
  This verifies real checkpoint shapes and FP16 representability, not GPU
  execution. The CPU audit explicitly simulates TP configuration and groups.

The same CPU test command now passes **142 tests**, with 8 GPU-only tests
skipped. Thirty-two FlashGen tests cover metadata errors, exact schedule/task
semantics, TP1/2/4 projection algebra, complete binding, missing original
restoration tensors, INT8 tensor preservation, scale-zero bypass and HTTP
defaults. All changed-file pre-commit checks pass. The first standalone binding
audit needed the owned worktree on `PYTHONPATH` and explicit simulated TP config;
those setup-only failures are retained separately from the successful audit.

Evidence under `/data/minimax-h3/workflows-lora-20260908/flashgen/`:
`manifest.json`, `files.json`, `cpu-all.log`, `precommit.log`,
`inspect_binding.py`, `production-binding-tp4.log`, and `production-binding.json`.
No FlashGen GPU result or output-quality acceptance is claimed.

The real native-engine API mixed-Ref2VA test was retried after an idle snapshot
and successful native GPU lease acquisition. INT8 TP4, Ref2V four-step v0.1,
actual image/video/audio uploads, 1344x768, 107 frames at 24 FPS, seed 42 and
shifts 12/3 reached startup (107.824344 s), all-rank binding and 10,266-token
Qwen media encoding. The process then received SIGTERM (exit 143), at the
start of the four-call denoise loop. There is no completed media or valid
generation timing. The sender was not identified. `api-ref2va-run2.log` retains
the record; no owned workers or GPU leases remain. Do not repeat this unchanged
GPU run without a resource-ownership change or coordinated validation window.

Next acceptance: uninterrupted mixed-reference API generation and FlashGen
T2VA, with measured restoration residency, every-rank call counts, output decode
and video/audio review. FastH3 and the rest of the authorized workflow tracker
remain outstanding; this checkpoint is not full workflow completion.
