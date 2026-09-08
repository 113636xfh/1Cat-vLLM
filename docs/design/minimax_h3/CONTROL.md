# Native MiniMax H3 migration control

Status: implementation in progress; no video quality or 80 TFLOPS acceptance yet.

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

### Independent FlashInfer register accumulation

The selected SM70 kernel keeps QK scores, online-softmax reductions and PV
accumulators in registers. Only cross-warp row partials and probabilities use
shared memory. A 128-query by 32-key tile uses 512 threads and 64 KiB shared
memory. The explicit backend continues to execute its own Volta WMMA kernel.

- Full FP32 checks at 17, 243, 1025 and 8192 tokens passed. Spread-out query
  checks at 12323 and 73483 tokens passed without a square attention matrix.
  Added regression coverage for batch indexing, 127/128/129 query boundaries,
  short-video token count and changing softmax maxima across key tiles.
- The final formatted implementation passes all 38 native video tests on GPU 0.
- At 12323 tokens (39-frame canvas), candidate median attention time was
  53.220 ms versus 61.108 ms for Flash-V100. At 73483 tokens the same isolated
  comparison was 1.857 seconds versus 2.246 seconds. No full long video was run.
- Cached-text TP4 INT8 denoise at 1344x768, 39 frames, seed 42 and three sigma
  positions (two forwards), after a one-forward warmup: Flash-V100 11.29494 s,
  FlashInfer candidate 10.55566 s. Each rank counted 346375340509184 useful
  FLOPs, giving 30.6664 and 32.8142 useful TFLOPS respectively. All four ranks
  matched bitwise within each backend and all latents were finite.
  This is development timing, not the full-schedule >80 TFLOPS acceptance.
- Rejected candidates: 128x64 requested too many launch resources; 32x64 was
  slower; keeping only PV in registers was slower than also reducing softmax
  in registers. Their code/binary and raw measurements remain in artifacts,
  while the public extension retains only the selected implementation.
- The first cached-text diagnostic attempted to export the raw BCTHW VAE
  tensor. Its script omitted the pipeline's output conversion to BTHWC RGB8.
  The native pipeline already performs that conversion; correct the diagnostic
  and save latents before export to avoid repeating denoise on export failures.

Artifacts: `attention-short-candidates.json`, `flashinfer-tested-candidates.cu`,
`flashinfer-tested-candidates.so`, `flashinfer-register-formatted-build.log`,
`kernel-register-native-tests.log`, `denoise-short-int8.log` and
`outputs/dev39-int8/{FLASH_ATTN_V100,FLASHINFER_SM70}/` under the task artifact
directory. Those output directories contain timing and NVML curves; the initial
export failed and must not be presented as generated-video quality evidence.

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
