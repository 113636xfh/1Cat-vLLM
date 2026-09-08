# H3 reproducible validation

This draft has not passed the full-video quality or per-GPU 80 TFLOPS gates.
Component numerics and short cached-text generation are working. FlashAttention-V100
is the user-selected mainline and the default denoiser. Keep profiler
diagnostics separate from the three formal timing runs.

The current short-development target is **under 50 seconds for 20 actual DiT
updates**, retaining 1344x768, 39 frames, seed42, INT8 ConvRot, TP4 GPU0-3 and
output quality, without a substantial memory increase. Exact full-tile
FlashAttention specialization and column-major INT8/Lt now complete this workload
in **69.923404 s**, or 3.496170 s/update and 49.536398 useful TFLOPS/card.
The preceding QK/RoPE baseline took 74.411059 s; the new route reduces that time
by 6.03%. Both use zero persistent FP16 cache. New Lt workspace is also zero.
Peak allocated denoise memory remains exactly 6.647851467 GiB on all four cards;
NVML peak device-used memory, including runtime allocations, is 8.583496 GiB.

Both video and audio latents are bitwise equal to the prior FlashAttention
baseline. Fresh VAE decoding produces identical PCM samples and MP4 bytes;
MP4 SHA256 is `17ac6de78b7bc280ce91a0c6ca018785d131b3798856ca3ea3cdc55a10811988`.
All automatic media checks pass. Human audiovisual quality scoring remains
pending. Earlier FlashInfer comparisons remain recorded in CONTROL.md.

These are individual unprofiled development measurements after a one-invocation
warmup with prompt-verified cached text embeddings. Do not interpret them as
end-to-end generation or the formal three-measurement 243-frame acceptance.
The separate low-memory prototype took 69.868952 s; combining different builds
into a formal median is not permitted.

The standard CMake components build. 86 targeted GPU/CPU tests pass, including
INT8 offsets/scales, padded-M12352 projection equality, fallback and CUDA Graph
replay. Isolated CUDA12.8 memcheck/racecheck/synccheck pass without errors or
hazards. Evidence is under `flashattention-lowmem50/`; serialized GPU test logs
are under `flashattention-under50/`. Media and NVML reports are under
`outputs/quality39-int8-flashattn-lowmem-native-20steps/FLASH_ATTN_V100/` in the
recorded artifact root. **Neither 50 seconds nor >80 TFLOPS/card is achieved.**

## Short development checks

Use the native entrypoint with the same spatial canvas and a short temporal
extent while developing kernels or debugging components:

```bash
vllm video generate --model /path/to/MiniMax-H3 \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --tensor-parallel-size 4 --attention-backend FLASH_ATTN_V100 \
  --int8-weight-layout column --fp16-weight-cache-gib 0 \
  --num-frames 39 --num-inference-steps 21 --output-dir /path/to/development
```

This produces 1.625 seconds and performs 20 DiT forwards. It is the short
quality/development workload; the sigma-point convention needs 21 positions
for 20 actual updates. One- or two-forward runs remain numerical/execution
diagnostics and must not be used to judge normal model image quality. The streaming VAE requires at least 22 frames.
Keep the requested 50 sigma positions when evaluating final quality. Do not run
the full acceptance runner for routine development.

## Fixed workload

The acceptance helper uses the Chinese paper-boat/duck prompt in
`H3Request`, 1344x768, 243 frames, 24 FPS, seed 42 and 50 sigma positions.
The checkpoint schedule performs 49 DiT calls. All four ranks must report
their actual completed call counts.

After installing 1Cat with `.[video]` and building the H3 extensions:

```bash
python tools/minimax_h3/fixtures.py --model-root /path/to/MiniMax-H3 \
  --output-dir /path/to/reference-fixtures
python tools/minimax_h3/benchmark.py --model /path/to/MiniMax-H3 \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --attention-backend FLASH_ATTN_V100 --output-dir /path/to/acceptance
```

The runner performs one warmup and three unprofiled requests in one engine.
The warmup's automatic media checks must pass before formal timing starts.
Use `FLASHINFER_SM70` for an explicit comparison or rollback. A nonzero FP16
weight-cache budget requires an explicit measured list of `--fp16-cache-layer`
arguments. The default budget is zero.

## Reports and interpretation

```bash
uv pip install --python .venv/bin/python matplotlib
python tools/minimax_h3/report.py /path/to/acceptance/run-1
```

Each request retains `video.mp4`, the original waveform, sampled screenshots,
`run.json`, and `nvml.jsonl`. The report helper exports rank throughput and phase
times as CSV, and six NVML curves as PNG/SVG. Phase times include overlapping
aggregate and component fields; do not sum all fields indiscriminately.

Useful FLOPs count the actual local model matrices and effective attention
lengths. Padding, Hadamard rotations, weight dequantization and extra output
rows do not inflate the numerator. Each rank uses the slowest rank's full
denoise wall time. Formal evaluation rejects mismatched requests/configurations,
profiled timing, duplicate ranks and incomplete schedules.

The performance gate requires all four per-rank medians to be strictly above
80 TFLOPS and the three denoise times' population coefficient of variation to
be at most 5%. Memory is a separate 30 GiB gate. NVML utilization is never
interpreted as Tensor Core activity. Retain Nsight Compute counter evidence and
Nsight Systems communication/stall traces separately.

The evaluator leaves overall `accepted` false. Final review must score prompt
following, subject consistency, temporal continuity, detail and audio at least
4/5 each and reject flicker, deformation or audio/video faults.

## Evidence completed so far

- Both fixed INT8 partition files are downloaded and SHA256 manifests saved.
  The original BF16 DiT partitions and shared components are downloaded, and
  their 98-file checksum manifest is complete.
- Real TP4 text-encoder outputs are finite and bitwise identical between ranks.
- Real 50-block INT8 DiT single-forward checks pass after preserving large
  condition projections, residuals, gated products and row-projection outputs
  in FP32. The matrix inputs/weights remain FP16 for Tensor Core execution.
- The current numerical/service suite passes 47 tests, including short clips,
  the optimized FlashInfer kernel, prefetch tails and unaligned storage views.
  Seven acceptance-contract tests pass, including excluded timing.
- A 256x256, 107-frame, one-DiT-call route exported finite video and audio and
  passed the automatic media checks. It is not a quality/performance baseline.
- The earlier full primary video was interrupted before completion and is
  retained as partial diagnostics only. Development now uses 39-frame clips.
  Cached-text INT8 TP4 denoise (two forwards after warmup) measured 11.29494 s
  with Flash-V100 and 10.55566 s with the FlashInfer candidate: 30.6664 versus
  32.8142 useful TFLOPS on each rank. Both produced finite, rank-identical
  latents. These are development results, not full-schedule acceptance.
- Original BF16 FL2VA text-to-video completed 20 updates at 39 frames through
  Flash-V100 with FP16/FP32 computation. All automatic checks pass; denoise took
  110.569167 s. Other original-checkpoint backend/partition combinations, mixed
  references, extra seeds, full quality and formal three-run timing remain pending.
- Standard CMake configuration, build and component installation of both H3
  extensions pass. Complete release-wheel validation remains pending.

The earlier two-forward short export contains 39 decoded frames and valid
audio, but its image has clear ghosting and grid artifacts. It is not quality
accepted. Other GPU workers were present during that export run, so its JSON
marks timing invalid. Both the evaluator and report helper reject explicitly
excluded timing. The earlier isolated candidate comparison is retained separately.

A subsequent matched 20-forward comparison completed on exclusively leased
GPUs 0–3. Both backends pass automatic media checks, and inspected frames no
longer show the severe two-forward artifacts. Flash-V100 took 113.459387 s
(5.672969 s/call); FlashInfer took 105.642574 s (5.282129 s/call). Both used the
same INT8 checkpoint, prompt, seed, 39 frames, cached text and shift rules.
Full primary acceptance and human audiovisual review remain pending.

The subsequent FlashInfer V-layout/KV-prefetch optimization completes the same
20-update short request in 90.880784 s (4.544039 s/update), with 38.113157 useful
TFLOPS on each rank. Video/audio latents and the exported MP4 are bitwise
identical to the previous FlashInfer result; automatic media checks pass.
CUDA 12.8 memcheck/synccheck each pass the six new tail/unaligned cases with
zero errors. The standard CMake FlashInfer component builds successfully.
The 6.647851 GiB Torch peak is measured during DiT only and must not be treated
as the full pipeline's memory gate. This single short run does not satisfy the
primary three-run timing contract or the >80 TFLOPS threshold.

The profiler-only run uses the first two updates of the unchanged 20-update
schedule. Its rank-0 trace attributes about 52% of the synchronized
span to attention, 25% to GEMM and 10% to communication before this optimization.
Explicit wall coverage and kernel service are retained separately. Profiling
results never replace unprofiled timing.

The selected cooperative-transpose kernel completes the same 20-update denoise
in 87.824099 s (4.391205 s/update), with 39.439671 useful TFLOPS on each rank.
Compared with the original 105.642574 s FlashInfer baseline, throughput improves
by 20.29%. Both final video and audio latent tensors are bitwise identical to
the previously decoded prefetch result. Only after checking both tensors and
the MP4 hash, this run reuses that validated decoder output. Its metadata marks
`short_clip_denoise_quality_with_reused_decode`; it does not provide a new VAE
or end-to-end timing measurement. Text embeddings are also cached and verified.

The final binary passes all 47 video tests and the ordinary CMake component
build. CUDA 12.8 memcheck, racecheck and synccheck each pass all six new
tail/unaligned cases without errors or hazards. Matched-shape Nsight Compute
reports 25.091% tensor-pipe activity, up from 16.976% before prefetch. This
counter is diagnostic, not effective model throughput. The 39-frame result
remains below 80 TFLOPS/card; primary-load quality, three formal timing runs
and the full-pipeline memory gate remain incomplete.

Fused FP32 input preparation and warp-shuffle ConvRot subsequently reduce the
same 20-update denoise to 86.200372 s (4.310019 s/update), or 40.182583 useful
TFLOPS on every rank. Final video/audio latents remain bitwise identical;
the run retains explicit cached-text/decoder-reuse labels. The video suite
passes 58 tests, with a subsequently added nonfinite-input test passing
separately. All eleven new rotation/scaling cases pass memcheck, racecheck
and synccheck, and the W8A16 CMake component builds.

A recovered ComfyUI V100 attention source was also audited. On aligned D128
controls it is about 1.9 times slower than current FlashInfer, and its raw
unaligned path has query-tail synchronization and numerical failures. Those
controls do not replace H3 quality or performance acceptance. The mainline
and >80 TFLOPS/card target remain unchanged; see CONTROL.md for evidence.
