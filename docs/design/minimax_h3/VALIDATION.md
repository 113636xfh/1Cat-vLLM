# H3 reproducible validation

This draft has not passed the full-video quality or per-GPU 80 TFLOPS gates.
Component numerics and a short end-to-end route are working. Keep profiler
diagnostics separate from the three formal timing runs.

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
Use `FLASHINFER_SM70` for the independent Volta attention path. A nonzero FP16
weight-cache budget requires an explicit measured list of `--fp16-cache-layer`
arguments. The default budget is zero.

## Reports and interpretation

```bash
python -m pip install matplotlib
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
  The original BF16 DiT partitions and shared components are downloaded; their
  full file manifest is being generated.
- Real TP4 text-encoder outputs are finite and bitwise identical between ranks.
- Real 50-block INT8 DiT single-forward checks pass after preserving large
  condition projections, residuals, gated products and row-projection outputs
  in FP32. The matrix inputs/weights remain FP16 for Tensor Core execution.
- The native numerical/service suite passed 23 tests before the additional
  request preflight changes. Six acceptance-contract tests pass.
- A 256x256, 107-frame, one-DiT-call route exported finite video and audio and
  passed the automatic media checks. It is not a quality/performance baseline.
- A full primary FL2VA INT8 / Flash-V100 eager video is running. Full quality,
  both-backend comparisons, original BF16 comparisons, mixed references,
  extra seeds and formal timing remain pending.
- Standard CMake configuration, build and component installation of both H3
  extensions pass. Complete release-wheel validation remains pending.
