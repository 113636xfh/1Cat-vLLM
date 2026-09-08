# Native MiniMax H3 (development)

This is an in-progress native integration. Full checkpoint video quality and
four-card 80 useful TFLOPS acceptance are not yet established. The control log
records completed tests and remaining gates. No separate vllm-omni installation
is required.

The supported deployment contract is Python 3.12, Torch 2.10.0+cu128, CUDA Toolkit
12.8 and V100/SM70. Install the normal 1Cat source build with its `video` extra;
`tools/minimax_h3/build_extensions.py` builds the two independent H3 operators
when working on top of an existing compatible 1Cat installation. It requires
`CUDA_HOME` and `TORCH_CUDA_ARCH_LIST=7.0`. It does not rebuild the rest of vLLM.
The development tests use Transformers 5.15.1 and Diffusers 0.40.0.

```bash
export CUDA_HOME=/path/to/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
uv pip install -e '.[video]' --no-build-isolation
vllm video generate \
  --model MiniMaxAI/MiniMax-H3 \
  --partition fl2va \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --tensor-parallel-size 4 \
  --attention-backend FLASHINFER_SM70 \
  --output-dir ./h3-output
```

The default prompt is the approved paper-boat/duck scene; defaults are 1344x768,
243 frames, 24 FPS, seed 42 and 50 sigma positions (49 actual DiT calls).
For short development quality checks, use
`--num-frames 39 --num-inference-steps 21`: this keeps the 1344x768 canvas,
generates 1.625 seconds and performs 20 DiT forwards. The sigma-point convention
means N positions give N-1 denoise updates for this checkpoint. The fixed
acceptance workload still uses 50 positions (49 forwards). One- or two-forward
runs are only execution/numeric diagnostics: they produced severe ghosting and
grid artifacts that disappeared from the inspected 20-forward samples. The minimum is 22 frames (0.917
seconds), required by the streaming VAE. Requests align upward to 17n+5 frames:
`--duration 1` produces 39 frames and `--duration 2` produces 56 frames.
Use an aligned `--num-frames` value when an exact short duration is wanted.
Omit `--transformer-path` to load the original BF16 checkpoint into the FP16/FP32
mixed-precision model. The reference checkpoint's VAE Python code executes from
its downloaded model directory. Frozen revisions and port licenses are recorded
in the model package's `UPSTREAM.md`. FFmpeg and FFprobe must be on `PATH` for
reference-video/audio processing.

Use `--image first.png --keyframe-indices 0`, `--image last.png
--keyframe-indices -1`, or two `--image` arguments with `--keyframe-indices 0 -1`
for FL2VA. A Ref2VA instance uses `--partition ref2va` and the matching transformer
file; `--image`, `--video` and `--audio` are repeatable reference arguments.
A serving instance binds one partition and processes one request at a time.

```bash
vllm video serve \
  --model /path/to/MiniMax-H3 \
  --partition fl2va \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --tensor-parallel-size 4 --host 127.0.0.1 --port 8000
curl -sS http://127.0.0.1:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"一艘红色纸船漂过公园浅水池，轻柔流水声。","seed":42}'
```

`POST /v1/videos` returns a job ID. Query `GET /v1/videos/{id}` and download
`GET /v1/videos/{id}/content` after completion. `/health` reports readiness;
`/metrics` reports the queue and job counters. Reference paths refer to files
available to the server process. Jobs are kept in memory for the current service
lifetime; generated files remain in the configured output directory.

`FLASHINFER_SM70` is the default denoiser and the main performance-development
path. It selects the independent dense Volta operator. `FLASH_ATTN_V100` remains
an explicit comparison and rollback option. The Torch
reference backend is for numerical investigation. Text encoding uses TP4 and
Flash-V100 causal GQA; the selectable denoiser backends use non-causal MHA.

FP16 weight caching defaults to zero. After measuring candidate layers, pass a
budget with `--fp16-weight-cache-gib` and repeat `--fp16-cache-layer` for the
fixed layer list. Cached weights retain ConvRot coordinates. Cache/staging
preparation is separately timed; dequantization remains inside denoise timing
for uncached weights. Both original INT8 tensors and FP32 scales are retained.
The measured 39-frame/20-update cache list and native cuBLASLt configuration
are documented in [FLASHINFER_TO50.md](FLASHINFER_TO50.md), including exact
commands, output comparison and the still-incomplete 50-second target.

Outputs include `video.mp4`, original decoded `audio.wav`, `run.json`, sampled
`nvml.jsonl`, `quality.json` and frame screenshots. Automatic checks do not
replace the five-axis human quality review. Useful TFLOPS use actual local
matrix dimensions and valid attention tokens, exclude padding/rotation/dequant,
and divide by the maximum complete denoise duration across all four ranks.
NVML utilization and standalone operator speed are diagnostic evidence only.

The native engine reserves a whole available GPU group (0–3 first, then 4–7)
with the shared 1Cat V100 per-card and group locks. The lease remains held until
its workers exit. A group reserved by another cooperating task is unavailable
even while that task is between CUDA processes. If both groups are occupied or
reserved, startup fails before model loading.
