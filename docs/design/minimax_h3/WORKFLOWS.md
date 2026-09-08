# Native H3 workflows and distilled LoRA

This extends the native 1Cat pipeline from PR #557. It does not require a
separate vllm-omni runtime. Upstream comparison is pinned to vLLM-Omni
`b58ff5cb8b17250b76f9cdf9b9b46385cdda4376` (2026-09-08):
[H3 recipe](https://github.com/vllm-project/vllm-omni/blob/b58ff5cb8b17250b76f9cdf9b9b46385cdda4376/recipes/MiniMaxAI/MiniMax-H3.md).
The wider model inventory is in [Omni coverage](../omni_workflow_coverage.md).

## Workflow contract

One engine serves one DiT partition. Both partitions use the shared FL2VA
tokenizer, Qwen3-VL encoder and VAEs. Use a separate engine configuration when
switching between FL2VA and Ref2VA; simply changing the request task cannot
change the loaded checkpoint.

| Workflow | Partition | Request fields / CLI arguments |
| --- | --- | --- |
| Text to video + audio | `fl2va` | `task=t2va`; no media |
| First frame to video + audio | `fl2va` | `task=fl2va`, one image, indices `[0]` |
| Last frame to video + audio | `fl2va` | `task=fl2va`, one image, indices `[-1]` |
| First and last frames | `fl2va` | `task=fl2va`, two images, indices `[0,-1]` |
| Image reference | `ref2va` | `task=ref2va`, one or more images |
| Image and audio reference | `ref2va` | images and standalone audio |
| Video reference | `ref2va` | video, with its embedded audio when present |
| Mixed reference | `ref2va` | images, videos and standalone audio |

Ref2VA requires a visual reference; audio-only is not an upstream-supported
workflow. Limits are 9 images, 3 videos, 3 standalone audio files and 12 total
files. Source videos/audio must each last 2–15 seconds, with a maximum total
duration of 15 seconds for each input type. Video start times must leave at
least two seconds in each source. Native validation uses the same metadata
checks as preprocessing, before worker dispatch. FFmpeg and FFprobe must be
on `PATH`. Corrupt media returns HTTP 422 without closing the worker group.

CLI media flags are repeatable: `--image`, `--video`, `--audio`.
`--keyframe-indices` selects first/last frames;
`--reference-video-start-times` takes one offset in seconds per video.
Explicit `--task`, `--flow-shift`, and `--audio-flow-shift` are now exposed.
Without `--task`, the existing partition/media-based task inference remains.

## LightX2V Turbo family

Download the **Diffusers export**, retaining its published filename:

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors \
  --revision 2f015e66b37c585cea9dc4ae6f1850ea8788e742 \
  --local-dir ./h3-turbo
```

Every filename below starts with `minimax_h3_` and ends in `.safetensors`.
All use rank 128 and audio flow shift 3. Alpha is read from file metadata,
with LightX2V's reference default 8 when metadata omits it.

| Filename middle | Tasks | Actual denoiser calls | Sigma points | Video shift | Alpha |
| --- | --- | ---: | ---: | ---: | ---: |
| `fl2v_turbo_4step_v0.1` | T2VA / FL2VA | 4 | 5 | 12 | absent → 8 |
| `fl2v_turbo_4step_v1.0_768p_bf16` | T2VA / FL2VA | 4 | 5 | 6 | 128 |
| `fl2v_turbo_4step_v1.1_768p_bf16` | T2VA / FL2VA | 4 | 5 | 6 | 128 |
| `fl2v_turbo_4step_v1.2_768p_bf16` | T2VA / FL2VA | 4 | 5 | 6 | 8 |
| `fl2v_turbo_8step_v1.0_bf16` | T2VA / FL2VA | 8 | 9 | 12 | 8 |
| `fl2v_turbo_8step_v1.0_768p_bf16` | T2VA / FL2VA | 8 | 9 | 6 | 8 |
| `ref2v_turbo_4step_v0.1_bf16` | Ref2VA | 4 | 5 | 12 | 8 |
| `ref2v_turbo_8step_v1.0_768p_bf16` | Ref2VA | 8 | 9 | 6 | 8 |

`--lora-path` accepts one local file, or a directory containing exactly one
recognized artifact. ComfyUI fused exports, renamed files, FlashGen native
adapters and FastH3 bundles are refused instead of being interpreted as this
layout. FL2V and Ref2V adapters require their matching base partition.

One immutable adapter is loaded at engine startup. Requests apply a multiplier
`--lora-scale` / `lora_scale` (default 1) to `alpha / rank`. Setting it to zero
skips all LoRA matmuls and uses base-model sampling defaults. There is no hot
loading of arbitrary files, composition of adapters, or weight prefusion.

When the CLI/HTTP request omits steps or shifts, native 1Cat selects the active
adapter's values and records them in the request. Explicit values are preserved
and checked: **four-step LightX2V needs `num_inference_steps=5`**, and eight-step
needs `9`. This is the inherited sigma-point convention. A base request still
defaults to 50 points / 49 updates. Python callers can use
`sampling_for_deployment(config, ...)`; directly constructed
`H3SamplingParams` retains its explicit/default 50-point contract.

```bash
vllm video generate \
  --model /path/to/MiniMax-H3 --partition fl2va \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --tensor-parallel-size 4 --attention-backend FLASHINFER_SM70 \
  --lora-path ./h3-turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors \
  --task fl2va --image first.png --image last.png --keyframe-indices 0 -1 \
  --num-frames 39 --output-dir ./h3-fl2va-turbo
```

Omit media and select `--task t2va` for text generation. Omit
`--transformer-path` for the original BF16 checkpoint's FP16/FP32 runtime.
The 39-frame command is a short development check, not the full quality gate.
For Ref2VA, select the Ref2VA partition, checkpoint and `ref2v` adapter together.
Use at least 56 output frames in short tests containing embedded reference
audio: the current pipeline's encoded reference-audio check requires 80 latent
positions, so the 39-frame audio truncation is too short. Official 4–15-second
output requests do not encounter that development-only boundary.

Serving uses the same deployment flags with `vllm video serve`. Native HTTP
accepts JSON and local reference paths; it is not the upstream multipart-upload
API. For example, on a Ref2VA server with its matching Turbo adapter:

```json
{
  "task": "ref2va",
  "prompt": "图中的纸船继续向右漂移，保留参考视频的环境和流水声。",
  "image": ["/path/to/boat.png"],
  "video": ["/path/to/reference.mp4"],
  "audio": ["/path/to/water.wav"],
  "reference_video_start_times": [0.0],
  "num_frames": 107,
  "lora_scale": 1.0
}
```

## Numerical and memory contract

The loader consumes all 624 tensors / 312 A/B pairs, covering 50 DiT blocks
and two token-refiner blocks. They bind to 208 native linears because Q/K/V
share one base projection. Q/K/V keep independent A/B pairs and local output
slices. MLP B rows are converted from `[value, gate]` to native `[gate, value]`
before TP partitioning. Row-parallel A matrices are sharded over input channels;
their deltas join the base result before the existing all-reduce.

For a ConvRot layer, the computation is:

```text
base = Linear(ConvRot(x), dequantized_rotated_INT8_weight)
output = base + lora_scale * alpha/rank * B(A(x))
```

The adapter sees the **unrotated** activation. No full weight delta is merged
into signed INT8. FP16 GEMM inputs and FP32 outputs use the existing SM70 H3
operator, including power-of-two scaling for wide-range intermediates. A/B
buffers are registered before the pinned stager snapshots the model, so they
follow its load/offload lifecycle. Dense shortcuts are disabled on adapted
layers so they cannot bypass the delta. FLOP accounting records active LoRA
matmuls separately with `.lora` keys and counts none when scale is zero.

## Validation and remaining work

See [CONTROL.md](CONTROL.md) for the exact environment, source base, commands,
measured results and evidence paths. Passing a parser or synthetic shape test
is not an end-to-end generation result. Each artifact/partition/quantization
combination requires its own evidence before being promoted.

FlashGen's fused native QKV/AdaLN tensors and pinned DMD2 schedule need a
separate loader; its target AdaLN shape also conflicts with pruned checkpoints.
FastH3 adds full-rank deltas and sampling/attention requirements, so it cannot
be accepted by renaming a LightX2V file. These two families, combined-partition
serving, upstream multipart uploads, step batching, DLO and approximate caches
remain outside this implementation.
