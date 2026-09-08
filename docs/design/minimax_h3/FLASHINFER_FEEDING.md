# H3 FlashInfer SM70 feeding follow-up

This branch continues the independent FlashInfer route. The concurrent
FlashAttention-V100 work belongs to another task. The overall requirement
remains greater than 80 useful model TFLOPS on **each** TP4 rank over complete
denoising; this follow-up does not meet that requirement.

## Implementation

The D128 non-causal operator uses Q128/K64 tiles with 512 threads. Two warps
own each 16-query group, each holding two K16 score fragments. Reusing the Q
operand across those fragments increases the key tile without doubling the
thread count. QK's limited loop unrolling avoids register spills. Each thread
prefetches two K/V vectors while PV consumes the current block; consumed P
storage still supplies the cooperative V transpose.

Named barriers join the two warps that exchange softmax maxima and sums.
CTA-wide barriers remain around shared K/V staging and scratch reuse. Every
thread, including those corresponding to padded query rows, participates.
Output, QK and softmax state remain FP32, with FP16 tensor operands. Original
scales, actual valid lengths, sampling and useful-FLOP accounting are unchanged.
This operator does not call FlashAttention or an external attention package.

CUDA 12.8 emits 126 registers/thread, no stack/spill, and 92,160 bytes of
dynamic shared memory. One CTA can reside per SM. The built extension SHA256 is
`23eb9d8a2f60ccf674f82bbfc8fe9bcb7af57ee693f1b8dddae3340cc227e238`.

## Validation and measured results

Environment: Python 3.12.13, Torch 2.10.0+cu128, CUDA Toolkit 12.8.93,
V100-SXM2-32GB. Dependency base is
`9bf2c79027e14f3e8fdcb233e60ae11c6512cb94`, within the native H3 campaign
based on `onecat/main@56f534e672657a6c7599afd6c0dcb2e2c211b2e3`.

All 65 `tests/video` tests pass. Added checks cover the two-vector prefetch
boundary, independent query-group softmax histories, and CUDA Graph replay
after input updates. Compute Sanitizer 12.8 memcheck, racecheck and synccheck
each pass the 11 focused prefetch/query-group cases without reported errors.

The isolated operator controls use B1/H14/D128 FP16 non-causal attention with
the exact H3 lengths, alternating CUDA event timings and sampled FP32
references using 65 query rows against all valid keys. These are **operator**
results, separate from model throughput:

| Valid length | Previous Q128/K32 | Candidate Q128/K64 |
| --- | ---: | ---: |
| 12,323 | 35.600 ms | 27.405 ms |
| 73,483 | 1,226.955 ms | 970.892 ms |

These candidate measurements used the pre-format prototype; the installed
source has the same operations and register/spill counts and was validated
with the suite, sanitizers and complete short denoise. Short-case observed
clocks were 1507–1522 MHz; in the long case the control ran at 1530 MHz and
the candidate at 1485 MHz. Clocks were not locked. Maximum sampled errors
versus FP32 were 3.05176e-5 and 1.52588e-5 respectively.

An independent Nsight Compute run on GPU 1 measured Tensor Core active cycles
at 32.332%, versus the earlier GPU 0 Q128/K32 profile's 25.091%. Shared-load
bank conflicts were about 603 million versus 872 million. The device change
and independent profiler runs limit this comparison; neither percentage is
useful model TFLOPS. The current profile still shows 25% maximum warp
occupancy, 14.229% short-scoreboard stalls and 8.138% barrier stalls.

The unprofiled model run uses GPU 0–3, Comfy INT8 ConvRot FL2VA, TP4,
1344x768, 39 frames, 24 FPS, seed 42, 21 sigma positions and **20 actual DiT
updates**, with video/audio shifts 12/3. Text conditioning is the previously
verified real TP4 encoder output for the fixed paper-boat/duck prompt.
There is one single-update warmup and one complete measured short schedule;
this is not the primary workload's full warmup plus three formal runs.

| Complete short-denoise metric | Previous | Candidate |
| --- | ---: | ---: |
| Denoise seconds | 86.200372 | 78.968862 |
| Seconds per actual update | 4.310019 | 3.948443 |
| Useful TFLOPS on every rank | 40.182583 | 43.862270 |
| DiT peak allocated memory per rank | 6.647851 GiB | 6.647851 GiB |

This is an 8.389% reduction in denoise time and a 9.157% increase in useful
throughput. Each rank's unchanged useful-FLOP numerator is
3,463,753,579,661,312. The denominator is the maximum synchronized rank time.
NVML peak memory during the monitored stage was 8.667480 GiB per GPU; this
does not establish the whole-pipeline 30-GiB memory gate.

GPU 0's median SM clock was 1402 MHz versus 1515–1522 MHz on GPUs 1–3.
All four have the same 300 W default/enforced/maximum power limit; no power
or clock settings were changed. NVML recorded software power-capping events.
GPU utilization medians of 100% are diagnostic and do not establish the
80-TFLOPS requirement.

## Quality limits

Both video and audio were freshly decoded; VAE time was 8.908984 seconds.
All automatic checks pass: complete 39-frame 1344x768/24-FPS video, valid
nonzero audio, no black frames or prolonged static run, and finite final
latents. The inspected first/last frames contain a consistent red paper boat
and yellow duck in the pool, with the boat moving rightward. This does not
replace full temporal/audio review.

The changed key partition changes floating-point rounding. Final latents
are **not bitwise identical** to the previous kernel: video relative L2 is
0.0742835 and audio relative L2 is 0.0170330. Decoded video SSIM versus the
previous same-seed video is 0.977513. Similarity and the automatic checks are
supporting evidence; human five-axis quality acceptance remains pending.
The output MP4 SHA256 is
`02ee9057bfa997ac578d8fdda11acd9770d99b86022207d1674a9dbc65eb50cb`.

## Reproduction and rejected paths

Run focused checks from this branch's SM70-built environment:

```bash
.venv/bin/python -m pytest -q tests/video
compute-sanitizer --tool racecheck --error-exitcode 1 \
  .venv/bin/python -m pytest -q tests/video/test_h3_numerics.py \
  -k 'prefetch_tail or query_groups'
```

The normal CLI can exercise the same short request with freshly encoded
text. Its end-to-end time should not be substituted for the cached-text
denoise benchmark:

```bash
vllm video generate --model "$H3_MODEL" --transformer-path "$H3_INT8" \
  --partition fl2va --tensor-parallel-size 4 \
  --attention-backend FLASHINFER_SM70 --fp16-weight-cache-gib 0 \
  --width 1344 --height 768 --num-frames 39 --num-inference-steps 21 \
  --seed 42 --output-dir h3-flashinfer-k64-short
```

Retained raw evidence is under
`/data/minimax-h3/native-h3-20260908/feeding-round3/`: `build_native.py`,
`make_pair_k64.py`, `build_final.py`, `probe.py`, `run_quality.py`,
`quality-summary.json`, `nvml-summary.json`, `video-tests.log`, the three
sanitizer logs, the NCU report/CSV, and `outputs/quality39-20steps/` containing
video, audio, screenshots and NVML curves. `ownership.json` identifies the
isolated worktree and dependencies. The old kernel worktree was not modified.

- Pair-only synchronization at K32: 34.974 -> 33.812 ms; long case
  1223.369 -> 1179.014 ms, bitwise equal. It is superseded by the K64 candidate.
- Manually swizzled WMMA operands: 47.027 ms, with spills. Limiting QK
  unrolling removed all spills but still took 47.542 ms. Reject the layout;
  spills alone did not explain its regression.
- K128 with Q/P and K/V shared-storage reuse: 42.971 ms versus a matched
  34.980-ms control; 44-byte spill stores and 84-byte spill loads. Reject.
- CUTLASS asymmetric FMHA controls: about 26.4–26.5 ms at the short shape,
  but are not installed, delegated to, or reported as FlashInfer performance.

Do not repeat these unchanged controls. Next work must address the retained
kernel's register/shared-memory pressure, matrix feeding, or critical TP
communication. Full 243-frame quality, additional seeds and partitions,
three formal unprofiled measurements, and >80 TFLOPS on each rank are still
outstanding. Keep this branch Draft. Rollback is the retained Q128/K32
FlashInfer kernel at the dependency base; the other attention route is managed
by its separate task.
