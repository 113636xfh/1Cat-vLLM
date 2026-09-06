# Experimental E4M3 grouped attention with FP32 partial state

## Scope and admission

**Do not merge for production or enable by default.** The latest
scaled-residual candidate passes operator/sanitizer checks but fails the
same-process 128K deterministic token gate described below. Human review and
whole-model quality/performance admission remain outstanding.

This change integrates the small-query part of a private E4M3 migration into
`onecat/main`, based on `755baae1d075ee04fa9096b23fc0225b23589a86`.
It does not change the default KV format, the existing prefill implementation,
the cache writer, B1 decode, or the default DFlash2 verifier route.

`VLLM_FLASH_V100_E4M3_GROUPED_FP32=1` opts into a rebuilt native entry for
explicit `fp8_e4m3` KV. The flag defaults to **off**. Older extensions without
the new entry retain the ordinary backend route. Set the flag before worker
initialization; rebuild and restart task-owned workers to change the route.
The extension must report `grouped_e4m3_fp32_precision_version() >= 2`.
Presence of the original forward symbol alone is insufficient: older builds
contained the repeatable numerical counterexample described below. A stale
binary retains the ordinary backend fallback and emits a rebuild warning
when the experimental flag is requested.

The supported contract is SM70, FP16 Q/output, 2–8 query rows, six query heads,
one KV head, D256, and page sizes 800/848/1616/1648/3296. The parent attention
metadata must describe exactly one request. Independent-request batches,
sliding windows, explicit partition overrides, incompatible layouts, and
unsupported shapes retain the existing fallback. Valid physical block IDs
and per-query lengths within the block-table capacity are caller invariants.
The wrapper is not an API for interpreting E5M2 bytes as E4M3.

The selected entry logs `experimental E4M3 grouped FP32 route selected` and
records `prefill_smallq_e4m3_grouped_fp32`. A requested environment flag alone
is not evidence that an inference used this entry.

## Design

We reuse the existing grouped Tensor Core dataflow to scan KV once for a
packed query/head group. The repaired precision revision keeps accumulation
and normalized partition outputs in FP32, with three additional safeguards:

1. QK uses K16 Tensor Core products and compensated FP32 summation across D.
   Explicit round-to-nearest additions preserve the correction under the
   standard fast-math build.
2. PV consumes both the high FP16 probability and its FP16 rounding residual.
   The latest experiment stores the residual multiplied by 2048 and divides
   the second product's E4M3-derived V operands by 2048. The inverse scaling
   is exact in FP16 for every finite E4M3 encoding; the purpose is to retain
   small residuals that otherwise underflow. This experiment's model and
   performance admission is separate from the recorded revision-2 results.
   The softmax denominator remains the original FP32 probability sum.
3. Each N32 PV tile starts with a fresh FP32 Tensor Core accumulator. We update
   the longer-lived FP32 online state with one scalar FMA per output element,
   combining rescaling and addition. This avoids repeatedly feeding a large
   accumulator into Tensor Core products of increasingly small corrections.

Tensor Core operands and the final output remain FP16. These changes reduce
arithmetic error on identical quantized KV; they do not remove FP8 quantization
loss or promise bitwise equivalence to FP64 on arbitrary inputs.

For q5, only 30 packed rows are live. We skip the wholly padded third M16 tile
in QK, PV, and softmax. A separate padded residual panel adds 3.75 KiB of
shared memory per CTA without additional KV reads or global workspace.
The measured q5 build still uses 128 registers/thread and one CTA per SM.
The q5 timing evidence is not a speed claim for every admitted query width.

The new entry consumes the metadata builder's per-query device lengths.
Those lengths, not the padded Q shape, determine each causal prefix. The
maximum live length controls partition assignment. Inactive rows explicitly
write zero before reading partial state, including when all rows are empty.
The same captured graph can therefore replay after padding or length changes.

The FP32 workspace is keyed separately from the old FP16 workspace, including
device and stream. Its partial tensor is `[80, 8, 6, 256]` FP32 (3.75 MiB),
plus `[80, 8, 6]` FP32 statistics (15 KiB). The legacy partial tensor is
1.875 MiB. Keeping a reference to an existing graph also keeps its workspace
alive; this is not a claim of zero concurrent-memory cost.

Existing E5M2 q8/q16 and sparse-page4 instantiations retain their template
defaults and FP16 partial storage. The new entry does not modify their
public call signatures. Unlike PR #517's E4M3 DFlash2 q8 port, this scope adds
FP32 partial state and explicit row lengths for dense small-query MTP input;
it does not duplicate its logits, sampling, or context-pipeline changes.

## Evidence boundaries

The private prototype and this integration build are different artifacts.
Private measurements motivated this implementation; they do not automatically
approve the rebuilt integration artifact or its end-to-end performance.

The retained private bundle `e4m3-precision-parity-20260906` includes 24 actual
MTP q5/page848 groups (four context lengths, three layers, two rounds). The
FP32-partial prototype had arithmetic relative-L2 0.0199%–0.0224% against
PyTorch FP64 on the identical quantized KV, lower than its captured scalar
control in every group. This is **not total FP8 error**. Four TP4 short
retrieval cases matched the control's 11 output token IDs, but that is not a
full long-generation quality gate. Private activations, model data, binaries,
paper drafts, and machine-specific paths are deliberately excluded from Git.

The integration build passes 39 GPU tests (12 new FP32 cases, 25 existing
grouped cases, and two sparse-page4 cases) on physical GPU4. Its 12 new cases
also pass Compute Sanitizer memcheck on GPU5 with zero errors. An additional
24-group private real-input replay on GPU6 gives arithmetic relative-L2
0.0198771%–0.0224338%, lower than the captured scalar control in every group.
Only one of those 24 outputs is bitwise identical to the private prototype.
Current-main dataflow and the standard build's `--use_fast_math` differ from
the private artifact; this replay does not establish the cause of every
rounding difference and does not transfer its model-quality approval.

The complete routing-policy selection passes 138 tests on physical GPU4,
including explicit-length forwarding, disabled-entry fallback and multi-request
rejection. An earlier CPU-hidden run had two existing stream/device-query
failures (132 passed, one skipped); rerunning with an idle visible GPU resolves
both without modifying those tests or changing their assertions.

The native library SHA256 is
`953924a40df189dea60b4bbc7624bdb1bd6e1460176f2f111b21f3b2f9c959f0`.
The real-input replay result SHA256 is
`f3a444a60aab55f67964d0cfb40db80d9297f2905278d296ebe0fa001fbc48b5`.
The memcheck log SHA256 is
`a9d061cee538182f762e132abeea7bac7bddb844338f25751bbbb043f242c425`.
Build/test artifacts remain in the owned worktree's `.artifacts/e4m3-fp32/`;
they are not distributed as source or installed into running services.

Environment: V100-SXM2-32GB, Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8
runtime, GCC/G++12, and the locally available CUDA 12.0 compiler. The build
logs retain the CUDA minor-version warning; this is not a CUDA 12.8 compiler
validation. Both Flash-V100 and `paged_kv_utils` build successfully. The
runtime import is verified to come from the task's build directory.

Default enablement requires a fresh matching model-quality and unprofiled
performance gate on the integration build, including longer outputs. There
is no new model-throughput claim in this change.

## Reproduce focused checks

Use a task-owned virtual environment and an idle SM70 GPU. Build Flash-V100
with the repository's `setup.py build_ext`, using separate build and library
directories. Point `PYTHONPATH` at that library directory and this checkout's
`flash-attention-v100` source package; importing an installed stale extension
does not validate this source change.

```bash
.venv/bin/python -m pytest --confcutdir=tests/kernels/attention -q \
  tests/kernels/attention/test_sm70_grouped_e4m3_fp32.py \
  tests/kernels/attention/test_sm70_flash_v100_grouped_verify.py \
  tests/kernels/attention/test_sm70_qsa_grouped_page4.py

.venv/bin/python -m pytest --confcutdir=tests/v1/attention -q \
  tests/v1/attention/test_sm70_e4m3_grouped.py \
  tests/v1/attention/test_sm70_flash_v100_policy.py
```

The new operator tests use randomly relocated pages, non-unit KV scales,
FP64 reference outputs, and one graph with live/all-zero/tail-zero/first-zero/
restored row lengths. They cover every admitted query length and page family,
and the exact 262144-token boundary. Workspace tests check that FP32 and FP16
buffers do not alias. Policy checks cover default-off behavior, native
capability detection, invalid shapes and independent-request rejection.

Run the same operator checks under Compute Sanitizer memcheck before promotion.
The test suite needs a visible idle GPU: some existing policy tests query a
CUDA stream even when their tensors are on CPU.

## Remaining gates and rejected routes

- This is an experimental code integration, not default deployment approval.
- Long-output model quality and current-main TP4 performance remain unapproved.
- A separate private paged SplitKV E4M3 port corrected an inherited merge-row
  mismatch but remained slower than its E5M2 counterpart. It is not included.
- MTP5 dual-CTA equivalence, independent batches, and other GQA ratios need
  their own evidence; they are not covered by this single-request gate.
- AI assistance was used. Human line-by-line review and confirmation of the
  relevant tests remain required before merging, per the repository policy.

## Model admission update: a repeatable counterexample

The original integration artifact (`953924a...959f0`, CUDA 12.0 compiler)
has now been exercised in Qwen3.8-27B-FP8, TP4, native MTP4, explicit E4M3
KV, FP16 activation/SSM cache, 262144 maximum length and CUDA graphs. At an
8192-token prompt, an 11-token retrieval response matches across
control--candidate--control, but this short smoke hides a longer-output
counterexample. Greedy sampling is an explicit regression contract, not an
official-sampling quality evaluation; EOS is not suppressed.

Initially, the two controls themselves diverged at output token 177, while
the candidate differed from the first control at token 229 (one-based).
The unstable controls make that initial comparison inconclusive; per-start
GEMM tuning is one possible confounder. With
FP8/AWQ small-shape tuning and FP8 coordinated tuning disabled **only for
diagnostic isolation**, two independent control processes reproduce all 256
output tokens and MTP acceptance length in six runs including warmups. The
FP32 candidate instead reproducibly differs at output token 102 in all
three runs. All recorded top-five logprobs are finite and both texts remain
coherent, but the no-token-divergence gate fails.

A separate counterfactual replaces only admitted small-Q attention with
PyTorch FP64 over the identical quantized KV, retaining FP16 output and all
other model computation. Its three runs match the control's entire 256-token
sequence. At the first divergence, the logprob margin for token `343` over
token `5604` is +0.046875 in the control, -0.046875 in the candidate, and
+0.093750 in this reference. This is an attention-only reference, not a
full-model FP64 or unquantized-KV oracle. The reference used GPU0--3; the
fixed-dispatch native runs used GPU4--7. No absolute timings are pooled.

This counterexample blocks promotion; lower aggregate operator L2 alone
does not grant model acceptance. It is not evidence that E4M3 is worse than
E5M2, nor a verdict on the private long-prefill implementation, which was not
loaded. Do not expand to the long-context speed sweep or change defaults
before localizing this repeatable failure. Raw bundle:
`e4m3-mainline-model-gate-20260906`, indexed in the private handoff.

## E4M3 bridge prerequisite, without default routing

The backend's existing prefill bridge is still E5M2-only. The explicit
`fp8_e4m3_paged_kv_to_fp16` entry adds the missing conversion building block,
sharing the paged scheduling and unit-scale specialization with E5M2. It
preserves signed zero, scales in FP32, rounds to FP16, and zeroes the live
prefix's 16-token padding. It does not reinterpret E5M2 bytes, select a
backend route, or change the global KV format. Old extensions fail with an
explicit rebuild message when this new entry is requested.

The bridge build uses CUDA 12.8.93/GCC12, Torch 2.10+cu128, SM70 and standard
fast-math flags. Its native SHA256 is
`c1ce4140fe82a94fba8c351e219e72ca6676f4ebb1fcc54fc7444679e08b2003`.
The 30 bridge cases cover both formats, all 256 byte encodings, five page
sizes, three scale pairs, relocated pages, signed zero and graph length
changes. Together with 39 grouped/sparse regressions and one missing-native
entry check, 70 tests pass; the routing-policy suite also passes 138 tests.
Memcheck covers these 30 cases and 12 FP32 grouped cases with zero
errors. The same 24-group activation replay has relative-L2
0.0198771%--0.0224338%, all below the captured scalar control. These are
operator results, **not model approval for this updated artifact**.

Artifact directory: `.artifacts/e4m3-bridge-fp32/`. To reproduce the added
conversion checks, run the ordinary GPU and memcheck workflows above with
`tests/kernels/attention/test_sm70_fp8_bridge_formats.py`. The small-Q
selection message is now process-scoped so future gates can check every TP
rank rather than mistaking rank-zero-only logging for missing execution.

## Precision-repair update

The repair separates errors hidden by the final FP16 output floor. A replay
of 200 retained real groups includes the previous 24 groups and 176 additional
rank-zero target-MTP groups near the natural-prose regression. Instrumenting
the model to capture those groups changes its generation trajectory; captured
inputs are used for same-input arithmetic analysis, not uninstrumented model
admission or timing.

Probability compensation alone reduces operator error but does not pass the
model gate. With compensated P and a long-lived Tensor Core accumulator,
the median partition-state relative-L2 at approximately 256K is `2.95e-5`
after removing QK and final FP16 rounding effects. Replacing only the merge
weights with FP64-reference weights barely changes that error. Tile-local PV
and FP32 online-state updates reduce it to `2.92e-7`. The biased-V synthetic
regression reproduces the old failure (`2.93e-5` against a `3e-6` bound), while
the repaired build passes. Zero-mean random V alone did not catch this issue.

A separate model counterfactual retains the native-order QK computation but
uses FP64 softmax/PV. It still diverges at output token 177 in the new GPU0--3
cohort, supporting the additional compensated QK summation. The full FP64
attention-only reference again matches the entire control output in three
runs. This is not a claim about all-model FP64 arithmetic.

With all three repairs, the 8K natural-prose counterexample matches all 256
reference/control token IDs in three candidate runs. The native controls on
either side are token-stable. In the 200-group operator replay, disagreements
with the FP64 result rounded to FP16 decrease from 197210 elements to 2092;
these counts describe attention tensor elements, **not token errors**. The
maximum output relative-L2 divided by the unavoidable FP16-rounding floor is
`1.000018`.

The same-GPU, q5/page848, 20-warmup/100-ABBA CUDA-graph comparison against the
original FP32-partial candidate measures:

| Actual KV length | Original latency (ms) | Repaired latency (ms) |
|---:|---:|---:|
| 8197 | 0.071053 | 0.071027 |
| 65541 | 0.356045 | 0.353997 |
| 131077 | 0.687693 | 0.685722 |
| 261893 | 1.338496 | 1.335757 |

This establishes essentially unchanged operator speed in this contract,
not a substantial new speedup, a 60-TFLOP/s prefill result, or production
end-to-end throughput. Model isolation still disables per-start GEMM tuning;
the diagnostic setting is not promoted to a production default.

The repair-algorithm DSO is
`2f48d070dbe7536c011a6585334da61afd943d70a6ff68dc024ca90ababd2f43`.
The capability-guarded build is
`252da72e37a65ca40b4b821960c0d62d11a705e0d08e2a651217ad28f9d13e86`.
Their 200 replay outputs are bitwise identical. The final four-context
model cohort, broader production sampling/performance admission, and human
review remain separate gates. The experimental flag and global KV-format
defaults are unchanged. Artifacts remain in the owned worktree's
`.artifacts/e4m3-output-repair/`, with large private data outside Git.

### Remaining long-context counterexample and residual experiment

The subsequent four-context cohort does not establish promotion. Against its
attention-only FP64 reference, revision 2 matches all 256 tokens at 8K/64K but
first diverges at token 151 at 128K. The approximately 256K candidate completes
generation, while the corresponding reference/control are incomplete. The
ordinary control also differs from the FP64 reference (token 177 at 8K and
248 at 128K), so it is not treated as an exact numerical oracle. Cross-process
results must be distinguished from same-process reference brackets.

Retained 128K diagnostic frames expose another avoidable source: direct FP16
storage of an unscaled probability residual. On one real layer-27 frame,
with all other arithmetic in FP64, the residual representation contributes
relative-L2 `1.98e-7`; power-of-two scaling reduces this to `5.87e-9`. This is
a representation-only counterfactual, not a kernel or model result. The new
regression isolates scores 0/-8 and inspects FP32 partition outputs before
final FP16 rounding, including positive/negative minimum and maximum finite
E4M3 values to exercise subnormal scaled V operands on actual hardware.

The scaled-residual experimental library is
`87ffc1fd626e3d0fa46f6e2bda741a88d79d53809685a4d569c37e4d35b325cd`.
Its 89 GPU checks pass. The new small-probability test rejects revision 2
(`2.2877e-5` relative error against a `3e-6` bound), while the scaled-residual
build passes all six signed/range cases. In the same 200-group replay,
FP16 element disagreements with the FP64-rounded reference decrease from
2092 to 1872; in 32 additional real 128K frames they decrease from 348 to 320.
The four same-GPU q5/page848 100-ABBA speed ratios versus revision 2 are
1.0007/1.0000/1.0007/1.0007, effectively unchanged. These improvements do not
establish bitwise FP64 equivalence or model quality.

A completed revision-2 same-process 128K reference--candidate--reference
bracket matches all 256 token IDs in all three requests; the two references
are stable. It does not reproduce the earlier cross-process token-151
difference, whose full cause remains unlocalized. This instrumented,
cached-prefix bracket is not a cold-process or throughput benchmark.
The scaled-residual build also passes 62 Compute Sanitizer checks with zero
memory errors. Its same-process 128K bracket has now completed and **fails**
the deterministic token gate: the two FP64 reference requests match all 256
tokens, while the candidate first differs at token 77. Both references select
token 799 (`one`) and the candidate selects 23438 (`engineers`). The two
reference top logprobs are tied; the candidate favors 23438 by 0.015625. A
near-tie does not waive the stated no-divergence gate or establish broad
semantic degradation.

All four TP workers select the native entry. Rank-zero instrumentation covers
all 16 full-attention layers in all three requests, with finite outputs.
The archived result hash is
`911c9906aed7ecf48d2673ce8267f2d27d2de22da65e55a0ec7ae36bbbff1926`.
The reference sequence also differs from the previous revision-2 process's
reference sequence. This prevents attributing the model difference solely to
the residual scaling; it does not turn this failing bracket into a pass.
Accumulation remains FP32. There is no production FP64 path, default-format
promotion, or transfer of the private 60-TFLOP/s prefill acceptance to this
small-query implementation.
