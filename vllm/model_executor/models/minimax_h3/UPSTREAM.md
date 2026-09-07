# Native MiniMax H3 provenance

The numerical model and media helpers are adapted from Apache-2.0 licensed
vLLM-Omni commit `7be014bce6374f06c95b703763bdbac4c6198f31`.
Comfy checkpoint handling is informed by vLLM-Omni PR #6894 at
`be89929001563a882f1f783fb9ee37b4d5a970f4` (not merged at port time).
Original copyright and SPDX notices remain on adapted files.

This package is part of 1Cat-vLLM. It does not import or install vllm-omni.
The initial scope excludes approximate diffusion caches, LoRA, step batching,
Ulysses/Ring parallelism, and other model families. Model weights and checkpoint
remote code retain their respective upstream terms; they are not vendored here.

Frozen model revisions:

- MiniMaxAI/MiniMax-H3: `42ed227ee7df40d41602854ae760620d6eb651fe`.
- Comfy-Org/MiniMax-H3: `a98869194787969724c7425d95d0ed73ce9202af`.
