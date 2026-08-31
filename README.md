# 1Cat-vLLM

> 一猫之下始终相信，V100 不该在今天的大模型浪潮里被轻易宣判“过时”。
>
> 1Cat-vLLM 是面向 **SM70 / Tesla V100** 的 vLLM 工程分支。项目围绕
> AWQ、注意力后端、长上下文稳定性、MTP 投机解码、运行时默认值和部署
> 路径做了成体系的优化，让更多现代模型场景在 V100 上真正变得可用、
> 好用、能持续部署。
>
> 我们希望把一猫之下在 V100 上的工程经验、优化成果和验证过程贡献给
> 开源社区，也欢迎继续使用 V100 的个人开发者、工作室和团队一起反馈、
> 复现和改进。

1Cat-vLLM is a **Tesla V100 / SM70** focused vLLM fork for serving modern
Qwen-class AWQ and experimental FP8 models on Volta GPUs. It integrates
TurboMind-derived SM70 kernels, a V100 FlashAttention path, runtime defaults
for long-context serving, and OpenAI-compatible API fixes for common clients.

## Project Focus

- **V100 / SM70 first**: optimized for Tesla V100 rather than being a generic
  multi-hardware fork.
- **AWQ on Volta**: AWQ 4-bit inference paths for dense and MoE Qwen models on
  SM70.
- **V100 FlashAttention path**: `FLASH_ATTN_V100` decode and prefill backend
  for Volta GPUs, with SM70 compile-graph, guarded XQA decode, and D=256
  paged-prefix low-smem fast paths enabled by default.
- **Long-context serving**: public profiles default to 256K context where the
  model and memory budget allow it.
- **MTP serving**: Qwen3.6-class MTP speculative decoding remains available as
  an explicit opt-in path; long-context public profiles default to no MTP.
- **Image inputs by default**: SM70 `FLASH_ATTN_V100` profiles allow one image
  per prompt by default; video inputs remain opt-in.
- **Tool calling and OpenAI API compatibility**: validated with OpenAI-style
  clients such as Cherry Studio, OpenClaw, and similar tools.
- **Validated Qwen3.8 DFlash2 profile**: the four-V100 NVFP4/TP4/B1 profile
  includes prefix caching, Mamba state alignment, tool/reasoning parsing,
  FP8 E5M2 target KV, and guarded Flash-V100 grouped verification.
- **Experimental FP8 work**: FP8 model and KV-cache paths are included for
  validation, but they are not production defaults.
- **Experimental speculative research**: DFlash1/DDTree, lookup-augmented q16,
  and n-gram chaining remain opt-in research paths.

## Recommended Model Providers

- `tclf90/Qwen3.6-27B-AWQ`
- `tclf90/Qwen3.6-35B-A3B-AWQ`
- `tclf90/Qwen3.5-122B-A10B-AWQ` for larger 4-GPU setups

The launch examples use local paths such as `/path/to/Qwen3.6-27B-AWQ`.
Replace them with your local model path or a Hugging Face repository id.

## Hardware Target

The public commands are written for V100 Qwen serving workloads. Image inputs
are enabled by default on the SM70 `FLASH_ATTN_V100` path; video inputs are
disabled by default and should be enabled explicitly only after local memory
validation.

| Host | Notes |
| --- | --- |
| 4 x Tesla V100 32 GB | Main public reference target |
| 2 x Tesla V100 32 GB | Supported for selected 27B profiles with lower concurrency |

Typical model placement:

- `Qwen3.6-27B-AWQ`: TP1/TP2/TP4 supported; TP4 is the public reference.
- `Qwen3.6-27B-AWQ + MTP`: explicit opt-in profile for local validation, not
  the long-context public default.
- `Qwen3.6-35B-A3B-AWQ`: TP4 recommended.
- `Qwen3.5-122B-A10B-AWQ`: TP4 supported for larger deployments.

Multimodal defaults:

- Default SM70 `FLASH_ATTN_V100` serving allows `image=1`, `video=0` when
  `--limit-mm-per-prompt` is not set.
- For text-only serving, pass `--limit-mm-per-prompt '{"image":0,"video":0}'`
  or use `--language-model-only`.
- For video workloads, pass an explicit limit such as
  `--limit-mm-per-prompt '{"image":1,"video":1}'` and retune memory settings.

## Validated Stack

The public wheel path is validated on:

- OS: Ubuntu 24.04 LTS
- Python: 3.12
- CUDA toolkit: 12.8
- PyTorch: CUDA 12.8 runtime wheels
- GPU: Tesla V100 32 GB

## Quick Start

### 1. Install CUDA 12.8

Use the official NVIDIA repository on Ubuntu 24.04:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

If the machine also has another CUDA toolkit installed, force build-time and
runtime CUDA to 12.8:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
hash -r
nvcc -V
```

### 2. Create the Python environment

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda create -y -n 1cat-vllm-sm70 python=3.12
conda activate 1cat-vllm-sm70

python -m pip install --upgrade pip setuptools wheel
```

### 3. Install from Prebuilt Wheels

Prebuilt wheels are the recommended installation path for public users. Source
builds are intended for kernel development.

Download the latest wheel assets from:

```text
https://github.com/1CatAI/1Cat-vLLM/releases/latest
```

Install the wheel from the directory where you downloaded it:

```bash
python -m pip install --prefer-binary --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  ./1cat_vllm-*.whl
```

Notes:

- The `1cat_vllm` wheel already bundles the `flash_attn_v100` Python package
  and SM70 CUDA extensions.
- Runtime installation from wheels does not require the bundled `lmdeploy`
  source tree.
- Use Python 3.12 and CUDA 12.8.
- If your shell has a broken local proxy configured, unset it before
  installing:
  `env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy ...`.
- After installing from wheels, run `python -m vllm...` from a directory
  outside this source checkout, such as `cd ~` or `cd /tmp`. Running inside the
  cloned repository makes Python import the local source tree instead of the
  wheel-installed CUDA extensions.

### 4. Verify the Environment

```bash
python - <<'PY'
import torch, triton, vllm, sys
import flash_attn_v100
from flash_attn_v100 import flash_attn_v100_cuda, paged_kv_utils
from flash_attn_v100 import flash_attn_grouped_verify_max_query_tokens
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("triton", triton.__version__)
print("vllm", vllm.__version__)
print("flash_attn_v100", flash_attn_v100.__version__)
print("dflash2_grouped_verify_max_q", flash_attn_grouped_verify_max_query_tokens())
PY
```

## Recommended Launch Commands

These are the recommended public serving commands for the 27B AWQ and 35B AWQ
V100 profiles. When using prebuilt wheels, run them outside the source checkout
so Python loads the installed package and its CUDA extensions.

Use `CUDA_VISIBLE_DEVICES=0,1,2,3` only when you need to select a specific
four-card V100 set.

### Qwen3.6-27B-AWQ, TP4

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b-awq \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port 8000
```

### Qwen3.6-35B-A3B-AWQ, TP4

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.6-35B-A3B-AWQ \
  --served-model-name qwen3.6-35b-a3b-awq \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --host 0.0.0.0 \
  --port 8000
```

### Qwen3.8-27B-NVFP4 + DFlash2, TP4 low-latency profile

This is the quality-audited four-V100 DFlash2 profile. The target checkpoint
must use compressed-tensors NVFP4 with an unquantized/BF16-compatible LM head;
the draft revision and selector top-K are part of the checkpoint contract.

```bash
python -m vllm.entrypoints.cli.main serve /path/to/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b-dflash2 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN_V100 \
  --kv-cache-dtype fp8_e5m2 \
  --max-model-len 262144 \
  --performance-mode interactivity \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","revision":"dedf8df68adfb1afeaf7b7480c0a0243108177b4","kv_cache_dtype":"auto"}' \
  --host 0.0.0.0 \
  --port 8000
```

On SM70, this command resolves the following values automatically unless an
explicit override is present:

- DFlash2 draft width `7`, from checkpoint block size `8`;
- probabilistic draft sampling and the draft `FLASH_ATTN_V100` backend;
- `max_num_seqs=1` and `max_num_batched_tokens=4096` for
  `--performance-mode interactivity`;
- prefix caching and Mamba `align` state caching for the API server;
- the quality-audited NVFP4 DFlash2 fast paths when the exact TP4/B1,
  selector-top16, q4096, and E5M2-KV contract is present.

Use explicit `--max-num-seqs` and `--max-num-batched-tokens` values for a
concurrent service. That preserves capacity but intentionally leaves the B1
17-18 ms admission contract. Every `VLLM_SM70_DFLASH2_*` environment value is
also an authoritative rollback. Do not enable candidate-order reranking;
dense-order reranking is the validated path.

## OpenAI-Compatible Request Example

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{
    "model": "qwen3.6-27b-awq",
    "messages": [{"role": "user", "content": "用一句话回答，2+2等于几？"}],
    "temperature": 0,
    "max_completion_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

If the response is coherent and short, the API path is basically healthy.

## Experimental Features

### FP8

FP8 support is included for validation and research. It is not the stable
public default.

- FP8 model execution on V100 is experimental.
- `fp8_e5m2` KV cache can be used experimentally on V100.
- `fp8_e4m3` is not the recommended V100 option in the current path.
- Do not add `--calculate-kv-scales` unless you are specifically testing KV
  scale calculation behavior.

Example:

```bash
--kv-cache-dtype fp8_e5m2
```

### Other DFlash modes

The Qwen3.8 DFlash2 command above is the validated release profile. DFlash1,
`dflash_ddtree`, lookup-augmented q16 verification, n-gram assist, and
drafter-free chaining remain workload-dependent opt-in modes. Validate their
acceptance, output quality, and end-to-end throughput before production use.

### MTP

MTP is not enabled by default in the V100 public serving profile. Long-context
decode on V100 can slow down significantly when MTP is enabled, so keep the
default no-MTP path for 128K/256K style serving unless your own workload proves
otherwise.

To explicitly test the previous automatic SM70 MTP4 profile:

```bash
export VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1
```

You can also pass an explicit `--speculative-config` when you want full control
over speculative decoding settings.

### Dense F16 Fast Path

`VLLM_SM70_ENABLE_DENSE_F16_FASTPATH=1` is intended for targeted experiments.
Keep it disabled for public MoE serving profiles unless you are explicitly
benchmarking that path.

## Source Build

Source build is supported, but it is **not recommended** for normal runtime
deployment. Install the release wheels first unless you are changing CUDA,
C++, or Triton code.

This repository includes the validated `lmdeploy` source tree under
`csrc/sm70_turbomind/lmdeploy`, which is needed by the SM70 AWQ build path.

```bash
cd /path/to/1Cat-vLLM/vllm
test -d csrc/sm70_turbomind/lmdeploy
```

Install build dependencies:

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate 1cat-vllm-sm70

python -m pip install -r requirements/build/cuda.txt
python -m pip install -r requirements/cuda.txt
python -m pip install -r requirements/common.txt
python -m pip install cmake build
```

Build wheels:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.0"
export FLASH_ATTN_V100_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=12
export NVCC_THREADS=1
# Use this for a pre-tag 1.5.0 release candidate. An exact v1.5.0 checkout
# derives the same version from its Git tag and does not need the override.
export VLLM_VERSION_OVERRIDE=1.5.0

rm -rf build vllm.egg-info
rm -rf .deps/*-build .deps/*-subbuild

pushd flash-attention-v100
python -m build --wheel --no-isolation --outdir ../dist-cu128-sm70
popd

python -m build --wheel --no-isolation --outdir dist-cu128-sm70
```

For editable development:

```bash
python -m pip install -e . --no-build-isolation
```

## Benchmarking Notes

- First-request warmup is slow on V100 and should not be included in
  steady-state throughput.
- Browser-side OpenAI streaming throughput includes request overhead and should
  not be compared directly with strict incremental decode TPS.
- Long-context throughput depends strongly on TP, `max_num_seqs`,
  `max_num_batched_tokens`, prompt shape, and attention backend.
- If you publish a baseline, include the full launch command, GPU model,
  driver, CUDA runtime, model checkpoint, sampling parameters, prompt length,
  and decode length.

## WeChat Community

**群聊：** 1Cat-vLLM 开源交流群

请使用微信扫描下方二维码加入群组：

<img width="966" height="1518" alt="e52eca541f6c191928963c97ac1b38e5" src="https://github.com/user-attachments/assets/73be9e1e-41c3-49d7-9d4e-db4ce97cce9b" />

> 提示：微信群二维码通常 7 天内有效。若扫描失败或提示过期，请微信搜索微信号添加：YM_isi。

## Repository Notes

- Upstream project: [vLLM](https://github.com/vllm-project/vllm)
- This fork focuses on SM70 AWQ support, V100-oriented attention/runtime
  tuning, and experimental FP8/MTP/DFlash validation paths.
- Prebuilt wheels are the public installation path.
- Source builds are for development and kernel work.

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm)
- [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)
- [flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)
- [marlin_v100](https://github.com/zhinianqin/marlin_v100)

## License

This repository follows the upstream vLLM license model. See [LICENSE](LICENSE).
