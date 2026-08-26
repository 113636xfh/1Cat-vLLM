#!/usr/bin/env bash
# Canonical launch for the local vLLM server (Qwen3.8 VL, 2x V100, TP=2).
#
# 2026-08-26 v2:
#   - --limit-mm-per-prompt: {"image":1,"video":0} -> {"image":4,"video":1}
#     (the 1-image cap 400'd any pi prompt carrying 2+ accumulated images;
#      video:1 added for single-clip vision input)
#   - Added the SM70 (V100) environment block. The first relaunch without it
#     failed with: "Quantization scheme is not supported ... Min capability:
#     75, Current: 70" — the custom 1cat vLLM build needs these for its SM70
#     AWQ (compressed-tensors/Turbomind) and flash-attention paths.
#     NOTE: /proc/PID/cmdline does NOT carry environment variables; capture
#     /proc/PID/environ before killing the old server when changing this file.
#
# Usage:  nohup bash /home/xfh/1cat-vllm/launch-vllm.sh > /home/xfh/1cat-vllm/vllm-server.log 2>&1 &
# Stop:   pkill -f '1cat-vllm/venv/bin/[p]ython'

# --- SM70 (V100) requirements, captured from the running server 2026-08-26 ---
export NCCL_P2P_DISABLE=1
export VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND=1
export VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1

exec /home/xfh/1cat-vllm/venv/bin/python -u -m vllm.entrypoints.openai.api_server \
  --model /mnt/ssd/home/xfh/models/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ \
  --served-model-name Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.97 \
  --max-model-len 145824 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --mamba-block-size 512 \
  --cpu-offload-gb 0.8 \
  --cpu-offload-params visual \
  --kv-offloading-size 12 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --limit-mm-per-prompt '{"image":4,"video":1}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --disable-custom-all-reduce \
  --host 0.0.0.0 \
  --port 8080
