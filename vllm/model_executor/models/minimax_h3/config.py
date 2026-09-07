# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request and deployment contracts for native MiniMax H3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

BASE_MODEL = "MiniMaxAI/MiniMax-H3"
BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
COMFY_MODEL = "Comfy-Org/MiniMax-H3"
COMFY_REVISION = "a98869194787969724c7425d95d0ed73ce9202af"
DEFAULT_PROMPT = (
    "一个连续的写实镜头。白天自然光下的公园浅水池，一艘红色纸船从画面左侧"
    "缓慢漂向右侧，一只黄色橡皮鸭从纸船后方经过。微风在水面形成细小涟漪，"
    "背景绿树保持稳定。镜头缓慢向前推进，纸船与橡皮鸭的颜色、形状和数量"
    "始终一致。音轨包含轻柔流水声和远处鸟鸣，无人声、无音乐、无字幕。"
)


class H3InputError(ValueError):
    """Invalid media or generation parameters, safe to report to a client."""


@dataclass(frozen=True)
class H3Config:
    model: str = BASE_MODEL
    revision: str = BASE_REVISION
    partition: Literal["fl2va", "ref2va"] = "fl2va"
    transformer_path: str | None = None
    tensor_parallel_size: int = 4
    attention_backend: str = "FLASH_ATTN_V100"
    fp16_weight_cache_gib: float = 0.0
    fp16_cache_layers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.partition not in ("fl2va", "ref2va"):
            raise H3InputError("partition must be fl2va or ref2va")
        if self.tensor_parallel_size not in (1, 2, 4):
            raise H3InputError("native H3 supports TP1, TP2, or TP4")
        if self.attention_backend not in (
            "FLASH_ATTN_V100",
            "FLASHINFER_SM70",
            "TORCH_SDPA",
        ):
            raise H3InputError(f"unsupported H3 attention: {self.attention_backend}")
        if (
            not math.isfinite(self.fp16_weight_cache_gib)
            or self.fp16_weight_cache_gib < 0
        ):
            raise H3InputError("FP16 weight-cache budget must be finite and >= 0")
        if self.fp16_weight_cache_gib and not self.fp16_cache_layers:
            raise H3InputError("select measured cache layers with --fp16-cache-layer")
        if len(set(self.fp16_cache_layers)) != len(self.fp16_cache_layers):
            raise H3InputError("FP16 cache layers must be unique")


@dataclass
class H3SamplingParams:
    height: int = 768
    width: int = 1344
    fps: int = 24
    num_frames: int = 243
    num_inference_steps: int = 50
    seed: int = 42
    num_outputs_per_prompt: int = 1
    quality: str | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise H3InputError("video dimensions must be positive")
        if self.height % 32 or self.width % 32:
            raise H3InputError("H3 video dimensions must be multiples of 32")
        if self.fps != 24:
            raise H3InputError("H3 generates at 24 FPS")
        if not 96 <= self.num_frames <= 362:
            raise H3InputError("H3 duration must be between 4 and 15 seconds")
        if self.num_inference_steps < 2:
            raise H3InputError("H3 needs at least two sigma positions")
        if self.num_outputs_per_prompt != 1:
            raise H3InputError("native H3 generates one video per request")
        if self.quality not in (None, "lossless"):
            raise H3InputError("native H3 does not enable approximate step caches")


@dataclass
class H3Request:
    prompt: str = DEFAULT_PROMPT
    sampling: H3SamplingParams = field(default_factory=H3SamplingParams)
    media: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise H3InputError("prompt must not be empty")
