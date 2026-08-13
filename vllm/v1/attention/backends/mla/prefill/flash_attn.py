# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention backend for MLA prefill."""

import functools
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None  # type: ignore[assignment]


@functools.cache
def _flash_v100_dense_prefill_usable() -> bool:
    """0014: True iff this device is SM70 with the in-tree Flash-V100 dense
    prefill op available. On SM70 the vendored FA2 called below can never run
    (fa_utils stub raises ImportError on first prefill), so this predicate gates
    an automatic route to the in-tree op. Exact (7,0) match: the flash_attn_v100
    kernel TORCH_CHECKs sm70 itself. Cached: device + op availability are
    process-invariant."""
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    if capability is None or capability[0] != 7 or capability[1] != 0:
        return False
    try:
        from vllm.v1.attention.backends.flash_attn_v100 import (
            flash_v100_dense_prefill_available,
        )
    except ImportError:
        return False
    return flash_v100_dense_prefill_available()


@functools.cache
def _flash_v100_dense_prefill_lse_usable() -> bool:
    """0014: True iff the in-tree Flash-V100 dense op can also emit the LSE.
    Same SM70 gate as above plus availability of the LSE-capable forward entry.
    Separate predicate so an older wheel degrades to no-LSE coverage instead of
    failing the whole route."""
    if not _flash_v100_dense_prefill_usable():
        return False
    try:
        from vllm.v1.attention.backends.flash_attn_v100 import (
            flash_v100_dense_prefill_lse_available,
        )
    except ImportError:
        return False
    return flash_v100_dense_prefill_lse_available()


# 0014: head-dim tiles the in-tree Flash-V100 dense prefill kernel instantiates
# (flash-attention-v100/kernel/fused_mha_forward.cu:1136-1142 `switch (D)`; anything
# else hits `default: TORCH_CHECK(false, "Unsupported D")`). A uniform MLA head dim
# outside this set would crash the kernel at runtime, so the SM70 route must not be
# offered for it. Kept beside the gate that reads it; update in lockstep with the
# kernel switch (adding `case 192:` for the DeepSeek slice would add 192 here too).
_SM70_DENSE_PREFILL_HEAD_DIMS = frozenset({16, 32, 64, 128, 256})


class FlashAttnPrefillBackend(MLAPrefillBackend):
    """FlashAttention backend for MLA prefill."""

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def is_available(cls) -> bool:
        # SM80+ builds: the vendored FA2/FA3 varlen extension.
        # SM70/V100 builds are compiled WITHOUT that extension (0013 makes
        # is_flash_attn_varlen_func_available() honestly False there); on SM70 this
        # backend serves MLA prefill through the in-tree Flash-V100 dense op instead
        # (0014, routed in _flash_attn_varlen_diff_headdims). Either presence means a
        # working MLA-prefill implementation exists. The SM70 route's shape/dtype
        # constraints are per-config, not device-global, so they are enforced in
        # validate_configuration (which sees the model's head dims and dtype), not
        # here.
        return (
            is_flash_attn_varlen_func_available()
            or _flash_v100_dense_prefill_lse_usable()
        )

    @classmethod
    def validate_configuration(cls, device_capability, selector_config) -> list[str]:
        invalid_reasons = super().validate_configuration(
            device_capability, selector_config
        )
        # 0014: when the ONLY available implementation is the SM70 in-tree dense
        # route (FA2 varlen absent, Flash-V100 dense op present), that route computes
        # only the uniform-head-dim, fp16 slice. Reject anything it cannot run here,
        # AT SELECTION TIME, so a config the route would fall through on gets a clean
        # load-time error (0013's guarantee) instead of loading and then crashing at
        # the first prefill in the raising FA2 stub. On SM80+ this block is inert
        # (is_flash_attn_varlen_func_available() is True).
        if (
            not is_flash_attn_varlen_func_available()
            and _flash_v100_dense_prefill_lse_usable()
        ):
            if selector_config.dtype != torch.float16:
                invalid_reasons.append(
                    "SM70 Flash-V100 MLA prefill route requires float16 "
                    f"(got {selector_config.dtype})"
                )
            if selector_config.qk_head_dim != selector_config.v_head_dim:
                invalid_reasons.append(
                    "SM70 Flash-V100 MLA prefill route requires uniform qk/v head "
                    f"dims (got qk={selector_config.qk_head_dim}, "
                    f"v={selector_config.v_head_dim}); DeepSeek-style asymmetric "
                    "MLA (qk=192/v=128) is 0014's deferred slice"
                )
            elif selector_config.qk_head_dim not in _SM70_DENSE_PREFILL_HEAD_DIMS:
                invalid_reasons.append(
                    "SM70 Flash-V100 MLA prefill route has no compiled kernel tile "
                    f"for head dim {selector_config.qk_head_dim} "
                    f"(compiled: {sorted(_SM70_DENSE_PREFILL_HEAD_DIMS)})"
                )
        return invalid_reasons

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

        # Handle the differences between the flash_attn_varlen from
        # flash_attn and the one from vllm_flash_attn
        if flash_attn_varlen_func is None:
            # 0014: SM70/V100 build without the FA2 varlen extension. This backend
            # serves MLA prefill through the in-tree Flash-V100 dense op instead
            # (routed at the top of _flash_attn_varlen_diff_headdims), so no FA2
            # setup is needed. validate_configuration has already guaranteed the
            # SM70 route is present and shape/dtype-valid for this model; a genuinely
            # absent op still fails loudly here rather than silently mis-routing.
            assert _flash_v100_dense_prefill_lse_usable(), (
                "FlashAttnPrefillBackend requires either flash_attn_varlen_func or "
                "the SM70 Flash-V100 dense prefill op. Ensure "
                "FlashAttnPrefillBackend.is_available() is checked first."
            )
            self.flash_attn_varlen_func = None
            self.vllm_flash_attn_version = None
            # Uniform head dims on this route (validate_configuration enforced qk==v),
            # so V never needs padding — and the route intercepts before the pad path.
            self.requires_v_padding = False
            self._is_vllm_fa = current_platform.is_cuda()
            return

        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.vllm_flash_attn_version = get_flash_attn_version(head_size=qk_head_dim)
        if self.vllm_flash_attn_version is not None:
            self.flash_attn_varlen_func = functools.partial(
                flash_attn_varlen_func, fa_version=self.vllm_flash_attn_version
            )

        # Determine if we need to pad V
        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim for attention backends that do
        # not support different headdims.
        # FA3 on Hopper (SM90) and FA4 natively handle diff headdims.
        device_capability = current_platform.get_device_capability()
        self.requires_v_padding = self.vllm_flash_attn_version is None or not (
            (
                self.vllm_flash_attn_version == 3
                and device_capability is not None
                and device_capability[0] == 9
            )
            or self.vllm_flash_attn_version == 4
        )

        # Track whether we're using vllm's FA or upstream (for ROCm)
        self._is_vllm_fa = current_platform.is_cuda() or current_platform.is_xpu()

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # ------------------------------------------------------------------
        # 0014: V100 (SM70) MLA prefill route (revives archived 0020 + 0043).
        # On SM70 the vendored FA2 call below is a guaranteed ImportError
        # (fa_utils stub), so when the in-tree Flash-V100 dense op can faithfully
        # compute this exact call we route to it. Guard sets are EXACT — any call
        # outside them falls through to the FA2 call, which FAILS LOUDLY on SM70
        # rather than silently computing wrong values.
        #
        # This slice covers the UNIFORM head-dim case (GLM-MoE-Lite MLA: qk==v,
        # e.g. 256==256). DeepSeek-style diff head dims (qk 192 / v 128) still
        # fall through and crash on SM70 — that case additionally needs the
        # FA_V100 `case 192:` build + a pad-V driver and is 0014's deferred slice.
        cu_q = kwargs.get("cu_seqlens_q")
        cu_k = kwargs.get("cu_seqlens_k")
        causal = bool(kwargs.get("causal", False))

        # (0043) LSE-emitting route — the context-chunk / prefix-cache path that
        # needs the softmax LSE and an INDEPENDENT cu_seqlens_k. Guard: LSE-capable
        # wheel, fp16, uniform head dims, both cu_seqlens present and same batch,
        # and causal only with cu_k IS cu_q (masking for M!=N is unvalidated here).
        if (
            _flash_v100_dense_prefill_lse_usable()
            and q.dtype == torch.float16
            and k.dtype == torch.float16
            and v.dtype == torch.float16
            and q.shape[-1] == v.shape[-1]
            and cu_q is not None
            and cu_k is not None
            and cu_q.numel() == cu_k.numel()
            and (cu_k is cu_q or not causal)
        ):
            from vllm.v1.attention.backends.flash_attn_v100 import (
                flash_v100_dense_prefill_lse,
            )

            q_c = q.contiguous()
            out = torch.empty_like(q_c)
            lse = torch.empty(
                (q_c.shape[1], q_c.shape[0]),
                dtype=torch.float32,
                device=q_c.device,
            )
            flash_v100_dense_prefill_lse(
                q_c,
                k.contiguous(),
                v.contiguous(),
                out,
                lse,
                cu_q,
                cu_k,
                q_c.shape[0],
                float(softmax_scale)
                if softmax_scale is not None
                else q.shape[-1] ** -0.5,
                causal=causal,
            )
            if return_softmax_lse:
                return out, lse
            return out

        # (0020) no-LSE route — the new-tokens self-attention path on wheels that
        # predate the LSE entry. Requires cu_k IS cu_q and no LSE requested.
        if (
            _flash_v100_dense_prefill_usable()
            and not return_softmax_lse
            and q.dtype == torch.float16
            and q.shape[-1] == v.shape[-1]
            and cu_q is not None
            and cu_k is cu_q
        ):
            from vllm.v1.attention.backends.flash_attn_v100 import (
                flash_v100_dense_prefill,
            )

            q_c = q.contiguous()
            out = torch.empty_like(q_c)
            flash_v100_dense_prefill(
                q_c,
                k.contiguous(),
                v.contiguous(),
                out,
                cu_q,
                q_c.shape[0],
                float(softmax_scale)
                if softmax_scale is not None
                else q.shape[-1] ** -0.5,
                causal=causal,
            )
            return out
        # --- end 0014 route -----------------------------------------------

        maybe_padded_v = v
        if self.requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        if self._is_vllm_fa:
            kwargs["return_softmax_lse"] = return_softmax_lse
        else:
            # ROCm leverages the upstream flash_attn, which takes a parameter
            # called "return_attn_probs" instead of return_softmax_lse
            kwargs["return_attn_probs"] = return_softmax_lse
        if envs.VLLM_BATCH_INVARIANT:
            kwargs["num_splits"] = 1

        attn_out = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=maybe_padded_v,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there are multiple results
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Unpad output back to v_head_dim if we padded V
        if self.requires_v_padding:
            attn_out = attn_out[..., : v.shape[-1]]

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._prefill_metadata.chunked_context is not None
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.chunked_context.cu_seq_lens[chunk_idx],
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.chunked_context.max_seq_lens[chunk_idx],
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
        )
