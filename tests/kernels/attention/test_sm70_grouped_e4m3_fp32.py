# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-quantized-KV FP64 oracle; these are not model-quality tests."""

import pytest
import torch


def _native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    module = pytest.importorskip("flash_attn_v100")
    if not module.flash_attn_grouped_e4m3_fp32_available():
        pytest.skip("rebuild native E4M3 FP32 entry")
    return module.flash_attn_grouped_e4m3_fp32_paged


@pytest.mark.parametrize(
    "rows,page,length",
    [
        *[(rows, 848, 8197) for rows in range(2, 9)],
        (5, 800, 8003),
        (8, 1616, 65536),
        (5, 1648, 131072),
        (5, 3296, 262144),
    ],
)
def test_fp32_grouped_row_lengths_graph(rows, page, length):
    op = _native()
    torch.manual_seed(20260906)
    pages = (length + page - 1) // page
    capacity = pages * page
    q = torch.randn((rows, 6, 256), device="cuda", dtype=torch.float16)
    raw = torch.randn((2, capacity, 1, 256), device="cuda", dtype=torch.float16)
    encoded = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    backing = torch.empty((pages, 2, page, 1, 256), device="cuda", dtype=torch.uint8)
    k, v = backing.unbind(1)
    order = torch.randperm(pages, device="cuda")
    k[order] = encoded[0].reshape_as(k)
    v[order] = encoded[1].reshape_as(v)
    table = order.int()[None].contiguous()
    lengths = torch.arange(
        length - rows + 1, length + 1, device="cuda", dtype=torch.int32
    )
    initial = lengths.clone()
    out = torch.empty_like(q)
    ks, vs = 0.5, 1.25
    rk = encoded[0, :length, 0].view(torch.float8_e4m3fn).double() * ks
    rv = encoded[1, :length, 0].view(torch.float8_e4m3fn).double() * vs

    def call():
        return op(
            q,
            k,
            v,
            table,
            lengths,
            out=out,
            softmax_scale=0.0625,
            k_scale=ks,
            v_scale=vs,
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for state in ("live", "all_zero", "tail_zero", "first_zero", "restore"):
        lengths.copy_(initial)
        if state == "all_zero":
            lengths.zero_()
        elif state == "tail_zero":
            lengths[-1] = 0
        elif state == "first_zero":
            lengths[0] = 0
        graph.replay()
        score = q.transpose(0, 1).double() @ rk.T * 0.0625
        mask = torch.arange(length, device="cuda")[None] >= lengths[:, None]
        score.masked_fill_(mask[None], -torch.inf)
        expected = (score.softmax(-1).nan_to_num(0.0) @ rv).transpose(0, 1)
        assert bool(torch.isfinite(out).all())
        assert bool((out[lengths == 0] == 0).all())
        if state != "all_zero":
            relative_l2 = (out.double() - expected).norm() / expected.norm()
            assert float(relative_l2) < 0.001
        if state == "live":
            original = out.clone()
        elif state == "restore":
            assert torch.equal(out, original)


def test_fp32_workspace_is_separate_from_legacy_half_workspace():
    _native()
    from flash_attn_v100.flash_attn_interface import _get_grouped_verify_workspace

    q = torch.zeros((5, 6, 256), dtype=torch.float16, device="cuda")
    half = _get_grouped_verify_workspace(q)
    full = _get_grouped_verify_workspace(q, partial_dtype=torch.float32)
    assert half.partial_out.dtype == torch.float16
    assert full.partial_out.dtype == torch.float32
    assert half.partial_out.data_ptr() != full.partial_out.data_ptr()
    assert _get_grouped_verify_workspace(q) is half
    assert _get_grouped_verify_workspace(q, partial_dtype=torch.float32) is full
