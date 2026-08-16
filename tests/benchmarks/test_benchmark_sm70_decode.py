# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

from benchmarks import benchmark_sm70_decode


def test_sm70_fa2_d256_prefill_status_reports_import_error(monkeypatch):
    def fail_import(_name):
        raise ImportError("missing vendored extension")

    monkeypatch.setattr(benchmark_sm70_decode.importlib, "import_module", fail_import)

    status = benchmark_sm70_decode._sm70_fa2_d256_prefill_status(
        types.SimpleNamespace(ops=types.SimpleNamespace())
    )

    assert status["available"] is False
    assert status["error"] == "ImportError: missing vendored extension"
    assert not any(status["required_ops"].values())


def test_sm70_fa2_d256_prefill_status_requires_dense_and_paged_ops(monkeypatch):
    monkeypatch.setattr(
        benchmark_sm70_decode.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(),
    )
    namespace = types.SimpleNamespace(
        sm70_d256_splitd_n32_dense_fwd=object(),
        sm70_d256_splitd_n32_paged_fwd=object(),
        sm70_d256_splitd_n32_dense_splitkv3_fwd=object(),
    )
    fake_torch = types.SimpleNamespace(ops=types.SimpleNamespace(_vllm_fa2_C=namespace))
    fake_extension = types.SimpleNamespace(__file__="/tmp/_vllm_fa2_C.abi3.so")
    monkeypatch.setitem(
        benchmark_sm70_decode.sys.modules,
        "vllm.vllm_flash_attn._vllm_fa2_C",
        fake_extension,
    )

    status = benchmark_sm70_decode._sm70_fa2_d256_prefill_status(fake_torch)

    assert status["available"] is True
    assert status["error"] is None
    assert all(status["required_ops"].values())
    assert all(status["optional_ops"].values())
