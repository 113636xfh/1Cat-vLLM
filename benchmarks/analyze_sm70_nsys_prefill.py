#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize one SM70 prefill request from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


def _category(name: str) -> str:
    lower = name.lower()
    if "turbomind::gemm" in lower and "fp4_e2m1_t" in lower:
        return "TurboMind MXFP4 MoE GEMM"
    if "turbomind::gemm" in lower and "__nv_fp8_e4m3" in lower:
        return "TurboMind FP8 dense GEMM"
    if "turbomind::gemm" in lower:
        return "TurboMind other GEMM"
    if "nccl" in lower:
        return "NCCL collectives"
    if "mhc_" in lower or "hc_prenorm" in lower or "hc_head" in lower:
        return "mHC"
    if any(
        marker in lower
        for marker in (
            "kv_compress",
            "insert_k",
            "gather_k",
            "slot_mapping",
            "qnorm_rope",
            "inverse_rope",
            "save_partial_states",
        )
    ):
        return "KV compression/indexer/rope"
    if "sparse_gathered" in lower or "sparse_attn" in lower:
        return "SM70 sparse MLA/SWA attention"
    if any(
        marker in lower
        for marker in (
            "moerouting",
            "topkgating",
            "experttoken",
            "expertfirsttoken",
            "expandinputrows",
            "radixsort",
        )
    ):
        return "MoE routing"
    if any(marker in lower for marker in ("volta_", "cublas", "cutlass")):
        return "FP16/CUTLASS GEMM"
    if any(marker in lower for marker in ("rms_norm", "rmsnorm", "act_and_mul")):
        return "Norm/activation"
    if any(marker in lower for marker in ("argmax", "softmax")):
        return "LM head/sample"
    return "Other"


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _short_name(name: str) -> str:
    return name if len(name) <= 120 else name[:117] + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument(
        "--before-first-graph",
        action="store_true",
        help="Exclude decode beginning at the first CUDA Graph kernel per GPU.",
    )
    parser.add_argument("--device", type=int)
    parser.add_argument("--top-kernels", type=int, default=20)
    args = parser.parse_args()

    connection = sqlite3.connect(args.sqlite)
    rows = list(
        connection.execute(
            """
            SELECT k.deviceId, k.start, k.end, k.graphNodeId, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.demangledName
            ORDER BY k.deviceId, k.start
            """
        )
    )
    by_device: dict[int, list[tuple[int, int, int | None, str]]] = defaultdict(list)
    for device, start, end, graph_node, name in rows:
        by_device[device].append((start, end, graph_node, name))

    selected: dict[int, list[tuple[int, int, str]]] = {}
    for device, kernels in by_device.items():
        graph_starts = [start for start, _, graph, _ in kernels if graph is not None]
        cutoff = min(graph_starts) if args.before_first_graph and graph_starts else None
        selected[device] = [
            (start, end, name)
            for start, end, graph, name in kernels
            if (cutoff is None or (graph is None and end <= cutoff))
        ]

    summaries = {}
    for device, kernels in selected.items():
        if not kernels:
            continue
        intervals = [(start, end) for start, end, _ in kernels]
        start = min(item[0] for item in intervals)
        end = max(item[1] for item in intervals)
        summaries[device] = {
            "count": len(kernels),
            "service_ns": sum(item[1] - item[0] for item in intervals),
            "busy_ns": _union_ns(intervals),
            "envelope_ns": end - start,
        }

    print("device  kernels  service_ms  busy_union_ms  envelope_ms  idle_gap_ms")
    for device, summary in sorted(summaries.items()):
        idle_ns = summary["envelope_ns"] - summary["busy_ns"]
        print(
            f"{device:>6}  {summary['count']:>7}  "
            f"{summary['service_ns'] / 1e6:>10.3f}  "
            f"{summary['busy_ns'] / 1e6:>13.3f}  "
            f"{summary['envelope_ns'] / 1e6:>11.3f}  "
            f"{idle_ns / 1e6:>11.3f}"
        )

    if not summaries:
        raise SystemExit("No CUDA kernels found in the selected window")
    device = args.device
    if device is None:
        device = max(summaries, key=lambda item: summaries[item]["envelope_ns"])
    kernels = selected[device]
    total_service_ns = sum(end - start for start, end, _ in kernels)

    categories: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    kernel_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for start, end, name in kernels:
        duration = end - start
        category = _category(name)
        categories[category][0] += duration
        categories[category][1] += 1
        kernel_totals[name][0] += duration
        kernel_totals[name][1] += 1

    print(f"\nCritical device: {device}")
    print("category                            service_ms  service_%  launches")
    for category, (duration, count) in sorted(
        categories.items(), key=lambda item: item[1][0], reverse=True
    ):
        print(
            f"{category:<35} {duration / 1e6:>10.3f}  "
            f"{duration / total_service_ns * 100:>8.2f}  {count:>8}"
        )

    print("\nTop kernels by summed service time")
    print("service_ms  service_%  launches  kernel")
    for name, (duration, count) in sorted(
        kernel_totals.items(), key=lambda item: item[1][0], reverse=True
    )[: args.top_kernels]:
        print(
            f"{duration / 1e6:>10.3f}  {duration / total_service_ns * 100:>8.2f}  "
            f"{count:>8}  {_short_name(name)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
