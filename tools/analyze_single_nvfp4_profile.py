#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Summarize the steady GPU/CPU gap from a vLLM torch trace.

The analyzer deliberately reports two separate views: CUDA kernel time and
host work that fits between consecutive GPU execution annotations.  Nested
Python events are retained because their source locations are the useful
callsite evidence when deciding whether to change vLLM worker code.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"trace has no traceEvents list: {path}")
    return [event for event in events if isinstance(event, dict)]


def _aggregate(
    events: list[dict[str, Any]],
    windows: list[tuple[float, float]],
    category: str,
    tid: int,
    limit: int,
) -> list[dict[str, Any]]:
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != category:
            continue
        if event.get("tid") != tid:
            continue
        start = float(event.get("ts", 0.0))
        duration = float(event.get("dur", 0.0))
        end = start + duration
        if not any(start >= left and end <= right for left, right in windows):
            continue
        name = str(event.get("name", "<unnamed>"))
        row = totals[name]
        row[0] += 1
        row[1] += duration
        row[2] = max(row[2], duration)
    return [
        {
            "name": name,
            "count": int(values[0]),
            "total_us": values[1],
            "max_us": values[2],
        }
        for name, values in sorted(
            totals.items(), key=lambda item: item[1][1], reverse=True
        )[:limit]
    ]


def analyze(path: Path, limit: int = 20) -> dict[str, Any]:
    events = _events(path)
    gpu = sorted(
        (
            event
            for event in events
            if event.get("ph") == "X"
            and event.get("cat") == "gpu_user_annotation"
            and str(event.get("name", "")).startswith("execute_context_0")
        ),
        key=lambda event: float(event.get("ts", 0.0)),
    )
    if len(gpu) < 2:
        raise ValueError("need at least two execute_context_0 GPU annotations")

    host_annotations = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("execute_context_0")
    ]
    if not host_annotations:
        raise ValueError("could not find host execute_context_0 annotation")
    host_tid = int(host_annotations[0]["tid"])
    gaps = [
        (
            float(previous["ts"]) + float(previous.get("dur", 0.0)),
            float(current["ts"]),
        )
        for previous, current in zip(gpu, gpu[1:])
        if float(current["ts"])
        > float(previous["ts"]) + float(previous.get("dur", 0.0))
    ]
    gap_us = [(right - left) for left, right in gaps]

    kernels: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        name = str(event.get("name", "<unnamed>"))
        row = kernels[name]
        row[0] += 1
        row[1] += float(event.get("dur", 0.0))
    kernel_top = [
        {"name": name, "count": int(values[0]), "total_us": values[1]}
        for name, values in sorted(
            kernels.items(), key=lambda item: item[1][1], reverse=True
        )[:limit]
    ]

    return {
        "trace": str(path),
        "steady_gpu_annotations": len(gpu),
        "host_tid": host_tid,
        "gpu_execute_us_mean": statistics.mean(
            float(event.get("dur", 0.0)) for event in gpu
        ),
        "gpu_execute_us_median": statistics.median(
            float(event.get("dur", 0.0)) for event in gpu
        ),
        "cpu_gap_us_mean": statistics.mean(gap_us),
        "cpu_gap_us_median": statistics.median(gap_us),
        "cpu_gap_us_min": min(gap_us),
        "cpu_gap_us_max": max(gap_us),
        "cpu_gap_python_top": _aggregate(
            events, gaps, "python_function", host_tid, limit
        ),
        "cpu_gap_torch_op_top": _aggregate(events, gaps, "cpu_op", host_tid, limit),
        "cpu_gap_cuda_runtime_top": _aggregate(
            events, gaps, "cuda_runtime", host_tid, limit
        ),
        "kernel_top": kernel_top,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.trace, limit=args.limit)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
