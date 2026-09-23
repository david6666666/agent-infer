#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Opt-in ring buffers for small model-runner CPU-to-GPU input copies.

The normal helper allocates a GPU output tensor whenever ``out`` is omitted.
This probe reuses both the pinned host staging tensor and the GPU destination
for small one-dimensional index inputs.  A CUDA event guards reuse; if a ring
slot is still in flight, the original vLLM helper is used for that call.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


_DEPTH = int(os.environ.get("AGENTINFER_INPUT_COPY_POOL", "0"))
_MAX_ELEMENTS = int(os.environ.get("AGENTINFER_INPUT_COPY_POOL_MAX_ELEMENTS", "4096"))
_LOG_PATH = os.environ.get("AGENTINFER_INPUT_COPY_POOL_LOG", "")
_ORIGINAL: Any = None
_LOCK = threading.Lock()
_POOLS: dict[tuple[str, tuple[int, ...], torch.dtype], list[dict[str, Any]]] = {}
_NEXT: defaultdict[tuple[str, tuple[int, ...], torch.dtype], int] = defaultdict(int)
_STATS = {"calls": 0, "reused": 0, "allocated": 0, "in_flight_fallback": 0, "shape_fallback": 0}


def _write_stats() -> None:
    if not _LOG_PATH:
        return
    path = Path(_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        payload = {"pid": os.getpid(), "depth": _DEPTH, "max_elements": _MAX_ELEMENTS, **_STATS}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _supported(source: torch.Tensor, device: torch.device | None) -> bool:
    return (
        device is not None
        and device.type == "cuda"
        and source.device.type == "cpu"
        and source.ndim == 1
        and source.numel() <= _MAX_ELEMENTS
        and source.dtype in (torch.int32, torch.int64)
    )


def _copy_to_gpu(
    x: torch.Tensor | Any,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    assert _ORIGINAL is not None
    if _DEPTH <= 0 or out is not None:
        return _ORIGINAL(x, out=out, device=device)
    try:
        source = torch.from_numpy(x) if not isinstance(x, torch.Tensor) else x
    except (TypeError, ValueError):
        return _ORIGINAL(x, out=out, device=device)
    with _LOCK:
        _STATS["calls"] += 1
        should_flush = _STATS["calls"] % 64 == 0
    if should_flush:
        _write_stats()
    if not _supported(source, device):
        with _LOCK:
            _STATS["shape_fallback"] += 1
        return _ORIGINAL(x, out=out, device=device)

    key = (str(device), tuple(int(item) for item in source.shape), source.dtype)
    with _LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = []
            _POOLS[key] = pool
        index = _NEXT[key] % _DEPTH
        _NEXT[key] += 1
        if len(pool) < _DEPTH:
            pool.append(
                {
                    "host": torch.empty_like(source, pin_memory=True),
                    "device": torch.empty_like(source, device=device),
                    "event": torch.cuda.Event(blocking=False),
                }
            )
            _STATS["allocated"] += 1
        slot = pool[index]

    if slot["event"].query() is False:
        with _LOCK:
            _STATS["in_flight_fallback"] += 1
        return _ORIGINAL(x, out=out, device=device)
    slot["host"].copy_(source)
    slot["device"].copy_(slot["host"], non_blocking=True)
    slot["event"].record()
    with _LOCK:
        _STATS["reused"] += 1
    return slot["device"]


if _DEPTH > 0:
    from vllm.v1.worker.gpu import buffer_utils

    _ORIGINAL = buffer_utils.async_copy_to_gpu
    buffer_utils.async_copy_to_gpu = _copy_to_gpu
    try:
        from vllm.v1.worker.gpu import model_runner
    except ImportError:
        model_runner = None
    if model_runner is not None:
        model_runner.async_copy_to_gpu = _copy_to_gpu
    atexit.register(_write_stats)
