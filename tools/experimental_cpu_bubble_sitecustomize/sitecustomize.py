"""Record the real per-iteration CPU bubble stages in the V2 runner.

This probe is intentionally measurement-only.  It wraps the scheduler and GPU
runner methods after import, aggregates wall-clock time by process, and writes
periodic snapshots so a harness that terminates workers still leaves evidence.
The wrappers do not change scheduling decisions or tensor contents.

Enable with ``AGENTINFER_CPU_BUBBLE_PROFILE=1`` and set
``AGENTINFER_CPU_BUBBLE_PROFILE_LOG`` to an append-only JSONL path.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


_ENABLED = os.environ.get("AGENTINFER_CPU_BUBBLE_PROFILE", "0") == "1"
_LOG_PATH = os.environ.get("AGENTINFER_CPU_BUBBLE_PROFILE_LOG", "")
_EVENTS = 0
_STATS: dict[str, dict[str, float | int]] = {}


def _record(name: str, elapsed_ns: int) -> None:
    global _EVENTS
    item = _STATS.setdefault(name, {"count": 0, "total_us": 0.0, "max_us": 0.0})
    elapsed_us = elapsed_ns / 1000.0
    item["count"] = int(item["count"]) + 1
    item["total_us"] = float(item["total_us"]) + elapsed_us
    item["max_us"] = max(float(item["max_us"]), elapsed_us)
    _EVENTS += 1
    if _EVENTS % 64 == 0:
        _write_stats()


def _write_stats() -> None:
    if not _LOG_PATH:
        return
    path = Path(_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "enabled": _ENABLED,
        "events": _EVENTS,
        "stats": _STATS,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _wrap_timed(cls: Any, method_name: str, stat_name: str) -> None:
    original = getattr(cls, method_name)
    if getattr(original, "_agentinfer_cpu_bubble_wrapped", False):
        return

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return original(self, *args, **kwargs)
        finally:
            _record(stat_name, time.perf_counter_ns() - started)

    wrapped._agentinfer_cpu_bubble_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapped)


def _wrap_scheduler_schedule(cls: Any) -> None:
    original = cls.schedule
    if getattr(original, "_agentinfer_cpu_bubble_wrapped", False):
        return

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return original(self, *args, **kwargs)
        finally:
            _record("scheduler.schedule", time.perf_counter_ns() - started)

    wrapped._agentinfer_cpu_bubble_wrapped = True  # type: ignore[attr-defined]
    cls.schedule = wrapped


def _wrap_scheduler_update(cls: Any) -> None:
    original = cls.update_from_output
    if getattr(original, "_agentinfer_cpu_bubble_wrapped", False):
        return

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return original(self, *args, **kwargs)
        finally:
            _record("scheduler.update_from_output", time.perf_counter_ns() - started)

    wrapped._agentinfer_cpu_bubble_wrapped = True  # type: ignore[attr-defined]
    cls.update_from_output = wrapped


def _wrap_batch_state(cls: Any) -> None:
    original = cls.gather_batch_req_state
    if getattr(original, "_agentinfer_cpu_bubble_wrapped", False):
        return

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return original(self, *args, **kwargs)
        finally:
            _record("gpu.gather_batch_req_state", time.perf_counter_ns() - started)

    wrapped._agentinfer_cpu_bubble_wrapped = True  # type: ignore[attr-defined]
    cls.gather_batch_req_state = wrapped


if _ENABLED:
    from vllm.v1.core.sched import scheduler as scheduler_module
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.worker.gpu import model_runner

    _wrap_scheduler_schedule(scheduler_module.Scheduler)
    _wrap_scheduler_update(scheduler_module.Scheduler)
    _wrap_timed(KVCacheManager, "allocate_slots", "scheduler.allocate_slots")

    _wrap_timed(
        model_runner.GPUModelRunner,
        "add_requests",
        "gpu.add_requests",
    )
    _wrap_batch_state(model_runner.GPUModelRunner)
    _wrap_timed(
        model_runner.GPUModelRunner,
        "prepare_inputs",
        "gpu.prepare_inputs",
    )
    _wrap_timed(
        model_runner.GPUModelRunner,
        "prepare_attn",
        "gpu.prepare_attn",
    )
    atexit.register(_write_stats)
