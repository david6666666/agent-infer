"""Split the hybrid-attention CPU preparation path into measured stages.

The outer GPU runner profile hides the cost of the model-state metadata pass.
This probe installs instance wrappers around the metadata builders used by the
Qwen hybrid model and records their wall-clock time by concrete builder type.
It is measurement-only and does not alter tensors or scheduling decisions.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path
from typing import Any


_ENABLED = os.environ.get("AGENTINFER_CPU_STAGE_PROFILE", "0") == "1"
_LOG_PATH = os.environ.get("AGENTINFER_CPU_STAGE_PROFILE_LOG", "")
_EVENTS = 0
_STATS: dict[str, dict[str, float | int]] = {}


def _record(name: str, elapsed_ns: int) -> None:
    global _EVENTS
    elapsed_us = elapsed_ns / 1000.0
    item = _STATS.setdefault(name, {"count": 0, "total_us": 0.0, "max_us": 0.0})
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
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"pid": os.getpid(), "enabled": _ENABLED, "events": _EVENTS, "stats": _STATS},
                sort_keys=True,
            )
            + "\n"
        )


def _wrap_instance_method(instance: Any, method_name: str, stat_name: str) -> None:
    original = getattr(instance, method_name, None)
    if original is None or getattr(original, "_agentinfer_cpu_stage_wrapped", False):
        return

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return original(*args, **kwargs)
        finally:
            _record(stat_name, time.perf_counter_ns() - started)

    wrapped._agentinfer_cpu_stage_wrapped = True  # type: ignore[attr-defined]
    setattr(instance, method_name, wrapped)


def _wrap_model_state_prepare_attn(instance: Any) -> None:
    original = instance.prepare_attn
    if getattr(original, "_agentinfer_cpu_stage_wrapped", False):
        return

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # ``original`` is already bound, so args[4] is attn_groups.
        attn_groups = kwargs.get("attn_groups")
        if attn_groups is None and len(args) >= 5:
            attn_groups = args[4]
        if attn_groups is not None:
            for groups in attn_groups:
                for group in groups:
                    builder = group.get_metadata_builder(0)
                    name = type(builder).__name__
                    _wrap_instance_method(
                        builder,
                        "build",
                        f"builder.{name}.build",
                    )
                    _wrap_instance_method(
                        builder,
                        "build_for_cudagraph_capture",
                        f"builder.{name}.build_for_cudagraph_capture",
                    )
        started = time.perf_counter_ns()
        try:
            return original(*args, **kwargs)
        finally:
            _record(
                f"model_state.{type(instance).__name__}.prepare_attn",
                time.perf_counter_ns() - started,
            )

    wrapped._agentinfer_cpu_stage_wrapped = True  # type: ignore[attr-defined]
    # Functions stored directly on an instance are not descriptor-bound.  The
    # closure calls the already-bound original method above.
    instance.prepare_attn = wrapped


if _ENABLED:
    # Patch the concrete class directly.  GPUModelRunner can be constructed in
    # a child process after sitecustomize has run, so an __init__ hook alone is
    # not a reliable attachment point for the measurement.
    from vllm.v1.worker.gpu.model_states import mamba_hybrid

    _original_prepare_attn = mamba_hybrid.MambaHybridModelState.prepare_attn

    def _prepare_attn_with_stage_probe(self: Any, *args: Any, **kwargs: Any) -> Any:
        attn_groups = kwargs.get("attn_groups")
        if attn_groups is None and len(args) >= 5:
            attn_groups = args[4]
        if attn_groups is not None:
            for groups in attn_groups:
                for group in groups:
                    builder = group.get_metadata_builder(0)
                    name = type(builder).__name__
                    _wrap_instance_method(builder, "build", f"builder.{name}.build")
                    _wrap_instance_method(
                        builder,
                        "build_for_cudagraph_capture",
                        f"builder.{name}.build_for_cudagraph_capture",
                    )
        started = time.perf_counter_ns()
        try:
            return _original_prepare_attn(self, *args, **kwargs)
        finally:
            _record(
                f"model_state.{type(self).__name__}.prepare_attn",
                time.perf_counter_ns() - started,
            )

    _prepare_attn_with_stage_probe._agentinfer_cpu_stage_wrapped = True  # type: ignore[attr-defined]
    mamba_hybrid.MambaHybridModelState.prepare_attn = _prepare_attn_with_stage_probe

    for _method_name in ("preprocess_state", "prepare_inputs"):
        _original_method = getattr(mamba_hybrid.MambaHybridModelState, _method_name)

        def _method_with_stage_probe(
            self: Any,
            *args: Any,
            _original: Any = _original_method,
            _name: str = _method_name,
            **kwargs: Any,
        ) -> Any:
            started = time.perf_counter_ns()
            try:
                return _original(self, *args, **kwargs)
            finally:
                _record(
                    f"model_state.{type(self).__name__}.{_name}",
                    time.perf_counter_ns() - started,
                )

        setattr(
            mamba_hybrid.MambaHybridModelState,
            _method_name,
            _method_with_stage_probe,
        )
    atexit.register(_write_stats)
