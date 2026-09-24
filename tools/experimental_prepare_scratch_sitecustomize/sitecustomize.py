"""Reuse the small NumPy scratch arrays in vLLM V2 ``prepare_inputs``.

The production profile attributes a recurring host gap to
``GPUModelRunner.prepare_inputs``.  This probe reuses only the arrays whose
contents are completely rewritten before the returned ``InputBatch`` is
consumed:

* ``cu_num_logits_np``;
* ``query_start_loc_np``;
* ``seq_lens_cpu_upper_bound_np``; and
* the common no-draft ``np.arange`` result.

The module's NumPy binding is replaced only while the original method runs;
all arithmetic, GPU copies, and downstream metadata builders stay on the
production path.  Enable with ``AGENTINFER_PREPARE_SCRATCH=1``.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any

import numpy as real_np


_ENABLED = os.environ.get("AGENTINFER_PREPARE_SCRATCH", "0") == "1"
_LOG_PATH = os.environ.get("AGENTINFER_PREPARE_SCRATCH_LOG", "")
_STATS = {
    "prepare_calls": 0,
    "empty_hits": 0,
    "zeros_hits": 0,
    "arange_hits": 0,
    "fallback_calls": 0,
}


def _write_stats() -> None:
    if not _LOG_PATH:
        return
    path = Path(_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "enabled": _ENABLED, **_STATS}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


class _ScratchNumpy:
    """Proxy only the allocation calls used by one prepare_inputs invocation."""

    def __init__(self, owner: Any, has_draft: bool):
        self._owner = owner
        self._has_draft = has_draft
        self._empty_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(real_np, name)

    @staticmethod
    def _scalar_shape(shape: Any) -> int | None:
        if isinstance(shape, int):
            return shape
        if isinstance(shape, tuple) and len(shape) == 1:
            return int(shape[0])
        return None

    def empty(self, shape: Any, *args: Any, **kwargs: Any) -> Any:
        n = self._scalar_shape(shape)
        dtype = kwargs.get("dtype", args[0] if args else None)
        if n is not None and dtype == real_np.int32:
            max_reqs = int(self._owner.max_num_reqs)
            # With draft tokens, cu_num_logits_np is the first empty array;
            # query_start_loc_np is always the following max-sized array.
            if self._has_draft and self._empty_calls == 0 and n <= max_reqs + 1:
                self._empty_calls += 1
                _STATS["empty_hits"] += 1
                return self._owner._agentinfer_cu_scratch[:n]
            if n == max_reqs + 1:
                self._empty_calls += 1
                _STATS["empty_hits"] += 1
                return self._owner._agentinfer_query_scratch
        _STATS["fallback_calls"] += 1
        self._empty_calls += 1
        return real_np.empty(shape, *args, **kwargs)

    def zeros(self, shape: Any, *args: Any, **kwargs: Any) -> Any:
        n = self._scalar_shape(shape)
        dtype = kwargs.get("dtype", args[0] if args else None)
        if n is not None and dtype == real_np.int32 and n <= int(
            self._owner.max_num_reqs
        ):
            out = self._owner._agentinfer_seq_scratch[:n]
            out.fill(0)
            _STATS["zeros_hits"] += 1
            return out
        _STATS["fallback_calls"] += 1
        return real_np.zeros(shape, *args, **kwargs)

    def arange(self, *args: Any, **kwargs: Any) -> Any:
        dtype = kwargs.get("dtype")
        if dtype == real_np.int32 and len(args) in (1, 2, 3):
            if len(args) == 1:
                start, stop, step = 0, int(args[0]), 1
            elif len(args) == 2:
                start, stop, step = int(args[0]), int(args[1]), 1
            else:
                start, stop, step = int(args[0]), int(args[1]), int(args[2])
            if (
                start == 0
                and step == 1
                and 0 <= stop <= int(self._owner.max_num_reqs) + 1
            ):
                _STATS["arange_hits"] += 1
                return self._owner._agentinfer_arange_scratch[:stop]
        _STATS["fallback_calls"] += 1
        return real_np.arange(*args, **kwargs)


if _ENABLED:
    from vllm.v1.worker.gpu import model_runner

    _original_prepare_inputs = model_runner.GPUModelRunner.prepare_inputs
    _original_np = model_runner.np

    def _ensure_scratch(self: Any) -> None:
        if hasattr(self, "_agentinfer_query_scratch"):
            return
        max_reqs = int(self.max_num_reqs)
        self._agentinfer_cu_scratch = _original_np.empty(max_reqs + 1, dtype=_original_np.int32)
        self._agentinfer_query_scratch = _original_np.empty(max_reqs + 1, dtype=_original_np.int32)
        self._agentinfer_seq_scratch = _original_np.empty(max_reqs, dtype=_original_np.int32)
        self._agentinfer_arange_scratch = _original_np.arange(
            max_reqs + 1, dtype=_original_np.int32
        )

    def _prepare_inputs_with_scratch(self: Any, *args: Any, **kwargs: Any) -> Any:
        _ensure_scratch(self)
        scheduler_output = args[0] if args else kwargs["scheduler_output"]
        has_draft = bool(scheduler_output.scheduled_spec_decode_tokens)
        proxy = _ScratchNumpy(self, has_draft)
        previous_np = model_runner.np
        model_runner.np = proxy
        _STATS["prepare_calls"] += 1
        if _STATS["prepare_calls"] % 64 == 0:
            _write_stats()
        try:
            return _original_prepare_inputs(self, *args, **kwargs)
        finally:
            model_runner.np = previous_np

    model_runner.GPUModelRunner.prepare_inputs = _prepare_inputs_with_scratch
    atexit.register(_write_stats)
