#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Probe the disabled FlashInfer SM103 NVFP4 dense GEMM path.

The installed FlashInfer wheel contains the SM103 3xFP4 kernel, but its
runner intentionally disables that kernel on SM103.  This sitecustomize keeps
the experiment outside the installed packages and routes one exact decode
shape through a manually selected SM103 tactic.  All other shapes retain the
normal SM100 heuristic.  It is therefore suitable for an A/B replay, not a
general runtime patch.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
from pathlib import Path


_lock = threading.Lock()
_seen: set[tuple[object, ...]] = set()


def _record(record: dict[str, object]) -> None:
    path_value = os.environ.get("AGENTINFER_SM103_FP4_LOG")
    if not path_value:
        return
    key = tuple(sorted(record.items()))
    with _lock:
        if key in _seen:
            return
        _seen.add(key)
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    record["pid"] = os.getpid()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _target_shape() -> tuple[int, int, int]:
    value = os.environ.get("AGENTINFER_SM103_FP4_TARGET", "40,5120,6144")
    try:
        m, n, real_k = (int(part) for part in value.split(","))
    except ValueError as exc:
        raise RuntimeError(
            "AGENTINFER_SM103_FP4_TARGET must be M,N,K, "
            f"got {value!r}"
        ) from exc
    return m, n, real_k


def _patch() -> None:
    import flashinfer.gemm.gemm_base as gemm_base

    if getattr(gemm_base, "_agentinfer_sm103_patch", False):
        return

    original_runner_factory = gemm_base._cute_dsl_gemm_fp4_runner
    source = inspect.getsource(original_runner_factory)
    disabled = "Sm103Kernel = None\n    if sm_version == 107:"
    enabled = "Sm103Kernel = None\n    if sm_version in [103, 107]:"
    if disabled not in source:
        raise RuntimeError("FlashInfer runner source changed; refusing to patch")
    source = source.replace(disabled, enabled, 1)

    # Execute the wheel's own runner source with the module globals.  This
    # preserves all layout checks and compilation/cache behavior; only the
    # SM103 import guard and selector below are changed.
    namespace = gemm_base.__dict__
    exec(compile(source, inspect.getsourcefile(original_runner_factory) or "<flashinfer>", "exec"), namespace)
    patched_runner_factory = namespace["_cute_dsl_gemm_fp4_runner"]

    original_selector = gemm_base._select_sm100_mm_fp4_cute_dsl_tactic
    target = _target_shape()
    sm103_tactic = ((256, 128), (2, 1), True, False, "sm103", True)

    def select_tactic(m, n, real_k, sm_count, sf_vec_size):
        if (int(m), int(n), int(real_k)) == target:
            _record(
                {
                    "kind": "sm103-selection",
                    "m": int(m),
                    "n": int(n),
                    "real_k": int(real_k),
                    "sm_count": int(sm_count),
                    "sf_vec_size": int(sf_vec_size),
                    "tactic": repr(sm103_tactic),
                }
            )
            return sm103_tactic
        return original_selector(m, n, real_k, sm_count, sf_vec_size)

    gemm_base._select_sm100_mm_fp4_cute_dsl_tactic = select_tactic
    gemm_base._cute_dsl_gemm_fp4_runner = patched_runner_factory
    gemm_base._agentinfer_sm103_patch = True
    _record({"kind": "patch-enabled", "target": target, "tactic": repr(sm103_tactic)})


try:
    _patch()
except Exception as exc:  # noqa: BLE001 - startup probe must remain diagnosable
    _record({"kind": "patch-error", "error": repr(exc)})
    raise
