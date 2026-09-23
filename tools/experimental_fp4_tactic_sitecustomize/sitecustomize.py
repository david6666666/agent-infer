#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Record the CuTeDSL NVFP4 tactic selected for each GEMM shape.

This is deliberately a read-only probe.  It patches the installed FlashInfer
Python dispatch after import and writes one JSON record per unique shape and
tactic.  The benchmark still runs the unmodified kernels; the output is used
to choose a narrowly scoped operator experiment.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path


_records: set[tuple[object, ...]] = set()
_records_lock = threading.Lock()


def _as_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def _load_overrides() -> dict[tuple[int, int, int, int], object]:
    raw = os.environ.get("AGENTINFER_FP4_TACTIC_OVERRIDES")
    if not raw:
        return {}
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("AGENTINFER_FP4_TACTIC_OVERRIDES must be a JSON object")
    overrides: dict[tuple[int, int, int, int], object] = {}
    for key, tactic in decoded.items():
        parts = [int(item) for item in key.split(",")]
        if len(parts) == 3:
            m, n, real_k = parts
            sf_vec_size = 16
        elif len(parts) == 4:
            m, n, real_k, sf_vec_size = parts
        else:
            raise ValueError(
                "override shape keys must be M,N,K or M,N,K,sf_vec_size"
            )
        overrides[(m, n, real_k, sf_vec_size)] = _as_tuple(tactic)
    return overrides


def _load_pdl_overrides() -> set[tuple[int, int, int]]:
    raw = os.environ.get("AGENTINFER_FP4_PDL_OVERRIDES")
    if not raw:
        return set()
    decoded = json.loads(raw)
    if not isinstance(decoded, list):
        raise ValueError("AGENTINFER_FP4_PDL_OVERRIDES must be a JSON list")
    return {tuple(int(item) for item in key.split(",")) for key in decoded}


def _tactic_key(tactic: object) -> str:
    return repr(tactic)


def _shape_key(inputs: object) -> tuple[object, ...]:
    if not isinstance(inputs, (list, tuple)):
        return ()
    shapes = []
    for tensor in inputs[:4]:
        shape = getattr(tensor, "shape", None)
        shapes.append(tuple(int(value) for value in shape) if shape is not None else None)
    return tuple(shapes)


def _write_record(record: dict[str, object], key: tuple[object, ...]) -> None:
    path_value = os.environ.get("AGENTINFER_FP4_TACTIC_LOG")
    if not path_value:
        return
    with _records_lock:
        if key in _records:
            return
        _records.add(key)
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    record["pid"] = os.getpid()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _patch() -> None:
    try:
        import flashinfer.autotuner.autotuner as autotuner_module
        import flashinfer.gemm.gemm_base as gemm_base
    except Exception:
        return

    if getattr(autotuner_module.AutoTuner, "_agentinfer_fp4_probe", False):
        return

    original_choose_one = autotuner_module.AutoTuner.choose_one

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        result = original_choose_one(
            self, custom_op, runners, tuning_config, inputs, **kwargs
        )
        if custom_op == "fp4_gemm":
            runner, tactic = result
            shape = _shape_key(inputs)
            _write_record(
                {
                    "kind": "autotuner-result",
                    "custom_op": custom_op,
                    "inputs": shape,
                    "runner": type(runner).__name__,
                    "tactic": _tactic_key(tactic),
                },
                ("autotuner-result", shape, type(runner).__name__, _tactic_key(tactic)),
            )
        return result

    autotuner_module.AutoTuner.choose_one = choose_one
    original_selector = gemm_base._select_sm100_mm_fp4_cute_dsl_tactic
    overrides = _load_overrides()
    pdl_overrides = _load_pdl_overrides()

    if pdl_overrides:
        import flashinfer

        original_mm_fp4 = flashinfer.mm_fp4

        def mm_fp4_with_shape_pdl(*args, **kwargs):
            if args and len(args) > 1:
                a, b = args[0], args[1]
                shape = (int(a.shape[0]), int(b.shape[1]), int(a.shape[1]) * 2)
                if shape in pdl_overrides:
                    kwargs["enable_pdl"] = False
            return original_mm_fp4(*args, **kwargs)

        flashinfer.mm_fp4 = mm_fp4_with_shape_pdl

    def select_tactic(m, n, real_k, sm_count, sf_vec_size):
        override_key = (int(m), int(n), int(real_k), int(sf_vec_size))
        default_tactic = original_selector(m, n, real_k, sm_count, sf_vec_size)
        tactic = overrides.get(override_key, default_tactic)
        _write_record(
            {
                "kind": "heuristic-selector",
                "m": int(m),
                "n": int(n),
                "real_k": int(real_k),
                "sm_count": int(sm_count),
                "sf_vec_size": int(sf_vec_size),
                "overridden": tactic != default_tactic,
                "tactic": _tactic_key(tactic),
            },
            (
                "heuristic-selector",
                int(m),
                int(n),
                int(real_k),
                int(sm_count),
                int(sf_vec_size),
                _tactic_key(tactic),
            ),
        )
        return tactic

    # gemm_base imported this function into its module namespace, so patch the
    # local binding that the actual runner calls.
    gemm_base._select_sm100_mm_fp4_cute_dsl_tactic = select_tactic
    autotuner_module.AutoTuner._agentinfer_fp4_probe = True


_patch()
