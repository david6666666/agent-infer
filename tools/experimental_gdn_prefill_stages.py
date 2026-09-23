#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Opt-in FlashInfer GDN prefill pipeline-stage probe.

The production profile shows the Blackwell chunked GDN prefill kernel using
about 227 KiB of shared memory per CTA.  This probe changes only the pipeline
stage counts after FlashInfer's own configuration has been constructed, so a
candidate can be measured without modifying the installed FlashInfer package.
It is intentionally disabled unless loaded through the companion
``sitecustomize.py`` directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from flashinfer.gdn_kernels.blackwell.gated_delta_net_chunked import (
    GatedDeltaNetChunkedKernel,
)


_ORIGINAL_SETUP = GatedDeltaNetChunkedKernel._setup_attributes
_CONFIG = os.environ.get("AGENTINFER_GDN_PREFILL_STAGES", "")


def _record(payload: dict[str, object]) -> None:
    path = os.environ.get("AGENTINFER_GDN_PREFILL_LOG")
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _patched_setup(self: GatedDeltaNetChunkedKernel) -> None:
    _ORIGINAL_SETUP(self)
    requested = {
        name: int(value)
        for item in _CONFIG.split(",")
        if item and "=" in item
        for name, value in [item.split("=", 1)]
    }
    for name in (
        "smem_q_stages",
        "smem_k_stages",
        "smem_v_stages",
        "smem_ainv_stages",
        "smem_qk_stages",
        "smem_o_stages",
        "smem_gate_stages",
        "smem_beta_stages",
    ):
        if name in requested:
            setattr(self, name, requested[name])
    _record(
        {
            "event": "gdn_prefill_attributes",
            "config": _CONFIG,
            "stages": {
                name: getattr(self, name)
                for name in (
                    "smem_q_stages",
                    "smem_k_stages",
                    "smem_v_stages",
                    "smem_ainv_stages",
                    "smem_qk_stages",
                    "smem_o_stages",
                    "smem_gate_stages",
                    "smem_beta_stages",
                )
            },
        }
    )


if _CONFIG:
    GatedDeltaNetChunkedKernel._setup_attributes = _patched_setup
