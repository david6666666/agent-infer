#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Opt-in reuse of the invariant Mamba aligned-index offsets tensor.

``mamba_get_block_table_tensor`` receives changing sequence lengths but the
offset vector depends only on device and the Mamba speculative-block count.
The production helper rebuilds that vector for every metadata build.  This
probe keeps only that immutable vector; the per-batch arithmetic and gather
remain unchanged.
"""

from __future__ import annotations

import atexit
import importlib
import json
import os
from pathlib import Path
from typing import Any

import torch


_ENABLED = os.environ.get("AGENTINFER_MAMBA_OFFSETS_CACHE", "0") == "1"
_LOG_PATH = os.environ.get("AGENTINFER_MAMBA_OFFSETS_CACHE_LOG", "")
_OFFSETS: dict[tuple[str, int], torch.Tensor] = {}
_STATS = {"calls": 0, "offset_hits": 0, "offset_allocations": 0, "passthrough": 0}


def _write_stats() -> None:
    if not _LOG_PATH:
        return
    path = Path(_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "enabled": _ENABLED, **_STATS}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_stats_periodically() -> None:
    if _STATS["calls"] and _STATS["calls"] % 64 == 0:
        _write_stats()


def _cached_mamba_get_block_table_tensor(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_spec: Any,
    mamba_cache_mode: str,
) -> torch.Tensor:
    _STATS["calls"] += 1
    _write_stats_periodically()
    if mamba_cache_mode in ("all", "none"):
        _STATS["passthrough"] += 1
        return block_table

    count = 1 + int(kv_cache_spec.num_speculative_blocks)
    key = (str(block_table.device), count)
    offsets = _OFFSETS.get(key)
    if offsets is None:
        offsets = torch.arange(count, device=block_table.device, dtype=torch.int32)
        _OFFSETS[key] = offsets
        _STATS["offset_allocations"] += 1
    else:
        _STATS["offset_hits"] += 1

    # Keep this expression identical to vLLM's helper.  In particular, the
    # gather still receives int64 indices and start_indices is clamped in place.
    start_indices = (seq_lens - 1) // kv_cache_spec.block_size
    start_indices.clamp_(min=0)
    indices_to_gather = (start_indices.unsqueeze(1) + offsets).to(torch.int64)
    return torch.gather(block_table, 1, indices_to_gather)


if _ENABLED:
    from vllm.v1.attention.backends import utils as attention_utils

    for module_name in (
        "vllm.v1.attention.backends.linear_attn",
        "vllm.v1.attention.backends.gdn_attn",
        "vllm.v1.attention.backends.short_conv_attn",
        "vllm.v1.attention.backends.mamba_attn",
    ):
        module = importlib.import_module(module_name)
        if hasattr(module, "mamba_get_block_table_tensor"):
            module.mamba_get_block_table_tensor = (
                _cached_mamba_get_block_table_tensor
            )
    attention_utils.mamba_get_block_table_tensor = _cached_mamba_get_block_table_tensor
    atexit.register(_write_stats)
