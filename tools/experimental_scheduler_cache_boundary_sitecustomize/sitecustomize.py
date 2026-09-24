"""Avoid redundant prefix-cache bookkeeping between scheduler block boundaries.

``AsyncScheduler._update_request_with_output`` calls ``cache_blocks`` for every
output delivery.  The cache manager itself discovers that no new full block is
available, but it still walks every KV manager.  This probe keeps a per-request
finalized-block watermark and calls the manager only after the watermark grows.

The watermark is reset when async preemption moves a request backwards.  The
actual cache operation and all token/stop handling remain the vLLM path.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any


_ENABLED = os.environ.get("AGENTINFER_CACHE_BOUNDARY_GUARD", "0") == "1"
_LOG_PATH = os.environ.get("AGENTINFER_CACHE_BOUNDARY_GUARD_LOG", "")
_STATS = {"calls": 0, "cache_calls": 0, "skips": 0, "resets": 0}


def _write_stats() -> None:
    if not _LOG_PATH:
        return
    path = Path(_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"pid": os.getpid(), "enabled": _ENABLED, **_STATS}, sort_keys=True)
            + "\n"
        )


if _ENABLED:
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    def _update_request_with_boundary_guard(
        self: Any,
        request: Any,
        new_token_ids: list[int],
        is_stale: bool = False,
    ) -> tuple[list[int], bool]:
        _STATS["calls"] += 1
        status_before_update = request.status
        new_token_ids, stopped = Scheduler._update_request_with_output(
            self, request, new_token_ids, is_stale=is_stale
        )

        # Keep the original AsyncScheduler placeholder accounting.
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        if status_before_update == RequestStatus.RUNNING:
            finalized_tokens = (
                request.num_computed_tokens - request.num_output_placeholders
            )
            block_size = int(
                self.kv_cache_manager.coordinator.scheduler_block_size
            )
            previous_tokens = getattr(request, "_agentinfer_cache_progress", -1)
            if finalized_tokens < previous_tokens:
                previous_tokens = -1
                request._agentinfer_cache_progress = -1
                _STATS["resets"] += 1
            finalized_blocks = finalized_tokens // block_size
            previous_blocks = previous_tokens // block_size
            if finalized_blocks > previous_blocks:
                self.kv_cache_manager.cache_blocks(request, finalized_tokens)
                request._agentinfer_cache_progress = finalized_tokens
                _STATS["cache_calls"] += 1
            else:
                _STATS["skips"] += 1
        if _STATS["calls"] % 64 == 0:
            _write_stats()
        return new_token_ids, stopped

    AsyncScheduler._update_request_with_output = _update_request_with_boundary_guard
    atexit.register(_write_stats)
