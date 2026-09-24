"""Inline the common no-repetition stop checks in the scheduler output loop.

The stock scheduler calls ``check_stop`` once per accepted token.  This probe
keeps the full function for requests using repetition detection or pooling and
only inlines the equivalent EOS, stop-token, and length checks for ordinary
generation requests.  Token append and block-hash behavior remain delegated to
``Request.append_output_token_ids``.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any


_ENABLED = os.environ.get("AGENTINFER_STOP_FASTPATH", "0") == "1"
_LOG_PATH = os.environ.get("AGENTINFER_STOP_FASTPATH_LOG", "")
_STATS = {"calls": 0, "fast_calls": 0, "fallback_calls": 0, "stops": 0}


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
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    _original = Scheduler._update_request_with_output

    def _fast_update_request_with_output(
        self: Any,
        request: Any,
        new_token_ids: list[int],
        is_stale: bool = False,
    ) -> tuple[list[int], bool]:
        _STATS["calls"] += 1
        params = request.sampling_params
        if (
            is_stale
            or request.pooling_params is not None
            or params is None
            or params.repetition_detection is not None
        ):
            _STATS["fallback_calls"] += 1
            return _original(self, request, new_token_ids, is_stale=is_stale)

        _STATS["fast_calls"] += 1
        eos_token_id = params.eos_token_id
        stop_token_ids = params.stop_token_ids
        max_model_len = self.max_model_len
        max_tokens = request.max_tokens
        output_token_ids = request._output_token_ids
        all_token_ids = request._all_token_ids
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)
            hit_eos = eos_token_id is not None and output_token_id == eos_token_id
            hit_stop = stop_token_ids is not None and output_token_id in stop_token_ids
            hit_length = len(all_token_ids) >= max_model_len or len(output_token_ids) >= max_tokens
            if hit_eos or hit_stop or hit_length:
                request.status = (
                    RequestStatus.FINISHED_STOPPED
                    if hit_eos or hit_stop
                    else RequestStatus.FINISHED_LENGTH_CAPPED
                )
                if hit_stop and not hit_eos:
                    request.stop_reason = output_token_id
                del new_token_ids[num_new:]
                _STATS["stops"] += 1
                stopped = True
                break
        if _STATS["calls"] % 64 == 0:
            _write_stats()
        return new_token_ids, stopped

    Scheduler._update_request_with_output = _fast_update_request_with_output
    atexit.register(_write_stats)
