#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Inject the experimental FlashInfer MTP metadata update in spawned workers."""

from __future__ import annotations

import os


_DEBUG = os.environ.get("AGENTINFER_FUSED_META_DEBUG") == "1"
_REPORTED = False

def _enable_sm100_fused_metadata() -> None:
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferDecodeKernel,
        FlashInferMetadataBuilder,
    )

    if getattr(FlashInferMetadataBuilder, "_agentinfer_fused_patch", False):
        return

    original_init = FlashInferMetadataBuilder.__init__

    def patched_init(self, *args, **kwargs):
        global _REPORTED
        original_init(self, *args, **kwargs)
        if (
            self.flashinfer_trtllm_api_decode_kernel
            == FlashInferDecodeKernel.TRTLLM_GEN
            and not self.use_dcp
        ):
            self.supports_draft_decode_metadata_update = True
            if _DEBUG and not _REPORTED:
                print(
                    "[agentinfer:fused-flashinfer] enabled "
                    "TRTLLM_GEN draft metadata update",
                    flush=True,
                )
                _REPORTED = True

    def update_draft_decode_metadata(self, _metadata) -> None:
        return None

    FlashInferMetadataBuilder.__init__ = patched_init
    FlashInferMetadataBuilder.update_draft_decode_metadata = (
        update_draft_decode_metadata
    )
    FlashInferMetadataBuilder._agentinfer_fused_patch = True


try:
    if _DEBUG:
        print("[agentinfer:fused-flashinfer] sitecustomize loaded", flush=True)
    _enable_sm100_fused_metadata()
except Exception:
    # Python imports sitecustomize before the application has configured its
    # logging.  Leave normal vLLM startup errors untouched; the server log will
    # show the original exception if the experiment cannot be injected.
    pass
