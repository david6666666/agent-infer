#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Run vLLM with the SM100 FlashInfer fused MTP metadata prototype.

The installed vLLM build uses FlashInfer's ``trtllm-gen`` decode API for the
NVFP4/FP8 path.  That API reads sequence lengths and block tables from the
persistent GPU buffers, so rebuilding the Python metadata object between MTP
draft steps is unnecessary for this configuration.  This wrapper enables the
prototype only for that API and leaves native FlashInfer paths unchanged.

It is an experiment harness, not a general vLLM compatibility layer.  The
benchmark must verify replay validity and GSM8K accuracy before this path can
be promoted into a vLLM patch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _enable_sm100_fused_metadata() -> None:
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferDecodeKernel,
        FlashInferMetadataBuilder,
    )

    if getattr(FlashInferMetadataBuilder, "_agentinfer_fused_patch", False):
        return

    original_init = FlashInferMetadataBuilder.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if (
            self.flashinfer_trtllm_api_decode_kernel
            == FlashInferDecodeKernel.TRTLLM_GEN
            and not self.use_dcp
        ):
            self.supports_draft_decode_metadata_update = True

    def update_draft_decode_metadata(self, _metadata) -> None:
        # The trtllm-gen path reads the mutable seq_lens and block-table buffers
        # by reference.  The speculator updates those buffers before calling
        # this hook; no wrapper re-plan is needed.
        return None

    FlashInferMetadataBuilder.__init__ = patched_init
    FlashInferMetadataBuilder.update_draft_decode_metadata = (
        update_draft_decode_metadata
    )
    FlashInferMetadataBuilder._agentinfer_fused_patch = True


if __name__ == "__main__":
    site_dir = Path(__file__).with_name("experimental_fused_flashinfer_sitecustomize")
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(site_dir), existing_pythonpath) if value
    )
    _enable_sm100_fused_metadata()
    from vllm.entrypoints.cli.main import main

    raise SystemExit(main())
