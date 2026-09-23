#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Opt-in FP8 GDN qkvz projection experiment.

The NVFP4 checkpoint intentionally keeps the GDN qkvz and ba projections in
BF16.  The decode profile shows the qkvz GEMM as one of the two largest
remaining operator groups.  This probe keeps the loaded BF16 weights for a
safe fallback, creates a per-tensor FP8 copy once after loading, and uses
Torch's fused scaled FP8 GEMM for small decode batches.

It is an experiment rather than a model conversion: prefill and unsupported
shapes use the original BF16 path, and the caller can disable the patch with
AGENTINFER_FP8_QKVZ_ENABLE=0.
"""

from __future__ import annotations

import json
import inspect
import logging
import os
from pathlib import Path
import textwrap
import types
from typing import Any

import torch


_LOG = logging.getLogger("agentinfer.experimental_gdn_fp8_qkvz")
_MAX_TOKENS = int(os.environ.get("AGENTINFER_FP8_QKVZ_MAX_TOKENS", "64"))
_ENABLED = os.environ.get("AGENTINFER_FP8_QKVZ_ENABLE", "1") != "0"
_LAYERS: dict[str, Any] = {}
_ACTIVATION_BUFFERS: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _record(event: str, **fields: Any) -> None:
    path = os.environ.get("AGENTINFER_FP8_QKVZ_LOG")
    if not path:
        return
    row = {"event": event, **fields}
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        _LOG.exception("failed to write FP8 qkvz experiment record")


def _is_qkvz_layer(layer: Any) -> bool:
    return str(getattr(layer, "prefix", "")).endswith(".in_proj_qkvz")


def _prepare_weight(layer: Any) -> None:
    if not _ENABLED or not _is_qkvz_layer(layer):
        return
    if hasattr(layer, "_agentinfer_fp8_qkvz_weight_t"):
        return

    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return
    if (
        weight.device.type != "cuda"
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or not weight.is_contiguous()
    ):
        _record(
            "skip_weight",
            prefix=getattr(layer, "prefix", ""),
            dtype=str(getattr(weight, "dtype", None)),
            shape=list(weight.shape) if isinstance(weight, torch.Tensor) else None,
        )
        return

    # scaled_fp8_quant uses this scale as the BF16-to-FP8 dequantization
    # factor.  Keep it per-tensor so torch._scaled_mm stays on its fastest
    # tensor-wise path for M=40 decode batches.
    scale = weight.float().abs().amax() / 448.0
    scale = torch.clamp(scale, min=torch.finfo(torch.float32).tiny)
    qweight, qscale = _quantize_static(weight, scale)

    # Store B as a contiguous [K, N] matrix.  The transposed view is valid but
    # costs a small amount of throughput on this shape.
    layer._agentinfer_fp8_qkvz_weight_t = qweight.t().contiguous()
    layer._agentinfer_fp8_qkvz_scale = qscale.reshape(1).contiguous()
    layer._agentinfer_fp8_qkvz_max_tokens = _MAX_TOKENS
    _LAYERS[getattr(layer, "prefix", "")] = layer
    _record(
        "weight_ready",
        prefix=getattr(layer, "prefix", ""),
        method=type(getattr(layer, "quant_method", None)).__name__,
        shape=list(weight.shape),
        qweight_shape=list(layer._agentinfer_fp8_qkvz_weight_t.shape),
        scale=float(layer._agentinfer_fp8_qkvz_scale.item()),
        max_tokens=_MAX_TOKENS,
    )
    _LOG.info(
        "Enabled experimental FP8 qkvz for %s, weight=%s, max_tokens=%d",
        getattr(layer, "prefix", ""),
        tuple(weight.shape),
        _MAX_TOKENS,
    )


def _patch_method_instance(method: Any) -> None:
    """Intercept a quant method even when a subclass overrides ``apply``."""

    if getattr(method, "_agentinfer_fp8_qkvz_instance_patched", False):
        return
    original_apply = method.apply

    def apply(
        method_self: Any,
        layer: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = fp8_qkvz_apply(layer, x, bias)
        if result is not None:
            return result
        return original_apply(layer, x, bias)

    method.apply = types.MethodType(apply, method)
    method._agentinfer_fp8_qkvz_instance_patched = True


def _quantize_static(
    input_tensor: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call vLLM's compiled quantizer without importing it at module load."""

    from vllm import _custom_ops as ops

    return ops.scaled_fp8_quant(input_tensor, scale=scale)


def _activation_buffers(
    x: torch.Tensor, max_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (x.device.index or 0, max_tokens, x.shape[-1])
    cached = _ACTIVATION_BUFFERS.get(key)
    if cached is not None:
        return cached
    qx = torch.empty(
        (max_tokens, x.shape[-1]), device=x.device, dtype=torch.float8_e4m3fn
    )
    scale = torch.empty((1,), device=x.device, dtype=torch.float32)
    _ACTIVATION_BUFFERS[key] = (qx, scale)
    return qx, scale


def _quantize_activation(
    x: torch.Tensor, max_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    qx, scale = _activation_buffers(x, max_tokens)
    if qx.shape[0] != x.shape[0]:
        qx = qx[: x.shape[0]]
    # Call the in-place custom op directly.  The public helper also allocates
    # and returns a scale tensor for every call; here both output addresses are
    # already stable and are deliberately reused across GDN layers.
    torch.ops._C.dynamic_scaled_fp8_quant(qx, x, scale)
    return qx, scale


def fp8_qkvz_apply(
    layer: Any,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return the FP8 result, or None when the BF16 caller must be used."""

    qweight_t = getattr(layer, "_agentinfer_fp8_qkvz_weight_t", None)
    if (
        not _ENABLED
        or qweight_t is None
        or bias is not None
        or x.dtype != torch.bfloat16
        or x.device.type != "cuda"
        or x.ndim != 2
        or x.shape[0] > getattr(layer, "_agentinfer_fp8_qkvz_max_tokens", 0)
        or not x.is_contiguous()
    ):
        return None

    qx, x_scale = _quantize_activation(
        x, getattr(layer, "_agentinfer_fp8_qkvz_max_tokens", _MAX_TOKENS)
    )
    result = torch._scaled_mm(
        qx,
        qweight_t,
        scale_a=x_scale,
        scale_b=layer._agentinfer_fp8_qkvz_scale,
        out_dtype=x.dtype,
    )
    if isinstance(result, tuple):
        result = result[0]
    return result


def _project_qkvz(attention: Any, x: torch.Tensor) -> torch.Tensor | None:
    # The Qwen GDN projection is materialized through a merged linear layer
    # whose quant method does not always receive
    # ``process_weights_after_loading``.  Prepare lazily on the first real
    # forward so the experiment cannot silently report a patch hit while
    # continuing to execute the BF16 path.
    layer = attention.in_proj_qkvz
    if getattr(layer, "_agentinfer_fp8_qkvz_weight_t", None) is None:
        _prepare_weight(layer)
    return fp8_qkvz_apply(layer, x)


def _patch_attention_forward() -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as module

    cls = module.QwenGatedDeltaNetAttention
    if getattr(cls, "_agentinfer_fp8_qkvz_forward_patched", False):
        return
    source = inspect.getsource(cls.forward_cuda)
    old = "mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)"
    new = (
        "mixed_qkvz = _agentinfer_fp8_qkvz_project(self, hidden_states)\n"
        "        if mixed_qkvz is None:\n"
        "            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)"
    )
    if old not in source:
        raise RuntimeError("Qwen GDN forward source changed; FP8 qkvz probe is stale")
    setattr(module, "_agentinfer_fp8_qkvz_project", _project_qkvz)
    namespace = dict(vars(module))
    namespace["_agentinfer_fp8_qkvz_project"] = _project_qkvz
    exec(
        compile(
            textwrap.dedent(source.replace(old, new)),
            "<agentinfer_experimental_gdn_fp8_qkvz>",
            "exec",
        ),
        namespace,
    )
    cls.forward_cuda = namespace["forward_cuda"]
    cls._agentinfer_fp8_qkvz_forward_patched = True
    _record("attention_forward_patched")


def patch() -> None:
    if not _ENABLED:
        _LOG.info("Experimental FP8 qkvz patch disabled")
        return

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if getattr(UnquantizedLinearMethod, "_agentinfer_fp8_qkvz_patched", False):
        return

    original_process = UnquantizedLinearMethod.process_weights_after_loading
    original_apply = UnquantizedLinearMethod.apply

    def process_weights_after_loading(self: Any, layer: Any) -> None:
        original_process(self, layer)
        _prepare_weight(layer)
        if _is_qkvz_layer(layer):
            _patch_method_instance(self)

    def apply(
        self: Any,
        layer: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = fp8_qkvz_apply(layer, x, bias)
        if result is not None:
            return result
        return original_apply(self, layer, x, bias)

    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._agentinfer_fp8_qkvz_patched = True
    _patch_attention_forward()
    _record("patch_enabled", max_tokens=_MAX_TOKENS)


patch()
