#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Load the experimental FP8 GDN qkvz projection patch at interpreter start."""

from pathlib import Path
import importlib.util
import sys


_probe_path = Path(__file__).resolve().parents[1] / "experimental_gdn_fp8_qkvz.py"
_module_name = "agentinfer_experimental_gdn_fp8_qkvz"
_spec = importlib.util.spec_from_file_location(_module_name, _probe_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load FP8 qkvz probe from {_probe_path}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_module_name] = _module
_spec.loader.exec_module(_module)
