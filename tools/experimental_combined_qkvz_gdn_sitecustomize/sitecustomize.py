#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Load the two profile-backed experimental probes in one server process."""

from pathlib import Path
import importlib.util
import sys


_tools = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


_load(
    "agentinfer_experimental_gdn_prefill_stages",
    _tools / "experimental_gdn_prefill_stages.py",
)
_load(
    "agentinfer_experimental_gdn_fp8_qkvz",
    _tools / "experimental_gdn_fp8_qkvz.py",
)
