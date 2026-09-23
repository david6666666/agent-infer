#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Collect a reproducible vLLM worker profile for the single-B300 NVFP4 path.

The workload is the same converted codex_swebenchpro trace used by the RSI
benchmark.  A warmup replay is completed before the vLLM profiler is started;
the profiled replay then records worker CPU/CUDA events, engine iteration logs,
and host/GPU utilization samples.  The script supports both the production
CUDA-graph path and an eager shadow run for operator attribution.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.rsi_single_nvfp4_sweep import (  # noqa: E402
    Profile,
    _server_command,
    _server_environment,
    run_replay,
    stop_process,
    wait_ready,
    write_config,
)

DEFAULT_MODEL = (
    "/home/zjy/.cache/huggingface/models--Inferact--Qwen3.8-27B-NVFP4/"
    "snapshots/6128240ebaf4eaa7bad2b3d1c72c37d677c5f462"
)
DEFAULT_TRACE = (
    "/home/zjy/code/hsliu/tmp/rsi-pr25-b300-20260922/codex-trace/"
    "codex_swebenchpro.json"
)
DEFAULT_CONVERTED = (
    "/home/zjy/code/hsliu/tmp/rsi-pr25-b300-20260922/results/"
    "replay-tp4-smoke/convert_result"
)
DEFAULT_OUTPUT = "/home/zjy/code/david/tmp/rsi-single-nvfp4-20260923/deep-profile"


def _post(url: str) -> None:
    with urlopen(Request(url, method="POST"), timeout=60) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"POST {url} returned HTTP {response.status}")


def _descendants(root_pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for line in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = line.read_text(encoding="utf-8").split()
            pid, parent = int(fields[0]), int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(parent, []).append(pid)
    result = [root_pid]
    queue = [root_pid]
    while queue:
        parent = queue.pop()
        for child in children.get(parent, []):
            result.append(child)
            queue.append(child)
    return sorted(set(result))


def _start_sampler(
    process: subprocess.Popen[bytes], output_dir: Path
) -> list[subprocess.Popen[bytes]]:
    pids = ",".join(str(pid) for pid in _descendants(process.pid))
    sampler_specs = [
        (
            "pidstat.log",
            ["pidstat", "-u", "-w", "-p", pids, "1"],
        ),
        (
            "gpu-dmon.log",
            ["nvidia-smi", "dmon", "-i", "0", "-s", "pucvmet", "-d", "1"],
        ),
    ]
    samplers: list[subprocess.Popen[bytes]] = []
    for name, command in sampler_specs:
        stream = (output_dir / name).open("wb")
        sampler = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        sampler._profile_stream = stream  # type: ignore[attr-defined]
        samplers.append(sampler)
    (output_dir / "sampled-pids.json").write_text(
        json.dumps({"root_pid": process.pid, "pids": _descendants(process.pid)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return samplers


def _stop_samplers(samplers: list[subprocess.Popen[bytes]]) -> None:
    for sampler in samplers:
        if sampler.poll() is None:
            try:
                os.killpg(sampler.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            sampler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(sampler.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            sampler.wait(timeout=10)
        stream = getattr(sampler, "_profile_stream", None)
        if stream is not None:
            stream.close()


def _profile_config(output_dir: Path, mode: str, with_stack: bool) -> str:
    config = {
        "profiler": "torch",
        "torch_profiler_dir": str(output_dir / "torch"),
        "torch_profiler_with_stack": with_stack,
        "torch_profiler_with_flops": False,
        "torch_profiler_use_gzip": False,
        "torch_profiler_dump_cuda_time_total": True,
        "torch_profiler_record_shapes": mode == "eager",
        "torch_profiler_with_memory": False,
        "ignore_frontend": True,
        "delay_iterations": 2,
        "max_iterations": 18,
    }
    return json.dumps(config, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("production", "eager"), default="production")
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--model-name", default="Inferact/Qwen3.8-27B-NVFP4")
    parser.add_argument("--trace", type=Path, default=Path(DEFAULT_TRACE))
    parser.add_argument("--converted-trace", type=Path, default=Path(DEFAULT_CONVERTED))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=228)
    parser.add_argument("--python", default=sys.executable)
    default_vllm = shutil.which("vllm") or str(Path(sys.executable).with_name("vllm"))
    parser.add_argument("--vllm", default=default_vllm)
    parser.add_argument("--readiness-timeout", type=float, default=1800)
    parser.add_argument("--task-num", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument(
        "--spec-tokens",
        type=int,
        default=4,
        help="MTP speculative token count used for the profiled replay.",
    )
    parser.add_argument(
        "--linear-backend",
        default="auto",
        help="Forward an NVFP4 linear backend selector to vLLM for comparison.",
    )
    parser.add_argument(
        "--mamba-ssm-cache-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Override the recurrent GDN state dtype for a kernel/profile comparison.",
    )
    parser.add_argument(
        "--mamba-cache-mode",
        choices=("auto", "none", "align"),
        default="auto",
        help="Override the GDN cache mode for a profile comparison.",
    )
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Environment variable forwarded to the vLLM server process.",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Record Python/C++ call stacks for CPU events in the torch trace.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"model snapshot does not exist: {args.model}")
    if not args.trace.is_file():
        raise SystemExit(f"trace does not exist: {args.trace}")
    if not args.converted_trace.is_dir():
        raise SystemExit(f"converted trace does not exist: {args.converted_trace}")

    output_dir = args.output / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{args.port}"
    profile = Profile(
        name=f"deep-profile-{args.mode}",
        layer="profile",
        hypothesis="Collect per-iteration and operator evidence before changing the hot path.",
        change=(
            f"MTP={args.spec_tokens} with the production path or eager shadow execution."
        ),
        server_args=(
            "--profiler-config",
            _profile_config(output_dir, args.mode, args.with_stack),
            "--enable-logging-iteration-details",
            "--cudagraph-metrics",
        )
        + (() if args.linear_backend == "auto" else ("--linear-backend", args.linear_backend))
        + (
            ("--mamba-ssm-cache-dtype", args.mamba_ssm_cache_dtype)
            if args.mamba_ssm_cache_dtype != "auto"
            else ()
        )
        + (
            ("--mamba-cache-mode", args.mamba_cache_mode)
            if args.mamba_cache_mode != "auto"
            else ()
        )
        + (("--enforce-eager", "--enable-layerwise-nvtx-tracing") if args.mode == "eager" else ()),
        spec_tokens=args.spec_tokens,
    )
    environment = _server_environment(profile)
    for assignment in args.extra_env:
        name, separator, value = assignment.partition("=")
        if not separator or not name:
            raise SystemExit(f"--extra-env must be NAME=VALUE, got {assignment!r}")
        environment[name] = value
    environment["VLLM_LOGGING_LEVEL"] = "INFO"
    command = _server_command(
        profile,
        vllm=args.vllm,
        model=args.model,
        model_name=args.model_name,
        port=args.port,
    )
    (output_dir / "command.txt").write_text(
        " ".join(command) + "\n", encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                key: environment[key]
                for key in ("CUDA_VISIBLE_DEVICES", "VLLM_LOGGING_LEVEL", "PYTHONUNBUFFERED")
                if key in environment
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    server_log = (output_dir / "server.log").open("wb")
    process: subprocess.Popen[bytes] | None = None
    samplers: list[subprocess.Popen[bytes]] = []
    profiling_started = False
    try:
        process = subprocess.Popen(
            command,
            cwd=_REPO_ROOT,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ready_seconds = wait_ready(base_url, process, args.readiness_timeout)
        (output_dir / "readiness.json").write_text(
            json.dumps({"seconds": ready_seconds}, indent=2) + "\n", encoding="utf-8"
        )

        warmup_dir = output_dir / "warmup"
        warmup_config = output_dir / "warmup.yaml"
        if not (warmup_dir / "summary.json").is_file():
            write_config(
                warmup_config,
                result_dir=warmup_dir,
                base_url=base_url,
                model_name=args.model_name,
                trace_path=args.trace,
                converted_trace_path=args.converted_trace,
                seed=args.seed,
                task_num=1,
                max_concurrency=1,
            )
            warmup_rc = run_replay(
                args.python, warmup_config, output_dir / "warmup.log", _REPO_ROOT
            )
            if warmup_rc != 0 or not (warmup_dir / "summary.json").is_file():
                raise RuntimeError(f"warmup replay failed with exit code {warmup_rc}")

        _post(f"{base_url}/start_profile")
        profiling_started = True
        samplers = _start_sampler(process, output_dir)
        measured_dir = output_dir / "measured"
        measured_config = output_dir / "measured.yaml"
        write_config(
            measured_config,
            result_dir=measured_dir,
            base_url=base_url,
            model_name=args.model_name,
            trace_path=args.trace,
            converted_trace_path=args.converted_trace,
            seed=args.seed,
            task_num=args.task_num,
            max_concurrency=args.max_concurrency,
        )
        replay_rc = run_replay(
            args.python, measured_config, output_dir / "measured.log", _REPO_ROOT
        )
        (output_dir / "replay-exit.json").write_text(
            json.dumps({"returncode": replay_rc}, indent=2) + "\n", encoding="utf-8"
        )
        if profiling_started:
            try:
                _post(f"{base_url}/stop_profile")
            except (OSError, URLError):
                pass
            profiling_started = False
        return replay_rc
    finally:
        if profiling_started:
            try:
                _post(f"{base_url}/stop_profile")
            except (OSError, URLError):
                pass
        _stop_samplers(samplers)
        stop_process(process)
        server_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
