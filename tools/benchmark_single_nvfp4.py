#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Measure workload utilization across concurrency and MTP settings.

Each MTP setting gets one cold vLLM process.  After a trace warmup, the same
converted trace is replayed at each requested concurrency so that the
profile-driven scheduler experiment is reproducible and resumable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.rsi_single_nvfp4_sweep import (  # noqa: E402
    Profile,
    _server_command,
    _server_environment,
    run_replay,
    stop_process,
    summarize,
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
DEFAULT_OUTPUT = "/home/zjy/code/david/tmp/rsi-single-nvfp4-20260923/deep-bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtp", type=int, nargs="+", default=[4])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--linear-backend", default="auto")
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("fp8", "bfloat16", "float16", "auto"),
        default="fp8",
        help="KV cache dtype; fp8 is the production baseline.",
    )
    parser.add_argument("--gdn-decode-kernel", choices=("cuda", "triton"), default="cuda")
    parser.add_argument(
        "--mamba-cache-mode",
        choices=("auto", "none", "align"),
        default="auto",
        help="Override the vLLM Mamba cache mode; auto keeps the serving default.",
    )
    parser.add_argument(
        "--mamba-ssm-cache-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Override the recurrent GDN state dtype; auto keeps the model config.",
    )
    parser.add_argument(
        "--performance-mode",
        choices=("balanced", "interactivity", "throughput"),
        default="balanced",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Override vLLM scheduler max_num_seqs for a graph/scheduler experiment.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Request a KV cache block size; hybrid alignment may raise it.",
    )
    parser.add_argument(
        "--compilation-config",
        default="",
        help="Raw JSON passed to vLLM --compilation-config for a guided fusion test.",
    )
    parser.add_argument(
        "--enable-flashinfer-fp4-autotune",
        action="store_true",
        help="Clear vLLM's CuTeDSL fp4_gemm autotune skip list for this run.",
    )
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Add an environment variable to the vLLM server; repeatable for probes.",
    )
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
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture for an operator correctness probe.",
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
    extra_env: dict[str, str] = {}
    for item in args.extra_env:
        name, separator, value = item.partition("=")
        if not separator or not name:
            raise SystemExit(f"--extra-env must be NAME=VALUE, got {item!r}")
        extra_env[name] = value

    args.output.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output / "results.json"
    if ledger_path.is_file():
        records = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        records = []
    completed = {
        (int(record["mtp"]), int(record["concurrency"])) for record in records
    }

    for mtp in args.mtp:
        profile_dir = args.output / f"mtp-{mtp}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile = Profile(
            name=f"benchmark-mtp-{mtp}-{args.linear_backend}",
            layer="benchmark-utilization",
            hypothesis="Higher admission concurrency should reduce the idle gaps seen in GPU dmon.",
            change=(
                f"MTP={mtp}; vary replay max_concurrency while holding trace seed "
                "and prompts fixed."
            ),
            server_args=(
                (("--enforce-eager",) if args.enforce_eager else ())
                +
                (() if args.linear_backend == "auto" else ("--linear-backend", args.linear_backend))
                + (
                    ("--kv-cache-dtype", args.kv_cache_dtype)
                    if args.kv_cache_dtype not in {"auto", "fp8"}
                    else ()
                )
                + (() if args.mamba_cache_mode == "auto" else ("--mamba-cache-mode", args.mamba_cache_mode))
                + (
                    ("--mamba-ssm-cache-dtype", args.mamba_ssm_cache_dtype)
                    if args.mamba_ssm_cache_dtype != "auto"
                    else ()
                )
                + (
                    ("--performance-mode", args.performance_mode)
                    if args.performance_mode != "balanced"
                    else ()
                )
                + (
                    ("--max-num-seqs", str(args.max_num_seqs))
                    if args.max_num_seqs is not None
                    else ()
                )
                + (
                    ("--block-size", str(args.block_size))
                    if args.block_size is not None
                    else ()
                )
                + (
                    ("--compilation-config", args.compilation_config)
                    if args.compilation_config
                    else ()
                )
            ),
            env=(
                (("VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS", ""),)
                if args.enable_flashinfer_fp4_autotune
                else ()
            )
            + (
                ()
                if args.gdn_decode_kernel == "cuda"
                else (("VLLM_GDN_DECODE_KERNEL", args.gdn_decode_kernel),)
            ) + tuple(extra_env.items()),
            spec_tokens=mtp,
        )
        base_url = f"http://127.0.0.1:{args.port}"
        command = _server_command(
            profile,
            vllm=args.vllm,
            model=args.model,
            model_name=args.model_name,
            port=args.port,
        )
        (profile_dir / "command.txt").write_text(
            " ".join(command) + "\n", encoding="utf-8"
        )
        process = None
        server_stream = None
        try:
            server_stream = (profile_dir / "server.log").open("ab")
            process = subprocess.Popen(
                command,
                cwd=_REPO_ROOT,
                env=_server_environment(profile),
                stdout=server_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            readiness = wait_ready(base_url, process, args.readiness_timeout)
            (profile_dir / "readiness.json").write_text(
                json.dumps({"seconds": readiness}, indent=2) + "\n", encoding="utf-8"
            )

            warmup_dir = profile_dir / "warmup"
            warmup_config = profile_dir / "warmup.yaml"
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
                rc = run_replay(
                    args.python, warmup_config, profile_dir / "warmup.log", _REPO_ROOT
                )
                if rc != 0 or not (warmup_dir / "summary.json").is_file():
                    raise RuntimeError(f"warmup replay failed with exit code {rc}")

            for concurrency in args.concurrency:
                key = (mtp, concurrency)
                if key in completed:
                    continue
                result_dir = profile_dir / f"concurrency-{concurrency}"
                config_path = profile_dir / f"concurrency-{concurrency}.yaml"
                write_config(
                    config_path,
                    result_dir=result_dir,
                    base_url=base_url,
                    model_name=args.model_name,
                    trace_path=args.trace,
                    converted_trace_path=args.converted_trace,
                    seed=args.seed,
                    task_num=args.task_num,
                    max_concurrency=concurrency,
                )
                rc = run_replay(
                    args.python,
                    config_path,
                    profile_dir / f"concurrency-{concurrency}.log",
                    _REPO_ROOT,
                )
                metrics, failure = summarize(result_dir)
                if rc != 0:
                    failure = failure or f"replay exited with code {rc}"
                record = {
                    "mtp": mtp,
                    "concurrency": concurrency,
                    "status": "measured" if failure is None else "failed",
                    "metrics": metrics,
                    "failure": failure,
                    "config": str(config_path),
                }
                records.append(record)
                completed.add(key)
                ledger_path.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"mtp={mtp} concurrency={concurrency} status={record['status']} "
                    f"output_tok_s={metrics.get('throughput_output_tokens_per_s')} "
                    f"prefix_hit={metrics.get('prefix_cache_hit_rate')}",
                    flush=True,
                )
        finally:
            stop_process(process)
            if server_stream is not None:
                server_stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
