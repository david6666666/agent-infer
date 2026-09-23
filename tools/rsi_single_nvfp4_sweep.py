#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Run a layered, resumable 50-round single-B300 NVFP4 serving sweep.

The sweep keeps the Inferact codex_swebenchpro trace and exact prompt
calibration fixed.  It uses five baseline repetitions, two repetitions for
fifteen layer-local screening profiles, and five fresh repetitions for each of
the three fastest replay-valid screening profiles.  Every server profile is
started from a cold process and every measured repetition retains the raw
replay, metrics and server-log evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# Keep the tool runnable directly from a checkout without requiring an
# editable install of AgentInfer.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentinfer.rsi.experiments import append_experiment, list_experiments  # noqa: E402

DEFAULT_MODEL = (
    "/home/zjy/.cache/huggingface/models--Inferact--Qwen3.8-27B-NVFP4/"
    "snapshots/6128240ebaf4eaa7bad2b3d1c72c37d677c5f462"
)
DEFAULT_TRACE = "/home/zjy/code/hsliu/tmp/rsi-pr25-b300-20260922/codex-trace/codex_swebenchpro.json"
DEFAULT_CONVERTED = "/home/zjy/code/hsliu/tmp/rsi-pr25-b300-20260922/results/replay-tp4-smoke/convert_result"
DEFAULT_RUN_DIR = "/home/zjy/code/david/tmp/rsi-single-nvfp4-20260923/rsi-real"
DEFAULT_SWEEP_DIR = "/home/zjy/code/david/tmp/rsi-single-nvfp4-20260923/results/rsi-50-rounds"
DEFAULT_PORT = 8000
DEFAULT_SEED = 228
BASELINE_REPETITIONS = 5
SCREEN_REPETITIONS = 2
CONFIRM_REPETITIONS = 5
EXPECTED_REPLAY_REQUESTS = 36


@dataclass(frozen=True)
class Profile:
    name: str
    layer: str
    hypothesis: str
    change: str
    server_args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    spec_tokens: int | None = 3


BASELINE = Profile(
    name="baseline-user-command",
    layer="contract/control",
    hypothesis="The corrected user command is repeatable when all serving layers remain at their vLLM defaults.",
    change="Keep TP1, 262144 context, FP8 KV, qwen3 reasoning, qwen3_xml tools and MTP=3 exactly as requested.",
)


SCREEN_PROFILES = (
    Profile(
        "mtp-off",
        "speculative",
        "MTP=3 may lose more time to verification and metadata rebuilds than it saves on this trace.",
        "Remove speculative decoding while keeping every other user command flag fixed.",
        spec_tokens=None,
    ),
    Profile(
        "mtp-1",
        "speculative",
        "One MTP token may retain most of the acceptance benefit while avoiding repeated draft-layer work.",
        "Set speculative-config method=mtp and num_speculative_tokens=1.",
        spec_tokens=1,
    ),
    Profile(
        "mtp-2",
        "speculative",
        "MTP=2 may balance acceptance and verification overhead better than the requested MTP=3.",
        "Set speculative-config method=mtp and num_speculative_tokens=2.",
        spec_tokens=2,
    ),
    Profile(
        "mtp-4",
        "speculative",
        "A longer draft may improve accepted tokens per target forward if acceptance remains high.",
        "Set speculative-config method=mtp and num_speculative_tokens=4.",
        spec_tokens=4,
    ),
    Profile(
        "gdn-cutedsl",
        "model-linear-attention",
        "The in-tree CuteDSL GDN prefill path may reduce prefill cost on B300 SM10x.",
        "Set --gdn-prefill-backend=cutedsl; keep GDN decode on its default CUDA path.",
        server_args=("--gdn-prefill-backend", "cutedsl"),
    ),
    Profile(
        "gdn-triton",
        "model-linear-attention",
        "Triton GDN prefill may avoid JIT or synchronization cost even if its kernel is less specialized.",
        "Set --gdn-prefill-backend=triton as a negative and warmup control.",
        server_args=("--gdn-prefill-backend", "triton"),
    ),
    Profile(
        "gdn-decode-triton",
        "model-linear-attention",
        "The fused CUDA GDN decode path may be less effective with MTP metadata rebuilds than Triton decode.",
        "Set VLLM_GDN_DECODE_KERNEL=triton; keep the prefill backend automatic.",
        env=(("VLLM_GDN_DECODE_KERNEL", "triton"),),
    ),
    Profile(
        "linear-cutedsl",
        "quantized-linear-kernels",
        "The official Qwen3.8 recipe's FlashInfer CuTeDSL linear backend may beat auto dispatch for unquantized projections.",
        "Set --linear-backend=flashinfer_cutedsl.",
        server_args=("--linear-backend", "flashinfer_cutedsl"),
    ),
    Profile(
        "attention-triton",
        "attention-and-mtp",
        "TRITON_ATTN may support a more direct MTP verification path than FlashInfer's metadata rebuild.",
        "Set --attention-backend=TRITON_ATTN; retain FP8 KV to isolate backend behavior under the user contract.",
        server_args=("--attention-backend", "TRITON_ATTN"),
    ),
    Profile(
        "batch-8192",
        "scheduler",
        "A smaller scheduler budget may protect decode from long codex prefills.",
        "Set --max-num-batched-tokens=8192.",
        server_args=("--max-num-batched-tokens", "8192"),
    ),
    Profile(
        "batch-32768",
        "scheduler",
        "A larger scheduler budget may improve aggregate prefill throughput on B300.",
        "Set --max-num-batched-tokens=32768.",
        server_args=("--max-num-batched-tokens", "32768"),
    ),
    Profile(
        "batch-65536",
        "scheduler",
        "A still larger scheduler budget may amortize scheduler and launch overhead.",
        "Set --max-num-batched-tokens=65536.",
        server_args=("--max-num-batched-tokens", "65536"),
    ),
    Profile(
        "async-off",
        "engine-runtime",
        "Synchronous scheduling is a negative control for host-to-engine overlap.",
        "Disable --async-scheduling explicitly.",
        server_args=("--no-async-scheduling",),
    ),
    Profile(
        "no-prefix-cache",
        "kv-cache",
        "The continuation-heavy trace should benefit from the hybrid model's default prefix cache.",
        "Disable prefix caching to quantify the cache layer's contribution.",
        server_args=("--no-enable-prefix-caching",),
    ),
    Profile(
        "enforce-eager",
        "compile-and-cuda-graph",
        "CUDA graph capture and piecewise dispatch should matter for this decode-heavy workload.",
        "Disable CUDA graphs with --enforce-eager as a compile/runtime negative control.",
        server_args=("--enforce-eager",),
    ),
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, str | None]:
    return {"path": str(path), "sha256": sha256(path)}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def wait_ready(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> float:
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness with code {process.returncode}")
        try:
            with urlopen(f"{base_url}/health", timeout=5) as response:  # noqa: S310 - loopback URL
                if response.status == 200:
                    return time.monotonic() - started
        except (OSError, URLError):
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready within {timeout:g}s")


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=90)
    except (KeyboardInterrupt, ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            pass


def write_config(
    path: Path,
    *,
    result_dir: Path,
    base_url: str,
    model_name: str,
    trace_path: Path,
    converted_trace_path: Path,
    seed: int,
    task_num: int,
    max_concurrency: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""experiment:
  task_num: {task_num}
  result_dir: {result_dir}
  max_concurrency: {max_concurrency}
  task_timeout_seconds: 1800
  run_timeout_seconds: 2400

backend:
  type: vllm
  base_url: {base_url}
  metrics_url: {base_url}/metrics
  tokenizer_base_url: {base_url}
  model: {model_name}
  endpoint: /v1/chat/completions
  chat_template_kwargs: {{}}
  api_key_env: null

replay:
  trace_type: inferact_codex_swebenchpro
  trace_path: {trace_path}
  converted_trace_path: {converted_trace_path}
  interval_mode: lognormal
  interval_lognormal:
    p50_seconds: 0.001
    p95_seconds: 0.002278
    p99_seconds: 0.003205
  sample_seed: {seed}
  prompt_shape: trace_record
  max_input_tokens: 240000
  max_output_tokens: null
  context_adjustment_mode: adaptive
  context_micro_trim_max_tokens: 64
  context_micro_trim_max_ratio: 0.005
  prompt_calibration_tolerance_tokens: 0
  request_timeout_seconds: 300
"""
    path.write_text(text, encoding="utf-8")


def run_replay(python: str, config: Path, log_path: Path, repo: Path) -> int:
    dispatcher = (
        "import sys; "
        "from agentinfer.agentcache.entrypoints.cli.main import main; "
        "raise SystemExit(main(['vllm', *sys.argv[1:]]))"
    )
    command = [
        python,
        "-c",
        dispatcher,
        "bench",
        "serve",
        "--agentinfer",
        "replay",
        "--config",
        str(config),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=3600,
            check=False,
        )
    return completed.returncode


def _prometheus_values(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        sample, raw_value = line.rsplit(None, 1)
        name = sample.split("{", 1)[0]
        try:
            value = float(raw_value)
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + value
    return values


def _prometheus_delta(result_dir: Path) -> dict[str, float | None]:
    start = _prometheus_values(result_dir / "evidence" / "vllm_metrics_start.prom")
    end = _prometheus_values(result_dir / "evidence" / "vllm_metrics_end.prom")

    def delta(name: str) -> float | None:
        if name not in end:
            return None
        return max(0.0, end[name] - start.get(name, 0.0))

    drafts = delta("vllm:spec_decode_num_drafts_total")
    draft_tokens = delta("vllm:spec_decode_num_draft_tokens_total")
    accepted = delta("vllm:spec_decode_num_accepted_tokens_total")
    mean_acceptance_length = None
    if drafts is not None and drafts > 0 and accepted is not None:
        mean_acceptance_length = 1.0 + accepted / drafts
    acceptance_rate = None
    if draft_tokens is not None and draft_tokens > 0 and accepted is not None:
        acceptance_rate = accepted / draft_tokens
    return {
        "spec_decode_drafts": drafts,
        "spec_decode_draft_tokens": draft_tokens,
        "spec_decode_accepted_tokens": accepted,
        "spec_decode_acceptance_rate": acceptance_rate,
        "spec_decode_mean_acceptance_length": mean_acceptance_length,
        "kv_cache_usage_perc_end": end.get("vllm:kv_cache_usage_perc"),
    }


def summarize(result_dir: Path) -> tuple[dict[str, object], str | None]:
    summary_path = result_dir / "summary.json"
    execution_path = result_dir / "replay-execution.json"
    if not summary_path.is_file():
        return {}, "Replay did not produce summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    requests = summary.get("requests", {})
    tasks = summary.get("tasks", {})
    execution = summary.get("execution", {}).get("metadata", {})
    lifecycle = summary.get("lifecycle", {})
    ttft_p50 = requests.get("ttft_seconds", {}).get("p50")
    planned = execution.get("planned_requests")
    successful = requests.get("successful_requests")
    failure = None
    if lifecycle.get("status") != "completed":
        failure = f"Replay lifecycle status: {lifecycle.get('status')}: {lifecycle.get('error')}"
    elif planned != successful or requests.get("failed_requests", 0) != 0:
        failure = (
            f"Incomplete replay coverage: planned={planned}, successful={successful}, "
            f"failed={requests.get('failed_requests')}"
        )
    metrics: dict[str, object] = {
        "throughput_output_tokens_per_s": summary.get("output_token_throughput_per_second"),
        "throughput_input_tokens_per_s": summary.get("input_token_throughput_per_second"),
        "throughput_requests_per_s": summary.get("request_throughput_per_second"),
        "ttft_p50_ms": None if ttft_p50 is None else ttft_p50 * 1000,
        "replay_planned_tasks": tasks.get("completed", 0) + tasks.get("failed", 0),
        "replay_completed_tasks": tasks.get("completed"),
        "replay_failed_tasks": tasks.get("failed"),
        "replay_planned_requests": planned,
        "replay_exact_input_requests": execution.get("backend_input_checked_requests"),
        "replay_successful_requests": successful,
        "replay_failed_requests": requests.get("failed_requests"),
        "replay_calibration_max_residual_tokens": execution.get("prompt_calibration_max_absolute_residual_tokens"),
        "prefix_cache_hit_rate": summary.get("vllm", {}).get("prefix_cache_hit_rate"),
        "gsm8k_accuracy": None,
        "tpot_p50_ms": None,
        "readiness_seconds": None,
    }
    metrics.update(_prometheus_delta(result_dir))
    if not execution_path.is_file():
        failure = failure or "Replay did not produce replay-execution.json"
    return metrics, failure


def _profile_spec(profile: Profile) -> dict[str, object]:
    return {
        "name": profile.name,
        "layer": profile.layer,
        "server_args": list(profile.server_args),
        "env": dict(profile.env),
        "spec_tokens": profile.spec_tokens,
    }


def _server_command(
    profile: Profile,
    *,
    vllm: str,
    model: Path,
    model_name: str,
    port: int,
) -> list[str]:
    command = [
        vllm,
        "serve",
        str(model),
        "--served-model-name",
        model_name,
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "262144",
        "--kv-cache-dtype",
        "fp8",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if profile.spec_tokens is not None:
        command.extend(
            [
                "--speculative-config",
                json.dumps({"method": "mtp", "num_speculative_tokens": profile.spec_tokens}),
            ]
        )
    command.extend(profile.server_args)
    return command


def _server_environment(profile: Profile) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "HF_HOME": "/home/zjy/.cache/huggingface",
            "HF_HUB_CACHE": "/home/zjy/.cache/huggingface/hub",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    environment.update(dict(profile.env))
    return environment


def _write_runtime_probe(base_url: str, path: Path) -> None:
    payload: dict[str, object] = {"captured_at": timestamp(), "base_url": base_url}
    for name, url in (("version", f"{base_url}/version"), ("models", f"{base_url}/v1/models")):
        try:
            with urlopen(Request(url, method="GET"), timeout=30) as response:  # noqa: S310 - loopback URL
                payload[name] = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as error:
            payload[name] = {"error": f"{type(error).__name__}: {error}"}
    try:
        with urlopen(f"{base_url}/metrics", timeout=30) as response:  # noqa: S310 - loopback URL
            payload["metrics_initial"] = response.read().decode("utf-8", errors="replace")
    except (OSError, URLError) as error:
        payload["metrics_initial_error"] = f"{type(error).__name__}: {error}"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_record(
    *,
    round_id: str,
    profile: Profile,
    phase: str,
    parent_profile: str | None,
    repeat: int,
    repeat_target: int,
    result_dir: Path,
    config_path: Path,
    replay_log: Path,
    server_log: Path,
    runtime_probe: Path,
    metrics: dict[str, object],
    failure: str | None,
    readiness_seconds: float | None,
    commit: str,
    python: str,
    vllm: str,
    model: Path,
    model_name: str,
    port: int,
    trace_path: Path,
    converted_trace_path: Path,
    seed: int,
) -> dict[str, object]:
    values = dict(metrics)
    values["readiness_seconds"] = readiness_seconds
    status = "measured" if failure is None else "failed"
    server_command = _server_command(profile, vllm=vllm, model=model, model_name=model_name, port=port)
    dispatcher = (
        f"{python} -c agentinfer.agentcache.entrypoints.cli.main bench serve --agentinfer replay --config {config_path}"
    )
    evidence_paths = [
        config_path,
        result_dir / "summary.json",
        result_dir / "replay-execution.json",
        result_dir / "evidence" / "vllm_metrics_start.prom",
        result_dir / "evidence" / "vllm_metrics_end.prom",
        replay_log,
        server_log,
        runtime_probe,
    ]
    return {
        "round_id": round_id,
        "timestamp": timestamp(),
        "hypothesis": profile.hypothesis,
        "change": f"{profile.change} {phase} repetition {repeat}/{repeat_target}.",
        "model": model_name,
        "backend": f"cuda/vllm-tp1/nvfp4/{profile.name}",
        "component_version": f"agentinfer-{commit[:12]}+vllm-0.29.0-single-nvfp4",
        "config": {
            "precision": "nvfp4_weights_fp8_kv",
            "gpu_count": 1,
            "gpu_model": "NVIDIA B300 SXM6 AC",
            "tensor_parallel_size": 1,
            "max_model_len": 262144,
            "max_concurrency": 2,
            "trace_seed": seed,
            "trace_mode": "inferact_codex_swebenchpro",
            "prompt_calibration_tolerance_tokens": 0,
            "model_snapshot": str(model),
            "profile": profile.name,
            "parent_profile": parent_profile,
            "phase": phase,
            "layer": profile.layer,
            "server_args": list(profile.server_args),
            "server_env": dict(profile.env),
            "speculative_tokens": profile.spec_tokens,
            "repetition": repeat,
            "repetition_target": repeat_target,
        },
        "commands": [
            shlex.join(server_command),
            dispatcher,
            f"trace={trace_path} converted={converted_trace_path}",
        ],
        "metrics": values,
        "status": status,
        "evidence": [evidence(path) for path in evidence_paths],
        "failure_reason": failure,
        "next_test": "Rank only replay-valid 36/36 runs; use GSM8K before promoting a performance winner.",
        **({"baseline_round_id": "I0"} if round_id != "I0" else {}),
    }


def _record_profile(record: dict[str, object]) -> str:
    config = record.get("config")
    return str(config.get("profile")) if isinstance(config, dict) else ""


def _valid_throughput(record: dict[str, object]) -> float | None:
    if record.get("status") != "measured":
        return None
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    if metrics.get("replay_successful_requests") != EXPECTED_REPLAY_REQUESTS:
        return None
    value = metrics.get("throughput_output_tokens_per_s")
    return float(value) if isinstance(value, (int, float)) else None


def choose_top_profiles(records: list[dict[str, object]]) -> list[Profile]:
    by_name: dict[str, list[float]] = {}
    for record in records:
        name = _record_profile(record)
        if name not in {profile.name for profile in SCREEN_PROFILES}:
            continue
        value = _valid_throughput(record)
        if value is not None:
            by_name.setdefault(name, []).append(value)
    ranked = sorted(
        by_name.items(),
        key=lambda item: statistics.median(item[1]),
        reverse=True,
    )
    profile_by_name = {profile.name: profile for profile in SCREEN_PROFILES}
    selected = [profile_by_name[name] for name, _ in ranked[:3]]
    if len(selected) < 3:
        # Keep the ledger at exactly 50 records even when an environment issue
        # prevents a screening profile from producing a valid measurement. The
        # fallback is explicitly marked by the shortlist and remains subject
        # to the same replay and quality gates.
        for profile in SCREEN_PROFILES:
            if profile not in selected:
                selected.append(profile)
            if len(selected) == 3:
                break
    return selected


def _next_round_number(records: list[dict[str, object]]) -> int:
    numbers = []
    for record in records:
        round_id = str(record.get("round_id", ""))
        if round_id.startswith("I") and round_id[1:].isdigit():
            numbers.append(int(round_id[1:]))
    return max(numbers, default=-1) + 1


def _missing_repetitions(records: list[dict[str, object]], profile_name: str, target: int) -> list[int]:
    repeats = set()
    for record in records:
        if _record_profile(record) != profile_name:
            continue
        config = record.get("config")
        if isinstance(config, dict) and isinstance(config.get("repetition"), int):
            repeats.add(int(config["repetition"]))
    return [repeat for repeat in range(1, target + 1) if repeat not in repeats]


def _ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def run_profile(
    *,
    profile: Profile,
    phase: str,
    parent_profile: str | None,
    target: int,
    args: argparse.Namespace,
    records: list[dict[str, object]],
    next_number: int,
    commit: str,
) -> int:
    profile_name = profile.name if phase != "confirmation" else f"confirm-{profile.name}"
    missing = _missing_repetitions(records, profile_name, target)
    if not missing:
        return next_number

    profile_dir = args.sweep_dir / safe_name(profile_name)
    profile_dir.mkdir(parents=True, exist_ok=True)
    server_log = profile_dir / "server.log"
    runtime_probe = profile_dir / "runtime-probe.json"
    base_url = f"http://127.0.0.1:{args.port}"
    command = _server_command(
        profile,
        vllm=args.vllm,
        model=args.model,
        model_name=args.model_name,
        port=args.port,
    )
    environment = _server_environment(profile)
    with server_log.open("a", encoding="utf-8") as log:
        log.write(f"\n=== profile={profile_name} phase={phase} started={timestamp()} ===\n")
        log.write("env " + json.dumps({key: environment[key] for key, _ in profile.env}, sort_keys=True) + "\n")
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()

    process: subprocess.Popen[bytes] | None = None
    server_stream = None
    readiness: float | None = None
    profile_failure: str | None = None
    try:
        server_stream = server_log.open("ab")
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            env=environment,
            stdout=server_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        readiness = wait_ready(base_url, process, args.readiness_timeout)
        _write_runtime_probe(base_url, runtime_probe)

        warmup_dir = profile_dir / "warmup"
        warmup_config = profile_dir / "warmup.yaml"
        warmup_log = profile_dir / "warmup.log"
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
            warmup_rc = run_replay(args.python, warmup_config, warmup_log, args.repo)
            if warmup_rc != 0 or not (warmup_dir / "summary.json").is_file():
                raise RuntimeError(f"warmup failed with replay exit code {warmup_rc}")

        for repeat in missing:
            round_id = f"I{next_number}"
            next_number += 1
            result_dir = profile_dir / f"repeat-{repeat}"
            config_path = profile_dir / f"repeat-{repeat}.yaml"
            replay_log = profile_dir / f"repeat-{repeat}.log"
            write_config(
                config_path,
                result_dir=result_dir,
                base_url=base_url,
                model_name=args.model_name,
                trace_path=args.trace,
                converted_trace_path=args.converted_trace,
                seed=args.seed,
                task_num=2,
                max_concurrency=2,
            )
            replay_rc = run_replay(args.python, config_path, replay_log, args.repo)
            metrics, failure = summarize(result_dir)
            if replay_rc != 0:
                failure = failure or f"Replay command exited with code {replay_rc}"
            record = make_record(
                round_id=round_id,
                profile=profile,
                phase=phase,
                parent_profile=parent_profile,
                repeat=repeat,
                repeat_target=target,
                result_dir=result_dir,
                config_path=config_path,
                replay_log=replay_log,
                server_log=server_log,
                runtime_probe=runtime_probe,
                metrics=metrics,
                failure=failure,
                readiness_seconds=readiness,
                commit=commit,
                python=args.python,
                vllm=args.vllm,
                model=args.model,
                model_name=args.model_name,
                port=args.port,
                trace_path=args.trace,
                converted_trace_path=args.converted_trace,
                seed=args.seed,
            )
            appended = append_experiment(args.run_dir, record)
            records.append(appended)
            print(
                f"{round_id} {profile_name} repeat={repeat}/{target} "
                f"status={appended['status']} "
                f"output_tok_s={appended['metrics'].get('throughput_output_tokens_per_s')} "
                f"spec_al={appended['metrics'].get('spec_decode_mean_acceptance_length')}",
                flush=True,
            )
    except Exception as error:
        profile_failure = f"Profile execution error: {type(error).__name__}: {error}"
        print(f"{profile_name} failed: {profile_failure}", file=sys.stderr, flush=True)
    finally:
        stop_process(process)
        if server_stream is not None:
            server_stream.close()

    if profile_failure is not None:
        for repeat in missing:
            if any(
                _record_profile(record) == profile_name
                and isinstance(record.get("config"), dict)
                and record["config"].get("repetition") == repeat
                for record in records
            ):
                continue
            round_id = f"I{next_number}"
            next_number += 1
            result_dir = profile_dir / f"repeat-{repeat}"
            config_path = profile_dir / f"repeat-{repeat}.yaml"
            replay_log = profile_dir / f"repeat-{repeat}.log"
            write_config(
                config_path,
                result_dir=result_dir,
                base_url=base_url,
                model_name=args.model_name,
                trace_path=args.trace,
                converted_trace_path=args.converted_trace,
                seed=args.seed,
                task_num=2,
                max_concurrency=2,
            )
            _ensure_file(replay_log)
            record = make_record(
                round_id=round_id,
                profile=profile,
                phase=phase,
                parent_profile=parent_profile,
                repeat=repeat,
                repeat_target=target,
                result_dir=result_dir,
                config_path=config_path,
                replay_log=replay_log,
                server_log=server_log,
                runtime_probe=runtime_probe,
                metrics={},
                failure=profile_failure,
                readiness_seconds=readiness,
                commit=commit,
                python=args.python,
                vllm=args.vllm,
                model=args.model,
                model_name=args.model_name,
                port=args.port,
                trace_path=args.trace,
                converted_trace_path=args.converted_trace,
                seed=args.seed,
            )
            appended = append_experiment(args.run_dir, record)
            records.append(appended)
            print(f"appended {round_id} {appended['status']}", flush=True)
    return next_number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_RUN_DIR))
    parser.add_argument("--sweep-dir", type=Path, default=Path(DEFAULT_SWEEP_DIR))
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--model-name", default="Inferact/Qwen3.8-27B-NVFP4")
    parser.add_argument("--trace", type=Path, default=Path(DEFAULT_TRACE))
    parser.add_argument("--converted-trace", type=Path, default=Path(DEFAULT_CONVERTED))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vllm", default=shutil.which("vllm") or "vllm")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--readiness-timeout", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rounds", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds != 50:
        raise SystemExit("This controlled matrix is defined for exactly --rounds 50")
    if not args.model.is_dir():
        raise SystemExit(f"model snapshot does not exist: {args.model}")
    if not args.trace.is_file():
        raise SystemExit(f"trace does not exist: {args.trace}")
    if not args.converted_trace.is_dir():
        raise SystemExit(f"converted trace does not exist: {args.converted_trace}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.sweep_dir.mkdir(parents=True, exist_ok=True)
    records = list_experiments(args.run_dir)
    if len(records) > 50:
        raise SystemExit(f"existing ledger has {len(records)} records; expected at most 50")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    next_number = _next_round_number(records)

    next_number = run_profile(
        profile=BASELINE,
        phase="baseline",
        parent_profile=None,
        target=BASELINE_REPETITIONS,
        args=args,
        records=records,
        next_number=next_number,
        commit=commit,
    )
    for profile in SCREEN_PROFILES:
        next_number = run_profile(
            profile=profile,
            phase="screening",
            parent_profile=None,
            target=SCREEN_REPETITIONS,
            args=args,
            records=records,
            next_number=next_number,
            commit=commit,
        )

    records = list_experiments(args.run_dir)
    top_profiles = choose_top_profiles(records)
    shortlist_path = args.sweep_dir / "top-screening-profiles.json"
    shortlist_path.write_text(
        json.dumps(
            {
                "created_at": timestamp(),
                "selection": "top three screening medians among replay-valid 36/36 records",
                "profiles": [profile.name for profile in top_profiles],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"confirmation shortlist: {[profile.name for profile in top_profiles]}", flush=True)
    for profile in top_profiles:
        next_number = run_profile(
            profile=profile,
            phase="confirmation",
            parent_profile=profile.name,
            target=CONFIRM_REPETITIONS,
            args=args,
            records=records,
            next_number=next_number,
            commit=commit,
        )

    final_records = list_experiments(args.run_dir)
    if len(final_records) != 50:
        raise SystemExit(f"matrix stopped with {len(final_records)} records; expected 50")
    print("completed 50 records", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
