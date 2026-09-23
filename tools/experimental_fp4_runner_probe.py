#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Benchmark a FlashInfer CuTeDSL FP4 runner without the mm_fp4 wrapper."""

from __future__ import annotations

import argparse
import ast
import statistics

import torch
from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.gemm import gemm_base


def _tuple(value):
    if isinstance(value, list):
        return tuple(_tuple(item) for item in value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=40)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--tactic", default="", help="Python literal/list tactic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    x = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    w = torch.randn((args.n, args.k), device=device, dtype=torch.bfloat16)
    one = torch.ones((), device=device, dtype=torch.float32)
    a, a_sf = nvfp4_quantize(
        x, one, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b, b_sf = nvfp4_quantize(
        w, one, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    out = torch.empty((args.m, args.n), device=device, dtype=torch.bfloat16)
    alpha = torch.ones(1, device=device, dtype=torch.float32)
    workspace = torch.empty(32 * 1024 * 1024, device=device, dtype=torch.uint8)
    inputs = [
        a,
        b.T,
        a_sf,
        b_sf.T,
        alpha,
        torch.bfloat16,
        out,
        16,
        True,
        workspace,
    ]
    runner = gemm_base._cute_dsl_gemm_fp4_runner(10, 3, True, torch.bfloat16, True)
    tactic = _tuple(ast.literal_eval(args.tactic)) if args.tactic else None
    if tactic is None:
        tactic = gemm_base._select_sm100_mm_fp4_cute_dsl_tactic(
            args.m, args.n, args.k, torch.cuda.get_device_properties(0).multi_processor_count, 16
        )
    for _ in range(args.warmup):
        runner(inputs=inputs, tactic=tactic)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.reps)]
    for index in range(args.reps):
        starts[index].record()
        runner(inputs=inputs, tactic=tactic)
        ends[index].record()
    torch.cuda.synchronize()
    gpu_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    print(
        {
            "shape": [args.m, args.n, args.k],
            "tactic": repr(tactic),
            "gpu_median_ms": statistics.median(gpu_ms),
            "gpu_min_ms": min(gpu_ms),
            "gpu_samples_ms": gpu_ms,
            "out_checksum": float(out.float().mean().item()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
