#!/usr/bin/env python3
"""Compare FlashInfer NVFP4 output allocation with a caller-owned output.

The production vLLM custom op currently invokes ``mm_fp4`` without ``out``.
This probe keeps the exact Qwen NVFP4 tensor layouts and separates allocator
and Python-wrapper time from the actual CuTeDSL kernel time.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from flashinfer import SfLayout, mm_fp4, nvfp4_quantize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=40)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--out-mode", choices=("none", "reuse"), default="none")
    parser.add_argument("--pdl", action=argparse.BooleanOptionalAction, default=True)
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
    alpha = torch.ones(1, device=device, dtype=torch.float32)
    out = (
        torch.empty((args.m, args.n), device=device, dtype=torch.bfloat16)
        if args.out_mode == "reuse"
        else None
    )

    def run() -> torch.Tensor:
        return mm_fp4(
            a,
            b.T,
            a_sf,
            b_sf.T,
            alpha=alpha,
            out_dtype=torch.bfloat16,
            out=out,
            backend="cute-dsl",
            block_size=16,
            use_nvfp4=True,
            enable_pdl=args.pdl,
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    gpu_starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.reps)]
    gpu_ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.reps)]
    host_ms: list[float] = []
    result: torch.Tensor | None = None
    host_start = time.perf_counter()
    for index in range(args.reps):
        # Model execution consumes a linear output before the next same-shape
        # projection is launched.  Drop the Python reference before entering
        # the next call so the pool probe can observe that normal lifetime.
        result = None
        start = time.perf_counter()
        gpu_starts[index].record()
        result = run()
        gpu_ends[index].record()
        host_ms.append((time.perf_counter() - start) * 1000.0)
    total_host_ms = (time.perf_counter() - host_start) * 1000.0
    torch.cuda.synchronize()
    gpu_ms = [
        start_event.elapsed_time(end_event)
        for start_event, end_event in zip(gpu_starts, gpu_ends)
    ]
    print(
        {
            "shape": [args.m, args.n, args.k],
            "out_mode": args.out_mode,
            "pdl": args.pdl,
            "reps": args.reps,
            "gpu_median_ms": statistics.median(gpu_ms),
            "gpu_p10_ms": sorted(gpu_ms)[max(0, args.reps // 10 - 1)],
            "host_median_ms": statistics.median(host_ms),
            "host_p90_ms": sorted(host_ms)[max(0, int(args.reps * 0.90) - 1)],
            "loop_wall_median_ms": total_host_ms / args.reps,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
            "out_checksum": float(result.float().mean().item()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
