#!/usr/bin/env python3
"""Run one exact NVFP4 CuTeDSL GEMM shape for Nsight Compute.

The tensors use the same 128x4 scale layout and weight transpose as the
Qwen3.8 NVFP4 linear path.  This isolates the hot GEMM from model serving so
Nsight Compute can collect hardware counters without replaying a full server.
"""

from __future__ import annotations

import argparse
import time

import torch
import flashinfer
from flashinfer import SfLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=40)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--reps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k % 16 or args.n % 16:
        raise SystemExit("K and N must be divisible by 16")
    device = torch.device("cuda")
    x = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    w = torch.randn((args.n, args.k), device=device, dtype=torch.bfloat16)
    # The global scale is irrelevant to the kernel counters; using one keeps
    # the probe deterministic and preserves the production tensor layout.
    one = torch.ones((), device=device, dtype=torch.float32)
    a, a_sf = flashinfer.nvfp4_quantize(
        x, one, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b, b_sf = flashinfer.nvfp4_quantize(
        w, one, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    out = torch.empty((args.m, args.n), device=device, dtype=torch.bfloat16)
    alpha = torch.ones(1, device=device, dtype=torch.float32)

    def run() -> None:
        flashinfer.mm_fp4(
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
            enable_pdl=True,
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    gpu_starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.reps)]
    gpu_ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.reps)]
    start = time.perf_counter()
    for index in range(args.reps):
        gpu_starts[index].record()
        run()
        gpu_ends[index].record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gpu_ms = [
        start_event.elapsed_time(end_event)
        for start_event, end_event in zip(gpu_starts, gpu_ends)
    ]
    print(
        {
            "shape": [args.m, args.n, args.k],
            "a": [list(a.shape), str(a.dtype), list(a_sf.shape), str(a_sf.dtype)],
            "b": [list(b.T.shape), str(b.T.dtype), list(b_sf.T.shape), str(b_sf.T.dtype)],
            "reps": args.reps,
            "wall_ms": elapsed * 1000.0 / args.reps,
            "gpu_ms": gpu_ms,
            "gpu_median_ms": float(torch.tensor(gpu_ms).median().item()),
            "out_checksum": float(out.float().mean().item()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
