#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Render the profile-first serving architecture as a dependency-free PNG."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 2400, 1500
BG = "#f5f7fa"
INK = "#253238"
MUTED = "#69777d"
GRID = "#d6dee2"
WHITE = "#ffffff"
BLUE = "#eaf1ff"
BLUE_EDGE = "#6b8fd6"
GREEN = "#e8f1e8"
GREEN_EDGE = "#5d9b6f"
GOLD = "#fff4df"
GOLD_EDGE = "#c58b2c"
RED = "#fbe8e8"
RED_EDGE = "#c46a6a"
ORANGE = "#b66a00"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


TITLE = _font(40, True)
SUBTITLE = _font(21)
GROUP = _font(21, True)
NODE = _font(18)
NODE_BOLD = _font(18, True)
SMALL = _font(15)


def _center_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str]) -> None:
    x1, y1, x2, y2 = box
    heights = [draw.textbbox((0, 0), line, font=NODE)[3] for line in lines]
    total = sum(heights) + 5 * (len(lines) - 1)
    y = y1 + (y2 - y1 - total) // 2
    for line, height in zip(lines, heights, strict=True):
        bbox = draw.textbbox((0, 0), line, font=NODE)
        x = x1 + (x2 - x1 - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, font=NODE_BOLD if y == y1 + (y2 - y1 - total) // 2 else NODE, fill=INK)
        y += height + 5


def _node(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str], fill: str, edge: str) -> None:
    draw.rounded_rectangle(box, radius=14, fill=fill, outline=edge, width=3)
    _center_text(draw, box, lines)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = INK, dashed: bool = False) -> None:
    x1, y1 = start
    x2, y2 = end
    if dashed:
        distance = max(abs(x2 - x1), abs(y2 - y1))
        steps = max(1, distance // 18)
        for index in range(0, steps, 2):
            a = index / steps
            b = min(1.0, (index + 1) / steps)
            draw.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=color, width=3)
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=4)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 12
    left = (x2 - size * math.cos(angle - 0.45), y2 - size * math.sin(angle - 0.45))
    right = (x2 - size * math.cos(angle + 0.45), y2 - size * math.sin(angle + 0.45))
    draw.polygon([(x2, y2), left, right], fill=color)


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, color: str = MUTED) -> None:
    draw.text(xy, value, font=SMALL, fill=color)


def render(output: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((52, 28), "Profile-first serving architecture · single B300 NVFP4", font=TITLE, fill=INK)
    draw.text((54, 82), "The feedback loop observes GPU kernels and host gaps before choosing the next layer-local optimization.", font=SUBTITLE, fill=MUTED)

    # Orchestration path.
    codex = (55, 185, 300, 300)
    trace = (365, 185, 690, 300)
    replay = (755, 185, 1110, 300)
    api = (1175, 185, 1530, 300)
    gpu = (1900, 185, 2315, 300)
    _node(draw, codex, ["Codex CLI", "experiment loop"], GREEN, GREEN_EDGE)
    _node(draw, trace, ["AgentBench trace", "codex_swebenchpro"], GREEN, GREEN_EDGE)
    _node(draw, replay, ["AgentInfer replay", "exact request gate"], GREEN, GREEN_EDGE)
    _node(draw, api, ["vLLM OpenAI API", "TP1 · MTP · FP8 KV"], GREEN, GREEN_EDGE)
    _node(draw, gpu, ["1 × NVIDIA B300", "SM103 · CUDA 13.0"], BLUE, BLUE_EDGE)
    _arrow(draw, (codex[2], 242), (trace[0], 242), GREEN_EDGE)
    _arrow(draw, (trace[2], 242), (replay[0], 242), GREEN_EDGE)
    _arrow(draw, (replay[2], 242), (api[0], 242), GREEN_EDGE)

    # Serving layers.
    serve = (1120, 370, 2335, 850)
    draw.rounded_rectangle(serve, radius=18, fill=WHITE, outline=BLUE_EDGE, width=3)
    draw.text((1150, 392), "vLLM serving layers · one cold profile per iteration", font=GROUP, fill=INK)
    nodes = {
        "engine": (1160, 465, 1390, 575),
        "scheduler": (1450, 465, 1680, 575),
        "kv": (1740, 465, 1970, 575),
        "runner": (1160, 660, 1390, 770),
        "hybrid": (1450, 660, 1680, 770),
        "kernel": (1740, 660, 1970, 770),
    }
    _node(draw, nodes["engine"], ["Engine / API", "tokenizer · metrics"], BLUE, BLUE_EDGE)
    _node(draw, nodes["scheduler"], ["Scheduler", "async · chunked prefill", "stop fast path opt-in"], BLUE, BLUE_EDGE)
    _node(draw, nodes["kv"], ["KV / cache", "prefix · FP8 · paged"], BLUE, BLUE_EDGE)
    _node(draw, nodes["runner"], ["Model runner", "input prep · CUDA graph"], BLUE, BLUE_EDGE)
    _node(draw, nodes["hybrid"], ["Hybrid Qwen3.8", "attention + GDN + MTP"], BLUE, BLUE_EDGE)
    _node(draw, nodes["kernel"], ["Operator dispatch", "NVFP4 · GDN · attention"], BLUE, BLUE_EDGE)
    _arrow(draw, (1390, 520), (1450, 520), BLUE_EDGE)
    _arrow(draw, (1680, 520), (1740, 520), BLUE_EDGE)
    _arrow(draw, (1275, 575), (1275, 660), BLUE_EDGE)
    _arrow(draw, (1565, 575), (1565, 660), BLUE_EDGE)
    _arrow(draw, (1855, 575), (1855, 660), BLUE_EDGE)
    _arrow(draw, (1390, 715), (1450, 715), BLUE_EDGE)
    _arrow(draw, (1680, 715), (1740, 715), BLUE_EDGE)
    _arrow(draw, (1970, 715), (2140, 300), BLUE_EDGE)
    _label(draw, (1990, 785), "steady GPU execution", BLUE_EDGE)

    # Profile split.
    profiler = (55, 420, 470, 560)
    host = (55, 625, 470, 765)
    bubble = (535, 420, 1040, 560)
    hotspot = (535, 625, 1040, 765)
    _node(draw, profiler, ["Torch profiler", "GPU execute spans"], GOLD, GOLD_EDGE)
    _node(draw, host, ["Host stack trace", "Python + torch ops"], GOLD, GOLD_EDGE)
    _node(draw, bubble, ["CPU bubble ledger", "prepare_inputs · scheduler", "output checks · metadata · UVA"], GOLD, GOLD_EDGE)
    _node(draw, hotspot, ["Operator ledger", "FP4 GEMM · qkvz · GDN", "conversion · postconv"], GOLD, GOLD_EDGE)
    _arrow(draw, (1275, 770), (470, 490), GOLD_EDGE, dashed=True)
    _arrow(draw, (1275, 770), (470, 695), GOLD_EDGE, dashed=True)
    _arrow(draw, (470, 490), (535, 490), GOLD_EDGE)
    _arrow(draw, (470, 695), (535, 695), GOLD_EDGE)

    # Gate and feedback path.
    oracle = (1120, 930, 1510, 1065)
    micro = (1580, 930, 1970, 1065)
    quality = (2040, 930, 2335, 1065)
    kb = (55, 930, 470, 1065)
    _node(draw, oracle, ["Layer hypothesis gate", "profile → candidate"], RED, RED_EDGE)
    _node(draw, micro, ["Correctness + microbench", "shape · dtype · occupancy"], GOLD, GOLD_EDGE)
    _node(draw, quality, ["GSM8K quality", "parsed · correct · errors"], GOLD, GOLD_EDGE)
    _node(draw, kb, ["Knowledge base", "dashboard · next hypothesis"], GREEN, GREEN_EDGE)
    _arrow(draw, (1040, 490), (1120, 980), RED_EDGE, dashed=True)
    _arrow(draw, (1040, 695), (1120, 1015), RED_EDGE, dashed=True)
    _arrow(draw, (1510, 997), (1580, 997), RED_EDGE)
    _arrow(draw, (1970, 997), (2040, 997), GOLD_EDGE)
    _arrow(draw, (2040, 1065), (470, 998), GREEN_EDGE, dashed=True)
    _arrow(draw, (470, 930), (250, 300), GREEN_EDGE, dashed=True)
    _label(draw, (745, 872), "reject preserves evidence; promote only after replay + quality", RED_EDGE)

    # Contract legend.
    legend = (55, 1170, 2335, 1400)
    draw.rounded_rectangle(legend, radius=14, fill=WHITE, outline=GRID, width=2)
    draw.text((85, 1195), "Frozen contract and scope", font=GROUP, fill=INK)
    draw.text((85, 1240), "Inferact/Qwen3.8-27B-NVFP4 · TP1 · max-model-len 262144 · FP8 KV · qwen3/qwen3_xml · prefix cache · AgentBench codex_swebenchpro · B300", font=SMALL, fill=INK)
    draw.text((85, 1280), "Short replay: 2 tasks / 36 requests. Sustained replay: 4 tasks / 89 requests. The 300 tok/s target is reported separately for each workload.", font=SMALL, fill=INK)
    draw.text((85, 1320), "Blue = serving/GPU layers   Gold = profile/evidence   Red = hypothesis/rejection gate   Green = orchestration and promoted knowledge", font=SMALL, fill=MUTED)
    draw.text((85, 1360), "Current candidate: scheduler output stop fast path + BF16/aligned state; measured host gap remains ~2.60 ms, so clean confirmation is required before promotion.", font=SMALL, fill=ORANGE)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
