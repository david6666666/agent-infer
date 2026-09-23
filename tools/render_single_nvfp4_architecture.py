#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Render a dependency-free PNG companion to the single-B300 Mermaid diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 2200, 1420
BG = "#f5f7fa"
INK = "#253238"
MUTED = "#69777d"
GRID = "#d6dee2"
BLUE = "#eaf1ff"
BLUE_EDGE = "#6b8fd6"
GREEN = "#e8f1e8"
GREEN_EDGE = "#5d9b6f"
GOLD = "#fff4df"
GOLD_EDGE = "#c58b2c"
RED = "#fbe8e8"
RED_EDGE = "#c46a6a"
WHITE = "#ffffff"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


TITLE = font(38, True)
SUBTITLE = font(20)
BOX = font(20)
BOX_BOLD = font(20, True)
SMALL = font(17)
TINY = font(15)


def center_text(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str], *, bold_first: bool = False
) -> None:
    x1, y1, x2, y2 = box
    heights = []
    for index, _ in enumerate(lines):
        heights.append((BOX_BOLD if bold_first and index == 0 else BOX).getbbox("Ag")[3])
    total = sum(heights) + (len(lines) - 1) * 5
    y = y1 + (y2 - y1 - total) // 2
    for index, value in enumerate(lines):
        current = BOX_BOLD if bold_first and index == 0 else BOX
        bbox = draw.textbbox((0, 0), value, font=current)
        x = x1 + (x2 - x1 - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), value, font=current, fill=INK)
        y += heights[index] + 5


def node(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str], fill: str, outline: str) -> None:
    draw.rounded_rectangle(box, radius=14, fill=fill, outline=outline, width=3)
    center_text(draw, box, lines)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = INK,
    dashed: bool = False,
    width: int = 4,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if dashed:
        steps = max(abs(x2 - x1), abs(y2 - y1)) // 16
        for index in range(0, steps, 2):
            a = index / max(1, steps)
            b = min(1, (index + 1) / max(1, steps))
            draw.line(
                (x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b),
                fill=color,
                width=width,
            )
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    size = 13
    left = (x2 - size * math.cos(angle - 0.45), y2 - size * math.sin(angle - 0.45))
    right = (x2 - size * math.cos(angle + 0.45), y2 - size * math.sin(angle + 0.45))
    draw.polygon([(x2, y2), left, right], fill=color)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=MUTED) -> None:
    draw.text(xy, text, font=TINY, fill=fill)


def render(output: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), "Single-B300 NVFP4 RSI architecture", font=TITLE, fill=INK)
    draw.text(
        (50, 82),
        "Layer-local hypotheses → exact replay evidence → independent quality gate → knowledge-base feedback",
        font=SUBTITLE,
        fill=MUTED,
    )

    # Input and serving path.
    nodes = {
        "codex": (50, 190, 300, 300),
        "trace": (360, 190, 650, 300),
        "replay": (710, 190, 1030, 300),
        "api": (1090, 190, 1400, 300),
        "gpu": (1620, 190, 2110, 300),
    }
    node(draw, nodes["codex"], ["Codex CLI", "operator / loop"], GREEN, GREEN_EDGE)
    node(draw, nodes["trace"], ["AgentBench trace", "codex_swebenchpro"], GREEN, GREEN_EDGE)
    node(draw, nodes["replay"], ["AgentInfer replay", "36 requests · exact input"], GREEN, GREEN_EDGE)
    node(draw, nodes["api"], ["vLLM OpenAI API", "reasoning + tools"], GREEN, GREEN_EDGE)
    node(draw, nodes["gpu"], ["1 × NVIDIA B300", "SM103 · CUDA 13.0"], BLUE, BLUE_EDGE)
    for left, right in (("codex", "trace"), ("trace", "replay"), ("replay", "api")):
        arrow(draw, (nodes[left][2], 245), (nodes[right][0], 245), color=GREEN_EDGE)

    # Main layered box.
    layer_box = (1060, 350, 2160, 790)
    draw.rounded_rectangle(layer_box, radius=18, fill=WHITE, outline=BLUE_EDGE, width=3)
    draw.text((1090, 370), "vLLM serving layers · one cold profile per iteration", font=BOX_BOLD, fill=INK)
    layer_nodes = [
        ((1100, 430, 1370, 550), ["Engine runtime", "API · tokenizer · metrics"]),
        ((1420, 430, 1690, 550), ["Scheduler", "chunked prefill · async", "CUDA graphs"]),
        ((1740, 430, 2010, 550), ["KV / cache", "prefix cache · FP8 KV", "paged layout"]),
        ((1100, 615, 1370, 735), ["Hybrid Qwen3.8", "full attention +", "GDN / linear attention"]),
        ((1420, 615, 1690, 735), ["Speculative", "MTP draft · verify", "acceptance telemetry"]),
        ((1740, 615, 2010, 735), ["Kernel dispatch", "FlashInfer / TRTLLM", "NVFP4 · CuTeDSL"]),
    ]
    for box, lines in layer_nodes:
        node(draw, box, lines, BLUE, BLUE_EDGE)
    arrow(draw, (1235, 550), (1235, 615), color=BLUE_EDGE)
    arrow(draw, (1555, 550), (1555, 615), color=BLUE_EDGE)
    arrow(draw, (1875, 550), (1875, 615), color=BLUE_EDGE)
    arrow(draw, (1370, 490), (1420, 490), color=BLUE_EDGE)
    arrow(draw, (1690, 490), (1740, 490), color=BLUE_EDGE)
    arrow(draw, (1370, 675), (1420, 675), color=BLUE_EDGE)
    arrow(draw, (1690, 675), (1740, 675), color=BLUE_EDGE)
    arrow(draw, (1555, 790), (1860, 300), color=BLUE_EDGE)
    label(draw, (1785, 785), "steady-state GPU execution", BLUE_EDGE)

    # Feedback/evaluation path.
    evidence = (50, 900, 480, 1040)
    gate = (560, 900, 900, 1040)
    ranking = (980, 900, 1320, 1040)
    quality = (1400, 900, 1740, 1040)
    knowledge = (1820, 900, 2150, 1040)
    node(draw, evidence, ["Raw evidence", "logs · Prometheus", "hashes · probes"], GOLD, GOLD_EDGE)
    node(draw, gate, ["Coverage gate", "36/36? residual 0?"], RED, RED_EDGE)
    node(draw, ranking, ["Dense feedback", "median · range", "screen → confirmation"], GOLD, GOLD_EDGE)
    node(draw, quality, ["GSM8K gate", "1319 rows · temp 0", "accuracy + errors"], GOLD, GOLD_EDGE)
    node(draw, knowledge, ["Knowledge base", "dashboard · architecture", "next hypothesis"], GREEN, GREEN_EDGE)
    arrow(draw, (1250, 790), (265, 900), color=GOLD_EDGE, dashed=True)
    arrow(draw, (480, 970), (560, 970), color=GOLD_EDGE)
    arrow(draw, (900, 970), (980, 970), color=GOLD_EDGE)
    arrow(draw, (1320, 970), (1400, 970), color=GOLD_EDGE)
    arrow(draw, (1740, 970), (1820, 970), color=GREEN_EDGE)
    label(draw, (495, 930), "invalid → retain failure", RED_EDGE)

    # Boundary/contract legend.
    legend = (50, 1140, 2150, 1330)
    draw.rounded_rectangle(legend, radius=14, fill=WHITE, outline=GRID, width=2)
    draw.text((80, 1165), "Frozen contract for this artifact", font=BOX_BOLD, fill=INK)
    draw.text(
        (80, 1210),
        "TP1 · max-model-len 262144 · FP8 KV · qwen3/qwen3_xml · prefix cache · MTP candidate · B300 · vLLM 0.29.0",
        font=SMALL,
        fill=INK,
    )
    draw.text(
        (80, 1250),
        "Only one layer-local serving change per profile; all raw evidence is append-only. Quality promotion is separate from throughput ranking.",
        font=SMALL,
        fill=MUTED,
    )
    draw.text(
        (80, 1290),
        "Blue = vLLM/model/GPU layers   Green = control, orchestration and promoted knowledge   Gold = evidence/evaluation   Red = rejection gate",
        font=SMALL,
        fill=MUTED,
    )
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
