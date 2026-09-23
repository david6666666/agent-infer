#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Render the profile-first deep iteration ledger as a reviewable PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 2400, 2020
BACKGROUND = "#f5f7fa"
PANEL = "#ffffff"
INK = "#253238"
MUTED = "#69777d"
GRID = "#d6dee2"
BLUE = "#4f75c9"
BLUE_LIGHT = "#eaf1ff"
GREEN = "#2e7d5b"
GREEN_LIGHT = "#e8f1e8"
ORANGE = "#b66a00"
ORANGE_LIGHT = "#fff4df"
RED = "#bd4b4b"
RED_LIGHT = "#fbe8e8"
PURPLE = "#7650a8"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


TITLE = _font(42, bold=True)
SUBTITLE = _font(22)
HEADING = _font(24, bold=True)
BODY = _font(18)
BODY_BOLD = _font(18, bold=True)
SMALL = _font(16)
SMALL_BOLD = _font(16, bold=True)
TINY = _font(14)
MICRO = _font(12)


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    face: ImageFont.ImageFont,
    fill: str = INK,
) -> None:
    draw.text(xy, value, font=face, fill=fill)


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = PANEL,
    outline: str = GRID,
) -> None:
    draw.rounded_rectangle(box, radius=16, fill=fill, outline=outline, width=2)


def _card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    detail: str,
    *,
    value_color: str = INK,
) -> None:
    _panel(draw, box)
    x, y, _, _ = box
    _text(draw, (x + 18, y + 14), label, SMALL_BOLD, MUTED)
    _text(draw, (x + 18, y + 44), value, HEADING, value_color)
    _text(draw, (x + 18, y + 82), detail, TINY, MUTED)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _short_rows(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = data["iterations"]
    short_control = next(row for row in rows if row["id"] == "D0")
    short_best = max(
        (
            row
            for row in rows
            if row["tasks"] == 2 and "invalid" not in str(row["decision"])
        ),
        key=lambda row: row["output_tok_s"],
    )
    return short_control, short_best


def _sustained_rows(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [row for row in data["iterations"] if row["tasks"] == 4]
    return min(rows, key=lambda row: row["output_tok_s"]), max(rows, key=lambda row: row["output_tok_s"])


def _draw_profile_panel(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    box = (60, 420, 1210, 1010)
    _panel(draw, box)
    _text(draw, (88, 446), "Profile evidence · where the time goes", HEADING)
    profile = data["profile"]
    rows = [
        ("GPU execute", float(profile["gpu_execute_mean_ms"]), "ms / iteration", BLUE),
        ("CPU gap", float(profile["cpu_gap_mean_ms"]), "ms / iteration", ORANGE),
        ("NVFP4 prefill GEMM", 86.205, "ms aggregate", BLUE),
        ("GDN qkvz", 55.129, "ms aggregate", PURPLE),
        ("NVFP4 decode GEMM", 43.416, "ms aggregate", BLUE),
        ("GDN chunked", 19.369, "ms aggregate", PURPLE),
        ("FP4 conversion", 16.920, "ms aggregate", ORANGE),
    ]
    max_value = max(value for _, value, _, _ in rows)
    chart_x, chart_y, chart_w = 330, 510, 760
    for index, (label, value, detail, color) in enumerate(rows):
        y = chart_y + index * 57
        _text(draw, (88, y + 5), label, SMALL, INK)
        bar_w = max(3, int(chart_w * value / max_value))
        draw.rounded_rectangle((chart_x, y, chart_x + bar_w, y + 28), radius=6, fill=color)
        _text(draw, (chart_x + chart_w + 18, y + 4), f"{value:.3f}  {detail}", SMALL, color)
    _text(draw, (88, 930), "CPU gap / GPU execute = 39.1%; host work is a first-class optimization layer.", SMALL_BOLD, ORANGE)
    _text(draw, (88, 963), "Kernel totals are aggregate over the profiled trace window, not per-token latency.", TINY, MUTED)


def _draw_callsite_panel(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    box = (1240, 420, 2340, 1010)
    _panel(draw, box)
    _text(draw, (1268, 446), "CPU bubble + operator decision", HEADING)
    callsites = data["profile"]["python_callsite_aggregate_ms"][:7]
    y = 505
    for row in callsites:
        name = row["name"].replace("model_runner.py:", "model_runner ").replace("scheduler.py:", "scheduler ")
        name = name.replace("attention/backends/utils.py:", "attention utils ").replace("buffer_utils.py:", "buffer ")
        name = name.replace("sampler.py:", "sampler ").replace("kv_cache_manager.py:", "KV cache ")
        _text(draw, (1268, y), name[:47], TINY, INK)
        _text(draw, (2135, y), f"{float(row['total_ms']):.3f} ms", TINY, ORANGE)
        y += 31
    draw.line((1268, 735, 2308, 735), fill=GRID, width=1)
    decisions = [
        ("BF16 state + aligned cache", "keep → sustained", GREEN),
        ("GDN FP8 qkvz / stages", "reject E2E", RED),
        ("GDN 128-thread postconv", "reject micro", RED),
        ("pinned copy pool", "reject gap", RED),
        ("fused metadata", "reject gap", RED),
        ("FP4 tactic peak", "invalid attribution", RED),
    ]
    _text(draw, (1268, 757), "Promotion ledger", SMALL_BOLD, MUTED)
    y = 792
    for label, decision, color in decisions:
        _text(draw, (1268, y), label, TINY, INK)
        _text(draw, (2135, y), decision, TINY, color)
        y += 31


def _draw_workload_chart(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    box = (60, 1045, 1010, 1900)
    _panel(draw, box)
    _text(draw, (88, 1070), "Target view · same target, two workload scopes", HEADING)
    _text(draw, (88, 1110), "300 tok/s line; short replay and sustained replay are ranked separately.", SMALL, MUTED)
    rows = [
        ("short control D0", 238.106, BLUE),
        ("short best D3", 250.491, GREEN),
        ("sustained control D10", 339.991, BLUE),
        ("sustained winner D11", 358.375, GREEN),
    ]
    chart_x, chart_y, chart_w = 300, 1190, 570
    max_value = 380.0
    target_x = chart_x + int(chart_w * 300 / max_value)
    draw.line((target_x, chart_y - 25, target_x, chart_y + len(rows) * 94), fill=RED, width=4)
    _text(draw, (target_x - 34, chart_y - 53), "300", SMALL_BOLD, RED)
    for index, (label, value, color) in enumerate(rows):
        y = chart_y + index * 94
        _text(draw, (88, y + 8), label, SMALL, INK)
        bar_w = int(chart_w * value / max_value)
        draw.rounded_rectangle((chart_x, y, chart_x + bar_w, y + 30), radius=6, fill=color)
        _text(draw, (chart_x + chart_w + 18, y + 7), f"{value:.1f} tok/s", SMALL_BOLD, color)
    draw.line((88, 1575, 940, 1575), fill=GRID, width=1)
    _text(draw, (88, 1610), "Interpretation", SMALL_BOLD, MUTED)
    interpretation = [
        "Short fixed replay: the best measured result is 250.5 tok/s, below target.",
        "Sustained 4-task replay: baseline already reaches 340.0 tok/s; D11 reaches 358.4.",
        "D11 is a current candidate, pending clean vLLM-only confirmation and GSM8K rerun.",
    ]
    for index, line in enumerate(interpretation):
        _text(draw, (88, 1650 + index * 34), "• " + line, TINY, INK)
    _text(draw, (88, 1775), "Target status: PASS for sustained workload / OPEN for short replay", SMALL_BOLD, ORANGE)
    _text(draw, (88, 1810), "Do not transfer the sustained result to a different concurrency or request mix.", TINY, MUTED)


def _draw_round_table(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    box = (1040, 1045, 2340, 1900)
    _panel(draw, box)
    _text(draw, (1068, 1070), "Deep iteration ledger · optimization point → measured effect", HEADING)
    headers = [("id", 60), ("layer", 170), ("change", 525), ("tok/s", 90), ("Δ", 75), ("decision", 300)]
    x0, y0 = 1070, 1122
    x = x0
    for label, width in headers:
        _text(draw, (x, y0), label, SMALL_BOLD, MUTED)
        x += width
    draw.line((x0, y0 + 28, 2310, y0 + 28), fill=GRID, width=1)
    rows = data["iterations"]
    base_by_tasks = {2: next(row["output_tok_s"] for row in rows if row["id"] == "D0"), 4: next(row["output_tok_s"] for row in rows if row["id"] == "D10")}
    for index, row in enumerate(rows):
        y = y0 + 44 + index * 56
        base = base_by_tasks[int(row["tasks"])]
        delta = (float(row["output_tok_s"]) / base - 1) * 100
        layer = str(row["layer"]).replace("hybrid model / ", "hybrid/")
        change = str(row["change"])
        if len(change) > 58:
            change = change[:55] + "..."
        decision = str(row["decision"])
        color = GREEN if row["id"] in {"D3", "D11"} else RED if "reject" in decision or "invalid" in decision else INK
        values = [row["id"], layer[:21], change, f"{float(row['output_tok_s']):.1f}", f"{delta:+.1f}%", decision[:34]]
        x = x0
        for value, (_, width) in zip(values, headers, strict=True):
            _text(draw, (x, y), str(value), MICRO, color if value == decision[:34] else INK)
            x += width
        draw.line((x0, y + 31, 2310, y + 31), fill="#edf1f3", width=1)
    _text(draw, (1068, 1848), "Every row is replay-valid unless the decision explicitly says invalid attribution; raw paths and gate data are in deep-evidence.json.", TINY, MUTED)


def render(data: dict[str, Any], output: Path) -> None:
    short_control, short_best = _short_rows(data)
    sustained_control, sustained_best = _sustained_rows(data)
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _text(draw, (60, 30), "Profile-first RSI dashboard · single B300 · Qwen3.8-27B NVFP4", TITLE)
    _text(draw, (62, 88), "Deep iteration after the 50-round sweep · GPU kernels + CPU bubble + replay/quality gates", SUBTITLE, MUTED)
    draw.rounded_rectangle((60, 135, 2340, 205), radius=12, fill=ORANGE_LIGHT, outline="#e4c58a", width=2)
    _text(draw, (86, 158), "300 tok/s: PASS on 4-task sustained replay; OPEN on the short 2-task replay. No single peak is promoted without profile attribution.", SMALL_BOLD, ORANGE)

    short_delta = (float(short_best["output_tok_s"]) / float(short_control["output_tok_s"]) - 1) * 100
    sustained_delta = (float(sustained_best["output_tok_s"]) / float(sustained_control["output_tok_s"]) - 1) * 100
    _card(draw, (60, 235, 410, 345), "SHORT CONTROL", f"{float(short_control['output_tok_s']):.1f} tok/s", "2 tasks · 36 requests")
    _card(draw, (430, 235, 780, 345), "SHORT BEST", f"{float(short_best['output_tok_s']):.1f} tok/s", f"D3 · {short_delta:+.2f}%", value_color=GREEN)
    _card(draw, (800, 235, 1150, 345), "SUSTAINED CONTROL", f"{float(sustained_control['output_tok_s']):.1f} tok/s", "4 tasks · 89 requests")
    _card(draw, (1170, 235, 1520, 345), "SUSTAINED WINNER", f"{float(sustained_best['output_tok_s']):.1f} tok/s", f"D11 · {sustained_delta:+.2f}%", value_color=GREEN)
    _card(draw, (1540, 235, 1890, 345), "CPU GAP", "2.598 ms", "39.1% of GPU execute", value_color=ORANGE)
    _card(draw, (1910, 235, 2340, 345), "GPU EXECUTE", "6.641 ms", "profile mean / iteration", value_color=BLUE)

    _draw_profile_panel(draw, data)
    _draw_callsite_panel(draw, data)
    _draw_workload_chart(draw, data)
    _draw_round_table(draw, data)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.evidence.read_text(encoding="utf-8"))
    render(data, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
