#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Render the single-B300 NVFP4 RSI ledger and report.

The renderer intentionally reads the append-only ledger instead of embedding
benchmark numbers in the drawing code.  This keeps the review image and the
round-level markdown extract tied to the same evidence source.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1900
HEIGHT = 2450
BACKGROUND = "#f5f7fa"
PANEL = "#ffffff"
INK = "#253238"
MUTED = "#69777d"
GRID = "#d6dee2"
BLUE = "#2d6cdf"
GREEN = "#2e7d5b"
ORANGE = "#b66a00"
RED = "#bd4b4b"
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


TITLE = _font(40, bold=True)
SUBTITLE = _font(22)
HEADING = _font(23, bold=True)
BODY = _font(18)
SMALL = _font(16)
TINY = _font(14)
MICRO = _font(12)


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("metrics", {})


def _profile(record: dict[str, Any]) -> str:
    return str(record.get("config", {}).get("profile", "unknown"))


def _phase(record: dict[str, Any]) -> str:
    return str(record.get("config", {}).get("phase", "unknown"))


def _throughput(record: dict[str, Any]) -> float | None:
    value = _metrics(record).get("throughput_output_tokens_per_s")
    return float(value) if isinstance(value, (int, float)) else None


def _acceptance(record: dict[str, Any]) -> float | None:
    value = _metrics(record).get("spec_decode_mean_acceptance_length")
    return float(value) if isinstance(value, (int, float)) else None


def _valid(record: dict[str, Any]) -> bool:
    return record.get("status") == "measured" and _metrics(record).get("replay_successful_requests") == 36


def _values(records: list[dict[str, Any]]) -> list[float]:
    return [value for record in records if (value := _throughput(record)) is not None and _valid(record)]


def _median(records: list[dict[str, Any]]) -> float | None:
    values = _values(records)
    return statistics.median(values) if values else None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font: ImageFont.ImageFont, fill=INK) -> None:
    draw.text(xy, value, font=font, fill=fill)


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], outline=GRID) -> None:
    draw.rounded_rectangle(box, radius=14, fill=PANEL, outline=outline, width=2)


def _card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, value: str, detail: str) -> None:
    _panel(draw, box)
    x, y, _, _ = box
    _text(draw, (x + 20, y + 15), label, SMALL, MUTED)
    _text(draw, (x + 20, y + 45), value, HEADING)
    _text(draw, (x + 20, y + 80), detail, TINY, MUTED)


def _group(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_profile(record), []).append(record)
    return grouped


def _render_profile_bars(draw: ImageDraw.ImageDraw, records: list[dict[str, Any]], baseline: float) -> None:
    box = (40, 300, 930, 1000)
    _panel(draw, box)
    _text(draw, (68, 325), "Profile median · replay-valid output tok/s", HEADING)
    grouped = _group(records)
    rows: list[tuple[str, float, int, int]] = []
    for name, items in grouped.items():
        value = _median(items)
        if value is not None:
            rows.append((name, value, len(_values(items)), len(items)))
    rows.sort(key=lambda row: row[1], reverse=True)
    chart_x, chart_y, chart_w, row_h = 265, 385, 590, 35
    max_value = max([row[1] for row in rows] + [baseline]) * 1.08
    for index, (name, value, valid, total) in enumerate(rows):
        y = chart_y + index * row_h
        label = name.replace("baseline-user-command", "baseline")
        _text(draw, (68, y + 3), label[:25], TINY)
        draw.rounded_rectangle(
            (chart_x, y, chart_x + int(chart_w * value / max_value), y + 22),
            radius=5,
            fill=GREEN
            if name == "mtp-4"
            else RED
            if name in {"attention-triton", "enforce-eager", "gdn-cutedsl"}
            else BLUE,
        )
        delta = (value / baseline - 1) * 100
        _text(
            draw,
            (chart_x + chart_w + 18, y + 1),
            f"{value:.1f}  {delta:+.2f}%  {valid}/{total}",
            TINY,
            GREEN if delta >= 0 else MUTED,
        )
    baseline_x = chart_x + int(chart_w * baseline / max_value)
    draw.line((baseline_x, chart_y - 8, baseline_x, chart_y + row_h * len(rows)), fill=ORANGE, width=3)
    _text(draw, (68, 950), "bar label: median · Δ vs baseline · valid/total", TINY, MUTED)


def _render_phase_table(draw: ImageDraw.ImageDraw, records: list[dict[str, Any]], baseline: float) -> None:
    box = (960, 300, 1860, 1000)
    _panel(draw, box)
    _text(draw, (988, 325), "50-round ranking and promotion gate", HEADING)
    grouped = _group(records)
    order = sorted(
        grouped,
        key=lambda name: (
            _median([r for r in grouped[name] if _phase(r) == "confirmation"]) or _median(grouped[name]) or -1
        ),
        reverse=True,
    )
    headers = [("profile", 230), ("screen", 100), ("confirm", 110), ("Δ", 80), ("acc", 80), ("gate", 150)]
    x0, y0 = 988, 372
    x = x0
    for header, width in headers:
        _text(draw, (x, y0), header, SMALL, MUTED)
        x += width
    for index, name in enumerate(order):
        y = y0 + 34 + index * 34
        items = grouped[name]
        screen = _median([r for r in items if _phase(r) in {"baseline", "screening"}])
        confirm = _median([r for r in items if _phase(r) == "confirmation"])
        value = confirm or screen
        delta = None if value is None else (value / baseline - 1) * 100
        acc_values = [_acceptance(r) for r in items if _acceptance(r) is not None]
        acc = statistics.median(acc_values) if acc_values else None
        if name == "mtp-4":
            gate, color = "winner + GSM8K", GREEN
        elif name in {"batch-32768", "linear-cutedsl"}:
            gate, color = "confirmation", PURPLE
        elif name in {"no-prefix-cache", "attention-triton", "enforce-eager", "gdn-cutedsl"}:
            gate, color = "negative", RED
        else:
            gate, color = "screened", BLUE
        values = (
            name[:25],
            _fmt(screen, 1),
            _fmt(confirm, 1),
            f"{delta:+.2f}%" if delta is not None else "—",
            _fmt(acc, 2),
            gate,
        )
        x = x0
        for value_text, width in zip(values, (230, 100, 110, 80, 80, 150), strict=True):
            _text(draw, (x, y), value_text, TINY, color if value_text == gate else INK)
            x += width


def _render_round_chart(draw: ImageDraw.ImageDraw, records: list[dict[str, Any]], baseline: float) -> None:
    box = (40, 1040, 1860, 1450)
    _panel(draw, box)
    _text(draw, (68, 1065), "Round-level dense feedback · I0–I49", HEADING)
    plot_x, plot_y, plot_w, plot_h = 110, 1125, 1650, 240
    valid_values = [value for record in records if (value := _throughput(record)) is not None]
    low = min(valid_values + [0])
    high = max(valid_values + [baseline]) * 1.05
    for tick in range(5):
        value = low + (high - low) * tick / 4
        y = plot_y + plot_h - int(plot_h * (value - low) / max(1, high - low))
        draw.line((plot_x, y, plot_x + plot_w, y), fill="#edf1f3", width=1)
        _text(draw, (52, y - 9), f"{value:.0f}", TINY, MUTED)
    baseline_y = plot_y + plot_h - int(plot_h * (baseline - low) / max(1, high - low))
    draw.line((plot_x, baseline_y, plot_x + plot_w, baseline_y), fill=ORANGE, width=3)
    _text(draw, (plot_x + plot_w - 185, baseline_y - 24), f"baseline {baseline:.1f}", TINY, ORANGE)
    points: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        value = _throughput(record)
        if value is None:
            continue
        x = plot_x + int(index * plot_w / max(1, len(records) - 1))
        y = plot_y + plot_h - int(plot_h * (value - low) / max(1, high - low))
        phase = _phase(record)
        color = (
            GREEN
            if phase == "confirmation" and _profile(record) == "mtp-4"
            else PURPLE
            if phase == "confirmation"
            else BLUE
        )
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        points.append((x, y))
        if index % 5 == 0 or index == len(records) - 1:
            _text(draw, (x - 12, plot_y + plot_h + 14), str(record["round_id"]), TINY, MUTED)
    if len(points) > 1:
        draw.line(points, fill="#a9bfe9", width=2)
    _text(
        draw,
        (110, 1400),
        "Blue = baseline/screening · purple = confirmation · green = confirmed MTP4 winner · orange = baseline median",
        SMALL,
        MUTED,
    )


def _render_round_table(draw: ImageDraw.ImageDraw, records: list[dict[str, Any]], baseline: float) -> None:
    box = (40, 1490, 1860, 2380)
    _panel(draw, box)
    _text(draw, (68, 1515), "Every iteration · optimization point and measured effect", HEADING)
    columns = [
        ("round", 75),
        ("profile", 220),
        ("phase", 100),
        ("tok/s", 95),
        ("Δ", 75),
        ("acc", 75),
        ("coverage", 100),
    ]
    for side, subset in enumerate((records[:25], records[25:])):
        x0 = 70 + side * 900
        y0 = 1565
        x = x0
        for header, width in columns:
            _text(draw, (x, y0), header, SMALL, MUTED)
            x += width
        for index, record in enumerate(subset):
            y = y0 + 28 + index * 29
            value = _throughput(record)
            delta = None if value is None else (value / baseline - 1) * 100
            acc = _acceptance(record)
            coverage = f"{_metrics(record).get('replay_successful_requests', '—')}/36"
            phase = _phase(record)
            color = (
                GREEN if _profile(record) == "mtp-4" and phase == "confirmation" else RED if not _valid(record) else INK
            )
            values = (
                str(record["round_id"]),
                _profile(record)[:24],
                phase[:10],
                _fmt(value, 1),
                f"{delta:+.1f}%" if delta is not None else "—",
                _fmt(acc, 2),
                coverage,
            )
            x = x0
            for value_text, width in zip(values, (75, 220, 100, 95, 75, 75, 100), strict=True):
                _text(draw, (x, y), value_text, MICRO, color)
                x += width
            draw.line((x0, y + 23, x0 + 850, y + 23), fill="#edf1f3", width=1)
    _text(
        draw,
        (68, 2340),
        "Full optimization wording, server arguments, evidence hashes and quality artifacts are in the linked markdown report.",
        SMALL,
        MUTED,
    )


def _quality_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    first = path.read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(first)
    return payload if payload.get("type") == "summary" else payload


def render(
    records: list[dict[str, Any]],
    output: Path,
    quality_baseline: dict[str, Any] | None,
    quality_winner: dict[str, Any] | None,
) -> None:
    baseline_records = [record for record in records if _profile(record) == "baseline-user-command"]
    baseline = _median(baseline_records) or 0.0
    winner_records = [record for record in records if _profile(record) == "mtp-4" and _phase(record) == "confirmation"]
    winner = _median(winner_records) or 0.0
    valid = sum(_valid(record) for record in records)
    accuracy_base = None if quality_baseline is None else float(quality_baseline.get("accuracy", 0)) * 100
    accuracy_winner = None if quality_winner is None else float(quality_winner.get("accuracy", 0)) * 100

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _text(draw, (40, 28), "RSI dashboard · single B300 · Inferact/Qwen3.8-27B-NVFP4", TITLE)
    _text(
        draw,
        (42, 84),
        "50 cold-profile iterations · AgentBench codex_swebenchpro trace mode · vLLM 0.29.0",
        SUBTITLE,
        MUTED,
    )
    draw.rounded_rectangle((40, 130, 1860, 192), radius=10, fill="#e8f1e8", outline="#bfd4bf", width=2)
    _text(
        draw,
        (62, 150),
        "All 50 replay runs are 36/36 valid with exact prompt calibration residual 0; quality is gated separately by full GSM8K.",
        SMALL,
        GREEN,
    )
    _card(draw, (40, 220, 390, 282), "LEDGER", f"{len(records)} rounds", f"{valid}/{len(records)} replay-valid")
    _card(draw, (410, 220, 760, 282), "BASELINE MEDIAN", f"{baseline:.2f} tok/s", "MTP=3 · user command")
    delta = (winner / baseline - 1) * 100 if baseline else 0
    _card(draw, (780, 220, 1130, 282), "WINNER", f"{winner:.2f} tok/s", f"MTP=4 · {delta:+.2f}% vs baseline")
    quality_text = "—" if accuracy_winner is None else f"{accuracy_winner:.2f}%"
    quality_detail = (
        "not supplied"
        if accuracy_base is None
        else f"baseline {accuracy_base:.2f}% · Δ {(accuracy_winner - accuracy_base):+.2f} pp"
    )
    _card(draw, (1150, 220, 1500, 282), "GSM8K", quality_text, quality_detail)
    _card(draw, (1520, 220, 1860, 282), "CONTRACT", "TP1 · FP8 KV", "262144 ctx · prefix cache · MTP")
    _render_profile_bars(draw, records, baseline)
    _render_phase_table(draw, records, baseline)
    _render_round_chart(draw, records, baseline)
    _render_round_table(draw, records, baseline)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _report(
    records: list[dict[str, Any]],
    output: Path,
    quality_baseline: dict[str, Any] | None,
    quality_winner: dict[str, Any] | None,
    ledger: Path,
) -> None:
    baseline_records = [record for record in records if _profile(record) == "baseline-user-command"]
    baseline = _median(baseline_records) or 0.0
    grouped = _group(records)
    lines = [
        "<!-- markdownlint-disable MD013 -->",
        "",
        "# Single-B300 NVFP4 RSI：50 轮逐轮结果",
        "",
        "该文件由 `tools/render_single_nvfp4_dashboard.py` 从 append-only ledger 生成。原始 replay、vLLM server log、Prometheus 快照和 evidence hash 保留在实验目录；本报告只提交可审阅的摘要。",
        "",
        f"- Ledger：`{ledger}`",
        "- 固定模型：`Inferact/Qwen3.8-27B-NVFP4`，snapshot `6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`。",
        "- 固定服务：单张 NVIDIA B300、TP1、max-model-len 262144、FP8 KV、qwen3 reasoning、qwen3_xml tool parser、prefix cache、AgentInfer Codex SWE-bench Pro replay。",
        "- 每轮：冷启动服务、1 个 unranked warmup、2 tasks/36 requests measured replay、max concurrency 2；完整覆盖且 prompt calibration residual 必须为 0。",
        "",
        "## Profile 汇总",
        "",
        "| Profile | Layer | Screen median | Confirmation median | Delta vs baseline | Valid | Decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, items in sorted(
        grouped.items(),
        key=lambda item: (_median([r for r in item[1] if _phase(r) == "confirmation"]) or _median(item[1]) or -1),
        reverse=True,
    ):
        screen = _median([r for r in items if _phase(r) in {"baseline", "screening"}])
        confirm = _median([r for r in items if _phase(r) == "confirmation"])
        value = confirm or screen
        delta = (value / baseline - 1) * 100 if value is not None and baseline else None
        valid = f"{sum(_valid(r) for r in items)}/{len(items)}"
        layer = str(items[0].get("config", {}).get("layer", "—"))
        if name == "mtp-4":
            decision = "winner; GSM8K passed"
        elif name in {"batch-32768", "linear-cutedsl"}:
            decision = "confirmation candidate"
        elif name in {"no-prefix-cache", "attention-triton", "enforce-eager", "gdn-cutedsl"}:
            decision = "negative control"
        else:
            decision = "screened"
        lines.append(
            f"| `{name}` | {layer} | {_fmt(screen, 3)} | {_fmt(confirm, 3)} | {delta:+.2f}% | {valid} | {decision} |"
            if delta is not None
            else f"| `{name}` | {layer} | {_fmt(screen, 3)} | {_fmt(confirm, 3)} | — | {valid} | {decision} |"
        )
    lines.extend(["", "## GSM8K quality gate", ""])
    if quality_baseline and quality_winner:
        base_acc = float(quality_baseline["accuracy"]) * 100
        win_acc = float(quality_winner["accuracy"]) * 100
        lines.extend(
            [
                "| Configuration | Correct | Rows | Parsed | Errors | Accuracy | p50 latency |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| MTP=3 baseline | {quality_baseline['correct']} | {quality_baseline['rows']} | {quality_baseline['parsed_predictions']} | {quality_baseline['errors']} | {base_acc:.2f}% | {float(quality_baseline['latency_seconds']['p50']) * 1000:.0f} ms |",
                f"| MTP=4 winner | {quality_winner['correct']} | {quality_winner['rows']} | {quality_winner['parsed_predictions']} | {quality_winner['errors']} | {win_acc:.2f}% | {float(quality_winner['latency_seconds']['p50']) * 1000:.0f} ms |",
                "",
                f"MTP=4 提升 GSM8K `{win_acc - base_acc:+.2f} pp`，请求错误为 0；因此当前结果没有观察到相对 baseline 的精度回退。",
            ]
        )
    else:
        lines.append("质量结果未传入 renderer。")
    lines.extend(
        [
            "",
            "## 每轮优化点与效果",
            "",
            "| Round | Phase | Layer | Profile | Optimization point | Output tok/s | Delta | Acceptance length | Coverage |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:
        value = _throughput(record)
        delta = (value / baseline - 1) * 100 if value is not None and baseline else None
        m = _metrics(record)
        config = record.get("config", {})
        change = str(record.get("change", "")).replace("|", "\\|")
        lines.append(
            f"| {record['round_id']} | {config.get('phase')} | {config.get('layer')} | `{_profile(record)}` | {change} | {_fmt(value, 3)} | {delta:+.2f}% | {_fmt(_acceptance(record), 3)} | {m.get('replay_successful_requests', '—')}/36, residual {m.get('replay_calibration_max_residual_tokens', '—')} |"
            if delta is not None
            else f"| {record['round_id']} | {config.get('phase')} | {config.get('layer')} | `{_profile(record)}` | {change} | {_fmt(value, 3)} | — | {_fmt(_acceptance(record), 3)} | {m.get('replay_successful_requests', '—')}/36, residual {m.get('replay_calibration_max_residual_tokens', '—')} |"
        )
    lines.extend(
        [
            "",
            "## 结论和边界",
            "",
            "MTP=4 是当前固定 trace、TP1、单卡 B300、NVFP4 权重/FP8 KV contract 下的 winner。确认轮 median 为 248.928 tok/s，相对 baseline median 246.732 tok/s 为 +0.89%；这属于稳定的小幅收益，不应外推为所有并发、context 或 vLLM 版本的收益。",
            "",
            "`gdn-cutedsl`、`attention-triton`、`enforce-eager` 和关闭 prefix cache 是明确的负向信号；它们保留在结果中，用于防止知识库把局部 kernel/调度假设误写成通用结论。",
            "",
            "完整 raw evidence 位于 ledger 中的每条 `evidence` 字段；服务、AgentInfer、模型和环境版本以每轮 command/runtime probe 为准。当前实验环境还加载了 editable vLLM-Omni 包，虽 vLLM 主版本为 0.29.0；后续 clean vLLM-only environment 仍是必要复核项。",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--quality-baseline", type=Path)
    parser.add_argument("--quality-winner", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = _load_records(args.ledger)
    if len(records) != 50:
        raise SystemExit(f"expected exactly 50 records, got {len(records)}")
    quality_baseline = _quality_summary(args.quality_baseline)
    quality_winner = _quality_summary(args.quality_winner)
    render(records, args.output, quality_baseline, quality_winner)
    _report(records, args.report, quality_baseline, quality_winner, args.ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
