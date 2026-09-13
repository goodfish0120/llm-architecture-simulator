from __future__ import annotations

import html
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


WIDTH = 1200
HEIGHT = 900
LEFT = 92
RIGHT = 32
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PANEL_HEIGHT = 190
PANEL_TOPS = (105, 365, 625)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _polyline(
    values: list[tuple[float, float]],
    x_max: float,
    y_max: float,
    panel_top: float,
) -> str:
    points = []
    for x, y in values:
        px = _log_x_position(x, x_max)
        py = panel_top + PANEL_HEIGHT - (y / y_max) * PANEL_HEIGHT
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points)


def _log_x_position(value: float, maximum: float) -> float:
    if maximum <= 1:
        return LEFT + PLOT_WIDTH / 2
    return LEFT + (math.log10(value) / math.log10(maximum)) * PLOT_WIDTH


def _panel(
    title: str,
    y_unit: str,
    values: list[tuple[float, float]],
    x_max: float,
    panel_top: float,
) -> str:
    y_max = max((value for _, value in values), default=1.0) * 1.08 or 1.0
    lines = [
        f'<text x="{LEFT}" y="{panel_top - 22}" class="panel-title">{html.escape(title)}</text>',
        f'<rect x="{LEFT}" y="{panel_top}" width="{PLOT_WIDTH}" height="{PANEL_HEIGHT}" class="frame"/>',
    ]
    for tick in range(5):
        fraction = tick / 4
        y = panel_top + PANEL_HEIGHT * (1 - fraction)
        value = y_max * fraction
        lines.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{LEFT - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.1f}</text>')
    lines.append(
        f'<polyline points="{_polyline(values, x_max, y_max, panel_top)}" class="series"/>'
    )
    for x, y in values:
        px = _log_x_position(x, x_max)
        py = panel_top + PANEL_HEIGHT - (y / y_max) * PANEL_HEIGHT
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" class="point"><title>{x:.0f} agents: {y:.3f} {html.escape(y_unit)}</title></circle>')
    lines.append(
        f'<text x="24" y="{panel_top + PANEL_HEIGHT / 2:.1f}" transform="rotate(-90 24 {panel_top + PANEL_HEIGHT / 2:.1f})" text-anchor="middle" class="axis-title">{html.escape(y_unit)}</text>'
    )
    return "\n".join(lines)


def _write_one_chart(path: Path, model: str, node_count: int, rows: list[object]) -> None:
    rows = sorted(rows, key=lambda row: row.agent_count)
    x_max = max(row.agent_count for row in rows)
    throughput = [(row.agent_count, row.tokens_per_second) for row in rows]
    latency = [(row.agent_count, row.mean_token_latency_ms) for row in rows]
    batch = [(row.agent_count, row.mean_expert_batch_size) for row in rows]
    x_ticks = sorted({1, x_max, *[row.agent_count for row in rows[:: max(1, len(rows) // 6)]]})
    tick_markup = []
    for value in x_ticks:
        x = _log_x_position(value, x_max)
        tick_markup.append(f'<text x="{x:.1f}" y="844" text-anchor="middle" class="tick">{value}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(model)}, {node_count} nodes: agent pressure curve</title>
<desc id="desc">Throughput, mean token latency, and expert batch size as active agent count rises to the memory limit.</desc>
<style>
  :root {{ color-scheme: light dark; }}
  svg {{ background: #ffffff; color: #172033; font-family: Inter, Segoe UI, sans-serif; }}
  .title {{ font-size: 24px; font-weight: 600; fill: currentColor; }}
  .subtitle, .tick {{ font-size: 13px; fill: #526174; }}
  .panel-title, .axis-title {{ font-size: 15px; font-weight: 600; fill: currentColor; }}
  .frame {{ fill: none; stroke: #9aa7b7; stroke-width: 1; }}
  .grid {{ stroke: #d8dee8; stroke-width: 1; }}
  .series {{ fill: none; stroke: #2563eb; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
  .point {{ fill: #2563eb; }}
</style>
<text x="{LEFT}" y="42" class="title">{html.escape(model)} · {node_count} node{'s' if node_count != 1 else ''}</text>
<text x="{LEFT}" y="68" class="subtitle">100K context tokens · 80% resident KV · active agents swept independently to capacity</text>
{_panel("Aggregate decode throughput", "tokens / second", throughput, x_max, PANEL_TOPS[0])}
{_panel("Mean single-token latency", "milliseconds", latency, x_max, PANEL_TOPS[1])}
{_panel("Mean executed expert batch", "expert branches / kernel", batch, x_max, PANEL_TOPS[2])}
{''.join(tick_markup)}
<text x="{LEFT + PLOT_WIDTH / 2:.1f}" y="878" text-anchor="middle" class="axis-title">Continuously active agents (log scale)</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def write_pressure_curve_charts(output_directory: Path, rows: Iterable[object]) -> None:
    chart_directory = output_directory / "pressure_curves"
    chart_directory.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int], list[object]] = defaultdict(list)
    for row in rows:
        grouped[(row.model, row.node_count)].append(row)

    _write_scaling_summary(
        output_directory / "mac_studio_m5_ultra_scaling.svg",
        grouped,
    )

    index_lines = ["# Per-configuration agent pressure curves", ""]
    for (model, node_count), group in sorted(grouped.items()):
        filename = f"{_slug(model)}-{node_count}-nodes.svg"
        _write_one_chart(chart_directory / filename, model, node_count, group)
        index_lines.extend(
            [
                f"## {model} — {node_count} node{'s' if node_count != 1 else ''}",
                "",
                f"![{model}, {node_count} nodes](./{filename})",
                "",
            ]
        )
    (chart_directory / "README.md").write_text("\n".join(index_lines), encoding="utf-8")


def _write_scaling_summary(
    path: Path,
    grouped: dict[tuple[str, int], list[object]],
) -> None:
    peak_by_model: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (model, node_count), rows in grouped.items():
        peak_by_model[model].append(
            (node_count, max(row.tokens_per_second for row in rows))
        )
    y_max = max(value for values in peak_by_model.values() for _, value in values) * 1.08
    colors = ("#2563eb", "#dc2626", "#059669")
    lines = []
    legend = []
    for color, (model, values) in zip(colors, sorted(peak_by_model.items())):
        values = sorted(values)
        points = []
        for node_count, throughput in values:
            x = LEFT + ((node_count - 1) / 6) * PLOT_WIDTH
            y = 100 + PANEL_HEIGHT * 2 - (throughput / y_max) * PANEL_HEIGHT * 2
            points.append(f"{x:.1f},{y:.1f}")
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"><title>{html.escape(model)}, {node_count} nodes: {throughput:.1f} tokens / second</title></circle>'
            )
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
        )
        legend.append(
            f'<line x1="{LEFT}" y1="{65 + len(legend) * 20}" x2="{LEFT + 28}" y2="{65 + len(legend) * 20}" stroke="{color}" stroke-width="3"/><text x="{LEFT + 38}" y="{69 + len(legend) * 20}" class="tick">{html.escape(model)}</text>'
        )
    grid = []
    for tick in range(6):
        fraction = tick / 5
        y = 100 + PANEL_HEIGHT * 2 * (1 - fraction)
        grid.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" class="grid"/>')
        grid.append(f'<text x="{LEFT - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{y_max * fraction:.0f}</text>')
    x_ticks = []
    for node_count in range(1, 8):
        x = LEFT + ((node_count - 1) / 6) * PLOT_WIDTH
        x_ticks.append(f'<text x="{x:.1f}" y="505" text-anchor="middle" class="tick">{node_count}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="550" viewBox="0 0 {WIDTH} 550" role="img" aria-labelledby="title desc">
<title id="title">Peak throughput after independent agent pressure sweeps</title>
<desc id="desc">Best aggregate decode throughput for each model and node count after independently increasing active agents to capacity.</desc>
<style>svg {{ background:#fff;color:#172033;font-family:Inter,Segoe UI,sans-serif }} .title {{ font-size:24px;font-weight:600;fill:currentColor }} .tick {{ font-size:13px;fill:#526174 }} .axis-title {{ font-size:15px;font-weight:600;fill:currentColor }} .grid {{ stroke:#d8dee8;stroke-width:1 }}</style>
<text x="{LEFT}" y="35" class="title">Peak throughput after independent agent pressure sweeps</text>
{"".join(legend)}
{"".join(grid)}
<rect x="{LEFT}" y="100" width="{PLOT_WIDTH}" height="{PANEL_HEIGHT * 2}" fill="none" stroke="#9aa7b7"/>
{"".join(lines)}
{"".join(x_ticks)}
<text x="{LEFT + PLOT_WIDTH / 2:.1f}" y="538" text-anchor="middle" class="axis-title">Mac Studio M5 Ultra nodes</text>
<text x="24" y="290" transform="rotate(-90 24 290)" text-anchor="middle" class="axis-title">Peak aggregate tokens / second</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")
