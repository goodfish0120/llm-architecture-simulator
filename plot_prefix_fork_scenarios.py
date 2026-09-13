from __future__ import annotations

import csv
import html
import math
import re
from collections import defaultdict
from pathlib import Path


WIDTH = 1200
HEIGHT = 720
LEFT = 95
RIGHT = 30
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PANEL_HEIGHT = 220
COLORS = ("#64748b", "#2563eb", "#059669", "#dc2626")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def x_position(agent_count: int, maximum_agent_count: int) -> float:
    return LEFT + math.log10(agent_count) / math.log10(maximum_agent_count) * PLOT_WIDTH


def write_model_chart(path: Path, model: str, rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)
    maximum_agents = max(int(row["agent_count"]) for row in rows)
    maximum_throughput = max(float(row["tokens_per_second"]) for row in rows) * 1.08
    maximum_latency = max(float(row["p95_token_latency_ms"]) for row in rows) * 1.08
    markup = []
    legend = []
    for color, (scenario, scenario_rows) in zip(COLORS, sorted(grouped.items())):
        scenario_rows.sort(key=lambda row: int(row["agent_count"]))
        legend_y = 68 + len(legend) * 20
        legend.append(
            f'<line x1="{LEFT}" y1="{legend_y}" x2="{LEFT + 26}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
            f'<text x="{LEFT + 36}" y="{legend_y + 4}" class="tick">{html.escape(scenario)}</text>'
        )
        for panel_top, field, y_max in (
            (150, "tokens_per_second", maximum_throughput),
            (445, "p95_token_latency_ms", maximum_latency),
        ):
            points = []
            for row in scenario_rows:
                x = x_position(int(row["agent_count"]), maximum_agents)
                y = panel_top + PANEL_HEIGHT * (
                    1 - float(row[field]) / y_max
                )
                points.append(f"{x:.1f},{y:.1f}")
            markup.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
            )
    grid = []
    for panel_top, y_max in ((150, maximum_throughput), (445, maximum_latency)):
        for tick in range(5):
            fraction = tick / 4
            y = panel_top + PANEL_HEIGHT * (1 - fraction)
            grid.append(
                f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" class="grid"/>'
                f'<text x="{LEFT - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{y_max * fraction:.0f}</text>'
            )
    x_ticks = []
    for agent_count in (1, 10, 100, 1000, maximum_agents):
        if agent_count > maximum_agents:
            continue
        x = x_position(agent_count, maximum_agents)
        x_ticks.append(
            f'<text x="{x:.1f}" y="696" text-anchor="middle" class="tick">{agent_count}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(model)} prefix-fork pressure curves</title>
<desc id="desc">Seven-node throughput and P95 token latency under independent and shared-prefix agent workloads.</desc>
<style>svg{{background:#fff;color:#172033;font-family:Inter,Segoe UI,sans-serif}}.title{{font-size:24px;font-weight:600;fill:currentColor}}.subtitle,.tick{{font-size:13px;fill:#526174}}.panel{{font-size:15px;font-weight:600;fill:currentColor}}.grid{{stroke:#d8dee8;stroke-width:1}}</style>
<text x="{LEFT}" y="34" class="title">{html.escape(model)} · seven-node prefix-fork scenarios</text>
<text x="{LEFT}" y="54" class="subtitle">100K context: 0–20K system prefix, 20K–60K family context, 60K–100K unique suffix</text>
{"".join(legend)}
<text x="{LEFT}" y="137" class="panel">Aggregate throughput (tokens / second)</text>
<text x="{LEFT}" y="432" class="panel">P95 single-token latency (milliseconds)</text>
{"".join(grid)}
<rect x="{LEFT}" y="150" width="{PLOT_WIDTH}" height="{PANEL_HEIGHT}" fill="none" stroke="#9aa7b7"/>
<rect x="{LEFT}" y="445" width="{PLOT_WIDTH}" height="{PANEL_HEIGHT}" fill="none" stroke="#9aa7b7"/>
{"".join(markup)}
{"".join(x_ticks)}
<text x="{LEFT + PLOT_WIDTH / 2:.1f}" y="716" text-anchor="middle" class="panel">Continuously active agents (log scale)</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    source = Path("results/prefix_fork_scenarios.csv")
    output_directory = Path("results/prefix_fork_scenarios")
    output_directory.mkdir(parents=True, exist_ok=True)
    rows_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    with source.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows_by_model[row["model"]].append(row)
    for model, rows in rows_by_model.items():
        write_model_chart(output_directory / f"{slug(model)}.svg", model, rows)


if __name__ == "__main__":
    main()
