"""Regenerate the figures in docs/images/.

    uv run --all-extras python scripts/make_figures.py                  # schematics
    uv run --all-extras python scripts/make_figures.py --data <ditec-wdn dir>   # + network topology

The topology plot reads only the small static DiTEC-WDN files (pipe_diameter, junction_elevation,
reservoir_base_head); node positions are a graph layout, not geographic coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
SUPPLY, RETURN, ACCENT, GREY = "#1f77b4", "#d95f02", "#c0392b", "#555555"


def _box(ax, x, y, w, h, text, fc="#f4f4f4", ec=GREY, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else None)


def _arrow(ax, p, q, color, lw=2.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw))


def _marker(ax, x, y, n):
    ax.add_patch(Circle((x, y), 0.16, fc=ACCENT, ec="white", lw=1.5, zorder=5))
    ax.text(x, y, str(n), color="white", ha="center", va="center", fontsize=9, weight="bold", zorder=6)


def _ets(ax, x, y_top, y_bot, label=True):
    """One consumer: supply node -> filter -> FCV -> heat exchanger -> return node."""
    ax.plot([x, x], [y_top, y_bot], color=GREY, lw=1.2, zorder=1)
    ax.add_patch(Circle((x, y_top), 0.07, fc=SUPPLY, ec="none", zorder=3))
    ax.add_patch(Circle((x, y_bot), 0.07, fc=RETURN, ec="none", zorder=3))
    h = (y_top - y_bot) / 3
    _box(ax, x - 0.32, y_top - 0.95 * h, 0.64, 0.6 * h, "filter*", fs=7.5)
    _box(ax, x - 0.32, y_top - 1.95 * h, 0.64, 0.6 * h, "FCV", fs=7.5, fc="#fdebd0")
    _box(ax, x - 0.32, y_top - 2.95 * h, 0.64, 0.6 * h, "HX", fs=7.5, fc="#d6eaf8")
    if label:
        ax.text(x + 0.45, (y_top + y_bot) / 2, "ETS", ha="left", fontsize=8, color=GREY, rotation=90, va="center")


def schematic() -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_title("District cooling network model (dcngen): closed supply/return loop", fontsize=13)

    ys, yr = 3.9, 1.3
    _box(ax, 0.3, 1.3, 1.7, 2.6, "", fc="#eef6ee")
    ax.text(1.15, 3.65, "Plant", ha="center", weight="bold", fontsize=10)
    ax.add_patch(Circle((1.15, 2.85), 0.32, fc="white", ec=GREY, lw=1.5))
    ax.text(1.15, 2.85, "P", ha="center", va="center", weight="bold")
    ax.text(1.15, 2.3, "pump + reservoir\n(supply setpoint 6 °C)", ha="center", va="center", fontsize=8)
    ax.text(1.15, 1.6, "return stub", ha="center", fontsize=8)

    ax.plot([2.0, 10.4], [ys, ys], color=SUPPLY, lw=4, solid_capstyle="round")
    ax.plot([10.4, 2.0], [yr, yr], color=RETURN, lw=4, solid_capstyle="round")
    ax.plot([2.0, 2.0], [yr, 2.85], color=RETURN, lw=4)
    ax.text(6.2, ys + 0.22, "supply pipes  (cold water, ~6 °C)", color=SUPPLY, ha="center", fontsize=10)
    ax.text(6.2, yr - 0.4, "return pipes  (mirrored, reversed direction, ~13 °C)", color=RETURN, ha="center", fontsize=10)
    for x in (4.0, 6.2):
        _arrow(ax, (x - 0.5, ys), (x + 0.5, ys), SUPPLY)
        _arrow(ax, (x + 0.5, yr), (x - 0.5, yr), RETURN)

    for x in (3.4, 5.6, 8.2):
        _ets(ax, x, ys, yr)
    ax.text(9.35, 2.6, "…", fontsize=22, color=GREY, ha="center")
    ax.text(9.75, 2.6, "one ETS per\njunction with\ndemand", fontsize=8, color=GREY, ha="left", va="center")

    ax.text(5.5, 0.35,
            "Every DiTEC junction j is mirrored into j_s (supply) and j_r (return); each pipe p becomes p_s and p_r.\n"
            "An ideal flow-control valve (FCV) carries each consumer's primary flow; a heat exchanger (HX) sets the return temperature.\n"
            "* the filter exists only in the clogged-filter experiment (a short minor-loss link ahead of the FCV).",
            ha="center", fontsize=8.5, color=GREY)
    path = OUT / "dcn_schematic.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def fault_points() -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_title("Where each FSED is injected", fontsize=13)

    ys, yr, x = 4.3, 0.8, 2.2
    ax.plot([0.4, 3.8], [ys, ys], color=SUPPLY, lw=4)
    ax.plot([3.8, 0.4], [yr, yr], color=RETURN, lw=4)
    _ets(ax, x, ys, yr, label=False)
    h = (ys - yr) / 3
    fy = ys - 0.65 * h
    ay = ys - 1.65 * h
    hy = ys - 2.65 * h
    _marker(ax, x + 0.62, fy, 4)
    _marker(ax, x + 0.62, ay, 2)
    _marker(ax, x + 1.02, ay, 3)
    _marker(ax, x + 0.62, hy, 5)
    _marker(ax, 0.9, ys + 0.5, 1)
    ax.text(1.15, ys + 0.5, "plant pump head", fontsize=8, va="center")

    rows = [
        ("1", "Excess control margin", "head_multiplier", "plant pump curve raised to h0 × m"),
        ("2", "Stuck actuator", "onset_fraction", "FCV frozen at its pre-fault setting"),
        ("3", "Poor balancing", "flow_multiplier", "FCV tracks m × requested flow"),
        ("4", "Clogged filter", "filter_K", "minor loss K in a link ahead of the FCV"),
        ("5", "HX fouling", "ua_multiplier", "installed UA scaled by m (<1)"),
    ]
    ax.text(4.6, 4.7, "#", weight="bold", fontsize=9)
    ax.text(5.0, 4.7, "FSED", weight="bold", fontsize=9)
    ax.text(8.2, 4.7, "severity parameter", weight="bold", fontsize=9)
    for i, (n, name, par, how) in enumerate(rows):
        y = 4.1 - i * 0.82
        _marker(ax, 4.7, y, int(n))
        ax.text(5.0, y + 0.08, name, fontsize=9.5, weight="bold", va="center")
        ax.text(5.0, y - 0.28, how, fontsize=8, va="center", color=GREY)
        ax.text(8.2, y + 0.08, par, fontsize=9, va="center", family="monospace")
    ax.add_patch(Rectangle((4.4, 0.2), 6.6, 4.35, fill=False, ec="#cccccc", lw=1))
    ax.text(2.2, 0.15, "Each pair simulates a healthy twin and a faulted twin\nwith identical loads, seed and network.",
            ha="center", fontsize=8.5, color=GREY)
    path = OUT / "fault_injection_points.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def workflow() -> Path:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    ax.set_title("Workflow", fontsize=13)
    steps = [
        ("0  Setup", "dcn setup-dcngen\ndcn download-data", "#f4f4f4"),
        ("1  Validate", "dcn validate\n5 FSED checks", "#d6eaf8"),
        ("2  Generate", "dcn generate\nresumable pairs", "#d5f5e3"),
        ("3  Consolidate", "dcn consolidate\none parquet", "#d5f5e3"),
        ("4  ML", "rq1_experiments/\nscenario-level split", "#fdebd0"),
    ]
    w, g = 1.8, 0.4
    for i, (title, body, fc) in enumerate(steps):
        x = 0.25 + i * (w + g)
        _box(ax, x, 1.5, w, 1.3, "", fc=fc)
        ax.text(x + w / 2, 2.52, title, ha="center", weight="bold", fontsize=10)
        ax.text(x + w / 2, 1.95, body, ha="center", va="center", fontsize=8.5)
        if i < len(steps) - 1:
            _arrow(ax, (x + w + 0.03, 2.15), (x + w + g - 0.03, 2.15), GREY, lw=1.6)
    ax.text(5.5, 0.95, "Outputs: reports/validation → datasets/<id>/pairs, manifest.csv → consolidated/ → results/ml",
            ha="center", fontsize=9, color=GREY)
    ax.text(5.5, 0.45, "Re-running any step skips finished work; scale up by raising n_seeds or appending levels in configs/datasets/*.yaml.",
            ha="center", fontsize=9, color=GREY)
    path = OUT / "workflow.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def topology(data_dir: Path) -> Path:
    folder = data_dir
    diameter = pq.read_table(folder / "pipe_diameter-0-static_input.parquet")
    adj = json.loads(diameter.schema.metadata[b"attrs"])["adj_list"]  # [start, end, pipe_id]
    d0 = diameter.to_pandas().set_index("scenario_id").loc[0.0]
    elev = pq.read_table(folder / "junction_elevation-0-static_input.parquet").to_pandas().set_index("scenario_id").loc[0.0]
    reservoirs = [c for c in pq.read_table(folder / "reservoir_base_head-0-static_input.parquet").column_names if c != "scenario_id"]

    g = nx.Graph()
    for a, b, pid in adj:
        g.add_edge(a, b, width=float(d0[str(pid)]))
    pos = nx.kamada_kawai_layout(g)

    fig, ax = plt.subplots(figsize=(8, 7))
    widths = [g[u][v]["width"] for u, v in g.edges]
    wmax = max(widths)
    nx.draw_networkx_edges(g, pos, ax=ax, width=[1 + 4 * w / wmax for w in widths], edge_color="#7f8c8d")
    junctions = [n for n in g.nodes if n not in reservoirs]
    elevations = [float(elev[n]) for n in junctions]
    nodes = nx.draw_networkx_nodes(g, pos, nodelist=junctions, node_color=elevations, cmap="viridis",
                                   node_size=260, ax=ax, edgecolors="white", linewidths=1)
    nx.draw_networkx_nodes(g, pos, nodelist=reservoirs, node_color=ACCENT, node_shape="s", node_size=520, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=7, font_color="white", ax=ax)
    fig.colorbar(nodes, ax=ax, shrink=0.7, label="junction elevation [m]")
    ax.set_title(f"hanoi_8GB_1Y topology: {len(junctions)} junctions, {g.number_of_edges()} pipes, "
                 f"source reservoir (red square)\nedge width ∝ pipe diameter; layout is schematic", fontsize=10)
    ax.axis("off")
    path = OUT / "network_topology.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="folder with the DiTEC-WDN static parquet files")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [schematic(), fault_points(), workflow()]
    if args.data:
        paths.append(topology(args.data))
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
