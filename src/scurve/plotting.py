"""Shared figure factory. All charts in the paper/post go through here.

Style follows the dataviz skill's validated reference palette (light mode):
- categorical hues in FIXED slot order (never cycled/generated)
- 2px lines, >=8px markers, recessive hairline grid, no top/right spines
- text wears ink tokens, never series colors; legend whenever >=2 series
- one axis only; CPR always in percent, incentive in bps
- every figure carries a source line
Output: 300dpi PNG, >=1600px wide (LinkedIn-safe), light surface.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

# validated reference palette — light mode, fixed order (dataviz skill)
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
          "#e87ba4", "#eb6834"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SOURCE_LINE = ("Source: Fannie Mae Single-Family Loan Performance data; "
               "author's calculations")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": BASELINE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
})


def _base(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def save_fig(fig, out_dir: Path, name: str) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.text(0.01, 0.005, SOURCE_LINE, fontsize=7, color=MUTED)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def scurve_chart(df: pl.DataFrame, title: str):
    """df: incentive_bps, cpr, series (one line per series, fixed slot order)."""
    fig, ax = _base()
    names = df["series"].unique(maintain_order=True).to_list()
    for i, name in enumerate(names):
        g = df.filter(pl.col("series") == name).sort("incentive_bps")
        ax.plot(g["incentive_bps"], g["cpr"] * 100, linewidth=2,
                marker="o", markersize=4, color=SERIES[i % len(SERIES)],
                label=str(name))
    ax.axvline(0, color=BASELINE, linewidth=0.8)
    ax.set_xlabel("Refi incentive (bps)")
    ax.set_ylabel("CPR (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    if len(names) >= 2:
        ax.legend(frameon=False, labelcolor=INK_2)
    return fig


def error_timeseries(df: pl.DataFrame, title: str, ylabel: str = "CPR (%)"):
    """df: month, value, series. One line per series in given order."""
    fig, ax = _base(figsize=(12, 5.5))
    names = df["series"].unique(maintain_order=True).to_list()
    for i, name in enumerate(names):
        g = df.filter(pl.col("series") == name).sort("month")
        ax.plot(g["month"], g["value"], linewidth=2,
                color=SERIES[i % len(SERIES)], label=str(name))
    ax.axhline(0, color=BASELINE, linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    months = sorted(df["month"].unique().to_list())
    step = max(1, len(months) // 12)
    ax.set_xticks(months[::step])
    ax.tick_params(axis="x", rotation=45)
    if len(names) >= 2:
        ax.legend(frameon=False, labelcolor=INK_2)
    return fig


def grouped_barh(labels: list[str], groups: dict[str, list[float]],
                 title: str, xlabel: str):
    """Horizontal grouped bars: one group per series, fixed slot order."""
    import numpy as np
    fig, ax = _base(figsize=(9, 6))
    y = np.arange(len(labels), dtype=float)
    n = len(groups)
    h = 0.8 / n
    for i, (name, vals) in enumerate(groups.items()):
        ax.barh(y + i * h, vals, height=h * 0.9, color=SERIES[i % len(SERIES)],
                label=name)
    ax.set_yticks(y + 0.4 - h / 2)
    ax.set_yticklabels(labels, color=INK_2)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontweight="bold")
    if n >= 2:
        ax.legend(frameon=False, labelcolor=INK_2)
    return fig
