"""Chart style for the GWP1 report.

Categorical hues are assigned in a fixed order and never cycled. Charts never use two
y-scales: where two measures live on different scales they are drawn as small multiples
or indexed to a common base. Correlation matrices use a diverging blue-red ramp with a
neutral grey midpoint.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8980"
GRID = "#e3e2dd"

# fixed categorical order (validated: adjacent CVD dE 9.1, normal-vision dE 19.6)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# first three slots also clear the all-pairs gate, for scatter and small multiples
SERIES_SAFE3 = SERIES[:3]

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#0d366b", "#256abf", "#86b6ef", "#f0efec",
                      "#f0a0a0", "#cf3736", "#8c1f1e"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "font.size": 9.5,
        "font.family": "DejaVu Sans",
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def subtitle(ax, text, fontsize=8.8):
    """Replace the axes title with an explicitly offset title plus a deck line.

    Matplotlib's `pad` argument does not reliably clear a second line of text, so both
    lines are drawn as offset annotations and the native title is removed.
    """
    t = ax.get_title(loc="left") or ax.get_title(loc="center")
    size = ax._left_title.get_fontsize()
    ax.set_title("", loc="left")
    ax.set_title("", loc="center")
    if t:
        ax.annotate(t, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 24), textcoords="offset points",
                    ha="left", va="bottom", fontsize=size, fontweight="bold", color=INK)
    ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, 8), textcoords="offset points",
                ha="left", va="bottom", fontsize=fontsize, color=INK_2)


def source_note(fig, text):
    fig.text(0.0, -0.02, text, fontsize=7.8, color=INK_MUTED, ha="left", va="top")
