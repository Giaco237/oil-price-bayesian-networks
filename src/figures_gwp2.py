"""Figures for GWP2. Same chart grammar as GWP1: one measure per axis, no dual scales,
a fixed categorical order, and a subtitle under every title saying what to take from it.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

import viz

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# bear, stagnant, bull, kept in this order everywhere
REGIME_COLOURS = [viz.SERIES[7], viz.INK_MUTED, viz.SERIES[2]]
REGIME_LABELS = ["Bear", "Stagnant", "Bull"]


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, f"{name}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def fig_regime_path(price: pd.Series, regimes: pd.Series, name="G1_regime_path"):
    """WTI coloured by the decoded regime, the plot Alvi builds in section 4.2.3."""
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    idx = price.index.intersection(regimes.index)
    p, r = price.loc[idx], regimes.loc[idx]
    x = mdates.date2num(p.index.to_pydatetime())
    pts = np.array([x, p.values]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap = ListedColormap(REGIME_COLOURS)
    lc = LineCollection(seg, cmap=cmap, norm=BoundaryNorm(range(4), cmap.N), linewidth=1.6)
    lc.set_array(r.values[:-1])
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, float(p.max()) * 1.08)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("USD per barrel")
    ax.set_title("WTI spot price by decoded regime, 1990 to 2026")
    viz.subtitle(ax, "Bear and bull runs line up with the dated events flagged in GWP1; "
                     "no stagnant episode lasts more than one month.")
    ax.legend(handles=[mpatches.Patch(color=c, label=l)
                       for c, l in zip(REGIME_COLOURS, REGIME_LABELS)],
              loc="upper left", frameon=False, ncol=3)
    viz.source_note(fig, "Source: sterilised monthly panel, GWP1; regimes decoded by "
                         "Baum-Welch and Viterbi (hmms 0.2.3).")
    return _save(fig, name)


def fig_emission_balance(shares: pd.Series, name="G2_emission_balance"):
    """Share of months carrying symbol 1. A variable near 0 or 1 carries no signal."""
    viz.apply_style()
    s = shares.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cols = [viz.SERIES[7] if (v < 0.25 or v > 0.75) else viz.SERIES[0] for v in s.values]
    ax.barh(s.index, s.values, color=cols)
    ax.axvline(0.5, color=viz.INK_2, lw=0.9, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of months encoded as a rise")
    ax.set_title("What the binary emission alphabet records")
    viz.subtitle(ax, "A series that rises almost every month gives the HMM a near-constant "
                     "sequence. M2 rises 89 % of months.")
    viz.source_note(fig, "Source: sterilised monthly panel, GWP1, 438 monthly changes.")
    return _save(fig, name)


def fig_regime_shares(diag: pd.DataFrame, name="G3_regime_shares"):
    """How the three states split per variable. Degenerate fits are visible at a glance."""
    viz.apply_style()
    d = diag.dropna(subset=["share_bear"]).set_index("variable")
    d = d[["share_bear", "share_stagnant", "share_bull"]]
    d = d.loc[d["share_stagnant"].sort_values().index]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    left = np.zeros(len(d))
    for col, colour, label in zip(d.columns, REGIME_COLOURS, REGIME_LABELS):
        ax.barh(d.index, d[col], left=left, color=colour, label=label)
        left += d[col].values
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of decoded months")
    ax.set_title("Decoded regime shares, by variable")
    viz.subtitle(ax, "Three states do not fit every series: industrial production lands "
                     "95 % stagnant, gold 72 %.")
    ax.legend(loc="lower right", frameon=False, ncol=3)
    viz.source_note(fig, "Source: own estimation on the GWP1 monthly panel.")
    return _save(fig, name)


def fig_transition(tm: pd.DataFrame, name="G4_transition_matrix"):
    """Empirical regime transition matrix for WTI."""
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(tm.values, cmap=viz.SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(len(tm.columns)), tm.columns)
    ax.set_yticks(range(len(tm.index)), tm.index)
    ax.set_xlabel("next month")
    ax.set_ylabel("current month")
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            v = tm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.55 else viz.INK, fontsize=10)
    ax.set_title("Regime transitions, WTI")
    viz.subtitle(ax, "Bull never steps straight to bear, and stagnant always leads to bear: "
                     "the third state is a bridge, not a market condition.")
    fig.colorbar(im, ax=ax, fraction=0.045)
    viz.source_note(fig, "Source: own estimation, 437 monthly transitions.")
    return _save(fig, name)


def fig_decoding_gap(agree: pd.Series, name="G5_decoding_gap"):
    """Share of months where smoothed and causal decoding give the same label."""
    viz.apply_style()
    s = agree.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(s.index, s.values, color=viz.SERIES[0])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("share of months labelled identically")
    ax.set_title("Smoothed Viterbi against causal decoding")
    viz.subtitle(ax, "Running Viterbi over the whole sample changes 14.5 % of labels on "
                     "average, and 22 % on WTI itself.")
    viz.source_note(fig, "Source: own estimation; same fitted HMMs, two decoding windows.")
    return _save(fig, name)


def fig_accuracy(res: dict, name="G6_accuracy"):
    """Network accuracy against its baselines, under both decoding rules."""
    viz.apply_style()
    labels = ["Network", "Persistence", "Majority class"]
    keys = ["network_accuracy", "persistence_accuracy", "majority_accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    for ax, split in zip(axes, ["valid", "test"]):
        x = np.arange(len(labels))
        w = 0.38
        for k, (tag, colour) in enumerate(zip(["lisse", "causal"],
                                              [viz.SERIES[1], viz.SERIES[0]])):
            vals = [res[tag][split][kk] for kk in keys]
            ax.bar(x + (k - 0.5) * w, vals, w, color=colour,
                   label="Smoothed decoding" if tag == "lisse" else "Causal decoding")
        ax.axhline(1 / 3, color=viz.INK_2, lw=0.9, ls="--")
        ax.set_xticks(x, labels, rotation=12)
        ax.set_ylim(0, 0.8)
        ax.set_title("Validation" if split == "valid" else "Test")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Where the forecasting performance comes from", y=1.02, fontsize=12)
    viz.source_note(fig, "Dashed line is the one-in-three chance level. "
                         "Source: own estimation on the GWP1 monthly panel.")
    return _save(fig, name)


def fig_network(edges, target="FORECAST", name="G7_network"):
    """The learnt belief network, target highlighted."""
    import networkx as nx
    viz.apply_style()
    G = nx.DiGraph()
    G.add_edges_from(edges)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    pos = nx.spring_layout(G, seed=7, k=1.1, iterations=200)
    blanket = set()
    if target in G:
        blanket |= set(G.predecessors(target))
        for ch in G.successors(target):
            blanket.add(ch)
            blanket |= set(G.predecessors(ch))
        blanket.discard(target)
    cols = [viz.SERIES[1] if n == target else
            (viz.SERIES[0] if n in blanket else "#d6d5d0") for n in G.nodes()]
    txt = [viz.SURFACE if (n == target or n in blanket) else viz.INK for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#bfbeb9", arrows=True,
                           arrowsize=12, width=1.3, node_size=1900,
                           min_source_margin=2, min_target_margin=2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cols, node_size=1900,
                           edgecolors=viz.SURFACE, linewidths=1.5)
    for (n, (x, y)), c in zip(pos.items(), txt):
        ax.text(x, y, n, ha="center", va="center", zorder=5, color=c,
                fontsize=8.2 if len(n) <= 7 else 7.0)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    px, py = (max(xs) - min(xs)) * 0.09, (max(ys) - min(ys)) * 0.12
    ax.set_xlim(min(xs) - px, max(xs) + px)
    ax.set_ylim(min(ys) - py, max(ys) + py)
    ax.axis("off")
    ax.set_title("Belief network learnt by Hill Climb, K2 score")
    viz.subtitle(ax, "Orange is the forecast target, blue its Markov blanket, "
                     "grey the variables that are irrelevant once the blanket is known.")
    viz.source_note(fig, "Source: own estimation, pgmpy 1.1.2, maximum in-degree 3.")
    return _save(fig, name)
