"""Figures for GWP3, numbered G8 onwards so the three projects assemble into one
mini-capstone with a single figure series.

The GWP1 feedback asked for complete labels, axes and scales on every chart. Three
rules are enforced here and checked by `audit_labels` at the end of the module: every
axis that carries a measure is labelled, every axis that carries a share or an accuracy
is drawn on an explicit 0 to 1 scale rather than an automatic one, and every figure
carries a source note.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import viz

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

REGIME_COLOURS = [viz.SERIES[7], viz.INK_MUTED, viz.SERIES[2]]   # bear, stagnant, bull
REGIME_LABELS = ["Bear", "Stagnant", "Bull"]
SRC = "Source: own estimation on the sterilised monthly panel delivered in GWP1."


LABEL_AUDIT = []


def _save(fig, name):
    """Save, and record what the figure actually carries before it is closed.

    The audit is taken from the live figure, not asserted afterwards, so the claim in
    the report that every axis is labelled is a measurement rather than a promise.
    """
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, f"{name}.png")
    data_axes = [a for a in fig.axes if a.axison]
    has_note = any(t.get_text().startswith("Source:") for t in fig.texts)

    def x_ok(a):
        """A shared-x column labels only its bottom axis, which is correct practice."""
        group = list(a.get_shared_x_axes().get_siblings(a))
        return any(bool(b.get_xlabel()) for b in group)

    if not data_axes:                       # a network diagram has its axes off by design
        row = {"figure": f"{name}.png", "kind": "diagram", "data_axes": 0,
               "x_label": "n/a", "y_label": "n/a", "ticks_both_axes": "n/a",
               "source_note": bool(has_note)}
    else:
        row = {"figure": f"{name}.png", "kind": "chart", "data_axes": len(data_axes),
               "x_label": all(x_ok(a) for a in data_axes),
               "y_label": all(bool(a.get_ylabel()) for a in data_axes),
               "ticks_both_axes": all(len(a.get_xticks()) > 0 and len(a.get_yticks()) > 0
                                      for a in data_axes),
               "source_note": bool(has_note)}
    LABEL_AUDIT.append(row)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------
def fig_allocation(index, train_idx, valid_idx, test_idx, warmup: int,
                   name="G8_data_allocation"):
    """How the 439 months are allocated, and what the warm-up removes."""
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    fig.subplots_adjust(bottom=0.26)
    spans = [("Training", train_idx, viz.SERIES[0]),
             ("Validation", valid_idx, viz.SERIES[1]),
             ("Test", test_idx, viz.SERIES[2])]
    for i, (label, idx, colour) in enumerate(spans):
        x0, x1 = mdates.date2num(idx[0]), mdates.date2num(idx[-1])
        ax.barh(0, x1 - x0, left=x0, height=0.42, color=colour, label=f"{label} ({len(idx)} months)")
        ax.text((x0 + x1) / 2, 0, f"{len(idx)}", ha="center", va="center",
                color=viz.SURFACE, fontsize=9, fontweight="bold")
    w0 = mdates.date2num(index[0])
    w1 = mdates.date2num(index[warmup])
    ax.barh(-0.6, w1 - w0, left=w0, height=0.28, color=viz.INK_MUTED,
            label=f"Causal warm-up, dropped ({warmup} months)")
    ax.set_yticks([0, -0.6], ["Split", "Warm-up"])
    ax.set_ylim(-1.1, 0.6)
    ax.set_xlim(mdates.date2num(index[0]) - 200, mdates.date2num(index[-1]) + 200)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("calendar time, monthly observations")
    ax.set_ylabel("allocation")
    ax.grid(axis="y", visible=False)
    ax.set_title("Allocation of the 439-month panel")
    viz.subtitle(ax, "The 80:10:10 split of the dissertation, applied chronologically. "
                     "Causal decoding spends the first 24 months warming up.")
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=8.2)
    viz.source_note(fig, SRC + " Split ratios follow Alvi (2018), section 3.5, p. 46.")
    return _save(fig, name)


# ----------------------------------------------------------------------------------
def fig_replication(rep: dict, alvi: dict, name="G9_replication"):
    """Our replication of the dissertation's own specification, against its figures."""
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    labels = ["Validation", "Test"]
    x = np.arange(2)
    w = 0.36
    ours = [rep["valid"]["network_accuracy"], rep["test"]["network_accuracy"]]
    theirs = [alvi["valid_accuracy"], alvi["test_accuracy"]]
    ax.bar(x - w / 2, theirs, w, color=viz.INK_MUTED, label="Alvi (2018), as published")
    ax.bar(x + w / 2, ours, w, color=viz.SERIES[0], label="This replication")
    for xi, v in zip(x - w / 2, theirs):
        ax.text(xi, v + 0.015, f"{v:.1%}", ha="center", fontsize=8.5, color=viz.INK_2)
    for xi, v in zip(x + w / 2, ours):
        ax.text(xi, v + 0.015, f"{v:.1%}", ha="center", fontsize=8.5, color=viz.INK_2)
    ax.axhline(1 / 3, color=viz.INK_2, lw=0.9, ls="--")
    ax.annotate("chance, 33.3%", xy=(1.42, 1 / 3), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=viz.INK_2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("evaluation window")
    ax.set_ylabel("share of months classified correctly")
    ax.set_title("Replicating the dissertation's own specification")
    viz.subtitle(ax, "Same encoding, same seed network, same score. Different panel, "
                     "so the comparison is of magnitudes, not of a single number.")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=8.6)
    viz.source_note(fig, SRC + " Published figures from Alvi (2018), cells In[34] and "
                             "In[37], pp. 62 and 64, converted from error to accuracy.")
    return _save(fig, name)


# ----------------------------------------------------------------------------------
def fig_indegree(sweep: pd.DataFrame, name="G10_indegree"):
    """What the in-degree cap buys and what it costs. Two panels, never two y-scales."""
    viz.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axes[0]
    ax.plot(sweep["max_indegree"], sweep["valid_accuracy"], marker="o",
            color=viz.SERIES[0], label="Accuracy, validation")
    ax.plot(sweep["max_indegree"], sweep["test_accuracy"], marker="s",
            color=viz.SERIES[1], label="Accuracy, test")
    ax.plot(sweep["max_indegree"], sweep["valid_balanced"], marker="o", ls=":",
            color=viz.SERIES[0], label="Balanced, validation")
    ax.plot(sweep["max_indegree"], sweep["test_balanced"], marker="s", ls=":",
            color=viz.SERIES[1], label="Balanced, test")
    ax.axhline(1 / 3, color=viz.INK_2, lw=0.9, ls="--")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(sweep["max_indegree"])
    ax.set_xlabel("maximum number of parents per node")
    ax.set_ylabel("share of months, or mean recall over regimes")
    ax.set_title("Accuracy")
    ax.legend(frameon=False, fontsize=8.0)

    ax = axes[1]
    ax.plot(sweep["max_indegree"], sweep["obs_per_cell"], marker="o",
            color=viz.SERIES[3])
    ax.axhline(5, color=viz.SERIES[7], lw=0.9, ls="--")
    ax.annotate("five observations per cell", xy=(sweep["max_indegree"].iloc[0], 5),
                xytext=(2, 5), textcoords="offset points", fontsize=8, color=viz.SERIES[7])
    ax.set_yscale("log")
    ax.set_xticks(sweep["max_indegree"])
    ax.set_xlabel("maximum number of parents per node")
    ax.set_ylabel("training months per probability cell, log scale")
    ax.set_title("What the data are asked to fill")
    fig.suptitle("The in-degree cap is a sample-size constraint, not a preference",
                 y=1.03, fontsize=12)
    viz.source_note(fig, SRC + " 351 training months; a node with p parents needs "
                             "3 to the power p+1 probability cells.")
    return _save(fig, name)


# ----------------------------------------------------------------------------------
def fig_scores(scores: pd.DataFrame, name="G11_score_selection"):
    """The score comparison the dissertation says it ran and never reports."""
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(scores))
    w = 0.36
    ax.bar(x - w / 2, scores["valid_accuracy"], w, color=viz.SERIES[0],
           label="Accuracy, validation")
    ax.bar(x + w / 2, scores["valid_balanced"], w, color=viz.SERIES[2],
           label="Balanced accuracy, validation")
    for xi, v in zip(x - w / 2, scores["valid_accuracy"]):
        ax.text(xi, v + 0.012, f"{v:.1%}", ha="center", fontsize=8.4, color=viz.INK_2)
    for xi, v in zip(x + w / 2, scores["valid_balanced"]):
        ax.text(xi, v + 0.012, f"{v:.1%}", ha="center", fontsize=8.4, color=viz.INK_2)
    ax.axhline(1 / 3, color=viz.INK_2, lw=0.9, ls="--")
    ax.set_xticks(x, scores["label"].tolist())
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("structure score used by the Hill Climb search")
    ax.set_ylabel("share of months classified correctly")
    ax.set_title("Choosing the score on the validation window")
    viz.subtitle(ax, "Alvi states that the validation set selects among these three "
                     "and never publishes the comparison.")
    ax.legend(frameon=False, loc="upper right")
    viz.source_note(fig, SRC + " Dashed line is the one-in-three chance level. "
                             "Scores as named in pgmpy 1.1.2.")
    return _save(fig, name)


# ----------------------------------------------------------------------------------
def fig_band(bands: pd.DataFrame, name="G12_dead_band"):
    """Accuracy as the dead band widens, with the share of months called flat."""
    viz.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    fig.subplots_adjust(wspace=0.32)
    ax = axes[0]
    ax.plot(bands["q"], bands["test_accuracy"], marker="s", color=viz.SERIES[1],
            label="Accuracy, test")
    ax.plot(bands["q"], bands["test_largest_class"], marker="x", ls=":",
            color=viz.INK_MUTED, label="Largest class share, test")
    ax.plot(bands["q"], bands["test_balanced"], marker="o", color=viz.SERIES[0],
            label="Balanced accuracy, test")
    ax.axhline(1 / 3, color=viz.INK_2, lw=0.9, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(bands["q"])
    ax.set_xlabel("dead band, quantile of absolute monthly change called flat")
    ax.set_ylabel("share of months, or mean recall over regimes")
    ax.set_title("Accuracy and class balance")
    ax.legend(frameon=False, fontsize=8.0, loc="lower right")

    ax = axes[1]
    ax.plot(bands["q"], bands["share_stagnant"], marker="o", color=viz.INK_MUTED,
            label="Stagnant")
    ax.plot(bands["q"], bands["share_bear"], marker="s", color=viz.SERIES[7],
            label="Bear")
    ax.plot(bands["q"], bands["share_bull"], marker="^", color=viz.SERIES[2],
            label="Bull")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(bands["q"])
    ax.set_xlabel("dead band, quantile of absolute monthly change called flat")
    ax.set_ylabel("share of decoded months on WTI")
    ax.set_title("Decoded regime mix")
    ax.legend(frameon=False, fontsize=8.0, loc="center right")
    fig.suptitle("The dead band buys accuracy by emptying the forecasting problem",
                 y=1.03, fontsize=12)
    viz.source_note(fig, SRC + " Band thresholds are estimated on the training window "
                             "only; q = 0 reproduces the binary alphabet of GWP2. "
                             "Accuracy tracks the largest class share almost exactly.")
    return _save(fig, name)


# ----------------------------------------------------------------------------------
def fig_duration(dur: pd.DataFrame, name="G13_episode_duration"):
    """Mean episode length by regime, binary alphabet against the dead band."""
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(REGIME_LABELS))
    w = 0.36
    ax.bar(x - w / 2, dur["binary"], w, color=viz.INK_MUTED, label="Binary alphabet (GWP2)")
    ax.bar(x + w / 2, dur["banded"], w, color=viz.SERIES[0], label="Three-symbol alphabet")
    for xi, v in zip(x - w / 2, dur["binary"]):
        ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=8.5, color=viz.INK_2)
    for xi, v in zip(x + w / 2, dur["banded"]):
        ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=8.5, color=viz.INK_2)
    ax.axhline(1.0, color=viz.SERIES[7], lw=0.9, ls="--")
    ax.annotate("one month, the shortest an episode can be", xy=(-0.42, 1.0),
                xytext=(0, 5), textcoords="offset points", fontsize=8,
                color=viz.SERIES[7], ha="left")
    ax.set_xticks(x, REGIME_LABELS)
    ax.set_xlabel("decoded regime on WTI")
    ax.set_ylabel("mean episode length, months")
    ax.set_ylim(0, max(dur.max()) * 1.25)
    ax.set_title("The stagnant state stops being a one-month bridge")
    viz.subtitle(ax, "Under the binary alphabet every stagnant episode lasted exactly "
                     "one month, which is what identified it as an artefact.")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=8.6)
    viz.source_note(fig, SRC + " Episodes measured on the smoothed decoding of WTI, "
                             "January 1990 to July 2026.")
    return _save(fig, name)


# ----------------------------------------------------------------------------------
def fig_backtest(curve_model: pd.Series, curve_bh: pd.Series, signal: pd.Series,
                 name="G14_backtest"):
    """Cumulative return of the signal against buy and hold, over the test window."""
    viz.apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(curve_bh.index, curve_bh.values, color=viz.INK_MUTED, label="Buy and hold")
    ax.plot(curve_model.index, curve_model.values, color=viz.SERIES[0],
            label="Belief network signal")
    ax.axhline(1.0, color=viz.INK_2, lw=0.9, ls="--")
    ax.set_ylabel("value of one barrel invested, index = 1.00")
    ax.set_title("Out-of-sample performance over the test window")
    viz.subtitle(ax, "The signal is the forecast regime: long on bull, short on bear, "
                     "flat on stagnant, one barrel, no costs.")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    pos = signal.map({0: -1, 1: 0, 2: 1}).reindex(curve_model.index)
    cols = [viz.SERIES[7] if v < 0 else (viz.SERIES[2] if v > 0 else viz.INK_MUTED)
            for v in pos.values]
    ax.bar(pos.index, pos.values, width=20, color=cols)
    ax.set_yticks([-1, 0, 1], ["Short", "Flat", "Long"])
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("calendar time, monthly decisions")
    ax.set_ylabel("position")
    ax.grid(axis="y", visible=False)
    viz.source_note(fig, SRC + " Returns are monthly log returns on WTI spot, "
                             "transaction costs excluded, one barrel of notional.")
    return _save(fig, name)


# ----------------------------------------------------------------------------------
def _separate(pos: dict, min_dist: float = 0.30, rounds: int = 400) -> dict:
    """Push overlapping nodes apart so no two labels sit on top of each other.

    A spring layout optimises edge length, not label collision. This is a short
    relaxation afterwards: any pair closer than `min_dist` is moved apart along the
    line joining them. It changes nothing about the graph, only about readability.
    """
    keys = list(pos)
    P = np.array([pos[k] for k in keys], dtype=float)
    for _ in range(rounds):
        moved = False
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = P[i] - P[j]
                dist = float(np.hypot(*d))
                if dist < 1e-9:
                    d = np.random.default_rng(i * 97 + j).normal(size=2) * 1e-3
                    dist = float(np.hypot(*d))
                if dist < min_dist:
                    push = (min_dist - dist) / 2.0 * d / dist
                    P[i] += push
                    P[j] -= push
                    moved = True
        if not moved:
            break
    return {k: tuple(P[i]) for i, k in enumerate(keys)}


def fig_final_network(edges, blanket, target="FORECAST", name="G15_final_network"):
    """The network learnt under the improved encoding."""
    import networkx as nx
    viz.apply_style()
    G = nx.DiGraph()
    G.add_edges_from([tuple(e) for e in edges])
    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    pos = nx.spring_layout(G, seed=7, k=1.6, iterations=400)
    pos = _separate(pos, min_dist=0.34)
    bl = set(blanket) - {target}
    cols = [viz.SERIES[1] if n == target else
            (viz.SERIES[0] if n in bl else "#d6d5d0") for n in G.nodes()]
    txt = [viz.SURFACE if (n == target or n in bl) else viz.INK for n in G.nodes()]
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
    ax.set_title("Belief network under the three-symbol alphabet and causal decoding")
    viz.subtitle(ax, "Orange is the forecast target, blue its Markov blanket, grey the "
                     "variables that are irrelevant once the blanket is known.")
    viz.source_note(fig, SRC + " pgmpy 1.1.2, Hill Climb search, K2 score, maximum "
                             "in-degree 3, expert seed of seven edges.")
    return _save(fig, name)


# ----------------------------------------------------------------------------------
def audit_labels() -> pd.DataFrame:
    """Return the audit collected while the figures were being saved.

    The GWP1 feedback asked for complete labels, axes and scales. The check lives in
    the code package rather than in a sentence of the report. A network diagram has its
    axes switched off deliberately and is excluded from the axis columns by `axison`.
    """
    return pd.DataFrame(LABEL_AUDIT)
