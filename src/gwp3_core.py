"""
GWP3 - MScFE 660 Risk Management - Group 18580
Interpretation of results and improvement of the model built in GWP2.

Four things happen here that did not happen in GWP2.

1. Replication. The dissertation's own specification is run on our panel so that its
   two published error rates can be compared with something.
2. Model selection. Alvi states that the validation window chooses between the BDeu,
   K2 and BIC scores, and never reports the comparison. It is reported here.
3. Capacity. The in-degree cap that GWP2 imposed by hand is justified by a sweep
   rather than by assertion.
4. Encoding. The binary alphabet is replaced by a three-symbol alphabet with a dead
   band, which is the first improvement GWP2 named in its bridge.

Everything imports the GWP2 core rather than restating it, so the two packages cannot
drift apart.
"""
from __future__ import annotations

import contextlib
import io
import os
import warnings

import numpy as np
import pandas as pd

import gwp2_core as g

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_STATES = 3
BAUM_WELCH_ITERS = g.BAUM_WELCH_ITERS
N_CHUNKS = g.N_CHUNKS
WARMUP = 24

# Alvi's published figures, chapter 4.3 of the dissertation, pages 62 and 64.
# He reports error; accuracy is one minus that.
ALVI = {
    "valid_error": 0.6786, "test_error": 0.4286,
    "valid_accuracy": 0.3214, "test_accuracy": 0.5714,
    "n_valid": 28, "n_test": 28,
}


# ----------------------------------------------------------------------------------
# 1. the three-symbol alphabet
# ----------------------------------------------------------------------------------
def band_threshold(series: pd.Series, train_idx, q: float) -> float:
    """Half-width of the dead band, read off the training window only.

    q is a quantile of the absolute monthly change. q = 0 gives a threshold of zero,
    which collapses the alphabet back to the sign of the change and reproduces GWP2.
    q = 0.3 means the smallest 30 per cent of monthly moves are called flat.
    """
    tr = series.loc[series.index.isin(train_idx)].diff().dropna().abs()
    if len(tr) == 0 or q <= 0:
        return 0.0
    return float(tr.quantile(q))


def emission_sequence_band(series: pd.Series, thr: float) -> pd.Series:
    """0 fall, 1 flat, 2 rise, with flat defined by the dead band.

    The binary encoding of GWP2 records only the sign of the change, so a month up
    0.2 per cent and a month up 18 per cent are the same symbol. A dead band gives the
    model a way to say that nothing happened, which is what the third hidden state was
    standing in for.
    """
    diff = series.diff().iloc[1:]
    out = np.where(diff.values > thr, 2, np.where(diff.values < -thr, 0, 1))
    return pd.Series(out.astype(np.int64), index=diff.index)


def fit_hmm_m(emissions: pd.Series, n_symbols: int, seed: int = 0,
              iters: int = BAUM_WELCH_ITERS, n_chunks: int = N_CHUNKS):
    """Baum-Welch with an alphabet of arbitrary size, otherwise GWP2's routine."""
    try:
        import hmms
    except ImportError:
        import hmms_compat as hmms
    np.random.seed(seed)
    model = hmms.DtHMM.random(N_STATES, n_symbols)
    blocks = np.array_split(emissions.values.astype(np.int64), n_chunks)
    with contextlib.redirect_stdout(io.StringIO()):
        model.baum_welch(blocks, iters)
    return model


# ----------------------------------------------------------------------------------
# 2. discretisation under the banded alphabet, both decoding rules
# ----------------------------------------------------------------------------------
def discretise_banded(panel: pd.DataFrame, train_idx, q: float, passthrough=(),
                      seed0: int = 0, warmup: int = WARMUP):
    """Fit one HMM per variable on the training window, decode twice.

    Returns the smoothed panel, the causal panel, the fitted models and the band
    thresholds actually used. No estimator, the band thresholds included, ever reads
    the validation or the test window.
    """
    n_symbols = 3 if q > 0 else 2
    models, smoothed, causal, bands = {}, {}, {}, {}
    for k, col in enumerate(panel.columns):
        series = panel[col].dropna()
        if col in passthrough:
            smoothed[col] = series.iloc[1:].astype(int)
            causal[col] = series.iloc[1:].astype(int).iloc[warmup:]
            bands[col] = np.nan
            continue
        tr = series.loc[series.index.isin(train_idx)]
        if len(tr) < 60:
            continue
        thr = band_threshold(series, train_idx, q) if q > 0 else 0.0
        bands[col] = thr
        enc = (lambda s: emission_sequence_band(s, thr)) if q > 0 else g.emission_sequence
        model = fit_hmm_m(enc(tr), n_symbols, seed=seed0 + k)
        models[col] = model

        emis = enc(series)
        chg = series.diff().iloc[1:]
        raw = g.decode(model, emis)
        smoothed[col] = g.relabel(raw, chg)

        tr_raw = g.decode(model, enc(tr))
        tr_chg = tr.diff().iloc[1:]
        mapping = g.regime_map(tr_raw, tr_chg)
        raw_online = g.decode_online(model, emis, warmup=warmup)
        causal[col] = raw_online.map(mapping).fillna(1).astype(int)

    return (pd.DataFrame(smoothed), pd.DataFrame(causal), models,
            pd.Series(bands, name="band"))


# ----------------------------------------------------------------------------------
# 3. structure learning helpers
# ----------------------------------------------------------------------------------
def cpt_cells(dag, n_states: int = 3) -> int:
    """Number of conditional probability cells the DAG asks the data to fill.

    A node with p parents needs n_states**(p+1) cells. The sum is the honest measure
    of how much the structure is asking of 351 training months.
    """
    total = 0
    for node in dag.nodes():
        p = len(list(dag.predecessors(node)))
        total += n_states ** (p + 1)
    return int(total)


def class_metrics(pred, truth) -> dict:
    """Balanced accuracy and macro F1, plus the class mix of the window.

    Plain accuracy is the only figure the dissertation reports, and it is the figure
    GWP2 carried forward. It is safe only while the three regimes are roughly balanced.
    As soon as one class dominates a window, a model that always names that class
    scores well without forecasting anything, so the two metrics below are reported
    beside it. Balanced accuracy is the mean recall over the classes that actually
    occur; macro F1 is the unweighted mean of the per-class F1 scores.
    """
    pred = np.asarray(pred).astype(str)
    truth = np.asarray(truth).astype(str)
    classes = sorted(set(truth) | set(pred))
    recalls, f1s = [], []
    for k in classes:
        tp = float(((pred == k) & (truth == k)).sum())
        fn = float(((pred != k) & (truth == k)).sum())
        fp = float(((pred == k) & (truth != k)).sum())
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    mix = pd.Series(truth).value_counts(normalize=True).sort_index()
    observed = int(len(set(truth)))
    # A window holding a single regime is not a three-class problem at all. Every
    # metric is then trivially one, including balanced accuracy, so the window is
    # marked degenerate and excluded from any selection rather than celebrated.
    degenerate = observed < 2
    return {
        "balanced_accuracy": (np.nan if degenerate else
                              round(float(np.mean(recalls)) if recalls else 0.0, 4)),
        "macro_f1": (np.nan if degenerate else
                     round(float(np.mean(f1s)) if f1s else 0.0, 4)),
        "classes_observed": observed,
        "largest_class_share": round(float(mix.max()), 4),
        "degenerate_window": bool(degenerate),
    }


def learn_and_score(full: pd.DataFrame, score: str = "k2", seed_edges=None,
                    max_indegree: int = 3, splits=None):
    """One complete pass: split, learn, fit, evaluate on validation and test."""
    a, b, c = splits if splits is not None else g.split_index(full.index)
    dag = g.learn_structure(full.loc[a], score=score, seed_edges=seed_edges,
                            max_indegree=max_indegree)
    model = g.fit_parameters(dag, full.loc[a])
    rv = g.evaluate(model, full.loc[b])
    rt = g.evaluate(model, full.loc[c])
    out_v = {k: v for k, v in rv.items() if k not in ("predictions", "truth")}
    out_t = {k: v for k, v in rt.items() if k not in ("predictions", "truth")}
    out_v.update(class_metrics(rv["predictions"], rv["truth"]))
    out_t.update(class_metrics(rt["predictions"], rt["truth"]))
    return {
        "score": score,
        "max_indegree": max_indegree,
        "n_edges": len(dag.edges()),
        "edges": [list(e) for e in dag.edges()],
        "blanket": sorted(model.get_markov_blanket("FORECAST")),
        "cpt_cells": cpt_cells(dag, n_states=full["FORECAST"].nunique()),
        "valid": out_v,
        "test": out_t,
        "_dag": dag, "_model": model,
        "_test_pred": rt["predictions"], "_test_truth": rt["truth"],
    }


def strip(res: dict) -> dict:
    """Drop the live objects so the result can be written to JSON."""
    return {k: v for k, v in res.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------------
# 4. the trading rule, rewritten
# ----------------------------------------------------------------------------------
def backtest(signal: pd.Series, price: pd.Series) -> dict:
    """Long, flat or short one barrel on next month's predicted regime.

    Alvi's own loop, cell In[39] of the dissertation, appends price levels to a list
    conditionally on the position, which mixes a position with a price and does not
    produce a return series. We keep his signal convention, 0 short, 1 flat, 2 long,
    and apply it the standard way: the position taken at the close of month t earns
    the log return of month t+1. Buy and hold is the benchmark a commodity desk would
    actually compare against.
    """
    px = price.reindex(signal.index).astype(float)
    ret = np.log(px).diff().shift(-1)            # return earned after the decision
    pos = signal.map({0: -1.0, 1: 0.0, 2: 1.0}).astype(float)
    pnl = (pos * ret).dropna()
    bh = ret.reindex(pnl.index)
    def stats(series, name, positions=None):
        cum = float(np.exp(series.sum()) - 1)
        vol = float(series.std() * np.sqrt(12))
        ann = float(series.mean() * 12)
        curve = np.exp(series.cumsum())
        dd = float((curve / curve.cummax() - 1).min())
        out = {"strategy": name, "cumulative_return": round(cum, 4),
               "annualised_return": round(ann, 4), "annualised_vol": round(vol, 4),
               "return_over_vol": round(ann / vol, 3) if vol > 0 else np.nan,
               "max_drawdown": round(dd, 4)}
        if positions is None:                 # buy and hold is long every month
            out.update({"months_long": int(len(series)), "months_short": 0,
                        "months_flat": 0})
        else:
            p = positions.reindex(series.index)
            out.update({"months_long": int((p > 0).sum()),
                        "months_short": int((p < 0).sum()),
                        "months_flat": int((p == 0).sum())})
        return out
    return {
        "model": stats(pnl, "Belief network signal", positions=pos),
        "buy_and_hold": stats(bh, "Buy and hold"),
        "curve_model": np.exp(pnl.cumsum()),
        "curve_bh": np.exp(bh.cumsum()),
    }


# ----------------------------------------------------------------------------------
# 5. the eight contributions, as the dissertation states them
# ----------------------------------------------------------------------------------
# Chapter 5, pages 66 and 67. The dissertation numbers them First to Lastly and uses
# "Secondly" twice; the duplicate claim about HMM discretisation is merged, which is
# what leaves exactly eight distinct claims.
CONTRIBUTIONS = [
    ("C1", "Bayesian views replace EGARCH-M views in Black-Litterman",
     "p. 66, citing Beach and Orlov, reference [15]"),
    ("C2", "Discretising time series with hidden Markov models as inputs to a belief "
           "network is novel", "p. 66, stated twice, as Secondly and as Fourthly"),
    ("C3", "A working trading mechanism, directly deployable, capable of independent "
           "decision making", "p. 66, and the trading loop of cell In[39], p. 64"),
    ("C4", "An almost autonomous system requiring no prior expert knowledge beyond "
           "dataset selection", "p. 66, against the expert seed of cell In[29], p. 58"),
    ("C5", "The abstraction offered by the Python modules clarified the design process",
     "pp. 66 to 67, and section 2.2, p. 28"),
    ("C6", "A systematic, event-driven global macro strategy that accounts for major "
           "geopolitical and macroeconomic change", "p. 67, and section 3.1, p. 34"),
    ("C7", "Better models of energy markets for policy makers",
     "p. 67, and the Contributions to Science preface, p. 2"),
    ("C8", "An amalgamation of multidisciplinary research applied to commodities to "
           "raise a trader's alpha", "p. 67"),
]
