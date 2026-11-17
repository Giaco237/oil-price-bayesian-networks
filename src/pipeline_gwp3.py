"""End to end run for GWP3. Produces every number, table and figure quoted in the report.

Five experiments, in the order the report reads them.

  E1  Replication of the dissertation's own specification.
  E2  In-degree sweep, which turns GWP2's hand-imposed cap into a measured choice.
  E3  Score selection on the validation window, the comparison Alvi says he ran.
  E4  The three-symbol alphabet with a dead band, the first improvement GWP2 named.
  E5  A trading backtest of the final model against buy and hold.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

import gwp2_core as g
import gwp3_core as c
import figures_gwp3 as fg
from pipeline_gwp2 import model_nodes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data_clean")
NAME = {0: "bear", 1: "stagnant", 2: "bull"}

# Alvi seeds Hill Climb with six edges on EIA series, cell In[29], p. 58. Three of the
# six have a counterpart on the FRED panel GWP1 was forced to build; the OPEC and OECD
# aggregates do not. The seventh edge ties the target to the price, as in GWP2.
ALVI_SEED_EDGES = [
    ("IP_OILGAS", "WTI_M"),     # his four production and consumption edges collapse
    ("IGREA", "WTI_M"),         # to one supply and one demand channel on our panel
    ("INDPRO", "IGREA"),        # his two GDP to consumption edges
    ("WTI_M", "FORECAST"),
]

SCORES = [("k2", "K2"), ("bic-d", "BIC"), ("bdeu", "BDeu")]
INDEGREES = [1, 2, 3, 4, 5]
BANDS = [0.0, 0.10, 0.20, 0.30, 0.40]


# ----------------------------------------------------------------------------------
def episode_lengths(regimes: pd.Series) -> dict:
    """Mean run length of each decoded regime, in months."""
    runs, cur, n = {0: [], 1: [], 2: []}, regimes.iloc[0], 0
    for v in regimes.values:
        if v == cur:
            n += 1
        else:
            runs[cur].append(n)
            cur, n = v, 1
    runs[cur].append(n)
    return {NAME[k]: (float(np.mean(v)) if v else 0.0) for k, v in runs.items()}


def _uncapped_child(q, full_records, columns, seed_edges):
    """Run an uncapped Hill Climb in a child process under a memory limit."""
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))
    import pandas as _pd
    import gwp2_core as _g
    import gwp3_core as _c
    try:
        full = _pd.DataFrame(full_records, columns=columns)
        dag = _g.learn_structure(full, score="k2", seed_edges=seed_edges,
                                 max_indegree=None)
        deg = max(len(list(dag.predecessors(n))) for n in dag.nodes())
        q.put({"status": "completed", "max_parents": int(deg),
               "cpt_cells": _c.cpt_cells(dag, 3), "n_edges": len(dag.edges())})
    except Exception as exc:                      # MemoryError included
        q.put({"status": "failed", "error": type(exc).__name__, "detail": str(exc)[:200]})


def uncapped_probe(train_frame: pd.DataFrame, seed_edges, timeout: int = 420) -> dict:
    """What an uncapped search does, measured rather than asserted.

    GWP2 states that a run with no cap produced a node with seventeen parents and a
    conditional probability table of 10.9 gigabytes. The claim is re-run here under a
    six gigabyte ceiling and a wall clock, so the report can say what happened instead
    of repeating an anecdote.
    """
    q = mp.Queue()
    p = mp.Process(target=_uncapped_child,
                   args=(q, train_frame.values.tolist(), list(train_frame.columns),
                         seed_edges))
    t0 = time.time()
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return {"status": "timed out", "seconds": timeout}
    out = q.get() if not q.empty() else {"status": "died", "error": "no result returned"}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ----------------------------------------------------------------------------------
def run(seed0: int = 0, verbose: bool = True, probe_uncapped: bool = True):
    t_start = time.time()
    monthly = g.load_monthly()
    panel = monthly[model_nodes()]
    train_idx, valid_idx, test_idx = g.split_index(panel.index)
    res: dict = {"panel": {"months": int(len(panel)), "variables": int(panel.shape[1]),
                           "train": len(train_idx), "valid": len(valid_idx),
                           "test": len(test_idx), "warmup": c.WARMUP}}

    # --- baseline, the GWP2 encoding, both decoding rules ---------------------------
    s0, c0, models0, _ = c.discretise_banded(panel, train_idx, q=0.0,
                                             passthrough=("GEOEVENT",), seed0=seed0)
    full_s0, full_c0 = g.add_target(s0), g.add_target(c0)
    base = {
        "smoothed": c.learn_and_score(full_s0, "k2", g.EXPERT_EDGES, 3),
        "causal": c.learn_and_score(full_c0, "k2", g.EXPERT_EDGES, 3),
    }
    res["baseline"] = {k: c.strip(v) for k, v in base.items()}
    if verbose:
        print(f"[baseline] smoothed test {base['smoothed']['test']['network_accuracy']:.3f} "
              f"| causal test {base['causal']['test']['network_accuracy']:.3f}")

    # --- E1 replication --------------------------------------------------------------
    rep = c.learn_and_score(full_s0, "k2", ALVI_SEED_EDGES, 3)
    res["replication"] = {"ours": c.strip(rep), "alvi": c.ALVI}
    if probe_uncapped:
        a, _, _ = g.split_index(full_s0.index)
        res["uncapped_probe"] = uncapped_probe(g.as_categorical(full_s0.loc[a]),
                                               ALVI_SEED_EDGES)
        if verbose:
            print(f"[E1] uncapped probe: {res['uncapped_probe']}")

    # --- E2 in-degree sweep ----------------------------------------------------------
    rows = []
    for d in INDEGREES:
        r = c.learn_and_score(full_c0, "k2", g.EXPERT_EDGES, d)
        rows.append({"max_indegree": d, "n_edges": r["n_edges"],
                     "cpt_cells": r["cpt_cells"],
                     "obs_per_cell": round(len(train_idx) / r["cpt_cells"], 4),
                     "valid_accuracy": r["valid"]["network_accuracy"],
                     "valid_balanced": r["valid"]["balanced_accuracy"],
                     "test_accuracy": r["test"]["network_accuracy"],
                     "test_balanced": r["test"]["balanced_accuracy"],
                     "blanket": ", ".join(r["blanket"])})
    sweep = pd.DataFrame(rows)
    res["indegree_sweep"] = sweep.to_dict("records")
    best_degree = int(sweep.loc[sweep.valid_balanced.idxmax(), "max_indegree"])
    if verbose:
        print(f"[E2] in-degree sweep done, validation best at {best_degree}")

    # --- E3 score selection ----------------------------------------------------------
    rows = []
    for key, label in SCORES:
        r = c.learn_and_score(full_c0, key, g.EXPERT_EDGES, 3)
        rows.append({"score": key, "label": label, "n_edges": r["n_edges"],
                     "blanket": ", ".join(r["blanket"]),
                     "valid_accuracy": r["valid"]["network_accuracy"],
                     "valid_balanced": r["valid"]["balanced_accuracy"],
                     "test_accuracy": r["test"]["network_accuracy"],
                     "test_balanced": r["test"]["balanced_accuracy"]})
    scores = pd.DataFrame(rows)
    res["score_selection"] = scores.to_dict("records")
    best_score = str(scores.loc[scores.valid_balanced.idxmax(), "score"])
    if verbose:
        print(f"[E3] score selection done, validation best is {best_score}")

    # --- E4 the dead band -------------------------------------------------------------
    rows, keep = [], {}
    for q in BANDS:
        s, cc, _, bands = c.discretise_banded(panel, train_idx, q=q,
                                              passthrough=("GEOEVENT",), seed0=seed0)
        full = g.add_target(cc)
        r = c.learn_and_score(full, best_score, g.EXPERT_EDGES, 3)
        sh = s["WTI_M"].value_counts(normalize=True).reindex([0, 1, 2]).fillna(0)
        rows.append({"q": q,
                     "band_wti": round(float(bands.get("WTI_M", np.nan)), 4),
                     "n_symbols": 3 if q > 0 else 2,
                     "n_edges": r["n_edges"],
                     "valid_accuracy": r["valid"]["network_accuracy"],
                     "valid_balanced": r["valid"]["balanced_accuracy"],
                     "valid_largest_class": r["valid"]["largest_class_share"],
                     "valid_degenerate": r["valid"]["degenerate_window"],
                     "test_degenerate": r["test"]["degenerate_window"],
                     "test_accuracy": r["test"]["network_accuracy"],
                     "test_balanced": r["test"]["balanced_accuracy"],
                     "test_macro_f1": r["test"]["macro_f1"],
                     "test_largest_class": r["test"]["largest_class_share"],
                     "test_classes_observed": r["test"]["classes_observed"],
                     "persistence_test": r["test"]["persistence_accuracy"],
                     "majority_test": r["test"]["majority_accuracy"],
                     "share_bear": round(float(sh[0]), 3),
                     "share_stagnant": round(float(sh[1]), 3),
                     "share_bull": round(float(sh[2]), 3)})
        keep[q] = {"result": r, "smoothed": s, "causal": cc, "full": full}
        if verbose:
            print(f"[E4] q={q:.2f} acc valid {r['valid']['network_accuracy']:.3f} "
                  f"test {r['test']['network_accuracy']:.3f} | balanced valid "
                  f"{r['valid']['balanced_accuracy']:.3f} test "
                  f"{r['test']['balanced_accuracy']:.3f} | largest class "
                  f"{r['test']['largest_class_share']:.3f}")
    band_tbl = pd.DataFrame(rows)
    res["dead_band"] = band_tbl.to_dict("records")
    # Selected on balanced accuracy, not on accuracy. Raw accuracy rises to one as the
    # band widens only because the target collapses onto a single regime, which is a
    # degenerate window rather than a forecast.
    best_q = float(band_tbl.loc[band_tbl.valid_balanced.idxmax(), "q"])
    final = keep[best_q]["result"]
    res["final"] = {"score": best_score, "max_indegree": 3, "dead_band_q": best_q,
                    **c.strip(final)}

    # --- episode lengths, binary against banded ---------------------------------------
    dur = pd.DataFrame({
        "binary": pd.Series(episode_lengths(keep[0.0]["smoothed"]["WTI_M"])),
        "banded": pd.Series(episode_lengths(keep[best_q]["smoothed"]["WTI_M"])),
    }).reindex(["bear", "stagnant", "bull"])
    res["episode_lengths"] = dur.round(2).to_dict()

    # --- E5 backtest --------------------------------------------------------------------
    full_best = keep[best_q]["full"]
    _, _, test_b = g.split_index(full_best.index)
    signal = pd.Series(final["_test_pred"].astype(int), index=test_b)
    bt = c.backtest(signal, monthly["WTI_M"])
    res["backtest"] = {"model": bt["model"], "buy_and_hold": bt["buy_and_hold"],
                       "n_months": int(len(bt["curve_model"]))}

    # --- figures --------------------------------------------------------------------------
    paths = [
        fg.fig_allocation(panel.index, train_idx, valid_idx, test_idx, c.WARMUP),
        fg.fig_replication(rep, c.ALVI),
        fg.fig_indegree(sweep),
        fg.fig_scores(scores),
        fg.fig_band(band_tbl),
        fg.fig_duration(dur),
        fg.fig_backtest(bt["curve_model"], bt["curve_bh"], signal),
        fg.fig_final_network(final["edges"], final["blanket"]),
    ]
    audit = fg.audit_labels()
    res["figure_audit"] = audit.to_dict("records")

    # --- save --------------------------------------------------------------------------
    os.makedirs(OUT, exist_ok=True)
    sweep.to_csv(os.path.join(OUT, "indegree_sweep.csv"), index=False)
    scores.to_csv(os.path.join(OUT, "score_selection.csv"), index=False)
    band_tbl.to_csv(os.path.join(OUT, "dead_band_sweep.csv"), index=False)
    audit.to_csv(os.path.join(OUT, "figure_label_audit.csv"), index=False)
    keep[best_q]["smoothed"].to_csv(os.path.join(OUT, "regimes_banded_smoothed.csv"))
    keep[best_q]["causal"].to_csv(os.path.join(OUT, "regimes_banded_causal.csv"))
    pd.DataFrame({"model": bt["curve_model"], "buy_and_hold": bt["curve_bh"],
                  "signal": signal.reindex(bt["curve_model"].index)}).to_csv(
        os.path.join(OUT, "backtest_test_window.csv"))
    res["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(OUT, "gwp3_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)

    if verbose:
        print(f"[final] score={best_score} q={best_q} "
              f"valid {final['valid']['network_accuracy']:.3f} "
              f"test {final['test']['network_accuracy']:.3f}")
        print("figures:", ", ".join(os.path.basename(p) for p in paths))
        print(f"runtime {res['runtime_seconds']} s")
    return res, band_tbl, sweep, scores, dur, bt, keep


if __name__ == "__main__":
    run()
