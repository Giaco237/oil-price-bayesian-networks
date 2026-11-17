"""End to end run for GWP2. Produces every number and figure quoted in the report."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import gwp2_core as g
import figures_gwp2 as fg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data_clean")
NAME = {0: "bear", 1: "stagnant", 2: "bull"}


def model_nodes():
    return [c for c in g.MACRO + g.MICRO + g.FINANCIAL
            if c not in g.TAUTOLOGIES and c not in g.LATE_START]


def regime_episodes(regimes: pd.Series, price: pd.Series) -> pd.DataFrame:
    runs, start, cur, prev = [], regimes.index[0], regimes.iloc[0], regimes.index[0]
    for d, v in regimes.items():
        if v != cur:
            runs.append((start, prev, cur))
            start, cur = d, v
        prev = d
    runs.append((start, regimes.index[-1], cur))
    R = pd.DataFrame(runs, columns=["start", "end", "regime"])
    R["months"] = ((R.end.dt.year - R.start.dt.year) * 12
                   + (R.end.dt.month - R.start.dt.month)) + 1
    R["wti_start"] = R.start.map(price).round(2)
    R["wti_end"] = R.end.map(price).round(2)
    R["move_pct"] = ((R.wti_end / R.wti_start - 1) * 100).round(1)
    R["regime"] = R.regime.map(NAME)
    return R


def run(seed0: int = 0, verbose: bool = True):
    monthly = g.load_monthly()
    nodes = model_nodes()
    panel = monthly[nodes]
    train_idx, valid_idx, test_idx = g.split_index(panel.index)

    # --- regime detection, both decoding rules -------------------------------------
    disc_s, models, diag = g.discretise_panel(panel, train_idx, passthrough=("GEOEVENT",),
                                              seed0=seed0)
    disc_c = g.discretise_panel_online(panel, train_idx, models,
                                       passthrough=("GEOEVENT",), warmup=24)
    agree = (disc_s.loc[disc_c.index] == disc_c).mean()

    # --- the belief network, once per decoding rule ---------------------------------
    results, dags = {}, {}
    for tag, dd in [("lisse", disc_s), ("causal", disc_c)]:
        full = g.add_target(dd)
        a, b, c = g.split_index(full.index)
        dag = g.learn_structure(full.loc[a], score="k2", seed_edges=g.EXPERT_EDGES)
        model = g.fit_parameters(dag, full.loc[a])
        rv, rt = g.evaluate(model, full.loc[b]), g.evaluate(model, full.loc[c])
        dags[tag] = dag
        results[tag] = {
            "n_edges": len(dag.edges()),
            "edges": [list(e) for e in dag.edges()],
            "blanket": sorted(model.get_markov_blanket("FORECAST")),
            "valid": {k: v for k, v in rv.items() if k not in ("predictions", "truth")},
            "test": {k: v for k, v in rt.items() if k not in ("predictions", "truth")},
        }

    # --- descriptive artefacts ------------------------------------------------------
    reg_wti = disc_s["WTI_M"]
    episodes = regime_episodes(reg_wti, monthly["WTI_M"])
    tm = pd.crosstab(reg_wti.shift().dropna().map(NAME), reg_wti.iloc[1:].map(NAME),
                     normalize="index").reindex(index=["bear", "stagnant", "bull"],
                                                columns=["bear", "stagnant", "bull"]).fillna(0)
    tm.index = [s.capitalize() for s in tm.index]
    tm.columns = [s.capitalize() for s in tm.columns]
    shares_up = (panel.diff().iloc[1:] > 0).mean()

    geo = monthly["GEOEVENT"]
    ev = geo[geo == 1].index
    event_mix = reg_wti.loc[reg_wti.index.isin(ev)].map(NAME).value_counts(normalize=True)
    base_mix = reg_wti.map(NAME).value_counts(normalize=True)

    # --- save ------------------------------------------------------------------------
    os.makedirs(OUT, exist_ok=True)
    disc_s.to_csv(os.path.join(OUT, "regimes_smoothed.csv"))
    disc_c.to_csv(os.path.join(OUT, "regimes_causal.csv"))
    episodes.to_csv(os.path.join(OUT, "regime_episodes_wti.csv"), index=False)
    diag.to_csv(os.path.join(OUT, "hmm_diagnostics.csv"), index=False)
    summary = {
        "panel": {"months": int(len(panel)), "variables": int(panel.shape[1]),
                  "train": len(train_idx), "valid": len(valid_idx), "test": len(test_idx)},
        "dropped": g.TAUTOLOGIES,
        "agreement_smoothed_vs_causal": agree.round(3).to_dict(),
        "mean_agreement": round(float(agree.mean()), 3),
        "share_up": shares_up.round(3).to_dict(),
        "regime_mix_all": base_mix.round(3).to_dict(),
        "regime_mix_event_months": event_mix.round(3).to_dict(),
        "transition_matrix": tm.round(3).to_dict(),
        "results": results,
    }
    with open(os.path.join(OUT, "gwp2_results.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # --- figures ----------------------------------------------------------------------
    paths = [
        fg.fig_regime_path(monthly["WTI_M"], reg_wti),
        fg.fig_emission_balance(shares_up),
        fg.fig_regime_shares(diag),
        fg.fig_transition(tm),
        fg.fig_decoding_gap(agree),
        fg.fig_accuracy(results),
        fg.fig_network(results["causal"]["edges"]),
    ]
    if verbose:
        print(f"panel {panel.shape[0]} months x {panel.shape[1]} variables")
        print(f"train {len(train_idx)} | valid {len(valid_idx)} | test {len(test_idx)}")
        print(f"mean agreement smoothed vs causal: {agree.mean():.3f}")
        for tag in ("lisse", "causal"):
            r = results[tag]
            print(f"{tag:7s} edges={r['n_edges']:2d} blanket={r['blanket']} "
                  f"valid={r['valid']['network_accuracy']:.3f} "
                  f"test={r['test']['network_accuracy']:.3f}")
        print("figures:", ", ".join(os.path.basename(p) for p in paths))
    return summary, disc_s, disc_c, diag, episodes, tm, models


def hill_climb_mwe(seed: int = 0):
    """Step 9a. Minimal working example of a Hill Climb search: Alvi's Listing 1
    rewritten for pgmpy 1.x. Eight variables are drawn independently, then two
    dependencies are imposed, so the recovered skeleton can be checked against a
    known truth. Returns the fitted DAG and its edges."""
    from pgmpy.estimators import HillClimbSearch

    rng = np.random.default_rng(seed)
    n = 2500
    d = pd.DataFrame(rng.integers(0, 3, size=(n, 8)), columns=list("ABCDEFGH"))
    d["A"] = d["A"] + d["B"] + d["C"]      # A depends on B and C
    d["H"] = d["G"] - d["A"]               # H depends on G and A
    train = d.iloc[: int(n * 0.75)].astype(str)
    hc = HillClimbSearch(train)
    dag = hc.estimate(scoring_method="k2", max_indegree=3, show_progress=False)
    return dag, sorted(dag.edges())


def hill_climb_blind_spot(seed: int = 0):
    """A companion to the example above. The same two dependencies are imposed
    modulo three, which leaves every pair marginally independent while the triples
    are not. A greedy single-edge search scores every first move at zero and stops
    immediately, returning the empty graph."""
    from pgmpy.estimators import HillClimbSearch

    rng = np.random.default_rng(seed)
    n = 2500
    d = pd.DataFrame(rng.integers(0, 3, size=(n, 8)), columns=list("ABCDEFGH"))
    d["A"] = (d["B"] + d["C"]) % 3
    d["H"] = (d["G"] - d["A"]) % 3
    train = d.iloc[: int(n * 0.75)].astype(str)
    hc = HillClimbSearch(train)
    dag = hc.estimate(scoring_method="k2", max_indegree=3, show_progress=False)
    return dag, sorted(dag.edges())


if __name__ == "__main__":
    run()
