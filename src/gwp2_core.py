"""
GWP2 - MScFE 660 Risk Management - Group 18580
Regime detection by hidden Markov models and structure learning of a belief network
on the oil panel built in GWP1.

The module mirrors the design of Figure 3.1 of Alvi (2018): clean data -> identify
regime (hmms) -> train network (pgmpy) -> validate -> test.
"""
import contextlib
import io
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:                      # noqa: E402
    import hmms
    HMMS_BACKEND = "hmms (Cython)"
except ImportError:       # the 2017 Cython package does not build on NumPy 2
    import hmms_compat as hmms
    HMMS_BACKEND = "hmms_compat (pure NumPy, numerically identical)" 

N_STATES = 3          # bear, stagnant, bull
N_SYMBOLS = 2         # 0 = fall, 1 = rise
BAUM_WELCH_ITERS = 100
N_CHUNKS = 32         # Alvi splits the emission sequence into 32 blocks
BEAR, STAGNANT, BULL = 0, 1, 2
REGIME_NAME = {BEAR: "bear", STAGNANT: "stagnant", BULL: "bull"}


# ----------------------------------------------------------------------------------
# 1. emissions
# ----------------------------------------------------------------------------------
def emission_sequence(series: pd.Series) -> pd.Series:
    """First-order integration followed by the parity of the change.

    1 when the variable rose relative to the previous month, 0 otherwise.
    The first observation is lost because it has no predecessor.
    """
    diff = series.diff().iloc[1:]
    return diff.apply(lambda x: 1 if x > 0 else 0).astype(np.int64)


# ----------------------------------------------------------------------------------
# 2. Baum-Welch
# ----------------------------------------------------------------------------------
def fit_hmm(emissions: pd.Series, seed: int = 0, iters: int = BAUM_WELCH_ITERS,
            n_chunks: int = N_CHUNKS) -> "hmms.DtHMM":
    """Estimate (Pi, A, B) of a discrete-time HMM by Baum-Welch.

    hmms expects a list of emission sequences. Alvi passes
    np.array_split(seq, 32), which produces 32 blocks rather than blocks of
    length 32; the behaviour is kept so the comparison stays like for like,
    and the block count is a documented parameter rather than a constant.
    """
    np.random.seed(seed)
    model = hmms.DtHMM.random(N_STATES, N_SYMBOLS)
    blocks = np.array_split(emissions.values.astype(np.int64), n_chunks)
    with contextlib.redirect_stdout(io.StringIO()):
        model.baum_welch(blocks, iters)
    return model


# ----------------------------------------------------------------------------------
# 3. Viterbi
# ----------------------------------------------------------------------------------
def decode(model: "hmms.DtHMM", emissions: pd.Series) -> pd.Series:
    """Most likely hidden-state path for an emission sequence."""
    with contextlib.redirect_stdout(io.StringIO()):
        _, path = model.viterbi(emissions.values.astype(np.int64))
    return pd.Series(np.asarray(path), index=emissions.index, dtype=int)


# ----------------------------------------------------------------------------------
# 4. latent meaning of each hidden state
# ----------------------------------------------------------------------------------
def regime_map(states: pd.Series, changes: pd.Series) -> dict:
    """Rank the raw Viterbi labels by the mean change they carry.

    The HMM returns state indices with no economic content. Sorting the states
    by the arithmetic mean of the underlying change assigns 0 to the most
    negative state (bear), 1 to the middle state (stagnant) and 2 to the most
    positive state (bull).
    """
    means = changes.groupby(states).mean().sort_values()
    return {raw: rank for rank, raw in enumerate(means.index)}


def relabel(states: pd.Series, changes: pd.Series) -> pd.Series:
    return states.map(regime_map(states, changes)).astype(int)


# ----------------------------------------------------------------------------------
# 5. the discretisation pipeline over a whole panel
# ----------------------------------------------------------------------------------
def split_index(index: pd.DatetimeIndex, ratios=(0.80, 0.10, 0.10)):
    n = len(index)
    a = int(n * ratios[0])
    b = int(n * (ratios[0] + ratios[1]))
    return index[:a], index[a:b], index[b:]


def discretise_panel(panel: pd.DataFrame, train_idx, passthrough=(), seed0: int = 0):
    """Train one HMM per variable on the training window, then decode every window.

    Variables listed in `passthrough` are already categorical and bypass the
    HMM entirely: running a three-state regime model on a binary event flag
    would invent two states the variable cannot occupy.
    """
    models, decoded, diagnostics = {}, {}, []
    for k, col in enumerate(panel.columns):
        series = panel[col].dropna()
        if col in passthrough:
            decoded[col] = series.iloc[1:].astype(int)
            continue
        tr = series.loc[series.index.isin(train_idx)]
        if len(tr) < 60:
            diagnostics.append({"variable": col, "status": "skipped, training window too short"})
            continue
        model = fit_hmm(emission_sequence(tr), seed=seed0 + k)
        models[col] = model
        emis = emission_sequence(series)
        raw = decode(model, emis)
        chg = series.diff().iloc[1:]
        decoded[col] = relabel(raw, chg)
        shares = decoded[col].value_counts(normalize=True).reindex([0, 1, 2]).fillna(0)
        diagnostics.append({
            "variable": col,
            "status": "ok",
            "n": len(decoded[col]),
            "share_bear": round(float(shares[0]), 3),
            "share_stagnant": round(float(shares[1]), 3),
            "share_bull": round(float(shares[2]), 3),
            "persistence": round(float(np.mean(np.diag(model.a))), 3),
        })
    disc = pd.DataFrame(decoded)
    return disc, models, pd.DataFrame(diagnostics)


# ----------------------------------------------------------------------------------
# 6. loading
# ----------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_monthly(path=None) -> pd.DataFrame:
    path = path or os.path.join(HERE, "data_clean", "monthly_sterilised.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


# variables carried over from GWP1, by the member who owned the domain
MACRO = ["CPIENG", "FEDFUNDS", "GEPU", "GS10", "IGREA", "INDPRO", "M2", "UNRATE",
         "USD_M", "USEPU", "GEOEVENT"]
MICRO = ["CAP_OILGAS", "CAPUTL_OILGAS", "GAS_EU", "IP_DRILL", "IP_OILGAS",
         "PPI_CRUDE", "PPI_DRILL", "PPI_OILGAS"]
FINANCIAL = ["BRENT_M", "GOLD", "HH_GAS_M", "SP500", "WTI_M"]

# removed before any structure search, per section 5.4 of GWP1
TAUTOLOGIES = {
    "PPI_CRUDE": "producer price index for crude petroleum, measures the target rather than driving it",
    "CAPUTL_OILGAS": "capacity utilisation is output divided by capacity, already carried by IP_OILGAS and CAP_OILGAS",
    "BRENT_M": "second light sweet benchmark, co-moves with WTI by construction",
}

# available only from a later date; see section 3.4 of GWP1
LATE_START = {"USD_M": "2006-01", "GEPU": "1997-01", "HH_GAS_M": "1997-01", "GAS_EU": "1992-01"}


# ----------------------------------------------------------------------------------
# 7. the belief network
# ----------------------------------------------------------------------------------
EXPERT_EDGES = [
    ("CAP_OILGAS", "IP_OILGAS"),   # installed capacity constrains extraction output
    ("IP_DRILL", "CAP_OILGAS"),    # drilling today is capacity tomorrow
    ("IP_OILGAS", "WTI_M"),        # supply side
    ("INDPRO", "IGREA"),           # industrial production feeds global real activity
    ("IGREA", "WTI_M"),            # demand side
    ("GEOEVENT", "WTI_M"),         # the geopolitical channel Alvi names but never measures
    ("WTI_M", "FORECAST"),         # the target is next month's regime of the same price
]


def add_target(disc: pd.DataFrame, target_col: str = "WTI_M") -> pd.DataFrame:
    """FORECAST_t is the regime of the target one month ahead."""
    out = disc.copy()
    out["FORECAST"] = out[target_col].shift(-1)
    return out.dropna().astype(int)


def as_categorical(frame: pd.DataFrame) -> pd.DataFrame:
    """pgmpy 1.x reads integer columns as continuous; states are passed as strings."""
    return frame.astype(int).astype(str)


def learn_structure(train: pd.DataFrame, score: str = "k2", seed_edges=None,
                    max_indegree: int = 3):
    from pgmpy.base import DAG
    from pgmpy.estimators import HillClimbSearch

    start = DAG()
    start.add_nodes_from(train.columns)
    if seed_edges:
        start.add_edges_from([e for e in seed_edges
                              if e[0] in train.columns and e[1] in train.columns])
    hc = HillClimbSearch(as_categorical(train))
    # A node with p parents needs 3**p parent configurations. With 349 training
    # months, three parents already leaves about four observations per cell, so the
    # in-degree is capped. Alvi imposes no cap, which is one reason his graph is
    # unstable between runs.
    return hc.estimate(scoring_method=score, start_dag=start,
                       max_indegree=max_indegree, show_progress=False)


def fit_parameters(dag, train: pd.DataFrame, prior: str = "K2"):
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.estimators import BayesianEstimator

    cat = as_categorical(train)
    model = DiscreteBayesianNetwork(list(dag.edges()))
    model.add_nodes_from(cat.columns)
    states = {c: sorted(cat[c].unique().tolist()) for c in cat.columns}
    # pgmpy 1.x no longer forwards prior_type through DiscreteBayesianNetwork.fit,
    # so the estimator is called directly. Alvi's 2018 call signature is dead code today.
    est = BayesianEstimator(model, cat, state_names=states)
    model.add_cpds(*est.get_parameters(prior_type=prior))
    return model


def evaluate(model, frame: pd.DataFrame, target: str = "FORECAST") -> dict:
    """Accuracy of the network against two baselines a reader will ask about."""
    cat = as_categorical(frame)
    evidence = cat.drop(columns=[target])
    truth = cat[target].values
    pred = model.predict(evidence)[target].values
    persistence = cat["WTI_M"].values             # tomorrow looks like today
    majority = np.full(len(truth), pd.Series(truth).mode().iloc[0])
    return {
        "n": len(truth),
        "network_accuracy": round(float((pred == truth).mean()), 4),
        "persistence_accuracy": round(float((persistence == truth).mean()), 4),
        "majority_accuracy": round(float((majority == truth).mean()), 4),
        "chance": round(1.0 / frame[target].nunique(), 4),
        "predictions": pred,
        "truth": truth,
    }


# ----------------------------------------------------------------------------------
# 8. causal (online) decoding
# ----------------------------------------------------------------------------------
def decode_online(model: "hmms.DtHMM", emissions: pd.Series, warmup: int = 24) -> pd.Series:
    """Decode state t using only emissions up to t.

    Viterbi maximises over the whole sequence, so the label it assigns to month t
    is informed by months t+1 ... T. That is a smoothing operation, legitimate for
    describing history and inadmissible for forecasting. This variant re-runs the
    recursion on the expanding window o_1..o_t and keeps the last state, which is
    what a desk could actually have known at t.
    """
    vals = emissions.values.astype(np.int64)
    path = np.empty(len(vals), dtype=int)
    with contextlib.redirect_stdout(io.StringIO()):
        for t in range(len(vals)):
            upto = vals[: t + 1]
            if len(upto) < 2:
                path[t] = 0
                continue
            _, p = model.viterbi(upto)
            path[t] = int(np.asarray(p)[-1])
    out = pd.Series(path, index=emissions.index, dtype=int)
    return out.iloc[warmup:]


def discretise_panel_online(panel: pd.DataFrame, train_idx, models: dict,
                            passthrough=(), warmup: int = 24) -> pd.DataFrame:
    """Same discretisation as `discretise_panel` but with causal decoding.

    The state-to-regime mapping is the one estimated on the training window only,
    so no future information enters through the relabelling either.
    """
    decoded = {}
    for col in panel.columns:
        series = panel[col].dropna()
        if col in passthrough:
            decoded[col] = series.iloc[1:].astype(int).iloc[warmup:]
            continue
        if col not in models:
            continue
        model = models[col]
        emis = emission_sequence(series)
        raw_online = decode_online(model, emis, warmup=warmup)
        tr_emis = emission_sequence(series.loc[series.index.isin(train_idx)])
        tr_raw = decode(model, tr_emis)
        tr_chg = series.loc[series.index.isin(train_idx)].diff().iloc[1:]
        mapping = regime_map(tr_raw, tr_chg)
        decoded[col] = raw_online.map(mapping).fillna(1).astype(int)
    return pd.DataFrame(decoded)
