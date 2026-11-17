"""
GWP1 - cleaning layer.

Section 4.1.2 of Alvi (2018) distinguishes three defects: formatting, holes marked by a
single dash, and series of uneven quality across providers. This module generalises that
treatment into the three responsibilities defined in Step 5 of the assignment.

  Student A - extreme outliers
  Student B - wrong, questionable or duplicated values
  Student C - missing values

Each routine returns the cleaned object and an audit record. Nothing is dropped silently:
every removal or substitution is appended to an elimination log that feeds Step 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# Audit log
# ----------------------------------------------------------------------------------
class EliminationLog:
    def __init__(self):
        self.rows = []

    def add(self, series, stage, rule, date, old, new, decision, rationale):
        self.rows.append(dict(Series=series, Stage=stage, Rule=rule,
                              Date=None if date is None else pd.Timestamp(date).date(),
                              Old=old, New=new, Decision=decision, Rationale=rationale))

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=["Series", "Stage", "Rule", "Date", "Old", "New",
                                         "Decision", "Rationale"])
        return pd.DataFrame(self.rows)


# ----------------------------------------------------------------------------------
# Student A - extreme outliers
# ----------------------------------------------------------------------------------
def log_returns(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(s / s.shift(1))
    return r.replace([np.inf, -np.inf], np.nan)


def modified_zscore(x: pd.Series) -> pd.Series:
    """Iglewicz and Hoaglin statistic. The MAD is robust to the very outliers we hunt,
    which the sample standard deviation is not."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.nan, index=x.index)
    return 0.6745 * (x - med) / mad


def detect_extremes(s: pd.Series, name: str, log: EliminationLog,
                    z_threshold: float = 8.0, iqr_k: float = 6.0) -> pd.DataFrame:
    """Flag extreme observations on the RETURN scale, not the level scale.

    A price of 140 dollars is not an outlier, it is a peak reached through a sequence of
    ordinary moves. What is extreme is a one-period move that the rest of the sample does
    not support. Three independent screens must agree before an observation is labelled.
    """
    r = log_returns(s).dropna()
    mz = modified_zscore(r)
    q1, q3 = r.quantile(0.25), r.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    sd = r.std()
    flags = pd.DataFrame({
        "return": r,
        "modified_z": mz,
        "sigma_multiple": r / sd,
        "by_modified_z": mz.abs() > z_threshold,
        "by_iqr": (r < lo) | (r > hi),
        "by_sigma": r.abs() > 5 * sd,
    })
    flags["n_screens"] = flags[["by_modified_z", "by_iqr", "by_sigma"]].sum(axis=1)
    ext = flags[flags["n_screens"] >= 2].copy()
    ext["level"] = s.reindex(ext.index)
    ext["prev_level"] = s.shift(1).reindex(ext.index)
    for d, row in ext.iterrows():
        log.add(name, "A - extreme outlier", f"{int(row.n_screens)}/3 screens", d,
                round(float(row.prev_level), 4), round(float(row.level), 4),
                "FLAGGED, retained",
                f"log return {row['return']:.4f} = {row.sigma_multiple:.1f} sigma")
    return ext.sort_values("return")


def negative_or_zero_levels(s: pd.Series, name: str, log: EliminationLog) -> pd.Series:
    bad = s[s <= 0]
    for d, v in bad.items():
        log.add(name, "A - extreme outlier", "non-positive price level", d, float(v), float(v),
                "RETAINED, excluded from log transform",
                "economically real but undefined under the logarithm")
    return bad


# ----------------------------------------------------------------------------------
# Student B - wrong, questionable or duplicated values
# ----------------------------------------------------------------------------------
DOMAIN = {
    # name -> (lower bound, upper bound, whether the bound is strict)
    "VIX": (0.0, 200.0),
    "OVX": (0.0, 400.0),
    "CAPUTL_OILGAS": (0.0, 105.0),
    "UNRATE": (0.0, 30.0),
    "GEPU": (0.0, 1000.0),
    "USEPU": (0.0, 1000.0),
}


def check_duplicates(df: pd.DataFrame, log: EliminationLog) -> pd.DataFrame:
    dup = df.index.duplicated(keep="first")
    if dup.any():
        for d in df.index[dup]:
            log.add("ALL", "B - bad data", "duplicated timestamp", d, None, None,
                    "REMOVED", "second occurrence of an existing date")
        df = df[~dup]
    return df


def check_domain(df: pd.DataFrame, log: EliminationLog) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in DOMAIN.items():
        if col not in out.columns:
            continue
        bad = out[col][(out[col] < lo) | (out[col] > hi)].dropna()
        for d, v in bad.items():
            log.add(col, "B - bad data", f"outside [{lo}, {hi}]", d, float(v), np.nan,
                    "SET TO MISSING", "value is impossible for this indicator")
            out.loc[d, col] = np.nan
    return out


def check_stale_runs(s: pd.Series, name: str, log: EliminationLog,
                     min_run: int = 5) -> pd.DataFrame:
    """A quoted price repeating identically for several consecutive observations is the
    classic signature of a carried-forward quote rather than a traded price."""
    v = s.dropna()
    grp = (v != v.shift(1)).cumsum()
    runs = v.groupby(grp).agg(start=lambda x: x.index[0], end=lambda x: x.index[-1],
                              n="size", value="first")
    runs = runs[runs["n"] >= min_run]
    for _, row in runs.iterrows():
        log.add(name, "B - bad data", f"stale run of {int(row.n)} observations", row.start,
                float(row.value), float(row.value), "FLAGGED, retained",
                f"identical quote repeated until {pd.Timestamp(row.end).date()}")
    return runs


def check_coverage(df: pd.DataFrame, log: EliminationLog, expected_end: pd.Timestamp,
                   tolerance_months: int = 6) -> pd.DataFrame:
    """A series that stops updating is not missing data, it is a discontinued series.
    Carrying it forward would inject a constant into the network."""
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        last = s.index.max()
        gap = (expected_end.to_period("M") - last.to_period("M")).n
        status = "active" if gap <= tolerance_months else "STOPS EARLY"
        rows.append(dict(Series=col, first=s.index.min().date(), last=last.date(),
                         months_behind=gap, status=status))
        if status != "active":
            log.add(col, "B - bad data", "coverage ends early", last, None, None,
                    "SERIES DROPPED", f"last observation {last.date()}, {gap} months behind")
    return pd.DataFrame(rows).sort_values("months_behind", ascending=False)


def check_cross_consistency(wti: pd.Series, brent: pd.Series, log: EliminationLog,
                            max_spread: float = 30.0) -> pd.DataFrame:
    """Brent and WTI are substitutable light sweet crudes. A spread far outside its own
    historical range points either to a genuine logistics dislocation or to a misaligned
    series, and in both cases must be surfaced rather than averaged away."""
    both = pd.concat([wti.rename("WTI"), brent.rename("BRENT")], axis=1).dropna()
    both["spread"] = both["BRENT"] - both["WTI"]
    bad = both[both["spread"].abs() > max_spread]
    for d, row in bad.iterrows():
        log.add("BRENT-WTI", "B - bad data", f"|spread| > {max_spread}", d,
                round(float(row.spread), 2), round(float(row.spread), 2),
                "FLAGGED, retained", "spread outside the plausible arbitrage band")
    return both


# ----------------------------------------------------------------------------------
# Student C - missing values
# ----------------------------------------------------------------------------------
def trading_calendar(target: pd.Series) -> pd.DatetimeIndex:
    """The analysis calendar is the set of dates on which the forecast target actually
    trades. On a calendar built from the union of every market, a US holiday looks like a
    missing value for WTI when in fact no price existed to be recorded. Treating a
    structural absence as a hole is the most common silent error in multi-market panels:
    it manufactures observations and deflates measured volatility."""
    return pd.DatetimeIndex(target.dropna().index)


def calendar_diagnostic(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Separate structural absence from genuine missingness for each series."""
    cal = trading_calendar(df[target_col])
    rows = []
    for col in df.columns:
        s = df[col]
        obs = s.dropna()
        if obs.empty:
            continue
        span = s.loc[obs.index.min():obs.index.max()]
        on_cal = s.reindex(cal).loc[obs.index.min():obs.index.max()]
        rows.append(dict(
            Series=col,
            obs=len(obs),
            holes_full_weekday_calendar=int(span.isna().sum()),
            holes_on_trading_calendar=int(on_cal.isna().sum()),
            pct_on_trading_calendar=round(100 * on_cal.isna().mean(), 2),
        ))
    out = pd.DataFrame(rows)
    out["spurious_holes_avoided"] = (out["holes_full_weekday_calendar"]
                                     - out["holes_on_trading_calendar"])
    return out.sort_values("pct_on_trading_calendar", ascending=False)


def naive_vs_calendar_diagnostic(target: pd.Series) -> pd.DataFrame:
    """Quantify what naive interpolation over non-trading days does to the object we model."""
    weekday = pd.date_range(target.dropna().index.min(), target.dropna().index.max(),
                            freq="B")
    naive = target.reindex(weekday).interpolate(method="time")
    honest = target.dropna()

    def moments(x, label):
        r = log_returns(x).dropna()
        return pd.Series(dict(n=len(r), std=r.std(), annualised_vol=r.std() * np.sqrt(252),
                              skewness=r.skew(), excess_kurtosis=r.kurtosis(),
                              worst_day=r.min(), best_day=r.max()), name=label)

    return pd.concat([moments(honest, "trading calendar, no imputation"),
                      moments(naive, "weekday calendar, interpolated")], axis=1)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        obs = s.dropna()
        if obs.empty:
            rows.append(dict(Series=col, n=len(s), n_missing=len(s), pct_missing=100.0,
                             interior_missing=0, longest_gap=0, leading=len(s), trailing=0))
            continue
        first, last = obs.index.min(), obs.index.max()
        interior = s.loc[first:last]
        isna = interior.isna()
        grp = (~isna).cumsum()[isna]
        longest = grp.value_counts().max() if len(grp) else 0
        rows.append(dict(Series=col, n=len(s), n_missing=int(s.isna().sum()),
                         pct_missing=round(100 * s.isna().mean(), 2),
                         interior_missing=int(isna.sum()), longest_gap=int(longest),
                         leading=int((s.index < first).sum()),
                         trailing=int((s.index > last).sum())))
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


def impute(df: pd.DataFrame, log: EliminationLog, max_interpolate: int = 5,
           flow_vars: tuple = ()) -> pd.DataFrame:
    """Imputation policy.

    Interior gaps shorter than `max_interpolate` observations are filled by time-weighted
    linear interpolation: for a price that is continuous in time, the straight line between
    two observed quotes is the minimum-assumption estimate.

    Longer interior gaps are left missing. Filling a long hole invents a trend the data
    never showed and would bias the volatility estimates downwards.

    Leading gaps are never filled. The series simply did not exist yet, and backfilling
    would create history out of nothing. Alvi backfills; we depart from that choice and
    say so.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        obs = s.dropna()
        if obs.empty:
            continue
        first, last = obs.index.min(), obs.index.max()
        interior = s.loc[first:last]
        isna = interior.isna()
        if not isna.any():
            continue
        grp = (~isna).cumsum()
        sizes = grp[isna].map(grp[isna].value_counts())
        short_idx = interior.index[isna & (sizes <= max_interpolate)]
        long_idx = interior.index[isna & (sizes > max_interpolate)]
        if len(short_idx):
            filled = s.loc[first:last].interpolate(method="time", limit_area="inside")
            out.loc[short_idx, col] = filled.loc[short_idx]
            log.add(col, "C - missing values", f"gap <= {max_interpolate}", short_idx.min(),
                    None, None, f"INTERPOLATED ({len(short_idx)} obs)",
                    "time-weighted linear interpolation inside the observed span")
        if len(long_idx):
            log.add(col, "C - missing values", f"gap > {max_interpolate}", long_idx.min(),
                    None, None, f"LEFT MISSING ({len(long_idx)} obs)",
                    "long interior gap, imputation would fabricate a trend")
    return out


def imputation_diagnostic(raw: pd.Series, imputed: pd.Series) -> pd.DataFrame:
    """Does the imputation change the object we are about to model? Compare the first four
    moments of the log-return distribution before and after."""
    def moments(x):
        r = log_returns(x).dropna()
        return dict(n=len(r), mean=r.mean(), std=r.std(), skew=r.skew(), kurtosis=r.kurtosis())
    return pd.DataFrame({"before": moments(raw), "after": moments(imputed)})
