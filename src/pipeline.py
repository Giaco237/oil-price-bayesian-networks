"""End-to-end build used both by the notebook and by the report generator."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import analysis as an  # noqa: E402
import cleaning as cl  # noqa: E402
import oil_data as od  # noqa: E402

CLEAN = os.path.join(ROOT, "data_clean")
FIG = os.path.join(ROOT, "figures")
os.makedirs(CLEAN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def build_raw():
    """Step 3: import and structure, by role."""
    gh = od.load_github_block()
    fred_m = od.load_fred_monthly()
    fred_d = od.load_fred_daily()

    monthly = fred_m.copy()
    # two series exist only in the mirrored block
    monthly["SP500"] = gh["SP500"].reindex(monthly.index)
    monthly["GOLD"] = gh["GOLD"].reindex(monthly.index)
    monthly["GEOEVENT"] = od.geopolitical_indicator(monthly.index)

    daily = fred_d.copy()
    # independent second source for the same quantity, kept for the cross-check in Step 5
    crosscheck = pd.DataFrame({
        "WTI_D_mirror": gh["WTI_D"].reindex(daily.index),
        "BRENT_D_mirror": gh["BRENT_D"].reindex(daily.index),
        "VIX_mirror": gh["VIX"].reindex(daily.index),
    })
    return monthly, daily, crosscheck


def clean_all(monthly: pd.DataFrame, daily: pd.DataFrame, crosscheck: pd.DataFrame):
    log = cl.EliminationLog()
    report = {}

    # ---- Student B: structural integrity first ---------------------------------
    monthly = cl.check_duplicates(monthly, log)
    daily = cl.check_duplicates(daily, log)
    monthly = cl.check_domain(monthly, log)
    daily = cl.check_domain(daily, log)

    report["coverage_monthly"] = cl.check_coverage(
        monthly, log, expected_end=monthly.index.max())
    report["coverage_daily"] = cl.check_coverage(
        daily, log, expected_end=daily.index.max())

    report["stale_wti"] = cl.check_stale_runs(daily["WTI_D"], "WTI_D", log, min_run=5)
    report["spread"] = cl.check_cross_consistency(daily["WTI_D"], daily["BRENT_D"], log,
                                                  max_spread=30.0)

    # source agreement: two independent mirrors of the same series
    agree = {}
    for a, b in (("WTI_D", "WTI_D_mirror"), ("BRENT_D", "BRENT_D_mirror"), ("VIX", "VIX_mirror")):
        joint = pd.concat([daily[a], crosscheck[b]], axis=1).dropna()
        diff = (joint[a] - joint[b]).abs()
        agree[a] = dict(overlapping_obs=len(joint), max_abs_diff=diff.max(),
                        n_disagreements=int((diff > 1e-6).sum()))
        if (diff > 1e-6).any():
            d0 = diff[diff > 1e-6].index.min()
            log.add(a, "B - bad data", "source disagreement", d0, None, None,
                    f"FLAGGED ({int((diff > 1e-6).sum())} obs)",
                    "the two independent mirrors of this series do not match")
    report["source_agreement"] = pd.DataFrame(agree).T

    # ---- Student A: extreme observations ----------------------------------------
    report["extremes_wti_d"] = cl.detect_extremes(daily["WTI_D"], "WTI_D", log)
    report["extremes_brent_d"] = cl.detect_extremes(daily["BRENT_D"], "BRENT_D", log)
    report["extremes_wti_m"] = cl.detect_extremes(monthly["WTI_M"], "WTI_M", log)
    report["nonpositive"] = cl.negative_or_zero_levels(daily["WTI_D"], "WTI_D", log)

    # ---- Student C: missing values ----------------------------------------------
    report["calendar_diagnostic"] = cl.calendar_diagnostic(daily, "WTI_D")
    report["naive_vs_calendar"] = cl.naive_vs_calendar_diagnostic(daily["WTI_D"])

    cal = cl.trading_calendar(daily["WTI_D"])
    daily = daily.reindex(cal)
    log.add("DAILY PANEL", "C - missing values", "analysis calendar", cal.min(), None, None,
            f"REINDEXED to {len(cal)} trading days",
            "non-trading days are structural absences, not holes to be filled")

    report["missing_monthly_before"] = cl.missingness_report(monthly)
    report["missing_daily_before"] = cl.missingness_report(daily)

    monthly_c = cl.impute(monthly, log, max_interpolate=2)
    # the target is never imputed: a return spanning a closed market is a multi-day
    # return, and inventing intermediate quotes would deflate the measured volatility
    protected = ["WTI_D"]
    daily_c = daily.copy()
    daily_c[[c for c in daily.columns if c not in protected]] = cl.impute(
        daily[[c for c in daily.columns if c not in protected]], log, max_interpolate=3)

    report["missing_monthly_after"] = cl.missingness_report(monthly_c)
    report["missing_daily_after"] = cl.missingness_report(daily_c)
    report["imputation_diagnostic"] = cl.imputation_diagnostic(daily["BRENT_D"],
                                                               daily_c["BRENT_D"])

    # ---- Step 6: sterilisation ---------------------------------------------------
    drop_m = [r.Series for r in report["coverage_monthly"].itertuples()
              if r.status != "active"]
    drop_d = [r.Series for r in report["coverage_daily"].itertuples()
              if r.status != "active"]

    # a series still riddled with interior holes after imputation cannot support a
    # conditional probability table: the evidence would be absent exactly when needed
    mis_d = report["missing_daily_after"].set_index("Series")
    patchy = [s for s in mis_d.index
              if mis_d.loc[s, "interior_missing"] / mis_d.loc[s, "n"] > 0.10]
    for s in patchy:
        log.add(s, "Step 6 - sterilisation", "interior missingness > 10%", None, None, None,
                "SERIES DROPPED",
                f"{int(mis_d.loc[s, 'interior_missing'])} interior holes remain after imputation")
    drop_d = sorted(set(drop_d) | set(patchy))

    monthly_s = monthly_c.drop(columns=drop_m, errors="ignore")
    daily_s = daily_c.drop(columns=drop_d, errors="ignore")

    # a series too patchy to survive at daily frequency can still be informative once
    # aggregated, provided the month is backed by enough quotes to be meaningful
    if "HH_GAS" in patchy and "HH_GAS" in daily_c.columns:
        g = daily_c["HH_GAS"]
        agg = g.groupby(g.index.to_period("M")).agg(["mean", "count"])
        hh = agg["mean"].where(agg["count"] >= 10)
        hh.index = hh.index.to_timestamp()
        monthly_s["HH_GAS_M"] = hh.reindex(monthly_s.index)
        log.add("HH_GAS", "Step 6 - sterilisation", "frequency reduction", None, None, None,
                "AGGREGATED to monthly",
                "monthly mean retained where the month holds at least 10 daily quotes")

    # leading gaps are availability, not defects: record when each survivor starts
    report["availability"] = pd.DataFrame([
        dict(Series=c, first_observation=monthly_s[c].dropna().index.min().date())
        for c in monthly_s.columns if monthly_s[c].notna().any()
    ]).sort_values("first_observation")

    # modelling window: start when the core oil and macro block is jointly observed
    core = ["WTI_M", "BRENT_M", "IGREA", "INDPRO", "CPIENG", "IP_OILGAS", "PPI_CRUDE"]
    core = [c for c in core if c in monthly_s.columns]
    first_core = monthly_s[core].dropna().index.min()
    monthly_s = monthly_s.loc[first_core:]
    log.add("MONTHLY PANEL", "Step 6 - sterilisation", "modelling window", first_core,
            None, None, "TRUNCATED",
            "sample starts when every core supply, demand and price variable is observed")

    # the last calendar month is almost always incomplete across providers
    last_full = monthly_s[core].dropna().index.max()
    if last_full < monthly_s.index.max():
        log.add("MONTHLY PANEL", "Step 6 - sterilisation", "ragged edge", last_full,
                None, None, "TRUNCATED",
                "months after this date are not yet published for every provider")
        monthly_s = monthly_s.loc[:last_full]

    report["dropped_monthly"] = drop_m
    report["dropped_daily"] = drop_d
    report["log"] = log.frame()
    return monthly_s, daily_s, report


def build_features(monthly_s: pd.DataFrame, daily_s: pd.DataFrame):
    """Derived quantities used by the EDA and by the belief network in GWP2."""
    m = monthly_s.copy()
    m["WTI_RET"] = an.log_returns(m["WTI_M"])
    m["WTI_REAL"] = m["WTI_M"] / m["CPIENG"] * 100.0
    m["BRENT_WTI_SPREAD"] = m["BRENT_M"] - m["WTI_M"]
    m["WTI_NEXT_RET"] = m["WTI_RET"].shift(-1)          # the forecasting target
    m["WTI_NEXT_DIR"] = np.sign(m["WTI_NEXT_RET"])

    d = daily_s.copy()
    d["WTI_RET"] = an.log_returns(d["WTI_D"])
    d["BRENT_RET"] = an.log_returns(d["BRENT_D"])
    d["NASDAQ_RET"] = an.log_returns(d["NASDAQ"])
    d["USD_RET"] = an.log_returns(d["USD_D"])
    d["WTI_VOL_63"] = d["WTI_RET"].rolling(63).std() * np.sqrt(252)
    d["NASDAQ_VOL_63"] = d["NASDAQ_RET"].rolling(63).std() * np.sqrt(252)
    return m, d


def run():
    monthly, daily, crosscheck = build_raw()
    monthly_s, daily_s, report = clean_all(monthly, daily, crosscheck)
    m, d = build_features(monthly_s, daily_s)
    m.to_csv(os.path.join(CLEAN, "monthly_sterilised.csv"))
    d.to_csv(os.path.join(CLEAN, "daily_sterilised.csv"))
    report["log"].to_csv(os.path.join(CLEAN, "elimination_log.csv"), index=False)
    return dict(monthly_raw=monthly, daily_raw=daily, crosscheck=crosscheck,
                monthly=m, daily=d, report=report)


if __name__ == "__main__":
    out = run()
    r = out["report"]
    pd.set_option("display.width", 220)
    print("monthly:", out["monthly"].shape, out["monthly"].index.min().date(),
          out["monthly"].index.max().date())
    print("daily:  ", out["daily"].shape, out["daily"].index.min().date(),
          out["daily"].index.max().date())
    print("\ndropped monthly:", r["dropped_monthly"], "| dropped daily:", r["dropped_daily"])
    print("\n--- coverage (monthly) ---\n", r["coverage_monthly"].to_string(index=False))
    print("\n--- source agreement ---\n", r["source_agreement"].to_string())
    print("\n--- elimination log (", len(r["log"]), "rows ) head ---")
    print(r["log"].head(25).to_string(index=False, max_colwidth=46))
