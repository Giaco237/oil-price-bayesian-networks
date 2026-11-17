"""Collect every number quoted in the report into one place, so the written text and the
notebook cannot drift apart."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import analysis as an  # noqa: E402
import pipeline as pl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect():
    out = pl.run()
    d, m, rep = out["daily"], out["monthly"], out["report"]
    R = {}

    R["sample"] = dict(
        monthly_start=str(m.index.min().date()), monthly_end=str(m.index.max().date()),
        monthly_obs=int(len(m)), monthly_vars=int(len(m.columns)),
        daily_start=str(d.index.min().date()), daily_end=str(d.index.max().date()),
        daily_obs=int(len(d)),
        raw_monthly_vars=int(out["monthly_raw"].shape[1]),
        raw_daily_vars=int(out["daily_raw"].shape[1]),
        weekday_rows=int(out["daily_raw"].shape[0]),
    )

    ret = {"WTI daily": d["WTI_RET"], "Brent daily": d["BRENT_RET"],
           "NASDAQ daily": d["NASDAQ_RET"], "WTI monthly": m["WTI_RET"]}
    dp = an.distribution_profile(ret)
    R["distribution"] = json.loads(dp.to_json(orient="index"))
    R["tails"] = {k: an.tail_comparison(v) for k, v in ret.items()}
    R["aggregation"] = json.loads(an.aggregational_gaussianity(d["WTI_D"]).to_json(orient="index"))

    ac = an.autocorrelation_profile(d["WTI_RET"])
    R["acf_wti"] = json.loads(ac.to_json(orient="index"))
    R["acf_ndx"] = json.loads(an.autocorrelation_profile(d["NASDAQ_RET"]).to_json(orient="index"))
    R["arch_wti"] = {k: float(v) for k, v in an.arch_test(d["WTI_RET"]).items()}
    R["arch_ndx"] = {k: float(v) for k, v in an.arch_test(d["NASDAQ_RET"]).items()}

    st = an.stationarity({"WTI level (monthly)": m["WTI_M"],
                          "WTI real (monthly)": m["WTI_REAL"],
                          "WTI log return (monthly)": m["WTI_RET"],
                          "WTI level (daily)": d["WTI_D"].dropna(),
                          "WTI log return (daily)": d["WTI_RET"],
                          "Brent-WTI spread": m["BRENT_WTI_SPREAD"]})
    R["stationarity"] = json.loads(st.to_json(orient="index"))

    seas = an.monthly_seasonality(m["WTI_RET"])
    R["seasonality"] = dict(table=json.loads(seas.to_json(orient="index")),
                            anova_p=float(seas.attrs["anova_p"]),
                            kruskal_p=float(seas.attrs["kruskal_p"]),
                            anova_F=float(seas.attrs["anova_F"]),
                            kruskal_H=float(seas.attrs["kruskal_H"]))

    R["leverage_oil"] = {k: float(v) for k, v in
                         an.volatility_asymmetry(d["WTI_RET"], d["OVX"]).items()}
    R["leverage_equity"] = {k: float(v) for k, v in
                            an.volatility_asymmetry(d["NASDAQ_RET"], d["VIX"]).items()}

    reg_o = an.regime_summary(d["WTI_RET"])
    reg_e = an.regime_summary(d["NASDAQ_RET"])
    R["regimes"] = dict(
        oil=json.loads(reg_o.to_json(orient="index")),
        equity=json.loads(reg_e.to_json(orient="index")),
        oil_median_vol=float(reg_o.attrs["median_annualised_vol"]),
        equity_median_vol=float(reg_e.attrs["median_annualised_vol"]),
        oil_expected_duration=float(1 / (1 - reg_o.loc["high vol", "to high vol"])),
        equity_expected_duration=float(1 / (1 - reg_e.loc["high vol", "to high vol"])),
    )

    neg = an.negative_price_episode(d["WTI_D"])
    R["negative_price"] = dict(rows=json.loads(neg.to_json(orient="records")),
                               log_returns_lost=int(neg.attrs["log_returns_lost"]))

    mm = pd.DataFrame({
        "WTI return": m["WTI_RET"],
        "global activity": m["IGREA"].diff(),
        "US dollar": np.log(m["USD_M"]).diff(),
        "S&P 500": np.log(m["SP500"]).diff(),
        "gold": np.log(m["GOLD"]).diff(),
        "oil and gas output": np.log(m["IP_OILGAS"]).diff(),
        "capacity use": m["CAPUTL_OILGAS"].diff(),
        "policy uncertainty": np.log(m["GEPU"]).diff(),
        "crude PPI": np.log(m["PPI_CRUDE"]).diff(),
        "10-year yield": m["GS10"].diff(),
    })
    corr = mm.corr(min_periods=120)
    R["correlation_with_wti"] = {k: (None if pd.isna(v) else round(float(v), 4))
                                 for k, v in corr["WTI return"].items()}
    R["collinear_pairs"] = [
        dict(a=a, b=b, corr=round(float(corr.loc[a, b]), 4))
        for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
        if not pd.isna(corr.loc[a, b]) and abs(corr.loc[a, b]) > 0.85]

    ll = an.lead_lag(m["IGREA"].diff(), m["WTI_RET"], 6)
    R["lead_lag_igrea"] = {int(k): round(float(v), 4) for k, v in ll["corr"].items()}

    dd = d[["WTI_RET", "BRENT_RET", "NASDAQ_RET", "USD_RET"]]
    R["daily_corr"] = json.loads(dd.corr().round(4).to_json(orient="index"))

    R["cleaning"] = dict(
        log_rows=int(len(rep["log"])),
        by_stage=json.loads(rep["log"].groupby("Stage").size().to_json()),
        extremes_wti_daily=int(len(rep["extremes_wti_d"])),
        extremes_brent_daily=int(len(rep["extremes_brent_d"])),
        extremes_wti_monthly=int(len(rep["extremes_wti_m"])),
        dropped_daily=rep["dropped_daily"], dropped_monthly=rep["dropped_monthly"],
        source_agreement=json.loads(rep["source_agreement"].to_json(orient="index")),
        calendar=json.loads(rep["calendar_diagnostic"].to_json(orient="records")),
        naive_vs_calendar=json.loads(rep["naive_vs_calendar"].to_json(orient="index")),
        spread_flags=int((rep["spread"]["spread"].abs() > 30).sum()),
        stale_runs=int(len(rep["stale_wti"])),
    )

    R["drawdown"] = json.loads(an.drawdown_profile(d["WTI_D"]).to_json(orient="records"))
    R["availability"] = json.loads(rep["availability"].to_json(orient="records"))

    with open(os.path.join(ROOT, "data_clean", "results.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    return R, out


if __name__ == "__main__":
    R, _ = collect()
    print(json.dumps(R, indent=1, default=str)[:6000])
