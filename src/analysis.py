"""
GWP1 - exploratory analysis and stylized facts (Steps 7 and 8).

Everything here operates on the sterilized dataset produced by the cleaning layer.
Tests are reported with their statistic and p-value so that the written answers in the
report can be traced back to a number rather than an impression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, adfuller, kpss


def log_returns(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(s / s.shift(1))
    return r.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------------
# 8b - distribution of returns
# ----------------------------------------------------------------------------------
def distribution_profile(returns: dict) -> pd.DataFrame:
    rows = []
    for name, r in returns.items():
        r = pd.Series(r).dropna()
        jb, jb_p = stats.jarque_bera(r)
        try:
            df_t, loc_t, scale_t = stats.t.fit(r)
        except Exception:
            df_t = np.nan
        ks_n = stats.kstest(((r - r.mean()) / r.std()).values, stats.norm.cdf)
        rows.append(dict(
            Series=name, n=len(r),
            mean=r.mean(), std=r.std(),
            skewness=stats.skew(r), excess_kurtosis=stats.kurtosis(r, fisher=True),
            JB_stat=jb, JB_p=jb_p,
            KS_vs_normal_p=ks_n.pvalue,
            student_t_df=df_t,
            q01=r.quantile(0.01), q99=r.quantile(0.99),
            min=r.min(), max=r.max(),
        ))
    return pd.DataFrame(rows).set_index("Series")


def tail_comparison(r: pd.Series, k: float = 4.0) -> dict:
    """How many observations beyond k sigma, against the Gaussian expectation."""
    r = pd.Series(r).dropna()
    z = (r - r.mean()) / r.std()
    observed = int((z.abs() > k).sum())
    expected = float(2 * stats.norm.sf(k) * len(z))
    return dict(k=k, n=len(z), observed=observed, expected_if_normal=round(expected, 2),
                ratio=np.nan if expected == 0 else round(observed / expected, 1))


def aggregational_gaussianity(price: pd.Series, horizons=(1, 5, 21, 63)) -> pd.DataFrame:
    """Excess kurtosis should fall as the return horizon lengthens."""
    rows = []
    s = price.dropna()
    for h in horizons:
        r = np.log(s / s.shift(h)).dropna()
        rows.append(dict(horizon_days=h, n=len(r), excess_kurtosis=stats.kurtosis(r),
                         skewness=stats.skew(r), std=r.std()))
    return pd.DataFrame(rows).set_index("horizon_days")


# ----------------------------------------------------------------------------------
# 8c - autocorrelation
# ----------------------------------------------------------------------------------
def autocorrelation_profile(r: pd.Series, lags=(1, 5, 10, 22)) -> pd.DataFrame:
    r = pd.Series(r).dropna()
    out = {}
    for label, x in (("returns", r), ("absolute returns", r.abs()),
                     ("squared returns", r ** 2)):
        lb = acorr_ljungbox(x, lags=list(lags), return_df=True)
        ac = acf(x, nlags=max(lags), fft=True)
        out[label] = {f"acf_lag{l}": ac[l] for l in lags}
        out[label].update({f"LB_p_lag{l}": lb.loc[l, "lb_pvalue"] for l in lags})
    return pd.DataFrame(out).T


def arch_test(r: pd.Series, lags: int = 12) -> dict:
    r = pd.Series(r).dropna()
    stat, p, f_stat, f_p = het_arch(r, nlags=lags)
    return dict(lags=lags, LM_stat=stat, LM_p=p, F_stat=f_stat, F_p=f_p)


def acf_table(r: pd.Series, nlags: int = 30) -> pd.DataFrame:
    r = pd.Series(r).dropna()
    n = len(r)
    band = 1.96 / np.sqrt(n)
    return pd.DataFrame({
        "lag": np.arange(nlags + 1),
        "returns": acf(r, nlags=nlags, fft=True),
        "absolute_returns": acf(r.abs(), nlags=nlags, fft=True),
        "squared_returns": acf(r ** 2, nlags=nlags, fft=True),
    }).assign(band=band).set_index("lag")


# ----------------------------------------------------------------------------------
# 8a / 8d - other stylized facts
# ----------------------------------------------------------------------------------
def stationarity(series: dict) -> pd.DataFrame:
    rows = []
    for name, s in series.items():
        s = pd.Series(s).dropna()
        adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
        try:
            kpss_stat, kpss_p = kpss(s, regression="c", nlags="auto")[:2]
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan
        rows.append(dict(Series=name, n=len(s), ADF_stat=adf_stat, ADF_p=adf_p,
                         KPSS_stat=kpss_stat, KPSS_p=kpss_p,
                         verdict=("stationary" if adf_p < 0.05 and (np.isnan(kpss_p) or kpss_p > 0.05)
                                  else "unit root / inconclusive")))
    return pd.DataFrame(rows).set_index("Series")


def monthly_seasonality(r: pd.Series) -> pd.DataFrame:
    r = pd.Series(r).dropna()
    g = r.groupby(r.index.month)
    tab = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.size()})
    tab.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(tab)]
    groups = [v.values for _, v in g]
    f_stat, f_p = stats.f_oneway(*groups)
    k_stat, k_p = stats.kruskal(*groups)
    tab.attrs["anova_F"] = f_stat
    tab.attrs["anova_p"] = f_p
    tab.attrs["kruskal_H"] = k_stat
    tab.attrs["kruskal_p"] = k_p
    return tab


def volatility_asymmetry(r: pd.Series, vol_index: pd.Series) -> dict:
    """Equities show a leverage effect: negative returns raise implied volatility more
    than positive returns lower it. Oil is reported to behave differently because an
    upward price shock is itself a supply-scarcity signal."""
    # the change in the volatility index is taken on its own calendar FIRST. Aligning
    # before differencing would let a hole in the return series, such as the two days
    # the April 2020 negative settlement removes, be bridged by a multi-day change.
    df = pd.concat([pd.Series(r).rename("r"),
                    pd.Series(vol_index).diff().rename("dv")], axis=1).dropna()
    up = df[df["r"] > 0]
    down = df[df["r"] < 0]
    return dict(
        n=len(df),
        corr_return_vs_dvol=df["r"].corr(df["dv"]),
        mean_dvol_after_up=up["dv"].mean(),
        mean_dvol_after_down=down["dv"].mean(),
        asymmetry=down["dv"].mean() + up["dv"].mean(),
    )


def drawdown_profile(price: pd.Series) -> pd.DataFrame:
    s = price.dropna()
    peak = s.cummax()
    dd = s / peak - 1.0
    trough = dd.idxmin()
    return pd.DataFrame({
        "metric": ["max drawdown", "date of trough", "peak before trough",
                   "level at trough", "largest 1-day fall", "largest 1-day rise"],
        "value": [f"{dd.min():.1%}", str(trough.date()), f"{peak.loc[trough]:.2f}",
                  f"{s.loc[trough]:.2f}",
                  f"{log_returns(s).min():.2%}", f"{log_returns(s).max():.2%}"],
    })


def simple_returns(s: pd.Series) -> pd.Series:
    return s.astype(float).pct_change()


def negative_price_episode(price: pd.Series) -> pd.DataFrame:
    """A commodity price can cross zero; an equity price cannot. The log return, which is
    the workhorse of empirical finance, is undefined there. This table states exactly how
    many observations the logarithm discards and what the move actually was."""
    s = price.dropna()
    lr = log_returns(s)
    sr = simple_returns(s)
    neg = s[s <= 0]
    rows = []
    for d in neg.index:
        prev = s.shift(1).loc[d]
        nxt_idx = s.index[s.index.get_loc(d) + 1] if s.index.get_loc(d) + 1 < len(s) else None
        rows.append(dict(date=d.date(), previous_close=prev, close=s.loc[d],
                         simple_return=sr.loc[d], log_return=lr.loc[d],
                         next_close=None if nxt_idx is None else s.loc[nxt_idx]))
    out = pd.DataFrame(rows)
    out.attrs["log_returns_lost"] = int(lr.isna().sum() - 1)
    return out


def cross_asset_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna().corr()


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 252) -> pd.Series:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return df["a"].rolling(window).corr(df["b"])


def lead_lag(x: pd.Series, y: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    """Cross-correlation of y with x shifted by k. A positive k means x leads y."""
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    rows = []
    for k in range(-max_lag, max_lag + 1):
        rows.append(dict(lag=k, corr=df["x"].shift(k).corr(df["y"])))
    return pd.DataFrame(rows).set_index("lag")


def regime_summary(r: pd.Series, window: int = 63) -> pd.DataFrame:
    """Persistence of the volatility state, the empirical counterpart of the hidden
    Markov chain Alvi fits before discretising the data."""
    r = pd.Series(r).dropna()
    vol = r.rolling(window).std() * np.sqrt(252)
    med = vol.median()
    state = (vol > med).astype(float)
    state = state[vol.notna()]
    trans = pd.crosstab(state.shift(1).dropna(), state[1:], normalize="index")
    trans.index = ["low vol", "high vol"]
    trans.columns = ["to low vol", "to high vol"]
    trans.attrs["median_annualised_vol"] = med
    return trans
