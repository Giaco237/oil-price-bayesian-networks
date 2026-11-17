"""Figure generation for GWP1 (Step 7).

Student A - distributional plots
Student B - time-series plots
Student C - multivariate plots
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

import analysis as an
import viz

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG, exist_ok=True)
S = viz.SERIES


def _save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ==================================================================================
# Student A - distributional
# ==================================================================================
def fig_return_distributions(d, m):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    panels = [("WTI, daily", d["WTI_RET"], S[0]),
              ("NASDAQ Composite, daily", d["NASDAQ_RET"], S[1]),
              ("WTI, monthly", m["WTI_RET"], S[2])]
    for ax, (label, r, col) in zip(axes, panels):
        r = r.dropna()
        lo, hi = r.min(), r.max()
        pad = 0.04 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 80)
        dens, edges = np.histogram(r, bins=bins, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        nz = dens > 0
        floor = dens[nz].min() * 0.4
        ax.vlines(centres[nz], floor, dens[nz], color=col, lw=2.0, alpha=0.95,
                  label="observed")
        x = np.linspace(lo - pad, hi + pad, 600)
        ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=viz.INK, lw=1.8,
                ls="--", label="normal fit")
        df_t, loc_t, sc_t = stats.t.fit(r)
        ax.plot(x, stats.t.pdf(x, df_t, loc_t, sc_t), color=viz.INK_2, lw=1.8,
                label=f"Student t fit, df {df_t:.1f}")
        ax.set_yscale("log")
        ax.set_ylim(floor, dens.max() * 3)
        ax.set_title(label)
        ax.set_xlabel("log return")
        ax.set_ylabel("density, log scale")
        viz.subtitle(ax, f"excess kurtosis {stats.kurtosis(r):.1f}, skewness {stats.skew(r):.2f}")
    h, l = axes[0].get_legend_handles_labels()
    l = ["observed", "normal fit", "Student t fit"]
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=3, frameon=False,
               fontsize=9)
    fig.suptitle("Return distributions against the normal benchmark", x=0.0, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.14)
    viz.source_note(fig, "Sources: U.S. EIA (WTI), Nasdaq Inc. (NASDAQ Composite), "
                         "retrieved from FRED. Log density: the normal curve collapses in the tails "
                         "where observations remain.")
    fig.tight_layout()
    return _save(fig, "A1_return_distributions.png")


def fig_qq(d):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1))
    r = d["WTI_RET"].dropna()
    z = (r - r.mean()) / r.std()
    for ax, (dist, args, label, col) in zip(axes, [
            (stats.norm, (), "normal", S[0]),
            (stats.t, (stats.t.fit(z)[0],), "Student t", S[1])]):
        osm, osr = stats.probplot(z, dist=dist, sparams=args, fit=False)
        ax.scatter(osm, osr, s=9, color=col, alpha=0.6, edgecolors="none")
        lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        ax.plot(lim, lim, color=viz.INK, lw=1.4, ls="--")
        ax.set_title(f"WTI daily returns against the {label}")
        ax.set_xlabel(f"{label} quantiles")
        ax.set_ylabel("standardised empirical quantiles")
        if label == "Student t":
            viz.subtitle(ax, f"fitted degrees of freedom {stats.t.fit(z)[0]:.2f}")
        else:
            viz.subtitle(ax, "points leave the line at both ends")
    fig.suptitle("Quantile plots: where the normal assumption fails", x=0.0, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.10)
    viz.source_note(fig, "Source: U.S. EIA WTI spot price via FRED, 1990-2026, trading calendar.")
    fig.tight_layout()
    return _save(fig, "A2_qq_plots.png")


def fig_decade_boxplot(d):
    r = d["WTI_RET"].dropna()
    blocks = (r.index.year // 5) * 5
    rows = []
    for k in sorted(set(blocks)):
        x = r[blocks == k]
        rows.append(dict(label=f"{k}-{k+4}", vol=x.std() * np.sqrt(252) * 100,
                         worst=x.min() * 100, best=x.max() * 100,
                         kurt=stats.kurtosis(x), n=len(x)))
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    xs = np.arange(len(t))
    ax.bar(xs, t["vol"], width=0.58, color=S[0], linewidth=0, label="annualised volatility")
    ax.scatter(xs, t["worst"].abs(), s=34, color=S[7], zorder=5,
               edgecolors=viz.SURFACE, linewidths=1.0, label="largest one-day fall")
    ax.scatter(xs, t["best"], s=34, color=S[3], zorder=5, marker="D",
               edgecolors=viz.SURFACE, linewidths=1.0, label="largest one-day rise")
    for x, v in zip(xs, t["vol"]):
        ax.text(x, v + 1.6, f"{v:.0f}", ha="center", va="bottom", fontsize=8.4,
                color=viz.INK_2)
    ax.set_xticks(xs)
    ax.set_xticklabels(t["label"])
    ax.set_title("Dispersion of WTI daily returns by five-year block")
    viz.subtitle(ax, "volatility is not a constant of the series: it is a state that persists "
                     "for years at a time")
    ax.set_ylabel("percent")
    ax.legend(loc="upper left")
    viz.source_note(fig, "Volatility is annualised; the two markers are single-day moves in "
                         "absolute size. Source: U.S. EIA WTI spot price via FRED. The April 2020 negative "
                         "settlement is absent from every statistic here: its log return does "
                         "not exist.")
    return _save(fig, "A3_returns_by_block.png")


# ==================================================================================
# Student B - time series
# ==================================================================================
def fig_price_history(d, extremes):
    fig, ax = plt.subplots(figsize=(12.5, 4.3))
    ax.plot(d.index, d["WTI_D"], color=S[0], lw=1.3, label="WTI, Cushing")
    ax.plot(d.index, d["BRENT_D"], color=S[1], lw=1.1, label="Brent, Europe", alpha=0.9)
    ax.axhline(0, color=viz.INK_2, lw=0.9)
    flagged = extremes.index
    ax.scatter(flagged, d["WTI_D"].reindex(flagged), s=16, color=S[7], zorder=5,
               edgecolors=viz.SURFACE, linewidths=0.8, label="flagged extreme move")
    neg = d.index[d["WTI_D"] <= 0]
    if len(neg):
        ax.annotate("20 Apr 2020: -36.98", xy=(neg[0], d["WTI_D"].loc[neg[0]]),
                    xytext=(neg[0] - pd.Timedelta(days=2400), -70),
                    color=viz.INK, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=viz.INK_2, lw=0.9))
    ax.set_title("Crude oil spot prices, 1990 to 2026")
    viz.subtitle(ax, "dollars per barrel, daily, trading calendar of the WTI contract")
    ax.set_ylabel("USD per barrel")
    ax.legend(loc="upper left")
    viz.source_note(fig, "Source: U.S. EIA, retrieved from FRED. Flagged moves are returns that "
                         "at least two of three independent outlier screens identify.")
    return _save(fig, "B1_price_history.png")


def fig_real_vs_nominal(m):
    base = m["WTI_M"].first_valid_index()
    fig, ax = plt.subplots(figsize=(12.5, 4.0))
    nom = m["WTI_M"] / m["WTI_M"].loc[base] * 100
    real = m["WTI_REAL"] / m["WTI_REAL"].loc[base] * 100
    ax.plot(m.index, nom, color=S[0], lw=1.6, label="nominal WTI")
    ax.plot(m.index, real, color=S[2], lw=1.6, label="WTI deflated by the energy CPI")
    ax.axhline(100, color=viz.INK_2, lw=0.9, ls=":")
    ax.set_title("Nominal and real crude oil price, indexed to January 1990 = 100")
    viz.subtitle(ax, "both series on one scale: indexing removes the need for a second axis")
    ax.set_ylabel("index, Jan 1990 = 100")
    ax.legend(loc="upper left")
    viz.source_note(fig, "Sources: U.S. EIA (WTI) and U.S. Bureau of Labor Statistics "
                         "(CPI energy, CPIENGSL), retrieved from FRED.")
    return _save(fig, "B2_real_vs_nominal.png")


def fig_volatility(d):
    fig, ax = plt.subplots(figsize=(12.5, 4.0))
    ax.plot(d.index, d["WTI_VOL_63"] * 100, color=S[0], lw=1.5, label="WTI")
    ax.plot(d.index, d["NASDAQ_VOL_63"] * 100, color=S[1], lw=1.3, label="NASDAQ Composite")
    ax.set_title("Realised volatility, 63-day window, annualised")
    viz.subtitle(ax, "quiet periods and turbulent periods arrive in blocks, in both markets, "
                     "but oil reaches levels equities do not")
    ax.set_ylabel("percent per year")
    ax.legend(loc="upper left")
    viz.source_note(fig, "Sources: U.S. EIA and Nasdaq Inc. via FRED. Windows containing the "
                         "April 2020 negative settlement are undefined and left blank.")
    return _save(fig, "B3_volatility.png")


def fig_driver_panel(m):
    specs = [("IGREA", "Global real activity (Kilian index)", "percent deviation"),
             ("USD_M", "Broad US dollar index", "index, Jan 2006 = 100"),
             ("IP_OILGAS", "US oil and gas extraction output", "index, 2017 = 100"),
             ("CAPUTL_OILGAS", "Capacity utilisation, oil and gas", "percent"),
             ("GEPU", "Global economic policy uncertainty", "index"),
             ("GAS_EU", "European natural gas price", "USD per MMBtu")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.0), sharex=True)
    for ax, (col, title, unit) in zip(axes.ravel(), specs):
        ax.plot(m.index, m[col], color=S[0], lw=1.4)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit, fontsize=8.2)
        first = m[col].first_valid_index()
        if first is not None and first > m.index.min():
            ax.axvspan(m.index.min(), first, color=viz.GRID, alpha=0.7)
            ax.text(m.index.min(), ax.get_ylim()[1], " not published", fontsize=7.6,
                    color=viz.INK_MUTED, va="top")
    fig.suptitle("Explanatory blocks entering the belief network", x=0.0, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.0)
    viz.source_note(fig, "Sources: Dallas Fed, Board of Governors, U.S. BLS, Baker-Bloom-Davis "
                         "and the IMF, all retrieved from FRED. Shaded areas mark the period "
                         "before a series existed; they are availability, not missing data.")
    fig.tight_layout()
    return _save(fig, "B4_driver_panel.png")


# ==================================================================================
# Student C - multivariate
# ==================================================================================
def fig_correlation_heatmap(m):
    cols = {"WTI_RET": "WTI return", "IGREA": "global activity", "USD_M": "US dollar",
            "SP500": "S&P 500", "GOLD": "gold", "IP_OILGAS": "oil and gas output",
            "CAPUTL_OILGAS": "capacity use", "GEPU": "policy uncertainty",
            "PPI_CRUDE": "crude PPI", "GAS_EU": "EU gas price", "GS10": "10-year yield"}
    x = pd.DataFrame({v: (m[k] if k in ("WTI_RET", "GS10", "CAPUTL_OILGAS", "IGREA")
                          else np.log(m[k]).diff()) for k, v in cols.items()})
    x["global activity"] = m["IGREA"].diff()
    x["capacity use"] = m["CAPUTL_OILGAS"].diff()
    x["10-year yield"] = m["GS10"].diff()
    c = x.corr(min_periods=120)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(c.values, cmap=viz.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c.index)
    ax.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.4,
                    color=viz.SURFACE if abs(v) > 0.55 else viz.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("Pearson correlation", fontsize=8.6)
    cb.outline.set_visible(False)
    ax.set_title("Pairwise correlation of monthly changes")
    viz.source_note(fig, "Pairwise complete observations, minimum 120 months. Levels are "
                         "differenced or log-differenced so that the matrix describes changes, "
                         "not shared trends. Sources: FRED providers listed in the data dictionary.")
    return _save(fig, "C1_correlation_heatmap.png")


def fig_rolling_correlation(m):
    fig, ax = plt.subplots(figsize=(12.5, 4.0))
    usd = an.rolling_correlation(m["WTI_RET"], np.log(m["USD_M"]).diff(), 36)
    spx = an.rolling_correlation(m["WTI_RET"], np.log(m["SP500"]).diff(), 36)
    ax.plot(usd.index, usd, color=S[0], lw=1.6, label="WTI with the US dollar")
    ax.plot(spx.index, spx, color=S[1], lw=1.6, label="WTI with the S&P 500")
    ax.axhline(0, color=viz.INK_2, lw=0.9)
    ax.set_title("Rolling 36-month correlation of monthly returns")
    viz.subtitle(ax, "the dependence structure is not constant, which is what a static "
                     "covariance matrix cannot represent")
    ax.set_ylabel("correlation")
    ax.legend(loc="upper left", ncol=2)
    viz.source_note(fig, "Sources: U.S. EIA, Board of Governors and Shiller S&P 500 data. "
                         "The dollar index begins in 2006.")
    return _save(fig, "C2_rolling_correlation.png")


def fig_acf(d):
    tab = an.acf_table(d["WTI_RET"], nlags=30)
    band = tab["band"].iloc[0]
    fig, ax = plt.subplots(figsize=(12.0, 4.0))
    w = 0.26
    lags = tab.index[1:]
    for k, (col, label, colour) in enumerate([
            ("returns", "returns", S[0]),
            ("absolute_returns", "absolute returns", S[1]),
            ("squared_returns", "squared returns", S[2])]):
        ax.bar(lags + (k - 1) * w, tab[col].iloc[1:], width=w * 0.9, color=colour,
               label=label, linewidth=0)
    ax.axhline(0, color=viz.INK_2, lw=0.9)
    ax.axhline(band, color=viz.INK, lw=1.0, ls="--")
    ax.axhline(-band, color=viz.INK, lw=1.0, ls="--")
    ax.text(30.3, band, " 95% band", fontsize=8, color=viz.INK_2, va="center")
    ax.set_title("Autocorrelation of WTI daily returns and of their magnitude")
    viz.subtitle(ax, "returns are close to unpredictable in sign, their size is strongly "
                     "predictable")
    ax.set_xlabel("lag, trading days")
    ax.set_ylabel("autocorrelation")
    ax.legend(loc="upper right")
    viz.source_note(fig, "Source: U.S. EIA WTI spot price via FRED, 1990-2026.")
    return _save(fig, "C3_acf.png")


def fig_seasonality(m):
    tab = an.monthly_seasonality(m["WTI_RET"])
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    vals = tab["mean"] * 100
    colours = [S[0] if v >= 0 else S[7] for v in vals]
    ax.bar(tab.index, vals, color=colours, width=0.62, linewidth=0)
    ax.axhline(0, color=viz.INK_2, lw=0.9)
    for xpos, v in zip(range(len(vals)), vals):
        ax.text(xpos, v + (0.18 if v >= 0 else -0.18), f"{v:.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.2, color=viz.INK_2)
    ax.set_title("Average WTI log return by calendar month, 1990 to 2026")
    viz.subtitle(ax, f"Kruskal-Wallis p = {tab.attrs['kruskal_p']:.3f}, "
                     f"one-way ANOVA p = {tab.attrs['anova_p']:.3f}")
    ax.set_ylabel("percent per month")
    ax.legend(handles=[Patch(facecolor=S[0], label="positive average"),
                       Patch(facecolor=S[7], label="negative average")], loc="lower left")
    viz.source_note(fig, "Source: U.S. EIA WTI spot price via FRED. The nonparametric test is "
                         "the relevant one here because the returns are far from normal.")
    return _save(fig, "C4_seasonality.png")


def fig_leverage(d):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    pairs = [("WTI_RET", "OVX", "WTI return against the change in OVX", S[0]),
             ("NASDAQ_RET", "VIX", "NASDAQ return against the change in VIX", S[1])]
    for ax, (rc, vc, title, colour) in zip(axes, pairs):
        df = pd.concat([d[rc].rename("r"), d[vc].diff().rename("dv")], axis=1).dropna()
        ax.scatter(df["r"] * 100, df["dv"], s=6, color=colour, alpha=0.35, edgecolors="none")
        b, a = np.polyfit(df["r"] * 100, df["dv"], 1)
        xs = np.linspace(df["r"].min() * 100, df["r"].max() * 100, 50)
        ax.plot(xs, a + b * xs, color=viz.INK, lw=1.6)
        ax.axhline(0, color=viz.INK_2, lw=0.8)
        ax.axvline(0, color=viz.INK_2, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("daily return, percent")
        ax.set_ylabel("change in the implied volatility index")
        viz.subtitle(ax, f"correlation {df['r'].corr(df['dv']):.2f}")
    fig.suptitle("The leverage effect is far weaker in oil than in equities", x=0.0,
                 ha="left", fontsize=12.5, fontweight="bold", y=1.10)
    viz.source_note(fig, "Sources: CBOE (VIX, OVX), U.S. EIA and Nasdaq Inc. via FRED. "
                         "OVX begins in May 2007, so the two panels cover different samples.")
    fig.tight_layout()
    return _save(fig, "C5_leverage.png")


def fig_missingness(monthly_raw, daily_raw):
    from matplotlib.colors import ListedColormap
    binary = ListedColormap(["#f2f1ed", "#1c5cab"])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6),
                             gridspec_kw={"width_ratios": [1, 1]})
    for ax, (df, title) in zip(axes, [(monthly_raw, "Monthly panel"),
                                      (daily_raw, "Daily panel, full weekday calendar")]):
        mask = df.isna().T.astype(float).values
        ax.imshow(mask, aspect="auto", cmap=binary, vmin=0, vmax=1,
                  interpolation="nearest",
                  extent=[0, df.shape[0], df.shape[1], 0])
        ax.set_yticks(np.arange(df.shape[1]) + 0.5)
        ax.set_yticklabels(df.columns, fontsize=8.0)
        n = df.shape[0]
        ticks = np.linspace(0, n - 1, 6).astype(int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(df.index[t].year) for t in ticks])
        ax.grid(False)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Where the holes are, before cleaning", x=0.0, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.03)
    viz.source_note(fig, "Dark cells are absent observations. In the daily panel most of them "
                         "are closed markets rather than lost data, which is why the analysis "
                         "calendar is rebuilt on the trading days of the target.")
    fig.tight_layout()
    return _save(fig, "C6_missingness.png")


def build_all(out):
    viz.apply_style()
    d, m = out["daily"], out["monthly"]
    rep = out["report"]
    paths = [
        fig_return_distributions(d, m),
        fig_qq(d),
        fig_decade_boxplot(d),
        fig_price_history(d, rep["extremes_wti_d"]),
        fig_real_vs_nominal(m),
        fig_volatility(d),
        fig_driver_panel(m),
        fig_correlation_heatmap(m),
        fig_rolling_correlation(m),
        fig_acf(d),
        fig_seasonality(m),
        fig_leverage(d),
        fig_missingness(out["monthly_raw"], out["daily_raw"]),
    ]
    plt.close("all")
    return paths
