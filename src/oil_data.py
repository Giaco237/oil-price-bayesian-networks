"""
GWP1 - Application of Probabilistic Graphical Models in Forecasting Crude Oil Price
Data layer: retrieval, structuring, dictionary.

Two retrieval channels are supported:
  1. GitHub-mirrored, EIA/CBOE-sourced market data (always available offline in data_raw/)
  2. FRED multi-series CSV exports (fredgraph.csv), placed in data_raw/ by the user

Every series keeps its provider code so that the data dictionary is reproducible.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_raw")
CLEAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_clean")

START = "1990-01-01"

# ----------------------------------------------------------------------------------
# Variable catalogue: the single source of truth for the data dictionary (Step 4)
# ----------------------------------------------------------------------------------
# role: A = macroeconomic / geopolitical, B = microeconomic, C = financial
# block: thematic grouping used in the belief network
CATALOGUE = {
    # ---- C. Financial -------------------------------------------------------------
    "WTI_M": dict(role="C", block="Price (target)", label="Spot crude oil price, WTI, Cushing",
                  code="WTISPLC", provider="FRED / St. Louis Fed", freq="Monthly",
                  units="USD per barrel"),
    "BRENT_M": dict(role="C", block="Price", label="Spot crude oil price, Brent, Europe",
                    code="MCOILBRENTEU", provider="FRED / U.S. EIA", freq="Monthly",
                    units="USD per barrel"),
    "WTI_D": dict(role="C", block="Price (target)", label="Spot crude oil price, WTI, Cushing",
                  code="DCOILWTICO", provider="U.S. EIA", freq="Daily", units="USD per barrel"),
    "BRENT_D": dict(role="C", block="Price", label="Spot crude oil price, Brent, Europe",
                    code="DCOILBRENTEU", provider="U.S. EIA", freq="Daily", units="USD per barrel"),
    "VIX": dict(role="C", block="Market risk", label="CBOE Volatility Index (VIX)",
                code="VIXCLS", provider="CBOE", freq="Daily", units="Index"),
    "OVX": dict(role="C", block="Market risk", label="CBOE Crude Oil ETF Volatility Index",
                code="OVXCLS", provider="CBOE", freq="Daily", units="Index"),
    "NASDAQ": dict(role="C", block="Market risk", label="NASDAQ Composite Index",
                   code="NASDAQCOM", provider="Nasdaq, Inc.", freq="Daily", units="Index"),
    "SP500": dict(role="C", block="Market risk", label="S&P 500 composite index",
                  code="SP500 (Shiller)", provider="Shiller / S&P Dow Jones", freq="Monthly",
                  units="Index"),
    "GOLD": dict(role="C", block="Cross-commodity", label="Gold price, London fix",
                 code="gold-prices", provider="LBMA via Datahub", freq="Monthly",
                 units="USD per troy ounce"),
    "HH_GAS": dict(role="C", block="Cross-commodity", label="Henry Hub natural gas spot price",
                   code="DHHNGSP", provider="U.S. EIA", freq="Daily", units="USD per MMBtu"),
    "DGS10": dict(role="C", block="Rates", label="10-year Treasury constant maturity yield",
                  code="DGS10", provider="Board of Governors (US)", freq="Daily", units="Percent"),
    "USD_D": dict(role="C", block="FX", label="Nominal broad U.S. dollar index",
                  code="DTWEXBGS", provider="Board of Governors (US)", freq="Daily",
                  units="Index Jan 2006 = 100"),

    # ---- A. Macroeconomic / geopolitical ------------------------------------------
    "IGREA": dict(role="A", block="Global demand", label="Index of global real economic activity",
                  code="IGREA", provider="FRED / Dallas Fed (Kilian)", freq="Monthly",
                  units="Percent deviation from trend"),
    "INDPRO": dict(role="A", block="Global demand", label="Industrial production, total index (US)",
                   code="INDPRO", provider="Board of Governors (US)", freq="Monthly",
                   units="Index 2017 = 100"),
    "UNRATE": dict(role="A", block="Global demand", label="Unemployment rate (US)",
                   code="UNRATE", provider="U.S. BLS", freq="Monthly", units="Percent"),
    "CPIENG": dict(role="A", block="Inflation", label="CPI for all urban consumers: energy",
                   code="CPIENGSL", provider="U.S. BLS", freq="Monthly",
                   units="Index 1982-1984 = 100"),
    "FEDFUNDS": dict(role="A", block="Monetary policy", label="Federal funds effective rate",
                     code="FEDFUNDS", provider="Board of Governors (US)", freq="Monthly",
                     units="Percent"),
    "GS10": dict(role="A", block="Monetary policy", label="10-year Treasury constant maturity yield",
                 code="GS10", provider="Board of Governors (US)", freq="Monthly", units="Percent"),
    "M2": dict(role="A", block="Monetary policy", label="M2 money stock",
               code="M2SL", provider="Board of Governors (US)", freq="Monthly",
               units="Billions of USD"),
    "USD_M": dict(role="A", block="FX", label="Nominal broad U.S. dollar index",
                  code="TWEXBGSMTH", provider="Board of Governors (US)", freq="Monthly",
                  units="Index Jan 2006 = 100"),
    "GEPU": dict(role="A", block="Geopolitical / uncertainty",
                 label="Global economic policy uncertainty index, GDP-weighted",
                 code="GEPUCURRENT", provider="Baker, Bloom and Davis", freq="Monthly",
                 units="Index"),
    "USEPU": dict(role="A", block="Geopolitical / uncertainty",
                  label="Economic policy uncertainty index for the United States",
                  code="USEPUINDXM", provider="Baker, Bloom and Davis", freq="Monthly",
                  units="Index"),
    "GEOEVENT": dict(role="A", block="Geopolitical / uncertainty",
                     label="Oil-relevant geopolitical event indicator (constructed)",
                     code="constructed", provider="Authors, from dated public events",
                     freq="Monthly", units="Binary"),

    # ---- B. Microeconomic ----------------------------------------------------------
    "IP_OILGAS": dict(role="B", block="Upstream activity",
                      label="Industrial production: oil and gas extraction (NAICS 211)",
                      code="IPG211S", provider="Board of Governors (US)", freq="Monthly",
                      units="Index 2017 = 100"),
    "CAP_OILGAS": dict(role="B", block="Upstream activity",
                       label="Industrial capacity: oil and gas extraction (NAICS 211)",
                       code="CAPG211S", provider="Board of Governors (US)", freq="Monthly",
                       units="Index 2017 = 100"),
    "CAPUTL_OILGAS": dict(role="B", block="Upstream activity",
                          label="Capacity utilisation: oil and gas extraction (NAICS 211)",
                          code="CAPUTLG211S", provider="Board of Governors (US)", freq="Monthly",
                          units="Percent"),
    "IP_DRILL": dict(role="B", block="Upstream activity",
                     label="Industrial production: drilling oil and gas wells (NAICS 213111)",
                     code="IPN213111N", provider="Board of Governors (US)", freq="Monthly",
                     units="Index 2017 = 100"),
    "PPI_OILGAS": dict(role="B", block="Producer prices",
                       label="PPI by industry: oil and gas extraction",
                       code="PCU211211", provider="U.S. BLS", freq="Monthly",
                       units="Index Dec 1985 = 100"),
    "PPI_CRUDE": dict(role="B", block="Producer prices",
                      label="PPI by commodity: crude petroleum, domestic production",
                      code="WPU0561", provider="U.S. BLS", freq="Monthly", units="Index 1982 = 100"),
    "PPI_DRILL": dict(role="B", block="Producer prices",
                      label="PPI by industry: drilling oil and gas wells",
                      code="PCU213111213111", provider="U.S. BLS", freq="Monthly",
                      units="Index Dec 1985 = 100"),
    "GAS_EU": dict(role="B", block="Substitutes", label="Global price of natural gas, EU",
                   code="PNGASEUUSDM", provider="IMF", freq="Monthly",
                   units="USD per MMBtu"),
}

ROLE_NAME = {
    "A": "Macroeconomic / geopolitical (Student A)",
    "B": "Microeconomic (Student B)",
    "C": "Financial (Student C)",
}

# FRED series code -> internal name
FRED_MAP = {v["code"]: k for k, v in CATALOGUE.items() if "FRED" in v["provider"] or v["provider"] in {
    "Board of Governors (US)", "U.S. BLS", "U.S. EIA", "CBOE", "Nasdaq, Inc.",
    "Baker, Bloom and Davis", "IMF"}}


# ----------------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------------
def _read_datahub(name: str, col: str, value_name: str, monthly: bool) -> pd.Series:
    df = pd.read_csv(os.path.join(RAW, name))
    df.columns = [c.strip() for c in df.columns]
    df[col] = pd.to_datetime(df[col])
    s = pd.Series(pd.to_numeric(df[value_name], errors="coerce").values, index=df[col],
                  name=value_name)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if monthly:
        s.index = s.index.to_period("M").to_timestamp()
    return s


def load_github_block() -> dict:
    """EIA / CBOE / Shiller market data mirrored on GitHub."""
    out = {}
    out["WTI_D"] = _read_datahub("wti-daily.csv", "Date", "Price", monthly=False).rename("WTI_D")
    out["BRENT_D"] = _read_datahub("brent-daily.csv", "Date", "Price", monthly=False).rename("BRENT_D")
    out["WTI_M"] = _read_datahub("wti-monthly.csv", "Date", "Price", monthly=True).rename("WTI_M")
    out["BRENT_M"] = _read_datahub("brent-monthly.csv", "Date", "Price", monthly=True).rename("BRENT_M")

    vix = pd.read_csv(os.path.join(RAW, "vix-daily.csv"))
    vix["DATE"] = pd.to_datetime(vix["DATE"])
    out["VIX"] = pd.Series(pd.to_numeric(vix["CLOSE"], errors="coerce").values,
                           index=vix["DATE"], name="VIX").sort_index()

    sp = pd.read_csv(os.path.join(RAW, "sp500-monthly.csv"))
    sp["Date"] = pd.to_datetime(sp["Date"])
    s = pd.Series(pd.to_numeric(sp["SP500"], errors="coerce").values, index=sp["Date"], name="SP500")
    s.index = s.index.to_period("M").to_timestamp()
    out["SP500"] = s.sort_index()

    g = pd.read_csv(os.path.join(RAW, "gold-monthly.csv"))
    g["Date"] = pd.to_datetime(g["Date"] + "-01")
    out["GOLD"] = pd.Series(pd.to_numeric(g["Price"], errors="coerce").values,
                            index=g["Date"], name="GOLD").sort_index()
    return out


def _parse_fredgraph(path: str) -> pd.DataFrame:
    """fredgraph.csv exports: first column is the date, remaining columns are series codes.
    Missing observations are written as '.' by FRED."""
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"
    df = df.replace(".", np.nan)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


FRED_MONTHLY_FILES = ("fredgraph_a.csv", "fredgraph_b.csv", "fredgraph_c.csv")
FRED_DAILY_FILES = ("fredgraph_d_daily.csv", "fredgraph_e_dailyclose.csv")


def _load_fred_files(files, start=START) -> pd.DataFrame:
    frames = []
    for f in files:
        path = os.path.join(RAW, f)
        if os.path.exists(path):
            frames.append(_parse_fredgraph(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.rename(columns={c: FRED_MAP.get(c, c) for c in df.columns})
    df = df.sort_index()
    return df.loc[df.index >= pd.Timestamp(start)]


def load_fred_monthly() -> pd.DataFrame:
    df = _load_fred_files(FRED_MONTHLY_FILES)
    if df.empty:
        return df
    df.index = df.index.to_period("M").to_timestamp()
    return df[~df.index.duplicated(keep="last")]


def load_fred_daily() -> pd.DataFrame:
    return _load_fred_files(FRED_DAILY_FILES)


# ----------------------------------------------------------------------------------
# Geopolitical event indicator (Student A, constructed variable)
# ----------------------------------------------------------------------------------
# Only events whose date and oil-market relevance are unambiguous in the public record
# are hard-coded. The list deliberately stops in October 2023; later shocks are picked
# up by the quantitative detector in the cleaning module, and the report says so.
GEO_EVENTS = [
    ("1990-08", "Iraq invades Kuwait"),
    ("1991-01", "Operation Desert Storm begins"),
    ("1998-12", "Trough of the Asian-crisis oil price collapse"),
    ("2001-09", "September 11 attacks"),
    ("2003-03", "US-led invasion of Iraq"),
    ("2005-08", "Hurricane Katrina shuts Gulf of Mexico output"),
    ("2008-09", "Lehman Brothers failure"),
    ("2011-02", "Libyan civil war halts exports"),
    ("2014-11", "OPEC declines to cut output at the November 2014 meeting"),
    ("2016-11", "First OPEC+ production-cut agreement"),
    ("2018-05", "US withdrawal from the Iran nuclear agreement"),
    ("2019-09", "Attack on the Abqaiq processing facility"),
    ("2020-03", "OPEC+ talks collapse and COVID-19 demand shock"),
    ("2020-04", "WTI front-month settles below zero"),
    ("2022-02", "Russian invasion of Ukraine"),
    ("2023-10", "Israel-Hamas conflict"),
]
GEO_EVENT_LIST_ENDS = "2023-10"


def geopolitical_indicator(index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(0, index=index, name="GEOEVENT", dtype="int64")
    for ym, _ in GEO_EVENTS:
        ts = pd.Timestamp(ym + "-01")
        if ts in s.index:
            s.loc[ts] = 1
    return s


def data_dictionary() -> pd.DataFrame:
    rows = []
    for name, m in CATALOGUE.items():
        rows.append(dict(Variable=name, Role=ROLE_NAME[m["role"]], Block=m["block"],
                         Description=m["label"], Code=m["code"], Source=m["provider"],
                         Frequency=m["freq"], Units=m["units"]))
    return pd.DataFrame(rows).sort_values(["Role", "Block", "Variable"]).reset_index(drop=True)
