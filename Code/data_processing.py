"""Lit les fichiers bruts de Data/Raw et enregistre les séries hebdomadaires
(jeudi) dans Data/Processed : prix FX, returns, volatilité réalisée, macro,
taux et financement (DFF)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from Function.functions import (
    PAIRS, PROCESSED, log_returns, read_raw_series, realized_volatility,
    save_frame, to_thursday,
)


def main():
    ## ---- fx ----
    tickers = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "CADUSD": "CADUSD=X",
               "CHFUSD": "CHFUSD=X", "JPYUSD": "JPYUSD=X"}
    fx_daily = {
        pair: read_raw_series(f"yahoo_{pair}", f"{ticker}.Adjusted").dropna().rename(pair)
        for pair, ticker in tickers.items()
    }
    fx_prices = pd.concat([to_thursday(x) for x in fx_daily.values()], axis=1)
    fx_returns = log_returns(fx_prices)

    # Volatilité réalisée hebdomadaire, mêmes jeudis que fx_returns
    fx_realized_vol = pd.concat([realized_volatility(x) for x in fx_daily.values()], axis=1)

    ## ---- macro ----
    macro = {}
    for name in ["gold", "sp500", "oil"]:
        raw = read_raw_series(f"yahoo_{name}")
        adjusted = [c for c in raw.columns if c.endswith(".Adjusted")][0]
        macro[name] = to_thursday(raw[adjusted].dropna().rename(name))
    macro_prices = pd.concat(macro.values(), axis=1)
    macro_returns = log_returns(macro_prices).add_suffix("_return")

    ## ---- rates ----
    def from_2004(x):
        return x[x.index >= "2004-01-01"]

    us2 = to_thursday(from_2004(read_raw_series("fred_DGS2", "DGS2").dropna()))
    us10 = to_thursday(from_2004(read_raw_series("fred_DGS10", "DGS10").dropna()))
    eur2 = to_thursday(read_raw_series("ecb_yield_2Y", "value").dropna())
    eur10 = to_thursday(read_raw_series("ecb_yield_10Y", "value").dropna())
    boe = read_raw_series("boe_yield_curve")
    gbp2, gbp10 = to_thursday(boe["m2"].dropna()), to_thursday(boe["m10"].dropna())

    cad = read_raw_series("cad_rates")
    stat = cad["Financial market statistics"]
    cad2 = to_thursday(cad.loc[stat.str.contains("2 year"), "VALUE"].dropna())
    cad10 = to_thursday(cad.loc[stat.str.contains("10 year"), "VALUE"].dropna())

    chf = read_raw_series("chf_rates")
    chf2, chf10 = to_thursday(chf["GSWISS02 Index"].dropna()), to_thursday(chf["GSWISS10 Index"].dropna())
    jpy = read_raw_series("jpy_rates")
    jpy2, jpy10 = to_thursday(jpy["GJGB2 Index"].dropna()), to_thursday(jpy["GJGB10 Index"].dropna())

    rates = pd.concat(
        [us2, eur2, cad2, gbp2, chf2, jpy2, us10, eur10, cad10, gbp10, chf10, jpy10], axis=1
    )
    rates.columns = ["US2", "EUR2", "CAD2", "GBP2", "CHF2", "JPY2",
                     "US10", "EUR10", "CAD10", "GBP10", "CHF10", "JPY10"]

    ## ---- funding ----
    # Fed Funds observé chaque semaine, puis taux annualisé (%) -> coût hebdomadaire
    dff = to_thursday(read_raw_series("fred_DFF", "DFF")).ffill()
    funding = pd.DataFrame({"Funding": dff, "Funding_weekly": np.log1p(dff / 100 * 7 / 360)})

    ## ---- save ----
    save_frame(fx_prices, PROCESSED / "fx_prices.csv")
    save_frame(fx_returns, PROCESSED / "fx_returns.csv")
    save_frame(fx_realized_vol, PROCESSED / "realized_vol.csv")
    save_frame(macro_prices, PROCESSED / "macro_prices.csv")
    save_frame(macro_returns, PROCESSED / "macro_returns.csv")
    save_frame(rates, PROCESSED / "rates.csv")
    save_frame(funding, PROCESSED / "funding.csv")


if __name__ == "__main__":
    main()
