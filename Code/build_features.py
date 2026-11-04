"""Construit les 15 features de chaque devise (Data/Features/<devise>_features.csv)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from Function.functions import FEATURES, PROCESSED, load_frame, save_frame
from Function.functions_feature import build_rate_features, build_ti_features

CURRENCIES = ["EUR", "CAD", "GBP", "CHF", "JPY"]

## ---- sample ----
START_DATE = "2004-01-01"
# Fin fixe : la période de test ne bouge pas si les données sont retéléchargées
END_DATE = "2026-08-27"


def main():
    fx_prices = load_frame(PROCESSED / "fx_prices.csv")
    fx_returns = load_frame(PROCESSED / "fx_returns.csv")
    rates = load_frame(PROCESSED / "rates.csv")
    macro_returns = load_frame(PROCESSED / "macro_returns.csv")

    assert fx_prices.index.max() > pd.Timestamp(END_DATE)

    def in_sample(x):
        return x[(x.index >= START_DATE) & (x.index <= END_DATE)]

    fx_prices, rates, macro_returns = in_sample(fx_prices), in_sample(rates), in_sample(macro_returns)

    # fx_prices a des NaN le 2008-08-14 et le 2008-08-21 (Yahoo sans prix > 5 jours) :
    # on reprend le dernier prix connu (aucun look-ahead)
    fx_prices = fx_prices.ffill()

    merged = {}
    for currency in CURRENCIES:
        rate = build_rate_features(currency, rates)
        ti = build_ti_features(currency, fx_prices, fx_returns)
        features = rate.join(macro_returns, how="inner").join(ti, how="inner")
        merged[currency] = features.dropna()

    ## ---- common_date_range ----
    # Les taux GBP (BoE) et CAD sont publiés avec du retard : on coupe toutes
    # les devises sur la même période (dernier début, première fin).
    common_start = max(x.index.min() for x in merged.values())
    common_end = min(x.index.max() for x in merged.values())
    for currency, x in merged.items():
        x = x[(x.index >= common_start) & (x.index <= common_end)]
        save_frame(x, FEATURES / f"{currency.lower()}_features.csv")


if __name__ == "__main__":
    main()
