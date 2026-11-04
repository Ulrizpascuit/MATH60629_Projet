"""Construction des features : écarts de taux et indicateurs techniques.

Les indicateurs reproduisent les fonctions du package R TTR (mêmes
initialisations), pour donner exactement les mêmes valeurs."""
import numpy as np
import pandas as pd


## ---- TTR equivalents ----
def ema(x, n, wilder=False):
    """TTR::EMA : premier point = moyenne des n premières valeurs non-NA,
    puis lissage exponentiel (ratio 2/(n+1), ou 1/n si wilder)."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    ratio = 1.0 / n if wilder else 2.0 / (n + 1)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) == 0:
        return out
    beg = valid[0]
    if np.isnan(x[beg:]).any():
        raise ValueError("Series contains non-leading NAs")
    seed = 0.0
    for i in range(beg, beg + n):
        seed += x[i] / n
    out[beg + n - 1] = seed
    for i in range(beg + n, len(x)):
        out[i] = x[i] * ratio + out[i - 1] * (1 - ratio)
    return out


def run_mean(x, n):
    return pd.Series(np.asarray(x, dtype=float)).rolling(n).mean().values


def run_sd(x, n):
    """TTR::runSD : écart-type échantillonnal (n - 1) sur une fenêtre de n."""
    return pd.Series(np.asarray(x, dtype=float)).rolling(n).std(ddof=1).values


def rsi(price, n=14):
    """TTR::RSI avec la moyenne de Wilder (défaut de TTR)."""
    up = np.diff(np.asarray(price, dtype=float), prepend=np.nan)
    dn = np.where(up < 0, -up, 0.0)
    dn[np.isnan(up)] = np.nan
    up = np.where(up < 0, 0.0, up)
    avg_up, avg_dn = ema(up, n, wilder=True), ema(dn, n, wilder=True)
    return 100 * avg_up / (avg_up + avg_dn)


def macd(price, fast=12, slow=26):
    """TTR::MACD (percent = TRUE) : 100 * (EMA rapide / EMA lente - 1)."""
    return 100 * (ema(price, fast) / ema(price, slow) - 1)


## ---- build_rate_features ----
def build_rate_features(currency, rates):
    c = currency.lower()
    spread2 = rates[f"{currency}2"] - rates["US2"]
    spread10 = rates[f"{currency}10"] - rates["US10"]
    features = pd.DataFrame({
        f"{c}_spread2": spread2,
        f"{c}_spread10": spread10,
        f"{c}_spread2_diff": spread2.diff(),
        f"{c}_spread10_diff": spread10.diff(),
        "dgs2_diff": rates["US2"].diff(),
        "dgs10_diff": rates["US10"].diff(),
    })
    return features.dropna()


## ---- build_technical_indicator_features ----
def build_ti_features(currency, fx_prices, fx_returns, vol_windows=(4, 12), rsi_n=14,
                      macd_fast=12, macd_slow=26, bb_n=20, bb_sd=2):
    pair = f"{currency}USD"
    # NaN internes remplis par le dernier prix connu (aucun look-ahead)
    price = fx_prices[pair].ffill()
    returns = fx_returns[pair]

    ti_price = pd.DataFrame(index=price.index)
    ti_price["rsi"] = rsi(price.values, rsi_n)
    ti_price["macd"] = macd(price.values, macd_fast, macd_slow)
    bb_mid = run_mean(price.values, bb_n)
    bb_sdv = run_sd(price.values, bb_n)
    bb_up, bb_lo = bb_mid + bb_sd * bb_sdv, bb_mid - bb_sd * bb_sdv
    ti_price["bb_pctB"] = (price.values - bb_lo) / (bb_up - bb_lo)

    ti_ret = pd.DataFrame(index=returns.index)
    ti_ret["lag_return"] = returns.values
    ti_ret["vol_4"] = run_sd(returns.values, vol_windows[0])
    ti_ret["vol_12"] = run_sd(returns.values, vol_windows[1])

    # Jointure externe sur les dates, comme merge() de xts
    features = ti_ret.join(ti_price, how="outer")
    return features[["lag_return", "vol_4", "vol_12", "rsi", "macd", "bb_pctB"]]
