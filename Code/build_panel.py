"""Panel de features pour les réseaux de neurones (Data/Features/panel.npz).

Une observation par (semaine, paire). Aux 15 features d'origine s'ajoutent des
features construites à partir des mêmes données brutes : momentum sur plusieurs
horizons, volatilité réalisée quotidienne, variations de taux sur 4 semaines,
pente relative de la courbe, conditions de marché (momentum et volatilité du
S&P 500, de l'or et du pétrole) et features transversales (écart à la moyenne
des 5 paires la même semaine). Toutes sont connues au jeudi t.

Cible : r(t -> t+1) / vol_12(t), le rendement de la semaine suivante divisé par
l'écart-type des 12 derniers rendements hebdomadaires (un « Sharpe réalisé »)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from Code.build_model import load_inputs
from Function.functions import (
    FEATURES, PAIRS, PROCESSED, load_frame, read_raw_series, realized_volatility,
)
from Function.functions_model import model_data

BASE = ["spread2", "spread10", "spread2_diff", "spread10_diff", "dgs2_diff", "dgs10_diff",
        "gold_return", "sp500_return", "oil_return", "lag_return", "vol_4", "vol_12",
        "rsi", "macd", "bb_pctB"]
NEW = ["mom_4", "mom_12", "mom_26", "rv_1", "rv_4", "vol_ratio", "spread2_chg4",
       "spread10_chg4", "slope_diff", "us2_chg4", "sp500_mom4", "sp500_rv4", "gold_mom4",
       "oil_mom4"]
CROSS = ["xs_mom_12", "xs_spread2", "xs_rv_4", "xs_lag_return"]
FEATURE_NAMES = BASE + NEW + CROSS


def pair_features(pair, base, fx_returns, fx_rv, rates, macro_ret, sp500_rv):
    cur = pair[:3]
    ret = fx_returns[pair]
    out = pd.DataFrame(index=base.index)
    cols = list(base.columns)
    out[BASE] = base[cols].values  # mêmes 15 features, noms génériques
    # Momentum : somme des log-rendements hebdomadaires (lignes de fx_returns)
    for k in (4, 12, 26):
        out[f"mom_{k}"] = ret.rolling(k).sum().reindex(out.index)
    # Volatilité réalisée quotidienne (semaine écoulée, et moyenne quadratique sur 4 semaines)
    # (semaines sans aucun prix, août 2008 : dernière valeur connue)
    rv = fx_rv[pair]
    out["rv_1"] = rv.ffill().reindex(out.index)
    out["rv_4"] = np.sqrt((rv ** 2).rolling(4, min_periods=1).mean()).ffill().reindex(out.index)
    out["vol_ratio"] = out["rv_4"] / out["vol_12"]
    # Taux : dernière valeur connue (jours fériés) ; variations sur 4 semaines
    # inconnues au tout début de l'échantillon mises à 0
    r = rates.ffill()
    spread2 = r[f"{cur}2"] - r["US2"]
    spread10 = r[f"{cur}10"] - r["US10"]
    out["spread2_chg4"] = spread2.diff(4).reindex(out.index).fillna(0.0)
    out["spread10_chg4"] = spread10.diff(4).reindex(out.index).fillna(0.0)
    out["slope_diff"] = out["spread10"] - out["spread2"]
    out["us2_chg4"] = r["US2"].diff(4).reindex(out.index).fillna(0.0)
    for name in ("sp500", "gold", "oil"):
        out[f"{name}_mom4"] = macro_ret[f"{name}_return"].rolling(4).sum().reindex(out.index)
    out["sp500_rv4"] = np.sqrt((sp500_rv ** 2).rolling(4, min_periods=1).mean()).ffill().reindex(out.index)
    return out


def main():
    features_by_pair, fx_targets, _ = load_inputs()
    fx_returns = load_frame(PROCESSED / "fx_returns.csv")
    fx_rv = load_frame(PROCESSED / "realized_vol.csv")
    rates = load_frame(PROCESSED / "rates.csv")
    macro_ret = load_frame(PROCESSED / "macro_returns.csv")
    sp = read_raw_series("yahoo_sp500", "GSPC.Adjusted").dropna()
    sp500_rv = realized_volatility(sp)

    # Mêmes semaines que le projet d'origine (1 143 semaines, mêmes 260 semaines de test)
    dates = model_data(features_by_pair["EURUSD"], fx_targets["EURUSD"]).index
    panel = {}
    for pair in PAIRS:
        base = features_by_pair[pair].reindex(dates)
        panel[pair] = pair_features(pair, base, fx_returns, fx_rv, rates, macro_ret, sp500_rv)

    # Features transversales : écart à la moyenne des 5 paires la même semaine
    for name, src in [("xs_mom_12", "mom_12"), ("xs_spread2", "spread2"),
                      ("xs_rv_4", "rv_4"), ("xs_lag_return", "lag_return")]:
        mean = pd.concat([panel[p][src] for p in PAIRS], axis=1).mean(axis=1)
        for p in PAIRS:
            panel[p][name] = panel[p][src] - mean

    X = np.stack([panel[p][FEATURE_NAMES].values for p in PAIRS], axis=1)  # (T, 5, F)
    ret = np.stack([fx_targets[p].reindex(dates).values for p in PAIRS], axis=1)
    vol = X[:, :, FEATURE_NAMES.index("vol_12")]
    z = ret / vol
    assert not np.isnan(X).any(), np.argwhere(np.isnan(X))[:5]
    assert not np.isnan(ret).any()
    np.savez(FEATURES / "panel.npz", X=X, ret=ret, z=z, vol=vol,
             dates=dates.values.astype("datetime64[D]"), features=np.array(FEATURE_NAMES),
             pairs=np.array(PAIRS))
    print(f"Panel : {X.shape[0]} semaines x {X.shape[1]} paires x {X.shape[2]} features")


if __name__ == "__main__":
    main()
