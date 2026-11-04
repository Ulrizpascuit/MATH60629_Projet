"""Fonctions de base : lecture des données brutes et alignement hebdomadaire (jeudi)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "Raw"
PROCESSED = ROOT / "Data" / "Processed"
FEATURES = ROOT / "Data" / "Features"
MODELS = ROOT / "Data" / "Models"
OUTPUTS = ROOT / "outputs"

PAIRS = ["EURUSD", "GBPUSD", "CADUSD", "CHFUSD", "JPYUSD"]


## ---- read_raw ----
def read_raw_series(name, column=None):
    """Série brute de Data/Raw/<name>.csv (index = dates).

    Les fichiers ont été téléchargés une fois (Yahoo, FRED, BCE, BoE) ou
    exportés à la main (CAD, CHF, JPY) ; ils sont relus tels quels.
    """
    x = pd.read_csv(RAW / f"{name}.csv", encoding="utf-8-sig")
    x.index = pd.to_datetime(x.iloc[:, 0])
    x = x.iloc[:, 1:]
    if column is not None:
        x = x[column]
    return x


## ---- to_thursday ----
def first_thursday(start="2004-01-01"):
    start = pd.Timestamp(start)
    return start + pd.Timedelta(days=(3 - start.weekday()) % 7)


def to_thursday(x, start="2004-01-01", max_days=5):
    """Pour chaque jeudi : dernière observation connue à ce jeudi ou avant,
    si elle date de max_days jours au plus (sinon NaN)."""
    frame = x.to_frame() if isinstance(x, pd.Series) else x
    frame = frame.dropna().sort_index()
    thursdays = pd.date_range(first_thursday(start), frame.index.max(), freq="7D")
    dates = frame.index.values
    pos = np.searchsorted(dates, thursdays.values, side="right") - 1
    ok = pos >= 0
    age = (thursdays.values[ok] - dates[pos[ok]]) / np.timedelta64(1, "D")
    ok[ok] = age <= max_days
    out = np.full((len(thursdays), frame.shape[1]), np.nan)
    out[ok] = frame.values[pos[ok]]
    out = pd.DataFrame(out, index=thursdays, columns=frame.columns)
    return out.iloc[:, 0].rename(x.name) if isinstance(x, pd.Series) else out


## ---- realized_volatility ----
def realized_volatility(x, start="2004-01-01"):
    """Volatilité réalisée de la semaine qui se termine à chaque jeudi :
    racine de la somme des log-returns quotidiens au carré, du jeudi précédent
    (exclu) à ce jeudi (inclus). NaN si aucun prix cette semaine."""
    x = x.dropna().sort_index()
    daily = np.diff(np.log(x.values))
    days = x.index[1:]
    thursdays = pd.date_range(first_thursday(start), x.index.max(), freq="7D")
    # Semaine de chaque jour : premier jeudi >= ce jour
    week = np.searchsorted(thursdays.values, days.values, side="left")
    ok = week < len(thursdays)
    sum_sq = np.bincount(week[ok], weights=daily[ok] ** 2, minlength=len(thursdays))
    has_obs = np.bincount(week[ok], minlength=len(thursdays)) > 0
    out = np.where(has_obs, np.sqrt(sum_sq), np.nan)
    return pd.Series(out, index=thursdays, name=x.name)


## ---- log_returns ----
def log_returns(prices):
    """Log-returns d'une semaine à l'autre ; lignes avec un NaN retirées."""
    return np.log(prices).diff().dropna()


## ---- save / load ----
def save_frame(x, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    x.to_csv(path, index_label="Date")


def load_frame(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)
