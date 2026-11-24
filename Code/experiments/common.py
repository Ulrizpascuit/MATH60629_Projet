"""Outils communs aux expériences (Code/experiments/).

Toutes les expériences lisent les données figées de Data/ et la référence
recalculée sur la machine de l'étude (experiments/E0_reference_mac/), et
écrivent leurs résultats dans experiments/<ID>/. Les sorties publiées de
l'ancien projet (outputs/, Data/Models/) ne sont jamais modifiées."""
import json
import pickle
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Code.build_model import TEST_WEEKS, load_inputs
from Function.functions import PAIRS
from Function.functions_model import forecast_rows, model_data

EXP = ROOT / "experiments"
REF_PKL = EXP / "E0_reference_mac" / "models" / "models.pkl"

METHODS = ["Naive", "Linear", "Stepwise", "Lasso", "Ridge", "Elastic_Net", "Random_Forest", "PCA"]
LEARNED = ["Linear", "Stepwise", "PCA", "Lasso", "Ridge", "Elastic_Net", "Random_Forest"]
PENALIZED = ["Lasso", "Ridge", "Elastic_Net"]
LABELS = {"Naive": "Naïf", "Linear": "MCO", "Stepwise": "Pas-à-pas", "Lasso": "Lasso",
          "Ridge": "Ridge", "Elastic_Net": "Elastic net", "Random_Forest": "Forêt",
          "PCA": "ACP", "MIXT": "MIXT", "Mean": "Moyenne", "Stacking": "Stacking"}
PAIR_LABELS = {p: f"{p[:3]}/{p[3:]}" for p in PAIRS}

# Palette catégorielle de référence (ordre fixe) et encres
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "figure.dpi": 150,
})


## ---- données ----
def load_frames():
    """Pour chaque paire : X, y, dates, colonnes, lignes de validation et de test
    (mêmes règles que Code/build_model.py)."""
    features, targets, _ = load_inputs()
    n = len(model_data(features[PAIRS[0]], targets[PAIRS[0]]))
    train_ratio = (n - TEST_WEEKS + 0.5) / n
    frames = {}
    for pair in PAIRS:
        data = model_data(features[pair], targets[pair])
        frames[pair] = {
            "X": np.ascontiguousarray(data.drop(columns="target").values),
            "y": data["target"].values,
            "dates": data.index,
            "columns": list(data.columns[1:]),
            "validation": forecast_rows(len(data), "validation", train_ratio),
            "test": forecast_rows(len(data), "test", train_ratio),
        }
    return frames


def load_reference():
    with open(REF_PKL, "rb") as f:
        return pickle.load(f)


def reference_predictions(ref, method, pair, period):
    """Prévisions de rendement et rendements réalisés de la référence E0."""
    if period == "validation":
        res = ref["validation_models"][method][pair]
        return np.asarray(res["predictions"]), np.asarray(res["actual"])
    res = ref["sharpe_models"][method][pair]
    return np.asarray(res["return_predictions"]), np.asarray(res["return_actual"])


## ---- métriques ----
def historical_mean(y, rows):
    """Moyenne historique connue à chaque date prévue (référence du R² hors échantillon)."""
    return np.array([y[:i].mean() for i in rows])


def clark_west(y, f_bench, f_model):
    """Test de Clark et West (2007), unilatéral : H0 le modèle n'améliore pas la référence emboîtée."""
    adj = (y - f_bench) ** 2 - ((y - f_model) ** 2 - (f_bench - f_model) ** 2)
    t = adj.mean() / (adj.std(ddof=1) / np.sqrt(len(adj)))
    return float(t), float(1 - stats.norm.cdf(t))


def accuracy(y, pred, hm):
    """R² (contre la moyenne de la période, comme le R² de validation de MIXT),
    R² hors échantillon (contre la moyenne historique), RMSE et Clark-West."""
    sse = np.sum((y - pred) ** 2)
    t, p = clark_west(y, hm, pred)
    return {"R2": 1 - sse / np.sum((y - y.mean()) ** 2),
            "R2_OS": 1 - sse / np.sum((y - hm) ** 2),
            "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
            "Hit": float(np.mean(np.sign(pred) == np.sign(y))),
            "CW_t": t, "CW_p": p}


def holm(pvalues):
    """Valeurs p ajustées par la procédure de Holm (1979)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def rank_ic(pred, actual):
    """Corrélation de Spearman hebdomadaire entre prévisions et réalisations des 5 paires
    (matrices semaines × paires) : moyenne et statistique t."""
    ic = np.array([stats.spearmanr(pred[t], actual[t])[0] for t in range(len(pred))])
    ic = ic[~np.isnan(ic)]
    return float(ic.mean()), float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))


## ---- sorties ----
def out_dir(exp_id):
    d = EXP / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig, dest, name):
    fig.savefig(dest / f"{name}.png", dpi=200)
    fig.savefig(dest / f"{name}.pdf")
    plt.close(fig)


def write_environment(dest, **extra):
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "numba", "joblib", "matplotlib"]
    env = {"python": platform.python_version(), "machine": platform.machine(),
           "system": f"{platform.system()} {platform.release()}",
           "packages": {p: metadata.version(p) for p in packages}, **extra}
    (dest / "environment.json").write_text(json.dumps(env, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
