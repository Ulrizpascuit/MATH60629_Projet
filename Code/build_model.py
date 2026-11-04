"""Estime tous les modèles en fenêtre croissante (validation et test), les
GARCH(1,1), le modèle MIXT et les Sharpe prévus. Enregistre tout dans
Data/Models/models.pkl."""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from Function.functions import FEATURES, MODELS, PAIRS, PROCESSED, load_frame
from Function.functions_model import (
    build_mixt_models, build_sharpe_models, build_walk_forward_garch_models,
    build_walk_forward_models, metric_table, model_data,
)
from Function.functions_strategy import (
    expanding_meta_cv, model_matrix, strategy_legs, threshold_diagnostics,
)

TEST_WEEKS = 260


def load_inputs():
    """Features, targets (return t -> t+1) et volatilité réalisée t -> t+1."""
    fx_returns = load_frame(PROCESSED / "fx_returns.csv")
    fx_realized_vol = load_frame(PROCESSED / "realized_vol.csv")
    features_by_pair = {
        pair: load_frame(FEATURES / f"{pair[:3].lower()}_features.csv") for pair in PAIRS
    }
    start = max(f.index.min() for f in features_by_pair.values())
    end = min(f.index.max() for f in features_by_pair.values())

    def in_range(x):
        return x[(x.index >= start) & (x.index <= end)]

    # Décalage d'une ligne : la valeur à t est le return de t à la semaine suivante
    fx_targets = in_range(fx_returns.shift(-1).dropna())
    fx_vol_targets = in_range(fx_realized_vol.shift(-1).dropna())
    return features_by_pair, fx_targets, fx_vol_targets


def main():
    t0 = time.time()
    features_by_pair, fx_targets, fx_vol_targets = load_inputs()

    ## ---- train_test_split ----
    n_weeks = len(model_data(features_by_pair["EURUSD"], fx_targets["EURUSD"]))
    assert n_weeks > TEST_WEEKS
    # Le +0.5 garantit que floor(train_ratio * n_weeks) donne exactement n_weeks - 260
    train_ratio = (n_weeks - TEST_WEEKS + 0.5) / n_weeks
    print(f"Entraînement : {n_weeks - TEST_WEEKS} semaines - Test : {TEST_WEEKS} semaines")

    ## ---- check_common_dates ----
    dates = [model_data(features_by_pair[p], fx_targets[p]).index for p in PAIRS]
    assert all(d.equals(dates[0]) for d in dates)

    ## ---- walk_forward ----
    models = build_walk_forward_models(features_by_pair, fx_targets, "test", train_ratio)
    validation_models = build_walk_forward_models(features_by_pair, fx_targets, "validation", train_ratio)
    print(f"Modèles de return : {time.time() - t0:.0f} s")

    garch = build_walk_forward_garch_models(fx_targets, fx_vol_targets, models["Linear"])
    validation_garch = build_walk_forward_garch_models(fx_targets, fx_vol_targets, validation_models["Linear"])
    print("Semaines où le GARCH n'a pas convergé :",
          {p: int(g["refit_failed"].sum()) for p, g in garch.items()},
          {p: int(g["refit_failed"].sum()) for p, g in validation_garch.items()})

    ## ---- validation_r2 / mixt ----
    validation_r2 = pd.DataFrame(
        {pair: [validation_models[m][pair]["r2"] for m in validation_models] for pair in PAIRS},
        index=list(validation_models),
    )
    print(validation_r2.round(4))
    best_models = validation_r2.idxmax(axis=0).to_dict()

    models["MIXT"] = build_mixt_models(models, best_models)
    mixt_selection = pd.DataFrame({
        "Currency": PAIRS,
        "Selected_Model": [best_models[p] for p in PAIRS],
        "Validation_R2": validation_r2.max(axis=0).values,
    })
    print(mixt_selection)

    ## ---- sharpe_models ----
    sharpe_models = {m: build_sharpe_models(models[m], garch) for m in models}
    validation_models["MIXT"] = build_mixt_models(validation_models, best_models)
    validation_sharpe_models = {
        m: build_sharpe_models(validation_models[m], validation_garch) for m in validation_models
    }

    ## ---- meta_threshold_diagnostics ----
    # Diagnostic seulement : les seuils sont fixés a priori à 0.5
    validation_legs = strategy_legs(
        model_matrix(validation_sharpe_models["MIXT"], "predictions"),
        model_matrix(validation_sharpe_models["MIXT"], "return_actual"),
    )
    threshold_table = threshold_diagnostics(
        expanding_meta_cv(validation_legs), thresholds=np.arange(0.35, 0.65 + 1e-9, 0.025)
    )

    ## ---- metrics ----
    vol_metrics = pd.DataFrame({
        "Currency": PAIRS,
        "RMSE": [garch[p]["rmse"] for p in PAIRS],
        "MAE": [garch[p]["mae"] for p in PAIRS],
        "R2": [garch[p]["r2"] for p in PAIRS],
    })

    results = {
        "test_weeks": TEST_WEEKS,
        "train_ratio": train_ratio,
        "sharpe_models": sharpe_models,
        "garch_models": garch,
        "validation_models": validation_models,
        "validation_r2": validation_r2,
        "mixt_selection": mixt_selection,
        "validation_sharpe_models": validation_sharpe_models,
        "validation_garch_models": validation_garch,
        "long_meta_threshold": 0.5,
        "short_meta_threshold": 0.5,
        "threshold_table": threshold_table,
        "rmse_comparison": metric_table(models, "rmse"),
        "sharpe_rmse_comparison": metric_table(sharpe_models, "rmse"),
        "sharpe_mae_comparison": metric_table(sharpe_models, "mae"),
        "sharpe_r2_comparison": metric_table(sharpe_models, "r2"),
        "vol_metrics": vol_metrics,
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / "models.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Modèles enregistrés ({time.time() - t0:.0f} s)")
    return results


if __name__ == "__main__":
    main()
