"""Features utilisées chaque semaine du test par la méthode du modèle mixte de
chaque paire (outputs/mixt_features.csv et outputs/mixt_features_summary.csv).
À lancer après run_strategy.py."""
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from Code.build_model import load_inputs
from Function.functions import MODELS, OUTPUTS
from Function.functions_model import (
    forecast_rows, model_data, ols_fit, ols_predict, select_stepwise_model,
)
from Function.glmnet import cv_glmnet


def used_features(X_train, y_train, X_test, model_type):
    """Features utilisées et prévision (mêmes estimations que fit_predict_model) :
    toutes pour Linear et Ridge, coefficients non nuls pour Lasso et Elastic_Net,
    variables retenues par le BIC pour Stepwise."""
    p = X_train.shape[1]
    if model_type == "Linear":
        return np.ones(p, bool), float(ols_predict(ols_fit(X_train, y_train), X_test)[0])
    if model_type == "Stepwise":
        selected = select_stepwise_model(X_train, y_train)
        used = np.zeros(p, bool)
        used[selected] = True
        coef = ols_fit(X_train[:, selected], y_train)
        return used, float(ols_predict(coef, X_test[:, selected])[0])
    if model_type in ("Lasso", "Ridge", "Elastic_Net"):
        alpha = {"Lasso": 1.0, "Ridge": 0.0, "Elastic_Net": 0.5}[model_type]
        a0, beta, _ = cv_glmnet(X_train, y_train, alpha)
        return beta != 0, float(a0 + X_test[0] @ beta)
    raise ValueError(f"Méthode sans features identifiables : {model_type}")


def main():
    with open(MODELS / "models.pkl", "rb") as f:
        models = pickle.load(f)
    features_by_pair, fx_targets, _ = load_inputs()
    selection = models["mixt_selection"]
    methods = dict(zip(selection["Currency"], selection["Selected_Model"]))

    usage, summary = {}, []
    for pair, method in methods.items():
        data = model_data(features_by_pair[pair], fx_targets[pair])
        y = data["target"].values
        X = np.ascontiguousarray(data.drop(columns="target").values)
        rows = forecast_rows(len(data), "test", models["train_ratio"])
        fits = [used_features(X[:i], y[:i], X[i:i + 1], method) for i in rows]

        # Mêmes prédictions que le modèle mixte sauvegardé
        saved = models["sharpe_models"]["MIXT"][pair]
        assert data.index[rows].equals(saved["dates"])
        assert np.allclose([f[1] for f in fits], saved["return_predictions"], rtol=1e-10, atol=0)

        used = np.array([f[0] for f in fits])
        names = [re.sub(r"^[a-z]{3}_spread", "spread", c) for c in data.columns[1:]]
        usage[pair] = pd.Series(np.round(100 * used.mean(axis=0), 1), index=names)
        counts = used.sum(axis=1)
        summary.append({"Currency": pair, "Selected_Model": method,
                        "Mean_Features": round(float(counts.mean()), 1),
                        "Min_Features": int(counts.min()), "Max_Features": int(counts.max())})

    mixt_features = pd.DataFrame(usage).rename_axis("Feature").reset_index()
    mixt_summary = pd.DataFrame(summary)
    print(mixt_summary.to_string(index=False))
    print(mixt_features.to_string(index=False))
    mixt_features.to_csv(OUTPUTS / "mixt_features.csv", index=False)
    mixt_summary.to_csv(OUTPUTS / "mixt_features_summary.csv", index=False)


if __name__ == "__main__":
    main()
