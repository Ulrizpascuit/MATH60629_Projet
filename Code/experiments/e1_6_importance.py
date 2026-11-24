"""E1.6 — Interprétation : quelles variables les modèles utilisent-ils ?

1) Forêt réglée (E1.3 : feuilles >= 25, 3 variables, 500 arbres) entraînée sur les
   529 premières semaines ; importance par permutation (perte de R²) sur les 354
   semaines de validation, 30 répétitions (Breiman, 2001). Le test n'est pas utilisé.
2) Fréquence de sélection des variables par le Lasso, l'Elastic net et le pas-à-pas
   sur les semaines de test (référence E0, reprise telle quelle).

Usage : .venv/bin/python Code/experiments/e1_6_importance.py   (après E1.3)
Sorties : experiments/E1_6_importance/"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from common import (EXP, INK, INK2, PAIR_LABELS, PAIRS, SERIES, load_frames, out_dir, plt,
                    save_fig, write_environment)

DEST = out_dir("E1_6_importance")
REPEATS, SEED = 30, 123
FEATURE_LABELS = {
    "spread2": "Écart 2 ans", "spread10": "Écart 10 ans", "spread2_diff": "Δ écart 2 ans",
    "spread10_diff": "Δ écart 10 ans", "dgs2_diff": "Δ taux US 2 ans",
    "dgs10_diff": "Δ taux US 10 ans", "gold_return": "Rend. or", "sp500_return": "Rend. S&P 500",
    "oil_return": "Rend. pétrole", "lag_return": "Rend. retardé", "vol_4": "Vol. 4 sem.",
    "vol_12": "Vol. 12 sem.", "rsi": "RSI", "macd": "MACD", "bb_pctB": "Bollinger %B",
}


def label(col):
    return FEATURE_LABELS[col.split("_", 1)[1] if col[:4] in ("eur_", "gbp_", "cad_", "chf_", "jpy_")
                          else col]


def main():
    frames = load_frames()
    choix = json.loads((EXP / "E1_3_rf_tuning" / "choix.json").read_text())
    leaf, mtry = choix["min_samples_leaf"], choix["max_features"]
    per_pair, base = {}, {}
    for pair in PAIRS:
        fr = frames[pair]
        v0 = fr["validation"][0]
        rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=leaf, max_features=mtry,
                                   bootstrap=True, random_state=SEED, n_jobs=-1)
        rf.fit(fr["X"][:v0], fr["y"][:v0])
        Xv, yv = fr["X"][fr["validation"]], fr["y"][fr["validation"]]
        base[pair] = rf.score(Xv, yv)
        pi = permutation_importance(rf, Xv, yv, scoring="r2", n_repeats=REPEATS,
                                    random_state=SEED, n_jobs=-1)
        per_pair[pair] = pd.DataFrame(pi.importances, index=[label(c) for c in fr["columns"]])
    # Importance moyenne sur les 5 paires, répétition par répétition
    stack = np.mean([per_pair[p].values for p in PAIRS], axis=0)
    names = per_pair[PAIRS[0]].index
    imp = pd.DataFrame({"Importance_moyenne": stack.mean(1),
                        "IC95_bas": np.quantile(stack, 0.025, axis=1),
                        "IC95_haut": np.quantile(stack, 0.975, axis=1)}, index=names)
    for p in PAIRS:
        imp[p] = per_pair[p].mean(axis=1).values
    imp = imp.sort_values("Importance_moyenne")
    imp.rename_axis("Variable").to_csv(DEST / "importance_permutation.csv")
    pd.Series(base, name="R2_validation_modele_unique").to_csv(DEST / "r2_validation_base.csv")
    print(pd.Series(base).round(4).to_string())
    print((100 * imp[["Importance_moyenne", "IC95_bas", "IC95_haut"]]).round(3).to_string())

    usage = pd.read_csv(EXP / "E0_reference_mac" / "outputs" / "ml" / "feature_usage.csv",
                        header=[0, 1], index_col=0)
    usage.to_csv(DEST / "frequence_selection_lineaire.csv")
    fig_permutation(imp)
    fig_usage(usage)
    write_environment(DEST, min_samples_leaf=leaf, max_features=mtry, repeats=REPEATS, seed=SEED)


def fig_permutation(imp):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    y = np.arange(len(imp))
    m = 100 * imp.Importance_moyenne.values
    ax.errorbar(m, y, xerr=[m - 100 * imp.IC95_bas.values, 100 * imp.IC95_haut.values - m],
                fmt="o", color=SERIES[0], ms=4, capsize=2, lw=1.2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, imp.index)
    ax.set_xlabel("Perte de $R^2$ de validation quand la variable est mélangée\n"
                  "(points de %, moyenne des 5 devises, intervalle à 95 % sur 30 répétitions)")
    ax.set_title("Forêt réglée : importance par permutation", loc="left")
    save_fig(fig, DEST, "fig_importance_permutation")


def fig_usage(usage):
    methods = ["Lasso", "Elastic_Net", "Stepwise"]
    titles = {"Lasso": "Lasso", "Elastic_Net": "Elastic net", "Stepwise": "Pas-à-pas"}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.6), sharey=True)
    for ax, m in zip(axes, methods):
        mat = usage[m][PAIRS].values
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if mat[i, j] > 60 else INK)
        ax.set_xticks(range(len(PAIRS)), [PAIR_LABELS[p][:3] for p in PAIRS], fontsize=7)
        ax.set_title(titles[m], loc="left")
        ax.spines[:].set_visible(False)
    axes[0].set_yticks(range(len(usage.index)), usage.index, fontsize=7.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("% des semaines de test où la variable est retenue")
    save_fig(fig, DEST, "fig_frequence_selection")


if __name__ == "__main__":
    main()
