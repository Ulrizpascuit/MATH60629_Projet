"""E1.3 — Réglage de la forêt aléatoire sur la validation.

Grille de 12 réglages (500 arbres) : taille minimale des feuilles dans
{5, 25, 50, 100} × nombre de variables candidates par division dans {3, 5, 15}
(√p, p/3 et toutes les variables). Chaque réglage est estimé en walk-forward
(fenêtre croissante) sur les 354 semaines de validation, pour les 5 devises.

Règle de choix fixée avant le calcul : un seul réglage pour toutes les devises,
celui qui a le meilleur R² de validation moyen sur les 5 devises. Seul ce réglage
est ensuite évalué, une seule fois, sur les 260 semaines de test.

Usage : .venv/bin/python Code/experiments/e1_3_rf_tuning.py   (≈ 30 à 60 min sur 10 cœurs)
Sorties : experiments/E1_3_rf_tuning/"""
import itertools
import json
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from common import (INK, PAIRS, accuracy, historical_mean, load_frames, load_reference,
                    out_dir, plt, reference_predictions, save_fig, write_environment)

DEST = out_dir("E1_3_rf_tuning")
LEAVES = [5, 25, 50, 100]
MTRY = [3, 5, 15]
GRID = list(itertools.product(LEAVES, MTRY))
N_TREES, SEED = 500, 123


def walk_forward(X, y, rows, leaf, mtry):
    preds = np.empty(len(rows))
    for k, i in enumerate(rows):
        model = RandomForestRegressor(n_estimators=N_TREES, max_features=mtry,
                                      min_samples_leaf=leaf, bootstrap=True,
                                      random_state=SEED, n_jobs=1)
        model.fit(X[:i], y[:i])
        preds[k] = model.predict(X[i:i + 1])[0]
    return preds


def run(frames, configs, period):
    tasks = [(pair, leaf, mtry) for (leaf, mtry) in configs for pair in PAIRS]
    # Les réglages les plus lents (toutes les variables, petites feuilles) d'abord
    tasks.sort(key=lambda t: (-t[2], t[1]))
    res = Parallel(n_jobs=-1)(
        delayed(walk_forward)(frames[p]["X"], frames[p]["y"], frames[p][period], leaf, mtry)
        for p, leaf, mtry in tasks)
    return {t: r for t, r in zip(tasks, res)}


def main():
    t0 = time.time()
    frames, ref = load_frames(), load_reference()

    # 1) Grille sur la validation seulement
    val = run(frames, GRID, "validation")
    rows = []
    for (pair, leaf, mtry), pred in val.items():
        fr = frames[pair]
        y = fr["y"][fr["validation"]]
        rows.append({"Leaf": leaf, "Mtry": mtry, "Pair": pair, "Period": "validation",
                     **accuracy(y, pred, historical_mean(fr["y"], fr["validation"]))})
    # Réglage de l'ancien projet (référence E0), pour comparaison
    for pair in PAIRS:
        fr = frames[pair]
        pred, y = reference_predictions(ref, "Random_Forest", pair, "validation")
        rows.append({"Leaf": "defaut", "Mtry": 3, "Pair": pair, "Period": "validation",
                     **accuracy(y, pred, historical_mean(fr["y"], fr["validation"]))})
    grid = pd.DataFrame(rows)
    mean_val = (grid[grid.Leaf != "defaut"].groupby(["Leaf", "Mtry"])["R2"].mean()
                .rename("R2_validation_moyen").reset_index())
    best = mean_val.loc[mean_val.R2_validation_moyen.idxmax()]
    leaf, mtry = int(best.Leaf), int(best.Mtry)
    print(mean_val.round(5).to_string(index=False))
    print(f"Réglage retenu (validation) : feuilles >= {leaf}, {mtry} variables par division")

    # 2) Test, une seule fois, pour le réglage retenu
    tst = run(frames, [(leaf, mtry)], "test")
    for (pair, _, _), pred in tst.items():
        fr = frames[pair]
        y = fr["y"][fr["test"]]
        grid.loc[len(grid)] = {"Leaf": leaf, "Mtry": mtry, "Pair": pair, "Period": "test",
                               **accuracy(y, pred, historical_mean(fr["y"], fr["test"]))}
    for pair in PAIRS:
        fr = frames[pair]
        pred, y = reference_predictions(ref, "Random_Forest", pair, "test")
        grid.loc[len(grid)] = {"Leaf": "defaut", "Mtry": 3, "Pair": pair, "Period": "test",
                               **accuracy(y, pred, historical_mean(fr["y"], fr["test"]))}
    grid.to_csv(DEST / "precision_grille.csv", index=False)
    mean_val.to_csv(DEST / "resume_grille_validation.csv", index=False)
    (DEST / "choix.json").write_text(json.dumps(
        {"min_samples_leaf": leaf, "max_features": mtry, "n_estimators": N_TREES,
         "regle": "meilleur R2 de validation moyen sur 5 devises"}, indent=1, ensure_ascii=False),
        encoding="utf-8")

    (DEST / "models").mkdir(exist_ok=True)
    with open(DEST / "models" / "predictions.pkl", "wb") as f:
        pickle.dump({"validation": val, "test": tst, "choix": (leaf, mtry)}, f)
    fig_grid(mean_val, grid, leaf, mtry)
    write_environment(DEST, seconds=round(time.time() - t0, 1),
                      regle_de_choix="meilleur R2 de validation moyen sur les 5 devises")


def fig_grid(mean_val, grid, leaf, mtry):
    """R² de validation moyen (5 devises) pour chaque réglage de la grille."""
    mat = mean_val.pivot(index="Leaf", columns="Mtry", values="R2_validation_moyen").loc[LEAVES, MTRY]
    ref = grid[(grid.Leaf == "defaut") & (grid.Period == "validation")]["R2"].mean()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    vals = 100 * mat.values
    im = ax.imshow(vals, cmap="Blues", aspect="auto")
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    for i, lf in enumerate(LEAVES):
        for j, mt in enumerate(MTRY):
            v = vals[i, j]
            ax.text(j, i, f"{v:.2f}" + (" ◆" if (lf, mt) == (leaf, mtry) else ""),
                    ha="center", va="center", fontsize=8,
                    color="white" if (v - lo) / (hi - lo) > 0.5 else INK)
    ax.set_xticks(range(len(MTRY)), [f"{m}\n({'√p' if m == 3 else 'p/3' if m == 5 else 'p'})"
                                     for m in MTRY])
    ax.set_yticks(range(len(LEAVES)), LEAVES)
    ax.set_xlabel("Variables candidates par division")
    ax.set_ylabel("Taille minimale des feuilles")
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("$R^2$ de validation moyen (%)")
    ax.set_title(f"◆ réglage retenu ; réglage de l'ancien projet : {100 * ref:.2f} %", loc="left",
                 fontsize=8)
    save_fig(fig, DEST, "fig_grille_foret")


def figures_only():
    """Refait la figure à partir des fichiers déjà écrits, sans ré-estimer la grille."""
    grid = pd.read_csv(DEST / "precision_grille.csv")
    mean_val = pd.read_csv(DEST / "resume_grille_validation.csv")
    choix = json.loads((DEST / "choix.json").read_text())
    fig_grid(mean_val, grid, choix["min_samples_leaf"], choix["max_features"])


if __name__ == "__main__":
    import sys
    figures_only() if "--figures" in sys.argv else main()
