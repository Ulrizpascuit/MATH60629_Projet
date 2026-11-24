"""E1.2 — Schéma de validation croisée pour choisir λ (Lasso, Ridge, Elastic net).

Trois schémas, à l'intérieur des données d'entraînement de chaque semaine :
  aleatoire : 5 plis tirés au hasard (version de l'ancien projet, cv_glmnet) ;
  blocs     : 5 blocs de semaines consécutives ; les 26 semaines collées avant et
              après le bloc évalué sont retirées de l'entraînement ;
  forward   : forward-chaining, 6 blocs consécutifs ; pour k = 1..5, entraînement
              sur les blocs précédents moins les 26 dernières semaines, évaluation
              sur le bloc k.
Dans tous les cas, λ = le plus grand λ du chemin du modèle complet qui minimise
l'erreur de validation croisée (lambda.min de glmnet).

Règle de choix fixée avant le calcul : le schéma retenu (un seul pour toutes les
devises) est celui qui a le meilleur R² de validation moyen sur les 15 cas
(3 modèles × 5 devises). Le test est calculé pour chaque schéma, une seule fois.

Usage : .venv/bin/python Code/experiments/e1_2_cv_lambda.py
Sorties : experiments/E1_2_cv_lambda/"""
import json
import pickle
import time

import matplotlib.ticker
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (GRID, INK2, LABELS, PAIR_LABELS, PAIRS, PENALIZED, SERIES, accuracy,
                    historical_mean, load_frames, load_reference, out_dir, plt,
                    reference_predictions, save_fig, write_environment)
from Function.glmnet import cv_glmnet, glmnet_gaussian, predict_glmnet

DEST = out_dir("E1_2_cv_lambda")
SCHEMES = ["aleatoire", "blocs", "forward"]
SCHEME_LABELS = {"aleatoire": "Plis aléatoires (actuel)", "blocs": "Blocs + tampon 26 sem.",
                 "forward": "Forward-chaining + tampon 26 sem."}
ALPHA = {"Lasso": 1.0, "Ridge": 0.0, "Elastic_Net": 0.5}
NFOLDS, GAP = 5, 26


def temporal_folds(n, scheme, nfolds=NFOLDS, gap=GAP):
    """Liste de (indices d'entraînement, indices de validation) d'un schéma temporel."""
    idx = np.arange(n)
    folds = []
    if scheme == "blocs":
        for block in np.array_split(idx, nfolds):
            lo, hi = block[0] - gap, block[-1] + gap
            folds.append((idx[(idx < lo) | (idx > hi)], block))
    elif scheme == "forward":
        blocks = np.array_split(idx, nfolds + 1)
        for k in range(1, nfolds + 1):
            folds.append((idx[: max(blocks[k][0] - gap, 0)], blocks[k]))
    else:
        raise ValueError(scheme)
    return folds


def cv_lambda(x, y, alpha, scheme):
    """Coefficients au λ choisi par le schéma de CV, et ce λ."""
    if scheme == "aleatoire":
        a0, beta, info = cv_glmnet(x, y, alpha, nfolds=NFOLDS, seed=123)
        return a0, beta, float(info["lambda"][info["index"]])
    lam, a0, beta = glmnet_gaussian(x, y, alpha)
    sse, count = np.zeros(len(lam)), 0
    for tr, te in temporal_folds(len(y), scheme):
        lam_i, a0_i, beta_i = glmnet_gaussian(x[tr], y[tr], alpha)
        pred = predict_glmnet(lam_i, a0_i, beta_i, x[te], lam)
        sse += ((y[te, None] - pred) ** 2).sum(axis=0)
        count += len(te)
    cvm = sse / count
    idx = int(np.flatnonzero(cvm <= cvm.min())[0])  # plus grand λ qui atteint le minimum
    return a0[idx], beta[:, idx], float(lam[idx])


def walk_forward(X, y, rows, method, scheme):
    preds, lams = np.empty(len(rows)), np.empty(len(rows))
    for k, i in enumerate(rows):
        a0, beta, lam = cv_lambda(X[:i], y[:i], ALPHA[method], scheme)
        preds[k], lams[k] = a0 + X[i] @ beta, lam
    return preds, lams


def main():
    t0 = time.time()
    frames, ref = load_frames(), load_reference()
    tasks = [(pair, m, s, period) for pair in PAIRS for m in PENALIZED for s in SCHEMES
             for period in ["validation", "test"]]
    results = Parallel(n_jobs=-1)(
        delayed(walk_forward)(frames[p]["X"], frames[p]["y"], frames[p][per], m, s)
        for p, m, s, per in tasks)
    out = {t: r for t, r in zip(tasks, results)}

    # Contrôle : le schéma aléatoire doit redonner exactement la référence E0
    for pair in PAIRS:
        for m in PENALIZED:
            for period in ["validation", "test"]:
                ref_pred, _ = reference_predictions(ref, m, pair, period)
                assert np.allclose(out[(pair, m, "aleatoire", period)][0], ref_pred,
                                   rtol=0, atol=1e-10), (pair, m, period)

    rows = []
    for (pair, m, s, period), (pred, lams) in out.items():
        fr = frames[pair]
        y = fr["y"][fr[period]]
        rows.append({"Scheme": s, "Method": m, "Pair": pair, "Period": period,
                     **accuracy(y, pred, historical_mean(fr["y"], fr[period])),
                     "Lambda_median": float(np.median(lams))})
    acc = pd.DataFrame(rows)
    acc.to_csv(DEST / "precision_par_schema.csv", index=False)

    val = acc[acc.Period == "validation"].groupby("Scheme")["R2"].mean()
    tst = acc[acc.Period == "test"].groupby("Scheme")["R2_OS"].mean()
    summary = pd.DataFrame({"R2_validation_moyen": val, "R2_OS_test_moyen": tst}).loc[SCHEMES]
    chosen = summary["R2_validation_moyen"].idxmax()
    summary["Retenu"] = summary.index == chosen
    summary.to_csv(DEST / "resume_schemas.csv")
    (DEST / "choix.json").write_text(json.dumps({"schema_retenu": chosen,
                                                 "regle": "meilleur R2 de validation moyen sur 15 cas"},
                                                indent=1, ensure_ascii=False), encoding="utf-8")
    print(summary.round(5).to_string())
    print("Schéma retenu (sur la validation) :", chosen)

    (DEST / "models").mkdir(exist_ok=True)
    with open(DEST / "models" / "predictions.pkl", "wb") as f:
        pickle.dump({"predictions": out, "dates": {p: frames[p]["dates"] for p in PAIRS},
                     "rows": {p: {per: frames[p][per] for per in ["validation", "test"]}
                              for p in PAIRS}}, f)
    fig_lambda(out, frames)
    write_environment(DEST, seconds=round(time.time() - t0, 1), regle_de_choix=
                      "meilleur R2 de validation moyen sur 15 cas (3 modeles x 5 devises)")


def fig_lambda(out, frames):
    """λ choisi au fil du temps (médiane sur les 5 devises), par modèle et par schéma."""
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.8), sharey=False)
    for ax, m in zip(axes, PENALIZED):
        for color, s in zip(SERIES, SCHEMES):
            for period in ["validation", "test"]:
                lam = np.median([out[(p, m, s, period)][1] for p in PAIRS], axis=0)
                dates = frames[PAIRS[0]]["dates"][frames[PAIRS[0]][period]]
                ax.plot(dates, lam, color=color, lw=1.2,
                        label=SCHEME_LABELS[s] if period == "validation" else None)
        ax.axvline(frames[PAIRS[0]]["dates"][frames[PAIRS[0]]["test"][0]], color=INK2, lw=0.8,
                   ls="--")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:.0e}".replace("e-0", "e-")))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(LABELS[m], loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0].set_ylabel("λ choisi (médiane des 5 devises, échelle log)")
    axes[1].set_xlabel("Date de la prévision (trait pointillé : début du test)")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.75, -0.38), ncol=3)
    save_fig(fig, DEST, "fig_lambda_temps")


def figures_only():
    """Refait la figure à partir des prévisions déjà enregistrées, sans ré-estimer."""
    with open(DEST / "models" / "predictions.pkl", "rb") as f:
        out = pickle.load(f)["predictions"]
    fig_lambda(out, load_frames())


if __name__ == "__main__":
    import sys
    figures_only() if "--figures" in sys.argv else main()
