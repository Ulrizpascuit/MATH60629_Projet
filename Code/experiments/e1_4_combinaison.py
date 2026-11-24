"""E1.4 — Sélection d'un modèle (MIXT) contre combinaison des modèles.

Membres (7 modèles appris) : MCO, pas-à-pas et ACP de la référence E0 ; Lasso,
Ridge et Elastic net avec le schéma de CV retenu en E1.2 ; forêt avec le réglage
retenu en E1.3. Variantes comparées :
  - chaque membre seul ;
  - MIXT de l'ancien projet (référence E0) ;
  - MIXT mis à jour : même règle (meilleur R² de validation par devise) appliquée
    aux membres mis à jour et au Naïf ;
  - Moyenne : moyenne simple des 7 membres ;
  - Stacking : poids >= 0 de somme 1, estimés par devise une seule fois sur les
    prévisions de validation (moindres carrés contraints), gelés sur le test.

Usage : .venv/bin/python Code/experiments/e1_4_combinaison.py   (après E1.2 et E1.3)
Sorties : experiments/E1_4_combinaison/"""
import json
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import (EXP, INK, INK2, LABELS, LEARNED, PAIR_LABELS, PAIRS, PENALIZED, SERIES,
                    accuracy, historical_mean, load_frames, load_reference, out_dir, plt,
                    rank_ic, reference_predictions, save_fig, write_environment)

DEST = out_dir("E1_4_combinaison")
PERIODS = ["validation", "test"]


def load_members(ref):
    """Prévisions (validation, test) des 7 membres mis à jour, par devise."""
    scheme = json.loads((EXP / "E1_2_cv_lambda" / "choix.json").read_text())["schema_retenu"]
    with open(EXP / "E1_2_cv_lambda" / "models" / "predictions.pkl", "rb") as f:
        cv = pickle.load(f)["predictions"]
    with open(EXP / "E1_3_rf_tuning" / "models" / "predictions.pkl", "rb") as f:
        rf = pickle.load(f)
    leaf, mtry = rf["choix"]
    members = {}
    for pair in PAIRS:
        for period in PERIODS:
            d = {}
            for m in ["Linear", "Stepwise", "PCA"]:
                d[m] = reference_predictions(ref, m, pair, period)[0]
            for m in PENALIZED:
                d[m] = cv[(pair, m, scheme, period)][0]
            d["Random_Forest"] = rf[period][(pair, leaf, mtry)]
            d["Naive"] = reference_predictions(ref, "Naive", pair, period)[0]
            members[(pair, period)] = d
    return members, {"schema_cv": scheme, "foret": {"min_samples_leaf": leaf, "max_features": mtry}}


def r2(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def stacking_weights(F, y):
    """min ||y - F w||² sous w >= 0 et somme(w) = 1."""
    k = F.shape[1]
    res = minimize(lambda w: np.sum((y - F @ w) ** 2), np.full(k, 1 / k),
                   jac=lambda w: -2 * F.T @ (y - F @ w), method="SLSQP",
                   bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-14, "maxiter": 1000})
    assert res.success, res.message
    return np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()


def main():
    frames, ref = load_frames(), load_reference()
    members, setup = load_members(ref)

    preds, weights, mixt_new = {}, {}, {}
    for pair in PAIRS:
        fr = frames[pair]
        y_val = fr["y"][fr["validation"]]
        val = members[(pair, "validation")]
        # MIXT mis à jour : même règle que l'ancien projet (Naïf compris dans les candidats)
        cands = ["Naive"] + LEARNED
        mixt_new[pair] = max(cands, key=lambda m: r2(y_val, val[m]))
        F_val = np.column_stack([val[m] for m in LEARNED])
        w = stacking_weights(F_val, y_val)
        weights[pair] = dict(zip(LEARNED, w))
        for period in PERIODS:
            d = members[(pair, period)]
            F = np.column_stack([d[m] for m in LEARNED])
            out = {m: d[m] for m in LEARNED}
            out["MIXT_ancien"] = reference_predictions(ref, "MIXT", pair, period)[0]
            out["MIXT_maj"] = d[mixt_new[pair]]
            out["Mean"] = F.mean(axis=1)
            out["Stacking"] = F @ w
            preds[(pair, period)] = out

    variants = LEARNED + ["MIXT_ancien", "MIXT_maj", "Mean", "Stacking"]
    rows, ic_rows = [], []
    for period in PERIODS:
        for v in variants:
            for pair in PAIRS:
                fr = frames[pair]
                y = fr["y"][fr[period]]
                rows.append({"Period": period, "Variant": v, "Pair": pair,
                             **accuracy(y, preds[(pair, period)][v],
                                        historical_mean(fr["y"], fr[period]))})
            P = np.column_stack([preds[(p, period)][v] for p in PAIRS])
            A = np.column_stack([frames[p]["y"][frames[p][period]] for p in PAIRS])
            ic, ic_t = rank_ic(P, A)
            ic_rows.append({"Period": period, "Variant": v, "IC_moyen": ic, "IC_t": ic_t})
    acc = pd.DataFrame(rows)
    ic = pd.DataFrame(ic_rows)
    acc.to_csv(DEST / "precision_variantes.csv", index=False)
    summary = (acc.groupby(["Period", "Variant"])[["R2", "R2_OS"]].mean().reset_index()
               .merge(ic, on=["Period", "Variant"]))
    summary.to_csv(DEST / "resume_variantes.csv", index=False)
    w = pd.DataFrame(weights).T.rename_axis("Pair")
    w.to_csv(DEST / "poids_stacking.csv")
    (DEST / "choix.json").write_text(json.dumps(
        {**setup, "mixt_maj": mixt_new,
         "mixt_ancien": ref["mixt_selection"].set_index("Currency")["Selected_Model"].to_dict()},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(summary.pivot(index="Variant", columns="Period", values=["R2_OS", "IC_moyen"])
          .loc[variants].round(4).to_string())
    print("MIXT mis à jour :", mixt_new)
    print(w.round(3).to_string())

    fig_variants(acc, summary, variants)
    fig_weights(w)
    write_environment(DEST, **setup)


def fig_variants(acc, summary, variants):
    """R² hors échantillon sur le test : moyenne des 5 devises (barre) et chaque devise (points)."""
    t = acc[acc.Period == "test"]
    s = summary[summary.Period == "test"].set_index("Variant").loc[variants]
    labels = [LABELS.get(v, v) for v in LEARNED] + ["MIXT (ancien)", "MIXT (mis à jour)",
                                                    "Moyenne", "Stacking"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(variants))
    colors = [SERIES[0]] * len(LEARNED) + [SERIES[1]] * 2 + [SERIES[2]] * 2
    ax.bar(x, 100 * s.R2_OS.values, color=colors, width=0.65, edgecolor="white", linewidth=2)
    for j, v in enumerate(variants):
        vals = 100 * t[t.Variant == v].set_index("Pair").loc[PAIRS].R2_OS.values
        ax.plot(np.full(len(vals), j), vals, "o", ms=3.5, color=INK2, alpha=0.8,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("$R^2$ hors échantillon, test (%)")
    ax.set_ylim(max(ax.get_ylim()[0], -16), ax.get_ylim()[1])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SERIES[:3]] + [
        plt.Line2D([], [], marker="o", color=INK2, ls="", ms=4)]
    ax.legend(handles, ["Modèle seul", "Sélection (MIXT)", "Combinaison", "Une devise"],
              frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save_fig(fig, DEST, "fig_selection_vs_combinaison")


def fig_weights(w):
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    mat = 100 * w.loc[PAIRS, LEARNED].values
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(PAIRS)):
        for j in range(len(LEARNED)):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if mat[i, j] > 60 else INK)
    ax.set_xticks(range(len(LEARNED)), [LABELS[m] for m in LEARNED], rotation=30, ha="right")
    ax.set_yticks(range(len(PAIRS)), [PAIR_LABELS[p] for p in PAIRS])
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Poids du stacking (%)")
    save_fig(fig, DEST, "fig_poids_stacking")


if __name__ == "__main__":
    main()
