"""E1.1 — Précision hors échantillon des modèles de rendement, avec correction de Holm.

Lit les prévisions de la référence E0 (aucune ré-estimation). Pour chaque
modèle et chaque devise : R² hors échantillon contre la moyenne historique et
test de Clark-West, sur la validation et sur le test. Sur le test, les 40 tests
(7 modèles appris + MIXT, 5 devises) sont corrigés par la procédure de Holm.

Usage : .venv/bin/python Code/experiments/e1_1_precision.py
Sorties : experiments/E1_1_precision/"""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from common import (INK, LABELS, LEARNED, PAIR_LABELS, PAIRS, accuracy, historical_mean,
                    holm, load_frames, load_reference, out_dir, plt, reference_predictions,
                    save_fig, write_environment)

DEST = out_dir("E1_1_precision")
CELLS = LEARNED + ["MIXT"]  # 8 × 5 = 40 tests sur le test (le Naïf est exclu, comme l'ancien projet)


def main():
    frames, ref = load_frames(), load_reference()
    rows = []
    for period in ["validation", "test"]:
        for m in ["Naive"] + CELLS:
            for pair in PAIRS:
                pred, y = reference_predictions(ref, m, pair, period)
                hm = historical_mean(frames[pair]["y"], frames[pair][period])
                assert np.allclose(y, frames[pair]["y"][frames[pair][period]])
                rows.append({"Period": period, "Method": m, "Pair": pair, **accuracy(y, pred, hm)})
    acc = pd.DataFrame(rows)

    test = acc[(acc.Period == "test") & acc.Method.isin(CELLS)].copy()
    assert len(test) == 40
    test["CW_p_Holm"] = holm(test.CW_p.values)
    acc = acc.merge(test[["Method", "Pair", "CW_p_Holm"]].assign(Period="test"),
                    on=["Period", "Method", "Pair"], how="left")
    acc.to_csv(DEST / "precision_par_modele.csv", index=False)

    summary = pd.DataFrame([{
        "Seuil": a,
        "Succes_bruts": int((test.CW_p < a).sum()),
        "Succes_Holm": int((test.CW_p_Holm < a).sum()),
        "Tests": len(test),
        "R2_OS_positifs": int((test.R2_OS > 0).sum()),
    } for a in [0.05, 0.10]])
    summary.to_csv(DEST / "resume_holm.csv", index=False)
    print(summary.to_string(index=False))
    print(test.sort_values("CW_p")[["Method", "Pair", "R2_OS", "CW_p", "CW_p_Holm"]]
          .head(10).round(4).to_string(index=False))

    fig_heatmaps(acc)
    write_environment(DEST, source="experiments/E0_reference_mac/models/models.pkl")


def fig_heatmaps(acc):
    """R² hors échantillon par modèle et devise, validation et test ; * = significatif
    (Clark-West) à 10 % sans correction, ** = encore significatif après Holm à 10 %."""
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#e9e8e4", "#2a78d6"])
    norm = TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.05)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for ax, period, title in [(axes[0], "validation", "A. Validation (354 semaines)"),
                              (axes[1], "test", "B. Test (260 semaines)")]:
        d = acc[(acc.Period == period) & acc.Method.isin(CELLS)]
        mat = d.pivot(index="Method", columns="Pair", values="R2_OS").loc[CELLS, PAIRS]
        pv = d.pivot(index="Method", columns="Pair", values="CW_p").loc[CELLS, PAIRS]
        ph = (d.pivot(index="Method", columns="Pair", values="CW_p_Holm").loc[CELLS, PAIRS]
              if period == "test" else None)
        im = ax.imshow(mat.values, cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(CELLS)):
            for j in range(len(PAIRS)):
                star = ""
                if pv.values[i, j] < 0.10:
                    star = "*"
                if ph is not None and ph.values[i, j] < 0.10:
                    star = "**"
                ax.text(j, i, f"{100 * mat.values[i, j]:.1f}{star}", ha="center", va="center",
                        color=INK, fontsize=7)
        ax.set_xticks(range(len(PAIRS)), [PAIR_LABELS[p] for p in PAIRS], fontsize=7)
        ax.set_yticks(range(len(CELLS)), [LABELS[m] for m in CELLS])
        ax.set_title(title, loc="left")
        ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, extend="both")
    cb.set_label("$R^2$ hors échantillon (%)")
    cb.set_ticks([-0.05, -0.025, 0, 0.025, 0.05], labels=["−5", "−2,5", "0", "2,5", "5"])
    fig.text(0.01, -0.04, "Valeurs en %. * : Clark-West significatif à 10 % (non corrigé) ; "
             "** : encore significatif après correction de Holm sur les 40 tests du test.",
             fontsize=7, color=INK)
    save_fig(fig, DEST, "fig_r2os_heatmap")


if __name__ == "__main__":
    main()
