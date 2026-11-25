"""E7 — Paquet de résultats pour l'équipe (docs/paquet_equipe/).

Rassemble, sans rien recalculer :
  - figures/ : les figures finales des expériences, numérotées dans l'ordre logique (PDF,
    pour Overleaf), et deux schémas dessinés ici : le pipeline et le protocole (PDF + PNG) ;
  - tableaux/ : les tableaux principaux en CSV et en LaTeX (booktabs, virgule décimale),
    générés à partir des CSV des expériences ; aucun chiffre n'est recopié à la main.

Usage : .venv/bin/python Code/experiments/e7_paquet_equipe.py   (après E1 à E5)"""
import json
import shutil

import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import EXP, INK, INK2, ROOT, SERIES, plt, save_fig

DEST = ROOT / "docs" / "paquet_equipe"
FIG, TAB = DEST / "figures", DEST / "tableaux"

# (nom dans le paquet, source) — dans l'ordre du récit
FIGURES = [
    ("fig01_decoupage_walk_forward", ROOT / "docs/figures/fig01_decoupage_walk_forward.pdf"),
    ("fig04_bloc1_r2os_holm", EXP / "E1_1_precision/fig_r2os_heatmap.pdf"),
    ("fig05_bloc1_lambda_cv", EXP / "E1_2_cv_lambda/fig_lambda_temps.pdf"),
    ("fig06_bloc1_grille_foret", EXP / "E1_3_rf_tuning/fig_grille_foret.pdf"),
    ("fig07_bloc1_selection_vs_combinaison", EXP / "E1_4_combinaison/fig_selection_vs_combinaison.pdf"),
    ("fig08_bloc1_poids_stacking", EXP / "E1_4_combinaison/fig_poids_stacking.pdf"),
    ("fig09_bloc1_stabilite_mixt", EXP / "E1_5_stabilite_mixt/fig_stabilite_mixt.pdf"),
    ("fig10_bloc1_importance_permutation", EXP / "E1_6_importance/fig_importance_permutation.pdf"),
    ("fig11_bloc1_frequence_selection", EXP / "E1_6_importance/fig_frequence_selection.pdf"),
    ("fig12_bloc2_qlike", EXP / "E2_volatilite/fig_qlike.pdf"),
    ("fig13_bloc2_vol_eurusd", EXP / "E2_volatilite/fig_vol_eurusd.pdf"),
    ("fig14_bloc2_ablation_signal", EXP / "E2_volatilite/fig_ablation_signal.pdf"),
    ("fig15_bloc3_roc_calibration", EXP / "E3_meta/fig_roc_calibration.pdf"),
    ("fig16_bloc3_placebo", EXP / "E3_meta/fig_placebo.pdf"),
    ("fig17_bloc3_permutation_auc", EXP / "E3_meta/fig_permutation_auc.pdf"),
    ("fig18_bloc3_risque_couverture", EXP / "E3_meta/fig_risque_couverture.pdf"),
    ("fig19_ablation_escalier", EXP / "E4_ablation/fig_escalier_ablation.pdf"),
    ("fig20_ablation_valeur_cumulee", EXP / "E4_ablation/fig_valeur_cumulee.pdf"),
    ("fig21_lstm_contre_ridge", EXP / "E5_lstm/fig_lstm_contre_ridge.pdf"),
]


## ---- formatage ----
def num(x, d=2, pct=False):
    if pd.isna(x):
        return "--"
    v = 100 * x if pct else x
    s = f"{v:.{d}f}".replace(".", ",")
    return s.replace("-", "$-$") if s.startswith("-") else s


def write_table(df, name, caption, label, colfmt=None):
    df.to_csv(TAB / f"{name}.csv", index=False)
    cols = colfmt or "l" + "r" * (df.shape[1] - 1)
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\begin{{tabular}}{{{cols}}}",
             r"\toprule", " & ".join(df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(str(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


## ---- tableaux ----
def tables():
    holm = pd.read_csv(EXP / "E1_1_precision/resume_holm.csv")
    write_table(pd.DataFrame({
        "Seuil": [num(a, 0, True) + r" \%" for a in holm.Seuil],
        "Tests": holm.Tests, "Significatifs (bruts)": holm.Succes_bruts,
        "Significatifs (Holm)": holm.Succes_Holm, r"$R^2_{OS} > 0$": holm.R2_OS_positifs}),
        "tab01_bloc1_holm", "Tests de Clark-West sur le test (7 modèles et MIXT, 5 devises).",
        "tab:holm")

    cv = pd.read_csv(EXP / "E1_2_cv_lambda/resume_schemas.csv")
    names = {"aleatoire": "Plis aléatoires", "blocs": "Blocs + tampon 26 sem.",
             "forward": "Forward-chaining + tampon 26 sem."}
    write_table(pd.DataFrame({
        "Schéma de CV": [names[s] + (" (retenu)" if r else "") for s, r in zip(cv.Scheme, cv.Retenu)],
        r"$R^2$ validation (\%)": [num(v, 2, True) for v in cv.R2_validation_moyen],
        r"$R^2_{OS}$ test (\%)": [num(v, 2, True) for v in cv.R2_OS_test_moyen]}),
        "tab02_bloc1_cv_lambda", "Choix de $\\lambda$ : moyenne sur Lasso, Ridge et Elastic net, 5 devises.",
        "tab:cv")

    v = pd.read_csv(EXP / "E1_4_combinaison/resume_variantes.csv")
    lab = {"Linear": "MCO", "Stepwise": "Pas-à-pas", "PCA": "ACP", "Lasso": "Lasso", "Ridge": "Ridge",
           "Elastic_Net": "Elastic net", "Random_Forest": "Forêt (réglée)", "MIXT_ancien": "MIXT",
           "Mean": "Moyenne simple", "Stacking": "Stacking"}
    rows = []
    for k, name in lab.items():
        va = v[(v.Variant == k) & (v.Period == "validation")].iloc[0]
        te = v[(v.Variant == k) & (v.Period == "test")].iloc[0]
        rows.append({"Prévision": name, r"$R^2_{OS}$ validation (\%)": num(va.R2_OS, 2, True),
                     r"$R^2_{OS}$ test (\%)": num(te.R2_OS, 2, True), "IC de rang (test)": num(te.IC_moyen, 3)})
    write_table(pd.DataFrame(rows), "tab03_bloc1_selection_combinaison",
                "Sélection (MIXT) contre combinaison ; moyennes sur 5 devises. MIXT et le stacking sont "
                "ajustés sur la validation, leur $R^2$ de validation est donc optimiste.", "tab:combinaison")

    q = pd.read_csv(EXP / "E2_volatilite/qlike_moyenne.csv")
    acc = pd.read_csv(EXP / "E2_volatilite/precision_volatilite.csv")
    t = acc[acc.Period == "test"].groupby("Model")[["RMSE_pct", "Corr"]].mean()
    vl = {"GARCH": "GARCH(1,1)", "HAR": "HAR (log) (retenu)", "VR_derniere": "VR semaine passée",
          "Ecart_type_12": "Écart-type 12 sem."}
    write_table(pd.DataFrame([{"Modèle": vl[m], "QLIKE validation": num(r.validation, 3),
                               "QLIKE test": num(r.test, 3), "RMSE test (pts de \\%)": num(t.loc[m, "RMSE_pct"], 2),
                               "Corrélation test": num(t.loc[m, "Corr"], 2)}
                              for m, r in q.set_index("Model").iterrows()]),
                "tab04_bloc2_volatilite", "Prévision de la volatilité réalisée hebdomadaire (moyennes sur 5 devises).",
                "tab:vol")

    d = pd.read_csv(EXP / "E3_meta/discrimination_calibration.csv")
    pl = json.loads((EXP / "E3_meta/resume.json").read_text())["placebo"]
    write_table(pd.DataFrame([{
        "Côté": {"Long": "Achats", "Short": "Ventes"}[r.Side], "Jambes": r.N,
        "AUC": num(r.AUC, 3), "$p$ (permutation)": num(r.AUC_p_permutation, 3),
        "Brier": num(r.Brier, 4), "Brier de référence": num(r.Brier_reference, 4),
        r"Gain gardées / rejetées (\%)": f"{num(r.Gain_acceptees, 1, True)} / {num(r.Gain_rejetees, 1, True)}"}
        for r in d.itertuples()]), "tab05_bloc3_discrimination",
        f"Méta-classifieur sur le test. Placebo : Sharpe {num(pl['Sharpe_filtre'])} contre "
        f"{num(pl['Placebo_moyenne'])} en moyenne pour 10~000 filtres au hasard, $p = {num(pl['p_valeur'], 3)}$.",
        "tab:meta")

    a = pd.read_csv(EXP / "E3_meta/aurc.csv", index_col=0)
    sl = {"Meta_p": "Probabilité du méta-classifieur", "Abs_s": r"Force du signal $|\hat s|$", "Hasard": "Hasard"}
    write_table(pd.DataFrame([{"Score": sl[k], "AURC (erreur)": num(r.AURC_erreur, 3),
                               "Rendement moyen (pb/sem.)": num(1e4 * r.Rendement_moyen_aire, 1)}
                              for k, r in a.iterrows()]),
                "tab06_bloc3_risque_couverture", "Courbes risque-couverture sur les 392 jambes de test (diagnostic).",
                "tab:aurc")

    p = pd.read_csv(EXP / "E4_ablation/ablation_performance.csv")
    n = json.loads((EXP / "E4_ablation/choix.json").read_text())
    rows = []
    for r in p.itertuples():
        rows.append({"Variante": r.Strategy.replace("r̂", r"$\hat r$").replace("σ̂", r"$\hat\sigma$"),
                     "Sharpe": num(r.Sharpe), "IC à 95 \\%": f"[{num(r.Sharpe_IC95_bas)} ; {num(r.Sharpe_IC95_haut)}]",
                     "Placebo $p$": num(r.p_valeur, 3), "DSR": num(r.DSR, 2),
                     "Sharpe sans financement": num(r.Sharpe_sans_financement)})
    write_table(pd.DataFrame(rows), "tab07_ablation",
                f"Ablation de bout en bout sur le test (260 semaines). DSR avec $N = {n['N_essais_DSR']}$ essais.",
                "tab:ablation", colfmt="lrrrrr")

    s = pd.read_csv(EXP / "E5_lstm/resume.csv")
    ml = {"LSTM": "LSTM à 5 sorties (5 graines)", "Ridge_52": "Ridge"}
    write_table(pd.DataFrame([{"Modèle (réestimé toutes les 52 sem.)": ml[r.Model],
                               r"$R^2_{OS}$ test (\%)": num(r.R2_OS_test, 2, True),
                               "IC de rang (test)": num(r.IC_test, 3),
                               "Sharpe A1 [IC 95 \\%]": f"{num(r.Sharpe_A1_test)} [{num(r.Sharpe_IC95_bas)} ; {num(r.Sharpe_IC95_haut)}]"}
                              for r in s.itertuples()]),
                "tab08_lstm", "LSTM contre Ridge réestimée aux mêmes dates.", "tab:lstm")


## ---- schémas ----
def box(ax, x, y, w, h, text, color, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=INK2, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=INK2))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.5); ax.axis("off")
    light = {"data": "#eeede9", "b1": "#dbe7f7", "b2": "#fbe1d6", "b3": "#d3efe3", "out": "#eeede9"}
    box(ax, 0.05, 1.55, 1.75, 1.3, "Données\n5 devises / USD\n15 variables\nhebdomadaires", light["data"], fs=7)
    box(ax, 2.25, 2.55, 3.0, 1.6, "Bloc 1 : rendements\n7 modèles en walk-forward\n(MCO, pas-à-pas, ACP,\nLasso, Ridge, EN, forêt)\n→ MIXT / moyenne", light["b1"], fs=7)
    box(ax, 2.25, 0.25, 3.0, 1.6, "Bloc 2 : volatilité\nGARCH(1,1) / HAR\nen walk-forward", light["b2"], fs=7)
    box(ax, 5.7, 1.5, 1.6, 1.4, "Signal\n$\\hat s = \\hat r / \\hat\\sigma$\nclassement\ndes 5 paires", light["out"], fs=7)
    box(ax, 7.7, 1.5, 1.75, 1.4, "Bloc 3 :\nméta-classifieur\nP(jambe\ngagnante)", light["b3"], fs=7)
    box(ax, 9.8, 1.5, 1.15, 1.4, "Stratégie\nlong / short\n(test :\n260 sem.)", light["out"], fs=7)
    arrow(ax, (1.8, 2.5), (2.25, 3.2)); arrow(ax, (1.8, 1.9), (2.25, 1.1))
    arrow(ax, (5.25, 3.3), (5.7, 2.6)); arrow(ax, (5.25, 1.0), (5.7, 1.8))
    arrow(ax, (7.3, 2.2), (7.7, 2.2)); arrow(ax, (9.45, 2.2), (9.8, 2.2))
    ax.text(5.5, 4.4, "Chaque bloc ML est évalué seul, puis ajouté un à un (ablation)",
            ha="center", fontsize=8, color=INK2)
    save_fig(fig, FIG, "fig02_pipeline")


def fig_protocole():
    fig, ax = plt.subplots(figsize=(7.4, 2.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis("off")
    specs = [(0.1, SERIES[0], "Niveau 1 : CV interne",
              "dans chaque réestimation\nhebdomadaire, sur les\nsemaines < t seulement\n→ choisit $\\lambda$"),
             (3.45, SERIES[1], "Niveau 2 : validation (354 sem.)",
              "choisit MIXT, le schéma de CV,\nla forêt, les poids du stacking,\nla volatilité, la variante finale"),
             (6.8, SERIES[2], "Niveau 3 : test (260 sem.)",
              "aucun choix ;\nchaque variante évaluée\nune seule fois ;\ntout est rapporté")]
    for x, c, title, body in specs:
        ax.add_patch(FancyBboxPatch((x, 0.15), 3.1, 2.25, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec=c, lw=1.6))
        ax.text(x + 1.55, 2.05, title, ha="center", fontsize=7.2, color=INK, fontweight="bold")
        ax.text(x + 1.55, 1.0, body, ha="center", va="center", fontsize=7.5, color=INK, linespacing=1.35)
    arrow(ax, (3.2, 1.25), (3.45, 1.25)); arrow(ax, (6.55, 1.25), (6.8, 1.25))
    save_fig(fig, FIG, "fig03_protocole")


def main():
    for d in (FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    for name, src in FIGURES:
        assert src.exists(), src
        shutil.copy2(src, FIG / f"{name}.pdf")
    fig_pipeline()
    fig_protocole()
    tables()
    print(f"{len(list(FIG.glob('*.pdf')))} figures PDF, {len(list(TAB.glob('*.tex')))} tableaux LaTeX dans {DEST}")


if __name__ == "__main__":
    main()
