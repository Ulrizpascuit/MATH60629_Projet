"""Figures et tableaux du réseau factorisé (après nn_conditional.py)."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from Code.nn_analysis import FEATURE_LABELS, block_bootstrap_sharpe, fmt, long_short, sharpe
from Code.nn_classification import ensemble_probs
from Code.nn_conditional import KS, base_rates, cond_log_losses, ensemble, selected_features
from Code.nn_experiments import INITIAL, OUT, TRAIN, load_panel, weekly_ic
from Function.functions import OUTPUTS, PAIRS, ROOT

FIG = ROOT / "latex" / "figures"
TAB = ROOT / "latex" / "tables"
CSV = OUTPUTS / "nn"
NAMES = {"logit": "Logistique", "mlp": "MLP", "gru": "GRU", "ens": "Ensemble MLP+GRU"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    with open(OUT / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def week_corr(a, b):
    return np.nanmean([np.corrcoef(a[t], b[t])[0, 1] for t in range(len(a))])


def intensity_rule(p_up, p_down, ret):
    """Acheter argmax P(forte hausse), vendre argmax P(forte baisse) parmi les autres paires."""
    rows = np.arange(len(ret))
    long = p_up.argmax(1)
    pdn = p_down.copy()
    pdn[rows, long] = -np.inf
    short = pdn.argmax(1)
    return ret[rows, long] - ret[rows, short]


def collect(d):
    final, logit = load("cond_final"), load("cond_logit")
    probs = {}
    for feat in ("all", "sel"):
        for k in KS:
            for period in ("val", "test"):
                mlp = ensemble(final[f"mlp_k{k}_{feat}"][period])
                gru = ensemble(final[f"gru_k{k}_{feat}"][period])
                probs[("mlp", k, feat, period)] = mlp
                probs[("gru", k, feat, period)] = gru
                probs[("ens", k, feat, period)] = ((mlp[0] + gru[0]) / 2, (mlp[1] + gru[1]) / 2)
                probs[("logit", k, feat, period)] = logit[f"logit_k{k}_{feat}"][period]
    return probs, final, logit


def metrics(d, probs):
    rows = []
    for (model, k, feat, period), (p_big, q) in probs.items():
        a, b = (INITIAL, TRAIN) if period == "val" else (TRAIN, len(d["X"]))
        z, ret = d["z"][a:b], d["ret"][a:b]
        big, up = np.abs(z) > k, z > 0
        bp, bq = base_rates(d["z"], a, b, k)
        ll, llb = cond_log_losses(p_big, q, z, k), cond_log_losses(bp, bq, z, k)
        p_up, p_down = p_big * q, p_big * (1 - q)
        r_dir = long_short(q, ret)
        r_int = intensity_rule(p_up, p_down, ret)
        r_exp = long_short(p_big * (2 * q - 1), ret)
        row = {"Model": model, "k": k, "Features": feat, "Period": period,
               "Skill_amp": 1 - ll[0] / llb[0], "Skill_dir": 1 - ll[1] / llb[1],
               "AUC_amp": roc_auc_score(big.ravel(), p_big.ravel()),
               "AUC_dir": roc_auc_score(up[big], q[big]),
               "Corr_q_p": week_corr(q, p_big), "Corr_up_down": week_corr(p_up, p_down),
               "IC_q": np.nanmean(weekly_ic(q, ret))}
        for name, r in [("dir", r_dir), ("int", r_int), ("exp", r_exp)]:
            lo, hi = block_bootstrap_sharpe(r)
            row.update({f"Sharpe_{name}": sharpe(r), f"Sharpe_{name}_lo": lo, f"Sharpe_{name}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["k", "Features", "Model", "Period"])


## ---- figures ----
def fig_importance(imp, keep):
    names = imp["features"]
    m, se = imp["mean"] * 1e3, imp["se"] * 1e3
    order = np.argsort(m[:, 1])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 5.6), sharey=True)
    y = np.arange(len(names))
    for ax, j, title in zip(axes, (1, 0), ("Tête de direction  P(hausse | grand mvt)",
                                           "Tête d'amplitude  P(grand mouvement)")):
        colors = ["steelblue" if names[i] in keep else "0.75" for i in order]
        ax.barh(y, m[order, j], xerr=se[order, j], color=colors, error_kw={"lw": 0.6})
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel(r"Hausse de la log-loss de validation ($\times 10^{-3}$)")
    axes[0].set_yticks(y, [FEATURE_LABELS[names[i]] for i in order], fontsize=7)
    fig.suptitle(f"Importance par permutation (bleu : {len(keep)} variables gardées, gris : retirées)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "cond_importance.pdf")
    plt.close(fig)


def fig_before_after(probs):
    """Réseau à 3 classes (section précédente) contre réseau factorisé, k = 1, test."""
    old = ensemble_probs(load("cls_final")["mlp_k1.0"]["test"])[0]
    p_big, q = probs[("mlp", 1.0, "all", "test")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    axes[0].scatter(old[..., 2].ravel(), old[..., 0].ravel(), s=3, alpha=0.4, color="0.4")
    axes[0].set_title(f"3 classes : corr. {week_corr(old[..., 2], old[..., 0]):.2f}")
    axes[1].scatter((p_big * q).ravel(), (p_big * (1 - q)).ravel(), s=3, alpha=0.4, color="navy")
    axes[1].set_title(f"Factorisé : corr. {week_corr(p_big * q, p_big * (1 - q)):.2f}")
    for ax in axes[:2]:
        ax.set_xlabel(r"$\hat p$(forte hausse)")
        ax.set_ylabel(r"$\hat p$(forte baisse)")
    axes[2].scatter(p_big.ravel(), q.ravel(), s=3, alpha=0.4, color="darkred")
    axes[2].axhline(0.5, color="0.5", lw=0.8, ls=":")
    axes[2].set_xlabel(r"$\hat p$ = P(grand mouvement)")
    axes[2].set_ylabel(r"$\hat q$ = P(hausse | grand mvt)")
    axes[2].set_title(f"Têtes séparées : corr. {week_corr(q, p_big):.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "cond_before_after.pdf")
    plt.close(fig)


def fig_cumulative(d, probs, model, k, feat):
    dates = pd.to_datetime(d["dates"][TRAIN:])
    ret = d["ret"][TRAIN:]
    p_big, q = probs[(model, k, feat, "test")]
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    for label, r, c in [(r"Direction : classer selon $\hat q$", long_short(q, ret), "navy"),
                        (r"Espérance : $\hat p(2\hat q - 1)$", long_short(p_big * (2 * q - 1), ret), "tab:green"),
                        (r"Intensité : $\max \hat p\hat q$ / $\max \hat p(1-\hat q)$",
                         intensity_rule(p_big * q, p_big * (1 - q), ret), "darkorange")]:
        ax.plot(dates, np.cumsum(r) * 100, color=c, lw=1.3, label=label)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("Rendement cumulé (%, log)")
    ax.set_title(f"Règles de décision sur le test ({NAMES[model]}, k = {k})")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "cond_cumulative.pdf")
    plt.close(fig)


## ---- table ----
def latex_table(res, k):
    lines = [r"\begin{tabular}{ll|rr|rr|r|rrr}", r"\toprule",
             r" & & \multicolumn{2}{c|}{Direction cond.} & \multicolumn{2}{c|}{Amplitude} & Corr. & "
             r"\multicolumn{3}{c}{Sharpe [IC 90\,\%] de la règle} \\",
             r"Variables & Modèle & Gain LL & AUC & Gain LL & AUC & $(\hat q, \hat p)$ & Direction $\hat q$ & "
             r"Espérance & Intensité \\", r"\midrule"]
    t = res[(res.Period == "test") & (res.k == k)]
    for feat, label in [("all", "33 (toutes)"), ("sel", f"{SEL_N} (gardées)")]:
        for m in ("logit", "mlp", "gru", "ens"):
            r = t[(t.Features == feat) & (t.Model == m)].iloc[0]

            def s(name):
                return (f"{fmt(r[f'Sharpe_{name}'], 2)} [{fmt(r[f'Sharpe_{name}_lo'], 2)}; "
                        f"{fmt(r[f'Sharpe_{name}_hi'], 2)}]")

            lines.append(f"{label if m == 'logit' else ''} & {NAMES[m]} & {fmt(100 * r.Skill_dir, 1)} & "
                         f"{r.AUC_dir:.3f} & {fmt(100 * r.Skill_amp, 1)} & {r.AUC_amp:.3f} & "
                         f"{fmt(r.Corr_q_p, 2)} & {s('dir')} & {fmt(r.Sharpe_exp, 2)} & {fmt(r.Sharpe_int, 2)} \\\\")
        if feat == "all":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


SEL_N = None


def main():
    global SEL_N
    d = load_panel()
    imp = load("cond_importance")
    keep = selected_features(imp)
    SEL_N = len(keep)
    probs, final, logit = collect(d)
    res = metrics(d, probs)
    res.to_csv(CSV / "cond_results.csv", index=False)
    for k in KS:
        (TAB / f"cond_results_k{str(k).replace('.', '')}.tex").write_text(latex_table(res, k), encoding="utf-8")

    v = res[(res.Period == "val") & res.Model.isin(["mlp", "gru", "ens"])]
    best = v.loc[v.Skill_dir.idxmax()]
    print(f"Retenu sur la validation : {best.Model}, k = {best.k}, variables = {best.Features}")
    fig_importance(imp, keep)
    fig_before_after(probs)
    fig_cumulative(d, probs, best.Model, best.k, best.Features)

    pd.set_option("display.width", 250)
    cols = ["Model", "k", "Features", "Period", "Skill_dir", "AUC_dir", "Skill_amp", "AUC_amp", "Corr_q_p",
            "Corr_up_down", "IC_q", "Sharpe_dir", "Sharpe_exp", "Sharpe_int"]
    print(res[cols].round(3).to_string(index=False))
    print("Gardées :", keep)
    print("Retirées :", [f for f in imp["features"] if f not in keep])
    print({n: v["cfg"] for n, v in final.items() if n.endswith("_all")})
    print({n: v["C"] for n, v in logit.items()})


if __name__ == "__main__":
    main()
