"""Figures et tableaux de la classification des grands mouvements (après nn_classification.py)."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from Code.nn_analysis import block_bootstrap_sharpe, fmt, long_short, sharpe
from Code.nn_classification import KS, base_rates, ensemble_probs, labels, log_losses
from Code.nn_experiments import INITIAL, OUT, TRAIN, load_panel, weekly_ic
from Function.functions import MODELS, OUTPUTS, PAIRS, PROCESSED, ROOT, load_frame

FIG = ROOT / "latex" / "figures"
TAB = ROOT / "latex" / "tables"
CSV = OUTPUTS / "nn"
COVERAGES = [1.0, 0.75, 0.5, 0.25]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    with open(OUT / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def collect(d):
    """Probabilités (direction (T,P,3), amplitude (T,P)) de chaque modèle et période."""
    final, logit = load("cls_final"), load("cls_logit")
    with open(MODELS / "models.pkl", "rb") as f:
        old = pickle.load(f)
    probs = {}
    for k in KS:
        for period in ("val", "test"):
            mlp = ensemble_probs(final[f"mlp_k{k}"][period])
            gru = ensemble_probs(final[f"gru_k{k}"][period])
            probs[("mlp", k, period)] = mlp
            probs[("gru", k, period)] = gru
            probs[("ens", k, period)] = ((mlp[0] + gru[0]) / 2, (mlp[1] + gru[1]) / 2)
            probs[("logit", k, period)] = logit[f"logit_k{k}"][period]
            a, b = (INITIAL, TRAIN) if period == "val" else (TRAIN, len(d["X"]))
            probs[("base", k, period)] = base_rates(d["z"], a, b, k)
            # GARCH : P(|r| > k vol_12) si r ~ N(0, sigma_GARCH^2)
            src = old["validation_sharpe_models"] if period == "val" else old["sharpe_models"]
            sig = np.column_stack([src["Linear"][p]["vol_predictions"] for p in PAIRS])
            vol = d["vol"][a:b]
            probs[("garch", k, period)] = (None, 2 * (1 - norm.cdf(k * vol / sig)))
    return probs, final, logit


def decision(p_dir, ret, cover=None, thresholds=None, funding=None):
    """Acheter argmax P(forte hausse), vendre argmax P(forte baisse) parmi les autres paires.
    Avec seuils : une jambe n'est prise que si sa probabilité dépasse le seuil ; une
    seule jambe paie (achat) ou reçoit (vente) le financement."""
    T = len(ret)
    rows = np.arange(T)
    pu, pdn = p_dir[..., 2], p_dir[..., 0]
    long = pu.argmax(1)
    masked = pdn.copy()
    masked[rows, long] = -np.inf
    short = masked.argmax(1)
    leg_long, leg_short = ret[rows, long], -ret[rows, short]
    take_l = np.ones(T, bool) if thresholds is None else pu[rows, long] >= thresholds[0]
    take_s = np.ones(T, bool) if thresholds is None else pdn[rows, short] >= thresholds[1]
    f = np.zeros(T) if funding is None else funding
    r = np.where(take_l & take_s, leg_long + leg_short,
                 np.where(take_l, leg_long - f, np.where(take_s, leg_short + f, 0.0)))
    return {"returns": r, "long_win": np.mean(leg_long[take_l] > 0) if take_l.any() else np.nan,
            "short_win": np.mean(leg_short[take_s] > 0) if take_s.any() else np.nan,
            "long_cov": take_l.mean(), "short_cov": take_s.mean(),
            "p_long": pu[rows, long], "p_short": pdn[rows, short],
            "leg_long": leg_long, "leg_short": leg_short}


def metrics_table(d, probs):
    rows = []
    for k in KS:
        for period in ("val", "test"):
            a, b = (INITIAL, TRAIN) if period == "val" else (TRAIN, len(d["X"]))
            z, ret = d["z"][a:b], d["ret"][a:b]
            y_dir, y_mag = labels(z, k)
            bd, bm = probs[("base", k, period)]
            ll_base = log_losses(bd, bm, z, k)
            for model in ("logit", "mlp", "gru", "ens", "garch"):
                p_dir, p_mag = probs[(model, k, period)]
                row = {"k": k, "Period": period, "Model": model,
                       "AUC_mag": roc_auc_score(y_mag.ravel(), p_mag.ravel())}
                eps = 1e-12
                ll_mag = -np.mean(y_mag * np.log(p_mag + eps) + (1 - y_mag) * np.log(1 - p_mag + eps))
                row["Skill_mag"] = 1 - ll_mag / ll_base[1]
                if p_dir is not None:
                    ll = log_losses(p_dir, p_mag, z, k)
                    row["Skill_dir"] = 1 - ll[0] / ll_base[0]
                    row["AUC_up"] = roc_auc_score((y_dir == 2).ravel(), p_dir[..., 2].ravel())
                    row["AUC_down"] = roc_auc_score((y_dir == 0).ravel(), p_dir[..., 0].ravel())
                    dec = decision(p_dir, ret)
                    lo, hi = block_bootstrap_sharpe(dec["returns"])
                    row.update(Sharpe=sharpe(dec["returns"]), Sharpe_lo=lo, Sharpe_hi=hi,
                               Long_win=dec["long_win"], Short_win=dec["short_win"],
                               Ann_return=dec["returns"].mean() * 52)
                    # Diagnostic : P(hausse) et P(baisse) bougent-elles ensemble ?
                    pu, pdn = p_dir[..., 2], p_dir[..., 0]
                    row["Corr_up_down"] = np.nanmean([np.corrcoef(pu[t], pdn[t])[0, 1]
                                                      for t in range(len(pu))])
                    # Règle directionnelle : classer selon P(hausse) - P(baisse)
                    diff = pu - pdn
                    r_diff = long_short(diff, ret)
                    lo, hi = block_bootstrap_sharpe(r_diff)
                    row.update(IC_diff=np.nanmean(weekly_ic(diff, ret)), Sharpe_diff=sharpe(r_diff),
                               Sharpe_diff_lo=lo, Sharpe_diff_hi=hi)
                rows.append(row)
    return pd.DataFrame(rows)


def abstention(d, probs, model, k):
    """Seuils fixés sur la validation (quantiles des probabilités des jambes choisies)."""
    funding = load_frame(PROCESSED / "funding.csv")["Funding_weekly"]
    dates = pd.to_datetime(d["dates"])
    val = decision(probs[(model, k, "val")][0], d["ret"][INITIAL:TRAIN])
    rows = []
    for c in COVERAGES:
        th = (np.quantile(val["p_long"], 1 - c), np.quantile(val["p_short"], 1 - c))
        for period, (a, b) in {"val": (INITIAL, TRAIN), "test": (TRAIN, len(d["X"]))}.items():
            f = funding.reindex(dates[a:b]).values
            dec = decision(probs[(model, k, period)][0], d["ret"][a:b], thresholds=th, funding=f)
            r = dec["returns"]
            lo, hi = block_bootstrap_sharpe(r)
            rows.append({"Coverage": c, "Period": period, "tau_long": th[0], "tau_short": th[1],
                         "Long_cov": dec["long_cov"], "Short_cov": dec["short_cov"],
                         "Long_win": dec["long_win"], "Short_win": dec["short_win"],
                         "Sharpe": sharpe(r), "Sharpe_lo": lo, "Sharpe_hi": hi,
                         "Weeks_traded": np.mean(r != 0)})
    return pd.DataFrame(rows)


## ---- figures ----
NAMES = {"logit": "Logistique", "mlp": "MLP", "gru": "GRU", "ens": "Ensemble MLP+GRU",
         "garch": "GARCH (prob. implicite)"}
COLORS = {"logit": "0.55", "mlp": "navy", "gru": "darkred", "ens": "black", "garch": "tab:green"}


def fig_auc(res):
    t = res[res.Period == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, k in zip(axes, KS):
        tk = t[t.k == k]
        models = ["logit", "mlp", "gru", "ens", "garch"]
        x = np.arange(3)
        w = 0.16
        for j, m in enumerate(models):
            r = tk[tk.Model == m].iloc[0]
            vals = [r.get("AUC_up", np.nan), r.get("AUC_down", np.nan), r.AUC_mag]
            ax.bar(x + (j - 2) * w, vals, w, color=COLORS[m], label=NAMES[m])
        ax.axhline(0.5, color="0.3", ls="--", lw=0.8)
        ax.set_xticks(x, ["Forte hausse", "Forte baisse", "Grand mouvement\n(amplitude)"])
        ax.set_ylim(0.45, 0.66)
        ax.set_title(f"Seuil k = {k}σ (test)")
    axes[0].set_ylabel("AUC")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, frameon=False, fontsize=7, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "cls_auc.pdf")
    plt.close(fig)


def fig_calibration(d, probs, model, k):
    a, b = TRAIN, len(d["X"])
    y_dir, y_mag = labels(d["z"][a:b], k)
    p_dir, p_mag = probs[(model, k, "test")]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for label, p, y, c in [("P(forte hausse)", p_dir[..., 2], y_dir == 2, "tab:blue"),
                           ("P(forte baisse)", p_dir[..., 0], y_dir == 0, "tab:red"),
                           ("P(grand mouvement)", p_mag, y_mag == 1, "tab:green")]:
        q = pd.qcut(p.ravel(), 10, duplicates="drop")
        g = pd.DataFrame({"p": p.ravel(), "y": y.ravel()}).groupby(q, observed=True).mean()
        ax.plot(g.p, g.y, "o-", ms=3, color=c, label=label)
    lim = [0.1, 0.85] if k == 0.5 else [0.0, 0.6]
    ax.plot(lim, lim, color="0.6", ls=":")
    ax.set_xlabel("Probabilité prédite (décile)")
    ax.set_ylabel("Fréquence observée")
    ax.set_title(f"Calibration sur le test ({NAMES[model]}, k = {k})")
    ax.legend(frameon=False)
    fig.savefig(FIG / "cls_calibration.pdf")
    plt.close(fig)


def fig_abstention(abst, model, k):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for period, ls in [("val", "--"), ("test", "-")]:
        a = abst[abst.Period == period]
        axes[0].plot(a.Long_cov * 100, a.Long_win * 100, "o" + ls, color="tab:blue", ms=3,
                     label=f"Achat ({'validation' if period == 'val' else 'test'})")
        axes[0].plot(a.Short_cov * 100, a.Short_win * 100, "s" + ls, color="tab:red", ms=3,
                     label=f"Vente ({'validation' if period == 'val' else 'test'})")
        axes[1].plot(a.Coverage * 100, a.Sharpe, "o" + ls, color="k", ms=3,
                     label="validation" if period == "val" else "test")
    axes[0].axhline(50, color="0.5", lw=0.8)
    axes[0].set_xlabel("Semaines où la jambe est prise (%)")
    axes[0].set_ylabel("Jambes gagnantes (%)")
    axes[0].invert_xaxis()
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_xlabel("Couverture visée (%)")
    axes[1].set_ylabel("Ratio de Sharpe annualisé")
    axes[1].invert_xaxis()
    axes[1].legend(frameon=False)
    fig.suptitle(f"Abstention : seuils fixés sur la validation ({NAMES[model]}, k = {k})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "cls_abstention.pdf")
    plt.close(fig)


## ---- tables ----
def latex_results(res):
    lines = [r"\begin{tabular}{ll|rr|rrr|r|rr}", r"\toprule",
             r" & & \multicolumn{2}{c|}{Gain log-loss (\%)} & \multicolumn{3}{c|}{AUC} & Corr. & "
             r"\multicolumn{2}{c}{Sharpe [IC 90\,\%] de la règle} \\",
             r"$k$ & Modèle & Dir. & Ampl. & Hausse & Baisse & Ampl. & $(\hat p_\uparrow, \hat p_\downarrow)$ & "
             r"Intensité & Direction \\",
             r"\midrule"]
    t = res[res.Period == "test"]
    for k in KS:
        for m in ("logit", "mlp", "gru", "ens", "garch"):
            r = t[(t.k == k) & (t.Model == m)].iloc[0]
            if m == "garch":
                lines.append(f" & {NAMES[m]} & -- & {fmt(100 * r.Skill_mag, 1)} & -- & -- & {r.AUC_mag:.3f} "
                             r"& -- & -- & -- \\")
            else:
                lines.append(f"{k if m == 'logit' else ''} & {NAMES[m]} & {fmt(100 * r.Skill_dir, 1)} & "
                             f"{fmt(100 * r.Skill_mag, 1)} & {r.AUC_up:.3f} & {r.AUC_down:.3f} & {r.AUC_mag:.3f} & "
                             f"{fmt(r.Corr_up_down, 2)} & "
                             f"{fmt(r.Sharpe, 2)} [{fmt(r.Sharpe_lo, 2)}; {fmt(r.Sharpe_hi, 2)}] & "
                             f"{fmt(r.Sharpe_diff, 2)} [{fmt(r.Sharpe_diff_lo, 2)}; {fmt(r.Sharpe_diff_hi, 2)}] \\\\")
        if k != KS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def fig_up_down(probs):
    """P(forte hausse) contre P(forte baisse) sur le test : le réseau apprend surtout l'amplitude."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, k in zip(axes, KS):
        p_dir = probs[("mlp", k, "test")][0]
        for j, pair in enumerate(PAIRS):
            ax.scatter(p_dir[:, j, 2], p_dir[:, j, 0], s=4, alpha=0.5, label=f"{pair[:3]}/USD")
        lo = min(p_dir[..., 0].min(), p_dir[..., 2].min())
        hi = max(p_dir[..., 0].max(), p_dir[..., 2].max())
        ax.plot([lo, hi], [lo, hi], color="0.5", ls=":", lw=0.8)
        ax.set_xlabel(r"$\hat p$(forte hausse)")
        ax.set_ylabel(r"$\hat p$(forte baisse)")
        ax.set_title(f"MLP, k = {k} (test)")
    axes[0].legend(frameon=False, fontsize=6, markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG / "cls_up_down.pdf")
    plt.close(fig)


def latex_abstention(abst):
    lines = [r"\begin{tabular}{r|rrr|rrrr}", r"\toprule",
             r" & \multicolumn{3}{c|}{Validation} & \multicolumn{4}{c}{Test} \\",
             r"Couverture & Gagn.\ achat & Gagn.\ vente & Sharpe & Sem.\ actives & Gagn.\ achat & Gagn.\ vente & Sharpe [IC 90\,\%] \\",
             r"\midrule"]
    for c in COVERAGES:
        v = abst[(abst.Coverage == c) & (abst.Period == "val")].iloc[0]
        t = abst[(abst.Coverage == c) & (abst.Period == "test")].iloc[0]
        lines.append(f"{100 * c:.0f}\\,\\% & {100 * v.Long_win:.1f} & {100 * v.Short_win:.1f} & {fmt(v.Sharpe, 2)} & "
                     f"{100 * t.Weeks_traded:.0f}\\,\\% & {100 * t.Long_win:.1f} & {100 * t.Short_win:.1f} & "
                     f"{fmt(t.Sharpe, 2)} [{fmt(t.Sharpe_lo, 2)}; {fmt(t.Sharpe_hi, 2)}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    d = load_panel()
    probs, final, logit = collect(d)
    res = metrics_table(d, probs)
    res.to_csv(CSV / "cls_results.csv", index=False)
    (TAB / "cls_results.tex").write_text(latex_results(res), encoding="utf-8")

    # Modèle principal choisi sur la validation : meilleur gain de log-loss de direction
    v = res[(res.Period == "val") & res.Model.isin(["mlp", "gru", "ens"])]
    best = v.loc[v.Skill_dir.idxmax()]
    model, k = best.Model, best.k
    print(f"Modèle retenu sur la validation : {model}, k = {k}")
    abst = abstention(d, probs, model, k)
    abst.to_csv(CSV / "cls_abstention.csv", index=False)
    (TAB / "cls_abstention.tex").write_text(latex_abstention(abst), encoding="utf-8")

    fig_auc(res)
    fig_up_down(probs)
    fig_calibration(d, probs, model, k)
    fig_abstention(abst, model, k)

    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(abst.round(3).to_string(index=False))
    print({name: v["cfg"] for name, v in final.items()})
    print({name: v["C"] for name, v in logit.items()})


if __name__ == "__main__":
    main()
