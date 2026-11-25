"""Figures et tableaux du rapport sur les réseaux de neurones (après nn_experiments.py).

Toutes les méthodes sont comparées sur les mêmes semaines (validation : 354,
test : 260) avec les mêmes mesures : corrélation de rang transversale (IC),
précision du choix de la meilleure et de la pire paire, R2 hors échantillon de
la cible z (modèles MSE), et valeur de la règle « acheter la paire au plus
haut score, vendre celle au plus bas » (toujours investie)."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Code.nn_experiments import INITIAL, OUT, TRAIN, load_panel, weekly_ic
from Function.functions import MODELS, OUTPUTS, PAIRS, PROCESSED, ROOT, load_frame
from Function.functions_strategy import meta_expanding_filtered_returns, strategy_legs

FIG = ROOT / "latex" / "figures"
TAB = ROOT / "latex" / "tables"
CSV = OUTPUTS / "nn"
LABELS = {"ridge": "Ridge (panel)", "mlp_mse": "MLP -- MSE", "mlp_rank": "MLP -- rang",
          "gru_mse": "GRU -- MSE", "gru_rank": "GRU -- rang", "nn_ens": "Ensemble RN",
          "old_mixt": "MIXT (projet initial)", "old_stepwise": "Pas-à-pas (projet initial)"}
COLORS = {"ridge": "0.45", "mlp_mse": "tab:blue", "mlp_rank": "navy", "gru_mse": "tab:orange",
          "gru_rank": "darkred", "nn_ens": "black", "old_mixt": "tab:green",
          "old_stepwise": "olivedrab"}
FEATURE_LABELS = {
    "spread2": "Écart 2 ans", "spread10": "Écart 10 ans", "spread2_diff": "Δ écart 2 ans",
    "spread10_diff": "Δ écart 10 ans", "dgs2_diff": "Δ taux US 2 ans", "dgs10_diff": "Δ taux US 10 ans",
    "gold_return": "Rend. or", "sp500_return": "Rend. S&P 500", "oil_return": "Rend. pétrole",
    "lag_return": "Rend. retardé", "vol_4": "Vol. 4 sem.", "vol_12": "Vol. 12 sem.", "rsi": "RSI",
    "macd": "MACD", "bb_pctB": "Bollinger %B", "mom_4": "Momentum 4 sem.",
    "mom_12": "Momentum 12 sem.", "mom_26": "Momentum 26 sem.", "rv_1": "VR 1 sem.",
    "rv_4": "VR 4 sem.", "vol_ratio": "Ratio de vol.", "spread2_chg4": "Δ4 écart 2 ans",
    "spread10_chg4": "Δ4 écart 10 ans", "slope_diff": "Pente relative", "us2_chg4": "Δ4 taux US 2 ans",
    "sp500_mom4": "Momentum S&P 500", "sp500_rv4": "Vol. S&P 500", "gold_mom4": "Momentum or",
    "oil_mom4": "Momentum pétrole", "xs_mom_12": "Momentum 12 sem. relatif",
    "xs_spread2": "Écart 2 ans relatif", "xs_rv_4": "VR 4 sem. relative",
    "xs_lag_return": "Rend. retardé relatif",
}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    with open(OUT / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def fmt(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}".replace("-", "$-$")


def ensemble(seed_preds, seeds=None):
    seeds = list(seed_preds) if seeds is None else seeds
    return np.mean([seed_preds[s] for s in seeds], axis=0)


def rank_standardize(s):
    """Scores centrés-réduits chaque semaine (pour moyenner des modèles d'échelles différentes)."""
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-12)


def long_short(scores, ret):
    rows = np.arange(len(ret))
    return ret[rows, scores.argmax(1)] - ret[rows, scores.argmin(1)]


def sharpe(r):
    return r.mean() / r.std(ddof=1) * np.sqrt(52)


def block_bootstrap_sharpe(r, B=2000, block=8, seed=123):
    rng = np.random.default_rng(seed)
    n = len(r)
    out = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        out[b] = sharpe(r[idx])
    return np.quantile(out, [0.05, 0.95])


def metrics(scores, ret, z, z_hist_mean, mse_model):
    ic = weekly_ic(scores, ret)
    pnl = long_short(scores, ret)
    lo, hi = block_bootstrap_sharpe(pnl)
    r2 = (1 - np.sum((z - scores) ** 2) / np.sum((z - z_hist_mean) ** 2)) if mse_model else np.nan
    return {"IC": np.nanmean(ic), "IC_t": np.nanmean(ic) / np.nanstd(ic, ddof=1) * np.sqrt(np.sum(~np.isnan(ic))),
            "Top": np.mean(scores.argmax(1) == ret.argmax(1)),
            "Bottom": np.mean(scores.argmin(1) == ret.argmin(1)),
            "R2_OS": r2, "Ann_return": pnl.mean() * 52, "Sharpe": sharpe(pnl),
            "Sharpe_lo": lo, "Sharpe_hi": hi}


def collect_scores(d):
    final, ridge = load("final"), load("ridge")
    with open(MODELS / "models.pkl", "rb") as f:
        old = pickle.load(f)
    scores = {"val": {}, "test": {}}
    for period in ("val", "test"):
        scores[period]["ridge"] = ridge[period]
        for name, res in final.items():
            scores[period][name] = ensemble(res[period])
        scores[period]["nn_ens"] = np.mean(
            [rank_standardize(scores[period][n]) for n in ("mlp_mse", "mlp_rank", "gru_mse", "gru_rank")], axis=0)
        src = old["validation_sharpe_models"] if period == "val" else old["sharpe_models"]
        for name, m in [("old_mixt", "MIXT"), ("old_stepwise", "Stepwise")]:
            scores[period][name] = np.column_stack([src[m][p]["predictions"] for p in PAIRS])
    return scores, final, old


def results_table(d, scores):
    rows = []
    T = len(d["X"])
    for period, (a, b) in {"val": (INITIAL, TRAIN), "test": (TRAIN, T)}.items():
        ret, z = d["ret"][a:b], d["z"][a:b]
        hist = np.array([d["z"][:t].mean(0) for t in range(a, b)])
        for name, s in scores[period].items():
            mse = name in ("ridge", "mlp_mse", "gru_mse")
            rows.append({"Period": period, "Model": name, **metrics(s, ret, z, hist, mse)})
    return pd.DataFrame(rows)


## ---- figures ----
def fig_tuning(tune, best):
    df = pd.DataFrame(tune["results"])
    df["family"] = df["arch"] + "_" + df["loss"]
    fams = ["mlp_mse", "mlp_rank", "gru_mse", "gru_rank"]
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    rng = np.random.default_rng(0)
    for k, f in enumerate(fams):
        v = df[df.family == f]
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v.ic, s=12, alpha=0.6, color=COLORS[f])
        ax.scatter(k, v.ic.max(), marker="*", s=90, color="gold", edgecolor="k", zorder=3)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(4), [LABELS[f] for f in fams])
    ax.set_ylabel("IC moyen de validation")
    ax.set_title("Recherche d'hyperparamètres (112 configurations, 5 graines)")
    fig.savefig(FIG / "nn_tuning.pdf")
    plt.close(fig)


def fig_ensemble_size(d, final):
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    rng = np.random.default_rng(0)
    for name in ("mlp_mse", "mlp_rank", "gru_mse", "gru_rank"):
        test = final[name]["test"]
        ret = d["ret"][TRAIN:]
        means = []
        for k in range(1, 11):
            ics = [np.nanmean(weekly_ic(ensemble(test, list(rng.choice(10, k, replace=False))), ret))
                   for _ in range(30)]
            means.append(np.mean(ics))
        ax.plot(range(1, 11), means, "o-", ms=3, color=COLORS[name], label=LABELS[name])
    ax.set_xlabel("Nombre de graines dans l'ensemble")
    ax.set_ylabel("IC moyen sur le test")
    ax.set_title("Effet de la taille de l'ensemble")
    ax.legend(frameon=False)
    fig.savefig(FIG / "nn_ensemble_size.pdf")
    plt.close(fig)


def fig_learning_curves(imp):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, name in zip(axes, ("mlp_mse", "mlp_rank")):
        for k, h in enumerate(imp[name]["histories"][:5]):
            ax.plot(h[:, 0] / h[0, 0], color="tab:blue", alpha=0.6, lw=1,
                    label="Entraînement (80 %)" if k == 0 else None)
            ax.plot(h[:, 1] / h[0, 1], color="tab:red", alpha=0.6, lw=1,
                    label="Arrêt précoce (20 % récents)" if k == 0 else None)
            ax.axvline(imp[name]["best_epochs"][k], color="0.6", ls=":", lw=0.8)
        ax.set_title(f"{LABELS[name]} (5 graines)")
        ax.set_xlabel("Époque")
        ax.set_ylabel("Perte relative à l'époque 1")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "nn_learning_curves.pdf")
    plt.close(fig)


def fig_cumulative(d, scores):
    dates = pd.to_datetime(d["dates"][TRAIN:])
    ret = d["ret"][TRAIN:]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for name in ("ridge", "mlp_mse", "mlp_rank", "gru_mse", "gru_rank", "nn_ens", "old_mixt"):
        pnl = long_short(scores["test"][name], ret)
        ax.plot(dates, np.cumsum(pnl) * 100, color=COLORS[name], lw=2 if name == "nn_ens" else 1.1,
                label=LABELS[name])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("Rendement cumulé (%, log)")
    ax.set_title("Règle « meilleure contre pire paire » sur le test")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(FIG / "nn_cumulative.pdf")
    plt.close(fig)


def fig_rolling_ic(d, scores):
    dates = pd.to_datetime(d["dates"][INITIAL:])
    ret = d["ret"][INITIAL:]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    for name in ("ridge", "mlp_rank", "nn_ens", "old_mixt"):
        s = np.vstack([scores["val"][name], scores["test"][name]])
        ic = pd.Series(weekly_ic(s, ret), index=dates).rolling(52, min_periods=26).mean()
        ax.plot(ic.index, ic, color=COLORS[name], lw=1.6 if name == "nn_ens" else 1.1, label=LABELS[name])
    ax.axvline(pd.Timestamp(d["dates"][TRAIN]), color="0.3", ls="--", lw=0.8)
    ax.text(pd.Timestamp(d["dates"][TRAIN]), ax.get_ylim()[1] * 0.9, " test", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("IC moyen mobile (52 sem.)")
    ax.legend(frameon=False, ncol=4, loc="lower left")
    fig.savefig(FIG / "nn_rolling_ic.pdf")
    plt.close(fig)


def fig_importance(imp, names):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=False)
    for ax, name in zip(axes, ("mlp_mse", "mlp_rank")):
        s = pd.Series(imp[name]["drop"], index=[FEATURE_LABELS[n] for n in names]).sort_values()
        s = pd.concat([s.head(5), s.tail(10)]) if len(s) > 15 else s
        ax.barh(s.index, s.values, color=["firebrick" if v < 0 else "steelblue" for v in s.values])
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_title(f"{LABELS[name]} (IC de validation {imp[name]['ic']:.3f})")
        ax.set_xlabel("Baisse de l'IC si la variable est permutée")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "nn_importance.pdf")
    plt.close(fig)


## ---- tables ----
def latex_results(res):
    order = ["old_stepwise", "old_mixt", "ridge", "mlp_mse", "mlp_rank", "gru_mse", "gru_rank", "nn_ens"]
    lines = [r"\begin{tabular}{l|rr|rrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c|}{Validation} & \multicolumn{6}{c}{Test} \\",
             r"Modèle & IC & Sharpe & IC & $t$(IC) & Meill.\ (\%) & Pire (\%) & $R^2_{OS}(z)$ (\%) & Sharpe [IC 90\,\%] \\",
             r"\midrule"]
    for m in order:
        v = res[(res.Period == "val") & (res.Model == m)].iloc[0]
        t = res[(res.Period == "test") & (res.Model == m)].iloc[0]
        if m == "ridge":
            lines.append(r"\midrule")
        r2 = fmt(100 * t.R2_OS, 2) if not pd.isna(t.R2_OS) else "--"
        lines.append(f"{LABELS[m]} & {fmt(v.IC)} & {fmt(v.Sharpe, 2)} & {fmt(t.IC)} & {fmt(t.IC_t, 2)} & "
                     f"{100 * t.Top:.1f} & {100 * t.Bottom:.1f} & {r2} & "
                     f"{fmt(t.Sharpe, 2)} [{fmt(t.Sharpe_lo, 2)}; {fmt(t.Sharpe_hi, 2)}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_best(best):
    lines = [r"\begin{tabular}{llllll}", r"\toprule",
             r"Famille & Couches cachées & Dropout & Pénalité $L_2$ & Taux d'appr. & Séquence \\", r"\midrule"]
    for name, c in best.items():
        hidden = "--".join(map(str, c["hidden"])) if isinstance(c["hidden"], tuple) else f"GRU({c['hidden']})"
        seq = f"{c['seq_len']} sem." if c["arch"] == "gru" else "--"
        lines.append(f"{LABELS[name]} & {hidden} & {c['dropout']} & {c['weight_decay']:g} & {c['lr']:g} & {seq} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ablation_table(d, final, scores):
    ab = load("ablation_base15")
    pp = load("ablation_per_pair")
    from Code.nn_experiments import cached, stage_ridge
    ridge15 = cached("ablation_ridge_base15",
                     lambda: stage_ridge({**d, "X": d["X"][:, :, :15]}))
    rows = []
    T = len(d["X"])
    for period, (a, b) in {"val": (INITIAL, TRAIN), "test": (TRAIN, T)}.items():
        ret = d["ret"][a:b]

        def add(label, s):
            ic = np.nanmean(weekly_ic(s, ret))
            rows.append({"Period": period, "Variant": label, "IC": ic, "Sharpe": sharpe(long_short(s, ret))})

        for name in ("mlp_mse", "mlp_rank", "gru_mse", "gru_rank"):
            add(f"{name}|33 features, panel", scores[period][name])
            add(f"{name}|15 features d'origine", ensemble(ab[name + "_base15"][period]))
        add("mlp_mse|un réseau par paire", pp[period])
        add("ridge|33 features, panel", scores[period]["ridge"])
        add("ridge|15 features d'origine", ridge15[period])
    return pd.DataFrame(rows)


def latex_ablation(ab):
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Validation} & \multicolumn{2}{c}{Test} \\",
             r"Modèle & Variante & IC & Sharpe & IC & Sharpe \\", r"\midrule"]
    for var in ab.Variant.unique():
        name, label = var.split("|")
        v = ab[(ab.Period == "val") & (ab.Variant == var)].iloc[0]
        t = ab[(ab.Period == "test") & (ab.Variant == var)].iloc[0]
        lines.append(f"{LABELS[name]} & {label} & {fmt(v.IC)} & {fmt(v.Sharpe, 2)} & {fmt(t.IC)} & {fmt(t.Sharpe, 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def meta_on_nn(d, scores, name):
    """Méta-classifieur du projet initial appliqué aux scores d'un réseau."""
    dates_v = pd.to_datetime(d["dates"][INITIAL:TRAIN])
    dates_t = pd.to_datetime(d["dates"][TRAIN:])

    def frame(x, dates):
        x = x - x.mean(1, keepdims=True)  # scores centrés : la règle prend toujours position
        return pd.DataFrame(x, index=dates, columns=PAIRS)

    pv, pt = frame(scores["val"][name], dates_v), frame(scores["test"][name], dates_t)
    av = pd.DataFrame(d["ret"][INITIAL:TRAIN], index=dates_v, columns=PAIRS)
    at = pd.DataFrame(d["ret"][TRAIN:], index=dates_t, columns=PAIRS)
    legs = strategy_legs(pv, av)
    funding = load_frame(PROCESSED / "funding.csv")["Funding_weekly"]
    res = meta_expanding_filtered_returns(pt, at, legs, 0.5, 0.5, funding)
    r = res["returns"].values
    lo, hi = block_bootstrap_sharpe(r)
    dec = res["decisions"]
    return {"Sharpe": sharpe(r), "lo": lo, "hi": hi, "accept": dec.Accept.mean(),
            "Ann_return": r.mean() * 52}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    d = load_panel()
    tune = load("tune")
    from Code.nn_experiments import best_configs
    best = best_configs(tune)
    scores, final, old = collect_scores(d)

    res = results_table(d, scores)
    res.to_csv(CSV / "results.csv", index=False)
    (TAB / "nn_results.tex").write_text(latex_results(res), encoding="utf-8")
    (TAB / "nn_best_configs.tex").write_text(latex_best(best), encoding="utf-8")
    pd.DataFrame(tune["results"]).to_csv(CSV / "tuning.csv", index=False)

    ab = ablation_table(d, final, scores)
    ab.to_csv(CSV / "ablation.csv", index=False)
    (TAB / "nn_ablation.tex").write_text(latex_ablation(ab), encoding="utf-8")

    fig_tuning(tune, best)
    fig_ensemble_size(d, final)
    fig_cumulative(d, scores)
    fig_rolling_ic(d, scores)
    imp = load("importance")
    fig_learning_curves(imp)
    fig_importance(imp, list(d["features"]))

    meta = {name: meta_on_nn(d, scores, name) for name in ("mlp_rank", "nn_ens")}
    pd.DataFrame(meta).T.to_csv(CSV / "meta_on_nn.csv")

    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(ab.round(3).to_string(index=False))
    print(pd.DataFrame(meta).T.round(3))
    print({k: {"ic": round(float(v["ic"]), 4), "epochs": v["best_epochs"]} for k, v in imp.items()})
    print("Ridge alphas (test):", np.round(load("ridge")["test_alphas"], 0))


if __name__ == "__main__":
    main()
