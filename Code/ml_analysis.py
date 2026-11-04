"""Analyses d'apprentissage automatique pour le rapport (après run_strategy.py).

Produit les figures (latex/figures/) et les tableaux (outputs/ml/*.csv et
latex/tables/*.tex) : qualité des prévisions hors échantillon (R2 hors
échantillon contre la moyenne historique, test de Clark-West, précision
directionnelle, corrélation de rang), généralisation validation -> test,
parcimonie des modèles (lasso, stepwise, ACP), importance des variables,
volatilité GARCH et diagnostics du meta-classifieur."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from Code.backtest import naive_last_vol_pred, run_backtest
from Code.build_model import load_inputs
from Function.functions import MODELS, OUTPUTS, PAIRS, PROCESSED, ROOT, load_frame
from Function.functions_model import (
    forecast_rows, model_data, select_stepwise_model,
)
from Function.functions_strategy import (
    META_X, glm_logit, glm_predict, model_matrix, performance_summary, strategy_returns,
)
from Function.glmnet import cv_glmnet, glmnet_gaussian

FIG = ROOT / "latex" / "figures"
TAB = ROOT / "latex" / "tables"
CSV = OUTPUTS / "ml"
METHODS = ["Naive", "Linear", "Stepwise", "Lasso", "Ridge", "Elastic_Net", "Random_Forest", "PCA"]
LABELS = {"Naive": "Naïf", "Linear": "MCO", "Stepwise": "Pas-à-pas", "Lasso": "Lasso",
          "Ridge": "Ridge", "Elastic_Net": "Elastic net", "Random_Forest": "Forêt aléatoire",
          "PCA": "ACP", "MIXT": "MIXT", "HM": "Moy. hist."}
PAIR_LABELS = {p: f"{p[:3]}/{p[3:]}" for p in PAIRS}
FEATURE_LABELS = {
    "spread2": "Écart 2 ans", "spread10": "Écart 10 ans", "spread2_diff": "Δ écart 2 ans",
    "spread10_diff": "Δ écart 10 ans", "dgs2_diff": "Δ taux US 2 ans",
    "dgs10_diff": "Δ taux US 10 ans", "gold_return": "Rend. or", "sp500_return": "Rend. S&P 500",
    "oil_return": "Rend. pétrole", "lag_return": "Rend. retardé", "vol_4": "Vol. 4 sem.",
    "vol_12": "Vol. 12 sem.", "rsi": "RSI", "macd": "MACD", "bb_pctB": "Bollinger %B",
}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def feature_name(col):
    return FEATURE_LABELS[col.split("_", 1)[1] if col[:4] in ("eur_", "gbp_", "cad_", "chf_", "jpy_") else col]


def save_table(df, name, latex=None):
    CSV.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV / f"{name}.csv", index=False)
    if latex is not None:
        TAB.mkdir(parents=True, exist_ok=True)
        (TAB / f"{name}.tex").write_text(latex, encoding="utf-8")


def fmt(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}".replace("-", "$-$")


## ---- data ----
def walk_forward_frames(features_by_pair, fx_targets, train_ratio):
    """Pour chaque paire : X, y, colonnes, lignes de validation et de test."""
    out = {}
    for pair in PAIRS:
        data = model_data(features_by_pair[pair], fx_targets[pair])
        out[pair] = {
            "X": np.ascontiguousarray(data.drop(columns="target").values),
            "y": data["target"].values, "dates": data.index,
            "columns": list(data.columns[1:]),
            "test": forecast_rows(len(data), "test", train_ratio),
            "validation": forecast_rows(len(data), "validation", train_ratio),
        }
    return out


## ---- forecast accuracy ----
def clark_west(y, f_bench, f_model):
    """Test de Clark-West (2007) : H0 le modèle n'améliore pas la référence emboîtée."""
    adj = (y - f_bench) ** 2 - ((y - f_model) ** 2 - (f_bench - f_model) ** 2)
    t = adj.mean() / (adj.std(ddof=1) / np.sqrt(len(adj)))
    return t, 1 - stats.norm.cdf(t)


def forecast_accuracy(models, frames):
    rows = []
    for period, key in [("Validation", "validation_models"), ("Test", "sharpe_models")]:
        for m in METHODS + (["MIXT"] if period == "Test" else []):
            for pair in PAIRS:
                fr = frames[pair]
                res = models[key][m][pair]
                pred = res["return_predictions"] if period == "Test" else res["predictions"]
                y = res["return_actual"] if period == "Test" else res["actual"]
                rws = fr["test" if period == "Test" else "validation"]
                hm = np.array([fr["y"][:i].mean() for i in rws])
                sse_m, sse_hm = np.sum((y - pred) ** 2), np.sum((y - hm) ** 2)
                t, p = clark_west(y, hm, pred)
                rows.append({"Period": period, "Method": m, "Pair": pair,
                             "R2_OS": 1 - sse_m / sse_hm,
                             "R2": 1 - sse_m / np.sum((y - y.mean()) ** 2),
                             "RMSE": np.sqrt(np.mean((y - pred) ** 2)),
                             "Hit": np.mean(np.sign(pred) == np.sign(y)),
                             "CW_t": t, "CW_p": p})
    return pd.DataFrame(rows)


def rank_ic(models):
    """Corrélation de Spearman hebdomadaire entre le Sharpe prévu des 5 paires
    et leur rendement réalisé (qualité du classement transversal)."""
    fx_rv = load_frame(PROCESSED / "realized_vol.csv")
    rows = []
    for m in METHODS + ["MIXT"]:
        sm = models["sharpe_models"][m]
        pred = naive_last_vol_pred(sm, fx_rv) if m == "Naive" else model_matrix(sm)
        actual = model_matrix(sm, "return_actual")
        ic = np.array([stats.spearmanr(pred.values[t], actual.values[t])[0]
                       for t in range(len(pred))])
        top = np.mean(np.argmax(pred.values, 1) == np.argmax(actual.values, 1))
        bottom = np.mean(np.argmin(pred.values, 1) == np.argmin(actual.values, 1))
        rows.append({"Method": m, "IC_mean": np.nanmean(ic),
                     "IC_t": np.nanmean(ic) / (np.nanstd(ic, ddof=1) / np.sqrt(np.sum(~np.isnan(ic)))),
                     "Top_hit": top, "Bottom_hit": bottom})
    return pd.DataFrame(rows)


## ---- sparsity over the test period ----
def _sparsity_task(fr, pair, method):
    X, y, cols = fr["X"], fr["y"], fr["columns"]
    out = []
    for i in fr["test"]:
        if method == "Stepwise":
            sel = np.zeros(X.shape[1], bool)
            sel[select_stepwise_model(X[:i], y[:i])] = True
            out.append((pair, method, fr["dates"][i], sel, np.nan))
        elif method == "PCA":
            Z = (X[:i] - X[:i].mean(0)) / X[:i].std(0, ddof=1)
            eig = np.linalg.svd(Z, compute_uv=False) ** 2 / (i - 1)
            k = max(1, int(np.sum(eig > 1)))
            out.append((pair, method, fr["dates"][i], k, eig[:k].sum() / eig.sum()))
        else:
            alpha = {"Lasso": 1.0, "Elastic_Net": 0.5}[method]
            _, beta, info = cv_glmnet(X[:i], y[:i], alpha)
            out.append((pair, method, fr["dates"][i], beta != 0, info["lambda"][info["index"]]))
    return out


def sparsity(frames):
    tasks = [(p, m) for p in PAIRS for m in ["Lasso", "Elastic_Net", "Stepwise", "PCA"]]
    res = Parallel(n_jobs=-1)(delayed(_sparsity_task)(frames[p], p, m) for p, m in tasks)
    return [r for chunk in res for r in chunk]


## ---- figures ----
def fig_validation_heatmap(acc):
    # Critère de sélection du MIXT : R2 de validation (par rapport à la moyenne de la période)
    v = acc[acc.Period == "Validation"].pivot(index="Method", columns="Pair", values="R2")
    v = v.loc[[m for m in METHODS if m != "Naive"], PAIRS] * 100
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    lim = np.abs(v.values).max()
    im = ax.imshow(v.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    best = v.values.argmax(axis=0)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v.values[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if best[j] == i else "normal")
    ax.set_xticks(range(5), [PAIR_LABELS[p] for p in PAIRS])
    ax.set_yticks(range(v.shape[0]), [LABELS[m] for m in v.index])
    fig.colorbar(im, ax=ax, label=r"$R^2$ de validation (%)")
    fig.savefig(FIG / "validation_r2.pdf")
    plt.close(fig)


def fig_val_vs_test(acc):
    v = acc[acc.Period == "Validation"].set_index(["Method", "Pair"])["R2_OS"]
    t = acc[(acc.Period == "Test") & (acc.Method != "MIXT")].set_index(["Method", "Pair"])["R2_OS"]
    d = pd.concat({"val": v, "test": t}, axis=1).drop(index="Naive", level=0) * 100
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    markers = dict(zip(PAIRS, ["o", "s", "^", "D", "v"]))
    colors = dict(zip([m for m in METHODS if m != "Naive"], plt.cm.tab10.colors))
    for (m, p), r in d.iterrows():
        ax.scatter(r.val, r.test, marker=markers[p], color=colors[m], s=28)
    for m in colors:
        ax.scatter([], [], color=colors[m], marker="o", label=LABELS[m])
    for p in PAIRS:
        ax.scatter([], [], color="0.4", marker=markers[p], label=PAIR_LABELS[p])
    lim = [min(d.min()) - 0.5, max(d.max()) + 0.5]
    ax.plot(lim, lim, color="0.6", ls=":", lw=1)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvline(0, color="0.7", lw=0.8)
    rho = stats.spearmanr(d.val, d.test)[0]
    ax.set_title(f"Corrélation de rang validation–test : {rho:.2f}")
    ax.set_xlabel(r"$R^2_{OS}$ validation (%)")
    ax.set_ylabel(r"$R^2_{OS}$ test (%)")
    ax.legend(ncol=2, fontsize=6.5, loc="lower right", frameon=False)
    fig.savefig(FIG / "val_vs_test.pdf")
    plt.close(fig)
    return rho


def fig_lasso_path(frames, train_size):
    """Chemin de régularisation du lasso et erreur de validation croisée (EUR/USD)."""
    fr = frames["EURUSD"]
    X, y = fr["X"][:train_size], fr["y"][:train_size]
    lam, _, beta = glmnet_gaussian(X, y, 1.0)
    _, _, info = cv_glmnet(X, y, 1.0)
    sd = X.std(axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    std_beta = beta * sd[:, None] * 1e3
    # Étiquettes : les 6 variables qui entrent en premier dans le modèle
    entry = [np.argmax(std_beta[j] != 0) if np.any(std_beta[j] != 0) else len(lam) for j in range(len(sd))]
    first = np.argsort(entry)[:6]
    for j, col in enumerate(fr["columns"]):
        line, = ax1.plot(np.log(lam), std_beta[j], lw=1.4 if j in first else 0.7,
                         alpha=1 if j in first else 0.5,
                         label=feature_name(col) if j in first else None)
    ax1.legend(fontsize=6, frameon=False, loc="lower left", title="Premières variables actives",
               title_fontsize=6)
    lmin = np.log(info["lambda"][info["index"]])
    for ax in (ax1, ax2):
        ax.axvline(lmin, color="0.3", ls="--", lw=0.8)
    ax1.set_xlabel(r"$\log \lambda$")
    ax1.set_ylabel(r"Coefficient standardisé ($\times 10^{3}$)")
    ax1.set_title("Chemin de régularisation")
    ax2.plot(np.log(info["lambda"]), info["cvm"] * 1e4, color="firebrick")
    ax2.set_xlabel(r"$\log \lambda$")
    ax2.set_ylabel(r"EQM de validation croisée ($\times 10^{-4}$)")
    ax2.set_title("Validation croisée à 5 plis")
    fig.tight_layout()
    fig.savefig(FIG / "lasso_path.pdf")
    plt.close(fig)


def fig_sparsity(records, frames):
    df = pd.DataFrame(records, columns=["Pair", "Method", "Date", "Sel", "Extra"])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), sharex=True)
    for ax, m, ylabel in zip(axes, ["Lasso", "Stepwise", "PCA"],
                             ["Variables non nulles", "Variables retenues (BIC)", "Composantes retenues"]):
        for p in PAIRS:
            d = df[(df.Method == m) & (df.Pair == p)]
            k = d.Sel.apply(lambda s: int(np.sum(s)) if not np.isscalar(s) else s)
            if m == "Lasso":  # très variable d'une semaine à l'autre : moyenne mobile 13 semaines
                ax.plot(d.Date, k.rolling(13, min_periods=1).mean(), lw=1, label=PAIR_LABELS[p])
            else:
                ax.step(d.Date, k, where="post", lw=1, label=PAIR_LABELS[p])
        ax.set_title(LABELS[m] + (" (moyenne mobile 13 sem.)" if m == "Lasso" else ""))
        ax.set_ylabel(ylabel)
        if m != "Lasso":
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "sparsity.pdf")
    plt.close(fig)

    # Fréquence de sélection de chaque variable (lasso et pas-à-pas, toutes paires)
    cols = [feature_name(c) for c in frames["EURUSD"]["columns"]]
    usage = {}
    for m in ["Lasso", "Elastic_Net", "Stepwise"]:
        for p in PAIRS:
            d = df[(df.Method == m) & (df.Pair == p)]
            usage[(m, p)] = 100 * np.mean(np.vstack(d.Sel.values), axis=0)
    u = pd.DataFrame(usage, index=cols)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(u.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(len(cols)), cols)
    ax.set_xticks(range(u.shape[1]), [PAIR_LABELS[p][:3] for _, p in u.columns], fontsize=7)
    for k, m in enumerate(["Lasso", "Elastic_Net", "Stepwise"]):
        ax.text(2 + 5 * k, -1.2, LABELS[m], ha="center", fontsize=8, fontweight="bold")
        if k:
            ax.axvline(5 * k - 0.5, color="white", lw=3)
    fig.colorbar(im, ax=ax, label="% des semaines de test où la variable est retenue")
    fig.savefig(FIG / "feature_usage.pdf")
    plt.close(fig)
    lam = df[df.Method == "Lasso"].groupby("Pair")["Extra"].median()
    pca_var = df[df.Method == "PCA"].groupby("Pair")["Extra"].mean()
    return u, lam, pca_var


def fig_rf_importance(frames, train_size):
    imp = []
    for p in PAIRS:
        fr = frames[p]
        rf = RandomForestRegressor(n_estimators=500, max_features=3, min_samples_split=6,
                                   random_state=123, n_jobs=-1)
        rf.fit(fr["X"][:train_size], fr["y"][:train_size])
        imp.append(pd.Series(rf.feature_importances_, index=[feature_name(c) for c in fr["columns"]]))
    imp = pd.concat(imp, axis=1).mean(axis=1).sort_values()
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.barh(imp.index, imp.values * 100, color="steelblue")
    ax.set_xlabel("Importance (réduction d'impureté, %)")
    ax.set_title("Forêt aléatoire (moyenne des 5 paires)")
    fig.savefig(FIG / "rf_importance.pdf")
    plt.close(fig)
    return imp


def fig_garch(models):
    g = models["garch_models"]
    fx_rv = load_frame(PROCESSED / "realized_vol.csv")
    rows = []
    for p in PAIRS:
        d = g[p]["dates"]
        actual = g[p]["actual"]
        # Références : volatilité réalisée de la semaine passée et écart-type sur 12 semaines
        last_rv = fx_rv[p].reindex(d).values
        feat = load_frame(ROOT / "Data" / "Features" / f"{p[:3].lower()}_features.csv")
        vol12 = feat["vol_12"].reindex(d).values
        for name, f in [("GARCH(1,1)", g[p]["predictions"]), ("VR semaine passée", last_rv),
                        ("Écart-type 12 sem.", vol12)]:
            qlike = np.mean(actual ** 2 / f ** 2 - np.log(actual ** 2 / f ** 2) - 1)
            rows.append({"Pair": p, "Model": name,
                         "RMSE": np.sqrt(np.mean((actual - f) ** 2)) * 100,
                         "QLIKE": qlike,
                         "Corr": np.corrcoef(actual, f)[0, 1]})
    vol = pd.DataFrame(rows)

    p = "EURUSD"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax1.plot(g[p]["dates"], g[p]["actual"] * 100, color="0.65", lw=0.8, label="Volatilité réalisée")
    ax1.plot(g[p]["dates"], g[p]["predictions"] * 100, color="navy", lw=1.3, label="Prévision GARCH(1,1)")
    ax1.set_ylabel("Volatilité hebdomadaire (%)")
    ax1.set_title("EUR/USD, période de test")
    ax1.legend(frameon=False)
    for q in PAIRS:
        pers = g[q]["params"]["alpha"] + g[q]["params"]["beta"]
        ax2.plot(g[q]["dates"], pers, lw=1, label=PAIR_LABELS[q])
    ax2.set_title(r"Persistance $\alpha+\beta$ ré-estimée")
    ax2.tick_params(axis="x", rotation=45)
    ax1.tick_params(axis="x", rotation=45)
    ax2.legend(fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(FIG / "garch.pdf")
    plt.close(fig)
    return vol


def fig_meta(bt, models):
    dec = bt["mixt_meta"]["decisions"]
    val_legs = bt["validation_legs"]
    out = []
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
    for side, color in [("Long", "navy"), ("Short", "darkorange")]:
        d = dec[dec.Side == side]
        auc = roc_auc_score(d.Win, d.Probability)
        fpr, tpr, _ = roc_curve(d.Win, d.Probability)
        axes[0].plot(fpr, tpr, color=color, label=f"{side} (AUC = {auc:.2f})")
        base = d.Win.mean()
        out.append({"Side": side, "N_test": len(d), "Base_rate": base, "AUC": auc,
                    "Brier": brier_score_loss(d.Win, d.Probability),
                    "Brier_base": brier_score_loss(d.Win, np.full(len(d), base)),
                    "Accept_rate": d.Accept.mean(),
                    "Hit_accepted": d.Win[d.Accept].mean(), "Hit_rejected": d.Win[~d.Accept].mean(),
                    "AUC_p": stats.mannwhitneyu(d.Probability[d.Win == 1], d.Probability[d.Win == 0],
                                                alternative="greater").pvalue})
        # Calibration par quintile de probabilité
        q = pd.qcut(d.Probability, 5, duplicates="drop")
        cal = d.groupby(q, observed=True).agg(p=("Probability", "mean"), w=("Win", "mean"))
        axes[1].plot(cal.p, cal.w, "o-", color=color, ms=3, label=side)
        axes[2].hist(d.Probability, bins=20, alpha=0.6, color=color, label=side)
    axes[0].plot([0, 1], [0, 1], color="0.6", ls=":")
    axes[0].set_xlabel("Taux de faux positifs")
    axes[0].set_ylabel("Taux de vrais positifs")
    axes[0].set_title("Courbe ROC (test)")
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    lo, hi = dec.Probability.min(), dec.Probability.max()
    axes[1].plot([lo, hi], [lo, hi], color="0.6", ls=":")
    axes[1].set_xlabel("Probabilité prédite (quintile)")
    axes[1].set_ylabel("Fréquence de gain observée")
    axes[1].set_title("Calibration (test)")
    axes[1].legend(fontsize=6.5, frameon=False)
    axes[2].axvline(0.5, color="0.3", ls="--", lw=0.8)
    axes[2].set_xlabel(r"$\hat p$ (seuil $\tau = 0{,}5$)")
    axes[2].set_ylabel("Nombre de jambes")
    axes[2].set_title("Probabilités prédites")
    axes[2].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "meta_classifier.pdf")
    plt.close(fig)

    # Coefficients du classifieur estimé sur les jambes de validation (variables standardisées)
    coefs = []
    for side in ["Long", "Short"]:
        d = val_legs[val_legs.Side == side]
        X = d[META_X].values
        Xs = (X - X.mean(0)) / X.std(0, ddof=1)
        b = glm_logit(Xs, d.Win.values)
        A = np.column_stack([np.ones(len(Xs)), Xs])
        mu = glm_predict(b, Xs)
        cov = np.linalg.inv(A.T @ (A * (mu * (1 - mu))[:, None]))
        se = np.sqrt(np.diag(cov))
        for name, bk, sk in zip(["Constante"] + META_X, b, se):
            coefs.append({"Side": side, "Variable": name, "Coef": bk, "SE": sk, "z": bk / sk,
                          "N": len(d)})
    return pd.DataFrame(out), pd.DataFrame(coefs)


def fig_strategy(models, bt):
    """Sharpe annualisé de la règle de classement selon le modèle de prévision."""
    fx_rv = load_frame(PROCESSED / "realized_vol.csv")
    rets = {}
    for m in METHODS + ["MIXT"]:
        sm = models["sharpe_models"][m]
        pred = naive_last_vol_pred(sm, fx_rv) if m == "Naive" else model_matrix(sm)
        rets[LABELS[m]] = strategy_returns(pred, model_matrix(sm, "return_actual"))
    rets["MIXT + méta"] = bt["mixt_meta_returns"]
    perf = performance_summary(pd.DataFrame(rets))
    # Intervalle bootstrap par blocs (blocs de 8 semaines) du ratio de Sharpe
    rng = np.random.default_rng(123)
    R = pd.DataFrame(rets).values
    n, block, B = len(R), 8, 2000
    boots = np.empty((B, R.shape[1]))
    for b in range(B):
        starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        r = R[idx]
        boots[b] = r.mean(0) / r.std(0, ddof=1) * np.sqrt(52)
    perf["Sharpe_lo"] = np.quantile(boots, 0.05, axis=0)
    perf["Sharpe_hi"] = np.quantile(boots, 0.95, axis=0)

    rand = bt["random_returns"].values
    rand_sharpe = rand.mean(0) / rand.std(0, ddof=1) * np.sqrt(52)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    y = np.arange(len(perf))
    ax.axvspan(*np.quantile(rand_sharpe, [0.05, 0.95]), color="0.9",
               label="1 000 stratégies aléatoires (5–95 %)")
    ax.errorbar(perf.Sharpe, y, xerr=[perf.Sharpe - perf.Sharpe_lo, perf.Sharpe_hi - perf.Sharpe],
                fmt="o", color="navy", ms=4, capsize=2)
    ax.set_yticks(y, perf.Strategy)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("Ratio de Sharpe annualisé (IC bootstrap 90 %)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.18))
    fig.savefig(FIG / "strategy_sharpe.pdf")
    plt.close(fig)
    return perf


## ---- LaTeX tables ----
def latex_accuracy(acc, ic):
    t = acc[acc.Period == "Test"]
    r2 = t.pivot(index="Method", columns="Pair", values="R2_OS").loc[METHODS + ["MIXT"], PAIRS] * 100
    cwp = t.pivot(index="Method", columns="Pair", values="CW_p").loc[METHODS + ["MIXT"], PAIRS]
    hit = t.groupby("Method")["Hit"].mean().loc[METHODS + ["MIXT"]] * 100
    ic = ic.set_index("Method").loc[METHODS + ["MIXT"]]
    lines = [r"\begin{tabular}{l" + "r" * 5 + "|rrr}", r"\toprule",
             r" & \multicolumn{5}{c|}{$R^2_{OS}$ (\%) contre la moyenne historique} & Signe & IC & \\",
             "Méthode & " + " & ".join(PAIR_LABELS[p] for p in PAIRS)
             + r" & (\%) & moyen & $t$(IC) \\", r"\midrule"]
    for m in METHODS + ["MIXT"]:
        cells = []
        for p in PAIRS:
            star = "^{**}" if cwp.loc[m, p] < 0.05 else ("^{*}" if cwp.loc[m, p] < 0.10 else "")
            cells.append(f"${fmt(r2.loc[m, p], 2).replace('$', '')}{star}$")
        if m == "MIXT":
            lines.append(r"\midrule")
        lines.append(f"{LABELS[m]} & " + " & ".join(cells)
                     + f" & {hit[m]:.1f} & {fmt(ic.loc[m, 'IC_mean'])} & {fmt(ic.loc[m, 'IC_t'], 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_meta(meta, coefs):
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Jambe & $N$ & Taux de gain & AUC & Brier & Brier (base) & Gain acceptées & Gain rejetées \\",
             r"\midrule"]
    for _, r in meta.iterrows():
        lines.append(f"{r.Side} & {r.N_test} & {100 * r.Base_rate:.1f}\\% & {r.AUC:.3f} "
                     f"& {r.Brier:.4f} & {r.Brier_base:.4f} & {100 * r.Hit_accepted:.1f}\\% "
                     f"& {100 * r.Hit_rejected:.1f}\\% \\\\")
    lines += [r"\midrule", r"\multicolumn{8}{l}{\emph{Coefficients estimés sur la validation "
              r"(variables standardisées), statistique $z$ entre parenthèses}} \\", r"\midrule",
              r"Jambe & \multicolumn{2}{r}{Constante} & \multicolumn{2}{r}{$|\hat s|$} & "
              r"\multicolumn{2}{r}{Écart} & Dispersion \\", r"\midrule"]
    for side in ["Long", "Short"]:
        c = coefs[coefs.Side == side].set_index("Variable")
        cell = [f"{fmt(c.loc[v, 'Coef'], 2)} ({fmt(c.loc[v, 'z'], 1)})" for v in ["Constante"] + META_X]
        lines.append(f"{side} & \\multicolumn{{2}}{{r}}{{{cell[0]}}} & \\multicolumn{{2}}{{r}}{{{cell[1]}}} "
                     f"& \\multicolumn{{2}}{{r}}{{{cell[2]}}} & {cell[3]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_vol(vol):
    lines = [r"\begin{tabular}{l" + "rr" * 3 + "}", r"\toprule",
             r" & \multicolumn{2}{c}{GARCH(1,1)} & \multicolumn{2}{c}{VR semaine passée} "
             r"& \multicolumn{2}{c}{Écart-type 12 sem.} \\",
             r"Paire & RMSE & QLIKE & RMSE & QLIKE & RMSE & QLIKE \\", r"\midrule"]
    for p in PAIRS:
        d = vol[vol.Pair == p].set_index("Model")
        cells = []
        for mdl in ["GARCH(1,1)", "VR semaine passée", "Écart-type 12 sem."]:
            cells += [f"{d.loc[mdl, 'RMSE']:.3f}", f"{d.loc[mdl, 'QLIKE']:.3f}"]
        best_q = d.QLIKE.idxmin()
        k = ["GARCH(1,1)", "VR semaine passée", "Écart-type 12 sem."].index(best_q)
        cells[2 * k + 1] = r"\textbf{" + cells[2 * k + 1] + "}"
        lines.append(f"{PAIR_LABELS[p]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_strategy(perf):
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Modèle de prévision & Rend. ann. & Vol. ann. & Sharpe & IC 90\,\% & Perte max. \\",
             r"\midrule"]
    for _, r in perf.iterrows():
        if r.Strategy == "MIXT":
            lines.append(r"\midrule")
        lines.append(f"{r.Strategy} & {fmt(100 * r.Ann_return, 1)}\\% & {100 * r.Ann_vol:.1f}\\% & "
                     f"{fmt(r.Sharpe, 2)} & [{fmt(r.Sharpe_lo, 2)}; {fmt(r.Sharpe_hi, 2)}] & "
                     f"{fmt(100 * r.Max_drawdown, 1)}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    with open(MODELS / "models.pkl", "rb") as f:
        models = pickle.load(f)
    features_by_pair, fx_targets, _ = load_inputs()
    frames = walk_forward_frames(features_by_pair, fx_targets, models["train_ratio"])
    train_size = len(frames["EURUSD"]["y"]) - models["test_weeks"]
    bt = run_backtest(make_plots=False)

    acc = forecast_accuracy(models, frames)
    ic = rank_ic(models)
    save_table(acc, "forecast_accuracy")
    save_table(ic, "rank_ic")
    (TAB / "forecast_accuracy.tex").write_text(latex_accuracy(acc, ic), encoding="utf-8")
    fig_validation_heatmap(acc)
    rho = fig_val_vs_test(acc)
    fig_lasso_path(frames, train_size)

    records = sparsity(frames)
    usage, lam, pca_var = fig_sparsity(records, frames)
    save_table(usage.reset_index(), "feature_usage")
    imp = fig_rf_importance(frames, train_size)
    save_table(imp.rename("importance").reset_index(), "rf_importance")

    vol = fig_garch(models)
    save_table(vol, "volatility_benchmarks", latex_vol(vol))
    meta, coefs = fig_meta(bt, models)
    save_table(meta, "meta_classifier", latex_meta(meta, coefs))
    save_table(coefs, "meta_coefficients")
    perf = fig_strategy(models, bt)
    save_table(perf, "strategy_by_model", latex_strategy(perf))

    # Résumé pour le texte du rapport
    t = acc[acc.Period == "Test"]
    summary = {
        "rank_corr_val_test": rho,
        "test_r2os_mean_by_method": t.groupby("Method")["R2_OS"].mean().round(4).to_dict(),
        "share_positive_r2os_test": float((t[t.Method != "Naive"].R2_OS > 0).mean()),
        "cw_signif_10pct": int((t[t.Method != "Naive"].CW_p < 0.10).sum()),
        "n_cells": int((t.Method != "Naive").sum()),
        "lasso_median_lambda": lam.round(6).to_dict(),
        "pca_var_explained": pca_var.round(3).to_dict(),
        "garch_failures": {p: int(g["refit_failed"].sum()) for p, g in models["garch_models"].items()},
    }
    pd.Series(summary).to_json(CSV / "summary.json", force_ascii=False, indent=1)
    print(pd.Series(summary))
    print(ic.round(3))
    print(meta.round(3))
    print(perf.round(3))
    print(vol.groupby("Model")[["RMSE", "QLIKE", "Corr"]].mean().round(4))


if __name__ == "__main__":
    main()
