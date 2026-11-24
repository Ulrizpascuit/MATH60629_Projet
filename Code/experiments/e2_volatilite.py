"""E2 — Bloc ML 2 : prévoir la volatilité, et son apport au signal.

2.1 Quatre prévisions de la volatilité réalisée de la semaine suivante :
    GARCH(1,1) (référence E0), HAR sur le log de la volatilité réalisée
    (réestimé chaque semaine, fenêtre croissante), volatilité réalisée de la
    semaine passée, écart-type des rendements sur 12 semaines.
    Mesures : RMSE, QLIKE, corrélation ; test de Diebold-Mariano (HLN, Newey-West)
    contre le GARCH, correction de Holm sur les 15 tests du test.
    Règle fixée d'avance : modèle retenu = plus petite QLIKE moyenne de validation.
2.2 Ablation du signal, pour MIXT et pour la moyenne simple : r̂ seul, puis r̂/σ̂
    pour chaque prévision de volatilité.

Usage : .venv/bin/python Code/experiments/e2_volatilite.py   (après E1.4)
Sorties : experiments/E2_volatilite/"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import (INK, INK2, LEARNED, PAIR_LABELS, PAIRS, SERIES, holm, load_frames,
                    load_reference, out_dir, plt, rank_ic, reference_predictions, save_fig,
                    write_environment)
from e1_4_combinaison import load_members
from Function.functions import FEATURES, PROCESSED, load_frame
from Function.functions_model import ols_fit, ols_predict

DEST = out_dir("E2_volatilite")
PERIODS = ["validation", "test"]
VOL_MODELS = ["GARCH", "HAR", "VR_derniere", "Ecart_type_12"]
VOL_LABELS = {"GARCH": "GARCH(1,1)", "HAR": "HAR (log)", "VR_derniere": "VR semaine passée",
              "Ecart_type_12": "Écart-type 12 sem."}


## ---- 2.1 prévisions ----
def har_design(rv):
    """Régresseurs du HAR à chaque date t (connus à t) et cible log RV_{t+1}."""
    d = pd.DataFrame({"l1": np.log(rv), "l4": np.log(rv.rolling(4).mean()),
                      "l12": np.log(rv.rolling(12).mean())})
    d["target"] = d["l1"].shift(-1)
    return d


def har_forecasts(rv, dates, start):
    """Walk-forward : à chaque date t, OLS sur les dates s < t (cible connue à t), à partir de `start`."""
    d = har_design(rv)
    X, y = d[["l1", "l4", "l12"]].values, d["target"].values
    idx = d.index
    out = np.empty(len(dates))
    for k, t in enumerate(dates):
        train = (idx >= start) & (idx < t) & ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        coef = ols_fit(X[train], y[train])
        resid = y[train] - ols_predict(coef, X[train])
        s2 = resid @ resid / (train.sum() - 4)
        x_t = X[idx.get_loc(t)]
        out[k] = np.exp(ols_predict(coef, x_t)[0] + s2 / 2)
    return out


def vol_forecasts(ref, frames):
    rv_all = load_frame(PROCESSED / "realized_vol.csv")
    res = {}
    for pair in PAIRS:
        feat = load_frame(FEATURES / f"{pair[:3].lower()}_features.csv")
        start = frames[pair]["dates"][0]
        for period in PERIODS:
            g = ref["validation_garch_models" if period == "validation" else "garch_models"][pair]
            dates = pd.DatetimeIndex(g["dates"])
            assert dates.equals(frames[pair]["dates"][frames[pair][period]])
            actual = np.asarray(g["actual"])
            # Contrôle : la cible est bien la volatilité réalisée de la semaine suivante
            nxt = rv_all[pair].shift(-1).reindex(dates).values
            assert np.allclose(actual, nxt)
            res[(pair, period)] = {
                "actual": actual,
                "GARCH": np.asarray(g["predictions"]),
                "HAR": har_forecasts(rv_all[pair], dates, start),
                "VR_derniere": rv_all[pair].reindex(dates).values,
                "Ecart_type_12": feat["vol_12"].reindex(dates).values,
            }
            for m in VOL_MODELS:
                assert np.isfinite(res[(pair, period)][m]).all() and (res[(pair, period)][m] > 0).all()
    return res


def qlike(actual, f):
    r = actual ** 2 / f ** 2
    return r - np.log(r) - 1


def dm_test(loss_model, loss_bench):
    """Diebold-Mariano bilatéral, variance de long terme de Newey-West, correction HLN (h = 1)."""
    d = loss_model - loss_bench
    n = len(d)
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    u = d - d.mean()
    lrv = u @ u / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n) * np.sqrt((n - 1) / n)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1))), lags


## ---- 2.2 signaux ----
def return_forecasts(ref):
    """r̂ de MIXT (référence) et de la moyenne simple des 7 membres (E1.4), par (paire, période)."""
    members, _ = load_members(ref)
    out = {}
    for pair in PAIRS:
        for period in PERIODS:
            out[("MIXT", pair, period)] = reference_predictions(ref, "MIXT", pair, period)[0]
            out[("Moyenne", pair, period)] = np.column_stack(
                [members[(pair, period)][m] for m in LEARNED]).mean(axis=1)
    return out


def signal_metrics(signal, returns, rv_next):
    ic_r, ic_r_t = rank_ic(signal, returns)
    ic_s, ic_s_t = rank_ic(signal, returns / rv_next)
    top = returns[np.arange(len(signal)), np.argmax(signal, axis=1)]
    bottom = returns[np.arange(len(signal)), np.argmin(signal, axis=1)]
    spread = top - bottom
    return {"IC_rendement": ic_r, "IC_rendement_t": ic_r_t, "IC_sharpe": ic_s, "IC_sharpe_t": ic_s_t,
            "Ecart_haut_bas_pb": 1e4 * spread.mean(),
            "Ecart_haut_bas_t": spread.mean() / (spread.std(ddof=1) / np.sqrt(len(spread)))}


def main():
    frames, ref = load_frames(), load_reference()
    vol = vol_forecasts(ref, frames)

    # 2.1 Précision
    rows, dm_rows = [], []
    for (pair, period), d in vol.items():
        a = d["actual"]
        for m in VOL_MODELS:
            f = d[m]
            rows.append({"Period": period, "Model": m, "Pair": pair,
                         "RMSE_pct": 100 * np.sqrt(np.mean((a - f) ** 2)),
                         "QLIKE": qlike(a, f).mean(), "Corr": np.corrcoef(a, f)[0, 1]})
            if m != "GARCH":
                s, p, lags = dm_test(qlike(a, f), qlike(a, d["GARCH"]))
                dm_rows.append({"Period": period, "Model": m, "Pair": pair, "DM": s, "p": p,
                                "Retards_NW": lags})
    acc = pd.DataFrame(rows)
    dm = pd.DataFrame(dm_rows)
    t = dm.Period == "test"
    dm.loc[t, "p_Holm"] = holm(dm.loc[t, "p"].values)
    acc.to_csv(DEST / "precision_volatilite.csv", index=False)
    dm.to_csv(DEST / "diebold_mariano.csv", index=False)

    mean_q = acc.groupby(["Period", "Model"])["QLIKE"].mean().unstack(0)[PERIODS].loc[VOL_MODELS]
    chosen = mean_q["validation"].idxmin()
    resume_dm = pd.DataFrame([{
        "Seuil": a, "Tests": int(t.sum()),
        "Mieux_que_GARCH_brut": int(((dm.p < a) & (dm.DM < 0) & t).sum()),
        "Moins_bon_que_GARCH_brut": int(((dm.p < a) & (dm.DM > 0) & t).sum()),
        "Mieux_que_GARCH_Holm": int(((dm.p_Holm < a) & (dm.DM < 0) & t).sum()),
        "Moins_bon_que_GARCH_Holm": int(((dm.p_Holm < a) & (dm.DM > 0) & t).sum())}
        for a in [0.05, 0.10]])
    resume_dm.to_csv(DEST / "resume_diebold_mariano.csv", index=False)
    mean_q.to_csv(DEST / "qlike_moyenne.csv")
    print(mean_q.round(4).to_string())
    print("Modèle de volatilité retenu (validation) :", chosen)
    print(resume_dm.to_string(index=False))
    print(dm[t].round(4).to_string(index=False))

    # 2.2 Ablation du signal
    rhat = return_forecasts(ref)
    sig_rows = []
    for source in ["MIXT", "Moyenne"]:
        for period in PERIODS:
            R = np.column_stack([rhat[(source, p, period)] for p in PAIRS])
            A = np.column_stack([frames[p]["y"][frames[p][period]] for p in PAIRS])
            RVn = np.column_stack([vol[(p, period)]["actual"] for p in PAIRS])
            variants = {"r_seul": R}
            for m in VOL_MODELS:
                variants[f"r/{m}"] = R / np.column_stack([vol[(p, period)][m] for p in PAIRS])
            for name, S in variants.items():
                sig_rows.append({"Source": source, "Period": period, "Signal": name,
                                 **signal_metrics(S, A, RVn)})
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(DEST / "ablation_signal.csv", index=False)
    print(sig.round(4).to_string(index=False))

    (DEST / "choix.json").write_text(json.dumps(
        {"modele_volatilite_retenu": chosen,
         "regle": "plus petite QLIKE moyenne de validation sur 5 devises"},
        indent=1, ensure_ascii=False), encoding="utf-8")
    fig_qlike(acc, chosen)
    fig_series(vol, frames)
    fig_signal(sig)
    write_environment(DEST, modele_retenu=chosen)


## ---- figures ----
def fig_qlike(acc, chosen):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    w = 0.2
    for ax, period, title in [(axes[0], "validation", "A. Validation"), (axes[1], "test", "B. Test")]:
        d = acc[acc.Period == period]
        for k, m in enumerate(VOL_MODELS):
            v = d[d.Model == m].set_index("Pair").loc[PAIRS].QLIKE.values
            ax.bar(np.arange(len(PAIRS)) + (k - 1.5) * w, v, width=w, color=SERIES[k],
                   edgecolor="white", linewidth=1,
                   label=VOL_LABELS[m] + (" (retenu)" if m == chosen else ""))
        ax.set_xticks(range(len(PAIRS)), [PAIR_LABELS[p] for p in PAIRS], fontsize=7.5)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Perte QLIKE moyenne (plus bas = mieux)")
    axes[0].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(1.05, -0.12),
                   fontsize=7)
    save_fig(fig, DEST, "fig_qlike")


def fig_series(vol, frames):
    pair = "EURUSD"
    d = vol[(pair, "test")]
    dates = frames[pair]["dates"][frames[pair]["test"]]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(dates, 100 * d["actual"], color="#b5b3ad", lw=0.8, label="Volatilité réalisée")
    ax.plot(dates, 100 * d["GARCH"], color=SERIES[0], lw=1.4, label="GARCH(1,1)")
    ax.plot(dates, 100 * d["HAR"], color=SERIES[1], lw=1.4, label="HAR (log)")
    ax.set_ylabel("Volatilité hebdomadaire (%)")
    ax.set_xlabel("Date (période de test)")
    ax.set_title(f"{PAIR_LABELS[pair]} : volatilité prévue et réalisée", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5)
    save_fig(fig, DEST, "fig_vol_eurusd")


def fig_signal(sig):
    names = ["r_seul"] + [f"r/{m}" for m in VOL_MODELS]
    labels = ["r̂ seul"] + [f"r̂ / {VOL_LABELS[m]}" for m in VOL_MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    w = 0.38
    for ax, source in zip(axes, ["MIXT", "Moyenne"]):
        for k, period in enumerate(PERIODS):
            v = sig[(sig.Source == source) & (sig.Period == period)].set_index("Signal").loc[names]
            ax.bar(np.arange(len(names)) + (k - 0.5) * w, v.IC_rendement.values, width=w,
                   color=SERIES[k], edgecolor="white", linewidth=1,
                   label="Validation" if period == "validation" else "Test")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(names)), labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(f"r̂ = {source}", loc="left")
    axes[0].set_ylabel("Corrélation de rang moyenne\navec les rendements réalisés")
    axes[1].legend(frameon=False, loc="upper right")
    save_fig(fig, DEST, "fig_ablation_signal")


if __name__ == "__main__":
    main()
