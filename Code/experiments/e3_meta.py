"""E3 — Bloc ML 3 : le méta-classifieur sépare-t-il les bonnes jambes des mauvaises ?

Jambes analysées : celles de la stratégie actuelle (MIXT + GARCH) de la référence
E0, avec les deux méta-classifieurs logistiques (seuil 0,5) ré-estimés en fenêtre
croissante (jambes de validation + jambes de test déjà connues).

3.1 Discrimination (AUC) et calibration (Brier, courbe de calibration).
3.2 Placebo : 10 000 filtres au hasard gardant 78 achats et 146 ventes.
3.3 Permutation des étiquettes (Ojala et Garriga, 2010, test 1) : 1 000 fois,
    réapprentissage complet en fenêtre croissante, AUC sur le test.
3.4 Courbes risque-couverture : scores p̂, |ŝ| et hasard.
3.5 Sensibilité au seuil sur la validation (rapportée, jamais optimisée).

Usage : .venv/bin/python Code/experiments/e3_meta.py
Sorties : experiments/E3_meta/"""
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from common import (INK, INK2, SERIES, load_reference, out_dir, plt, save_fig,
                    write_environment)
from Function.functions import PROCESSED, load_frame
from Function.functions_strategy import (META_X, expanding_meta_cv, glm_logit, glm_predict,
                                         meta_expanding_filtered_returns, model_matrix,
                                         performance_summary, strategy_legs, strategy_returns,
                                         threshold_diagnostics)

DEST = out_dir("E3_meta")
SIDES = ["Long", "Short"]
TAU = 0.5
B_PLACEBO, B_PERM, SEED = 10_000, 1_000, 123


def sharpe(r):
    return r.mean() / r.std(ddof=1) * np.sqrt(52)


## ---- reconstruction de la stratégie de référence ----
def reference_strategy(ref):
    sm, vsm = ref["sharpe_models"]["MIXT"], ref["validation_sharpe_models"]["MIXT"]
    pred, actual = model_matrix(sm, "predictions"), model_matrix(sm, "return_actual")
    val_legs = strategy_legs(model_matrix(vsm, "predictions"), model_matrix(vsm, "return_actual"))
    dff = load_frame(PROCESSED / "funding.csv")["Funding_weekly"]
    meta = meta_expanding_filtered_returns(pred, actual, val_legs, TAU, TAU, dff)
    raw = strategy_returns(pred, actual)
    return meta, raw, val_legs


def weekly_arrays(dec, index):
    """Pour chaque semaine du test : PnL de la jambe longue, de la jambe courte, financement,
    et indicateur de semaine tradée (une jambe de chaque côté par semaine tradée)."""
    L = dec[dec.Side == "Long"].set_index("Date")
    S = dec[dec.Side == "Short"].set_index("Date")
    assert L.index.equals(S.index)
    traded = index.isin(L.index)
    return L, S, traded


def strategy_from_accept(aL, aS, L, S, traded, n_weeks):
    """Rendement hebdomadaire : a r_long + b r_short - (a - b) f (même règle que le backtest)."""
    r = np.zeros(n_weeks)
    f = L["Funding_weekly"].values
    r[traded] = aL * L["PnL"].values + aS * S["PnL"].values - (aL - aS) * f
    return r


## ---- 3.3 apprentissage rapide en fenêtre croissante ----
def expanding_probs(Xv, yv, Xt, yt):
    """Probabilités de test : à la jambe k, logistique ajustée sur la validation + jambes 0..k-1."""
    X = np.vstack([Xv, Xt])
    y = np.concatenate([yv, yt])
    nv = len(yv)
    return np.array([glm_predict(glm_logit(X[:nv + k], y[:nv + k]), Xt[k:k + 1])[0]
                     for k in range(len(yt))])


def perm_auc(seed, data):
    rng = np.random.default_rng(seed)
    out = {}
    for side, (Xv, yv, Xt, yt) in data.items():
        y = rng.permutation(np.concatenate([yv, yt]))
        yv_p, yt_p = y[:len(yv)], y[len(yv):]
        out[side] = roc_auc_score(yt_p, expanding_probs(Xv, yv_p, Xt, yt_p))
    return out


def main():
    t0 = time.time()
    ref = load_reference()
    meta, raw, val_legs = reference_strategy(ref)
    dec = meta["decisions"]
    idx = meta["returns"].index
    obs_meta, obs_raw = sharpe(meta["returns"].values), sharpe(raw.values)
    perf = performance_summary(pd.DataFrame({"MIXT_sans_filtre": raw, "MIXT_meta": meta["returns"]}))
    assert abs(perf.loc[1, "Sharpe"] - 0.8765461357071419) < 1e-6  # = référence publiée
    perf.to_csv(DEST / "performance_reference.csv", index=False)

    # 3.1 Discrimination et calibration
    disc = []
    for side in SIDES:
        d = dec[dec.Side == side]
        base = d.Win.mean()
        disc.append({"Side": side, "N": len(d), "Taux_de_gain": base,
                     "AUC": roc_auc_score(d.Win, d.Probability),
                     "Brier": brier_score_loss(d.Win, d.Probability),
                     "Brier_reference": brier_score_loss(d.Win, np.full(len(d), base)),
                     "Acceptees": int(d.Accept.sum()),
                     "Gain_acceptees": d.Win[d.Accept].mean(), "Gain_rejetees": d.Win[~d.Accept].mean()})
    disc = pd.DataFrame(disc)

    # 3.2 Placebo
    L, S, traded = weekly_arrays(dec, idx)
    nL, nS = int(L.Accept.sum()), int(S.Accept.sum())
    check = strategy_from_accept(L.Accept.values.astype(float), S.Accept.values.astype(float),
                                 L, S, traded, len(idx))
    assert np.allclose(check, meta["returns"].values)
    rng = np.random.default_rng(SEED)
    placebo = np.empty(B_PLACEBO)
    for b in range(B_PLACEBO):
        aL = np.zeros(len(L)); aL[rng.choice(len(L), nL, replace=False)] = 1
        aS = np.zeros(len(S)); aS[rng.choice(len(S), nS, replace=False)] = 1
        placebo[b] = sharpe(strategy_from_accept(aL, aS, L, S, traded, len(idx)))
    p_placebo = (1 + np.sum(placebo >= obs_meta)) / (B_PLACEBO + 1)
    placebo_res = {"Sharpe_filtre": obs_meta, "Sharpe_sans_filtre": obs_raw,
                   "Placebo_moyenne": placebo.mean(), "Placebo_q05": np.quantile(placebo, 0.05),
                   "Placebo_q95": np.quantile(placebo, 0.95), "p_valeur": p_placebo,
                   "Achats_gardes": nL, "Ventes_gardees": nS, "Tirages": B_PLACEBO}
    print("Placebo :", {k: round(v, 4) if isinstance(v, float) else v for k, v in placebo_res.items()})

    # 3.3 Permutation des étiquettes (réapprentissage complet)
    data = {}
    for side in SIDES:
        v = val_legs[val_legs.Side == side].sort_values("Date", kind="stable")
        t = dec[dec.Side == side].sort_values("Date", kind="stable")
        data[side] = (v[META_X].values, v.Win.values.astype(float),
                      t[META_X].values, t.Win.values.astype(float))
        probs = expanding_probs(*data[side])
        assert np.allclose(probs, t.Probability.values, atol=1e-10)  # = procédure du backtest
    perms = Parallel(n_jobs=-1)(delayed(perm_auc)(SEED + k, data) for k in range(B_PERM))
    perm_df = pd.DataFrame(perms)
    disc["AUC_p_permutation"] = [(1 + np.sum(perm_df[s] >= disc.set_index("Side").loc[s, "AUC"]))
                                 / (B_PERM + 1) for s in SIDES]
    disc["AUC_nulle_q95"] = [np.quantile(perm_df[s], 0.95) for s in SIDES]
    disc.to_csv(DEST / "discrimination_calibration.csv", index=False)
    perm_df.to_csv(DEST / "permutation_auc.csv", index=False)
    print(disc.round(4).to_string(index=False))

    # 3.4 Risque-couverture
    legs = dec.copy()
    n = len(legs)
    k = np.arange(1, n + 1)
    curves = {"Coverage": k / n}
    aurc = {}
    for name, score in [("Meta_p", legs.Probability.values), ("Abs_s", legs.Abs_Forecast.values)]:
        order = np.argsort(-score, kind="stable")
        err = 1 - np.cumsum(legs.Win.values[order]) / k
        pnl = np.cumsum(legs.PnL.values[order]) / k
        curves[f"{name}_erreur"], curves[f"{name}_rendement"] = err, pnl
        aurc[name] = {"AURC_erreur": err.mean(), "Rendement_moyen_aire": pnl.mean()}
    rnd_err = np.zeros(n); rnd_pnl = np.zeros(n)
    for _ in range(1000):
        order = rng.permutation(n)
        rnd_err += 1 - np.cumsum(legs.Win.values[order]) / k
        rnd_pnl += np.cumsum(legs.PnL.values[order]) / k
    curves["Hasard_erreur"], curves["Hasard_rendement"] = rnd_err / 1000, rnd_pnl / 1000
    aurc["Hasard"] = {"AURC_erreur": curves["Hasard_erreur"].mean(),
                      "Rendement_moyen_aire": curves["Hasard_rendement"].mean()}
    curves = pd.DataFrame(curves)
    curves.to_csv(DEST / "risque_couverture.csv", index=False)
    aurc = pd.DataFrame(aurc).T
    aurc.to_csv(DEST / "aurc.csv")
    print(aurc.round(5).to_string())

    # 3.5 Sensibilité au seuil (validation)
    thr = threshold_diagnostics(expanding_meta_cv(val_legs),
                                thresholds=np.arange(0.35, 0.65 + 1e-9, 0.025))
    thr.to_csv(DEST / "seuils_validation.csv", index=False)

    (DEST / "resume.json").write_text(json.dumps(
        {"placebo": placebo_res, "coverage_filtre": (nL + nS) / n}, indent=1, default=float,
        ensure_ascii=False), encoding="utf-8")
    fig_roc_calibration(dec)
    fig_placebo(placebo, obs_meta, obs_raw, p_placebo)
    fig_permutation(perm_df, disc)
    fig_risk_coverage(curves, (nL + nS) / n)
    write_environment(DEST, seconds=round(time.time() - t0, 1), B_placebo=B_PLACEBO,
                      B_permutation=B_PERM, seed=SEED)


## ---- figures ----
def fig_roc_calibration(dec):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for c, side, lab in [(SERIES[0], "Long", "Achats"), (SERIES[1], "Short", "Ventes")]:
        d = dec[dec.Side == side]
        fpr, tpr, _ = roc_curve(d.Win, d.Probability)
        axes[0].plot(fpr, tpr, color=c, lw=1.6,
                     label=f"{lab} (AUC = {roc_auc_score(d.Win, d.Probability):.2f})")
        q = pd.qcut(d.Probability, 5, duplicates="drop")
        cal = d.groupby(q, observed=True).agg(p=("Probability", "mean"), w=("Win", "mean"))
        axes[1].plot(cal.p, cal.w, "o-", color=c, ms=5, lw=1.4, label=lab)
    axes[0].plot([0, 1], [0, 1], color=INK2, ls=":", lw=0.8)
    axes[0].set_xlabel("Taux de faux positifs")
    axes[0].set_ylabel("Taux de vrais positifs")
    axes[0].set_title("A. Courbe ROC (test)", loc="left")
    axes[0].legend(frameon=False, loc="lower right")
    lo, hi = dec.Probability.min(), dec.Probability.max()
    axes[1].plot([lo, hi], [lo, hi], color=INK2, ls=":", lw=0.8, label="Calibration parfaite")
    axes[1].set_xlabel("Probabilité de gain prédite (moyenne par quintile)")
    axes[1].set_ylabel("Fréquence de gain observée")
    axes[1].set_title("B. Calibration (test)", loc="left")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, DEST, "fig_roc_calibration")


def fig_placebo(placebo, obs, raw, p):
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.hist(placebo, bins=60, color=SERIES[0], alpha=0.85, edgecolor="white", linewidth=0.5,
            label=f"{len(placebo):,} filtres au hasard".replace(",", " "))
    ax.axvline(obs, color=SERIES[1], lw=2, label=f"Méta-classifieur : {obs:.2f} (p = {p:.3f})".replace(".", ","))
    ax.axvline(raw, color=INK, lw=1.2, ls="--", label=f"Sans filtre : {raw:.2f}".replace(".", ","))
    ax.set_xlabel("Ratio de Sharpe annualisé de la stratégie (test)")
    ax.set_ylabel("Nombre de filtres")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    save_fig(fig, DEST, "fig_placebo")


def fig_permutation(perm_df, disc):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, side, lab in [(axes[0], "Long", "A. Achats"), (axes[1], "Short", "B. Ventes")]:
        r = disc.set_index("Side").loc[side]
        ax.hist(perm_df[side], bins=40, color=SERIES[0], alpha=0.85, edgecolor="white",
                linewidth=0.5, label="Étiquettes mélangées")
        ax.axvline(r.AUC, color=SERIES[1], lw=2,
                   label=f"AUC observée : {r.AUC:.3f}\n(p = {r.AUC_p_permutation:.3f})".replace(".", ","))
        ax.set_title(lab, loc="left")
        ax.set_xlabel("AUC sur le test")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_ylabel("Nombre de permutations")
    save_fig(fig, DEST, "fig_permutation_auc")


def fig_risk_coverage(curves, cov_filter):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw={"wspace": 0.45})
    specs = [("Meta_p", "Probabilité du méta-classifieur", SERIES[1]),
             ("Abs_s", "Force du signal |ŝ|", SERIES[0]), ("Hasard", "Hasard (moyenne)", INK2)]
    for name, lab, c in specs:
        ls = "--" if name == "Hasard" else "-"
        axes[0].plot(100 * curves.Coverage, 100 * curves[f"{name}_erreur"], color=c, ls=ls, lw=1.5,
                     label=lab)
        axes[1].plot(100 * curves.Coverage, 1e4 * curves[f"{name}_rendement"], color=c, ls=ls,
                     lw=1.5)
    for ax in axes:
        ax.axvline(100 * cov_filter, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("Couverture : % des jambes gardées")
        ax.set_xlim(5, 100)
    axes[0].set_ylim(25, 60)
    axes[1].set_ylim(-5, 65)
    axes[0].set_ylabel("Taux d'erreur des jambes gardées (%)")
    axes[1].set_ylabel("Rendement moyen des jambes gardées\n(points de base par semaine)")
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[0].set_title("A. Risque = erreur", loc="left")
    axes[1].set_title("B. Risque = rendement", loc="left")
    axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(1.25, -0.2),
                   ncol=3)
    fig.text(0.5, -0.2, f"Pointillé vertical : couverture du filtre au seuil 0,5 "
             f"({100 * cov_filter:.0f} %).", ha="center", fontsize=7, color=INK)
    save_fig(fig, DEST, "fig_risque_couverture")


if __name__ == "__main__":
    main()
