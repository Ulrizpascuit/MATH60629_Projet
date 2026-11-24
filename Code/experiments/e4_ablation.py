"""E4 — Ablation de bout en bout : que rapporte chaque bloc ML à la stratégie ?

Escalier, pour deux chaînes :
  A0 : Naïf (dernier rendement / dernière volatilité réalisée), sans filtre ;
  A1 : + prévision ML du rendement, signal = r̂ ;
  A2 : + volatilité, signal = r̂ / σ̂ ;
  A3 : + méta-classifieur (seuil 0,5) ; A3-placebo : filtre au hasard, mêmes effectifs.
Chaîne principale (fixée avant tout) : MIXT, GARCH, méta de l'ancien projet (référence E0).
Chaîne secondaire (exploratoire) : HAR (étape 2) et prévision de rendement choisie par
validation emboîtée entre MIXT, stacking et moyenne.

Mesures : Sharpe annualisé et IC à 95 % (bootstrap par blocs de 8 semaines, 2 000 tirages),
rang parmi 1 000 stratégies au hasard, Deflated Sharpe Ratio (N = tous les essais du
projet), rendements par année.

Usage : .venv/bin/python Code/experiments/e4_ablation.py   (après E1.4, E2 et E3)
Sorties : experiments/E4_ablation/"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import (EXP, INK, INK2, LEARNED, PAIRS, SERIES, load_frames, load_reference,
                    out_dir, plt, reference_predictions, save_fig, write_environment)
from Code.backtest import naive_last_vol_pred  # après common, qui ajoute la racine au chemin
from e1_4_combinaison import load_members, r2, stacking_weights
from e2_volatilite import vol_forecasts
from e3_meta import sharpe, strategy_from_accept, weekly_arrays
from Function.functions import PROCESSED, load_frame
from Function.functions_strategy import (meta_expanding_filtered_returns, model_matrix,
                                         performance_summary, random_strategy_returns,
                                         strategy_legs, strategy_returns)

DEST = out_dir("E4_ablation")
TAU, SEED = 0.5, 123
B_BOOT, BLOCK, B_PLACEBO = 2000, 8, 10_000
EULER = 0.5772156649015329


def frame(values_by_pair, dates):
    return pd.DataFrame({p: values_by_pair[p] for p in PAIRS}, index=dates)


## ---- choix emboîté de la prévision de rendement (chaîne secondaire) ----
def nested_choice(members, frames):
    rows = []
    for pair in PAIRS:
        fr = frames[pair]
        y = fr["y"][fr["validation"]]
        val = members[(pair, "validation")]
        h = len(y) // 2
        cands = ["Naive"] + LEARNED
        mixt = max(cands, key=lambda m: r2(y[:h], val[m][:h]))
        F = np.column_stack([val[m] for m in LEARNED])
        w = stacking_weights(F[:h], y[:h])
        rows.append({"Pair": pair, "MIXT_1re_moitie": mixt,
                     "R2_MIXT": r2(y[h:], val[mixt][h:]),
                     "R2_Stacking": r2(y[h:], F[h:] @ w),
                     "R2_Moyenne": r2(y[h:], F[h:].mean(axis=1))})
    res = pd.DataFrame(rows)
    means = res[["R2_MIXT", "R2_Stacking", "R2_Moyenne"]].mean()
    winner = means.idxmax().replace("R2_", "")
    return res, means, winner


def updated_forecasts(members, ref, frames, winner):
    """Prévisions (validation, test) du gagnant, réajusté sur toute la validation."""
    out = {}
    for pair in PAIRS:
        fr = frames[pair]
        y = fr["y"][fr["validation"]]
        val = members[(pair, "validation")]
        F_val = np.column_stack([val[m] for m in LEARNED])
        w = stacking_weights(F_val, y)
        mixt = max(["Naive"] + LEARNED, key=lambda m: r2(y, val[m]))
        for period in ["validation", "test"]:
            d = members[(pair, period)]
            F = np.column_stack([d[m] for m in LEARNED])
            out[(pair, period)] = {"MIXT": d[mixt], "Stacking": F @ w, "Moyenne": F.mean(axis=1)}[winner]
    return out


## ---- mesures ----
def block_bootstrap_sharpe(R, rng):
    n = len(R)
    boots = np.empty((B_BOOT, R.shape[1]))
    for b in range(B_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, int(np.ceil(n / BLOCK)))
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        r = R[idx]
        boots[b] = r.mean(0) / r.std(0, ddof=1) * np.sqrt(52)
    return np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)


def deflated_sharpe(r, sr0_weekly):
    """DSR de Bailey et López de Prado (2014) : P(vrai Sharpe > SR0), Sharpe hebdomadaire."""
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    z = (sr - sr0_weekly) * np.sqrt(len(r) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf(z))


def placebo(decisions, returns_index, rng):
    L, S, traded = weekly_arrays(decisions, returns_index)
    nL, nS = int(L.Accept.sum()), int(S.Accept.sum())
    out = np.empty(B_PLACEBO)
    for b in range(B_PLACEBO):
        aL = np.zeros(len(L)); aL[rng.choice(len(L), nL, replace=False)] = 1
        aS = np.zeros(len(S)); aS[rng.choice(len(S), nS, replace=False)] = 1
        out[b] = sharpe(strategy_from_accept(aL, aS, L, S, traded, len(returns_index)))
    return out


def trial_sharpes(new_weekly_sharpes):
    """Sharpe hebdomadaires de tous les essais du projet sur le test (pour le DSR)."""
    trials = []
    old = pd.read_csv(EXP / "E0_reference_mac" / "outputs" / "ml" / "strategy_by_model.csv")
    trials += [("Ancien projet : " + s, v / np.sqrt(52)) for s, v in zip(old.Strategy, old.Sharpe)]
    comp = pd.read_csv(EXP / "E0_reference_mac" / "outputs" / "comparison_performance.csv")
    corr = comp[comp.Strategy.str.contains("corrigee")]
    trials += [("Ancien projet : " + s, v / np.sqrt(52)) for s, v in zip(corr.Strategy, corr.Sharpe)]
    sig = pd.read_csv(EXP / "E2_volatilite" / "ablation_signal.csv")
    sig = sig[sig.Period == "test"]
    trials += [(f"E2.2 : {s} / {g}", t / np.sqrt(260))
               for s, g, t in zip(sig.Source, sig.Signal, sig.Ecart_haut_bas_t)]
    trials += [(f"E4 : {k}", v) for k, v in new_weekly_sharpes.items()]
    return pd.DataFrame(trials, columns=["Essai", "Sharpe_hebdo"])


def main():
    frames, ref = load_frames(), load_reference()
    rng = np.random.default_rng(SEED)
    fx_rv = load_frame(PROCESSED / "realized_vol.csv")
    dff = load_frame(PROCESSED / "funding.csv")["Funding_weekly"]
    sm = ref["sharpe_models"]
    actual = model_matrix(sm["MIXT"], "return_actual")
    dates = actual.index
    vdates = model_matrix(ref["validation_sharpe_models"]["MIXT"]).index
    actual_val = model_matrix(ref["validation_sharpe_models"]["MIXT"], "return_actual")

    # Chaîne principale (ancien projet)
    r_mixt = model_matrix(sm["MIXT"], "return_predictions")
    s_garch = model_matrix(sm["MIXT"], "predictions")
    val_legs = strategy_legs(model_matrix(ref["validation_sharpe_models"]["MIXT"]), actual_val)
    meta_o = meta_expanding_filtered_returns(s_garch, actual, val_legs, TAU, TAU, dff)

    # Chaîne secondaire (mise à jour)
    members, setup = load_members(ref)
    nested, nested_means, winner = nested_choice(members, frames)
    upd = updated_forecasts(members, ref, frames, winner)
    vol = vol_forecasts(ref, frames)
    r_upd = frame({p: upd[(p, "test")] for p in PAIRS}, dates)
    s_upd = frame({p: upd[(p, "test")] / vol[(p, "test")]["HAR"] for p in PAIRS}, dates)
    s_upd_val = frame({p: upd[(p, "validation")] / vol[(p, "validation")]["HAR"] for p in PAIRS},
                      vdates)
    val_legs_u = strategy_legs(s_upd_val, actual_val)
    meta_u = meta_expanding_filtered_returns(s_upd, actual, val_legs_u, TAU, TAU, dff)

    naive = strategy_returns(naive_last_vol_pred(sm["Naive"], fx_rv), actual)
    variants = {
        "A0 Naïf": naive,
        "A1 origine (r̂ MIXT)": strategy_returns(r_mixt, actual),
        "A2 origine (r̂/σ̂ GARCH)": strategy_returns(s_garch, actual),
        "A3 origine (+ méta)": meta_o["returns"],
        f"A1 mise à jour (r̂ {winner})": strategy_returns(r_upd, actual),
        f"A2 mise à jour (r̂/σ̂ HAR)": strategy_returns(s_upd, actual),
        "A3 mise à jour (+ méta)": meta_u["returns"],
    }
    R = pd.DataFrame(variants)
    assert abs(sharpe(R["A3 origine (+ méta)"].values) - 0.8765461357071419) < 1e-6
    assert abs(sharpe(R["A2 origine (r̂/σ̂ GARCH)"].values) - 0.45196467449418215) < 1e-6

    perf = performance_summary(R)
    lo, hi = block_bootstrap_sharpe(R.values, rng)
    perf["Sharpe_IC95_bas"], perf["Sharpe_IC95_haut"] = lo, hi
    totals = random_strategy_returns(actual, n_sim=1000, seed=SEED).sum(axis=0).values
    perf["Rang_parmi_1000_au_hasard"] = [np.mean(totals < R[c].sum()) for c in R]

    # Placebos des deux A3, et part du financement (jambe seule : on paie ou reçoit le DFF)
    plac = {}
    for name, meta in [("A3 origine (+ méta)", meta_o), ("A3 mise à jour (+ méta)", meta_u)]:
        p = placebo(meta["decisions"], dates, rng)
        obs = sharpe(R[name].values)
        L, S, traded = weekly_arrays(meta["decisions"], dates)
        aL, aS = L.Accept.values.astype(float), S.Accept.values.astype(float)
        fund = np.zeros(len(dates))
        fund[traded] = -(aL - aS) * L.Funding_weekly.values
        no_fund = R[name].values - fund
        plac[name] = {"Placebo_moyenne": p.mean(), "Placebo_q025": np.quantile(p, 0.025),
                      "Placebo_q975": np.quantile(p, 0.975),
                      "p_valeur": (1 + np.sum(p >= obs)) / (B_PLACEBO + 1),
                      "Semaines_deux_jambes": int(((aL == 1) & (aS == 1)).sum()),
                      "Semaines_achat_seul": int(((aL == 1) & (aS == 0)).sum()),
                      "Semaines_vente_seule": int(((aL == 0) & (aS == 1)).sum()),
                      "Financement_annualise": 52 * fund.mean(),
                      "Sharpe_sans_financement": sharpe(no_fund)}
    perf = perf.merge(pd.DataFrame(plac).T.rename_axis("Strategy").reset_index(), on="Strategy",
                      how="left")

    # Deflated Sharpe Ratio, N = tous les essais du projet
    new = {k: R[k].mean() / R[k].std(ddof=1) for k in
           ["A1 origine (r̂ MIXT)", f"A1 mise à jour (r̂ {winner})", "A2 mise à jour (r̂/σ̂ HAR)",
            "A3 mise à jour (+ méta)"]}
    trials = trial_sharpes(new)
    N, V = len(trials), trials.Sharpe_hebdo.var(ddof=1)
    sr0 = np.sqrt(V) * ((1 - EULER) * stats.norm.ppf(1 - 1 / N)
                        + EULER * stats.norm.ppf(1 - 1 / (N * np.e)))
    perf["DSR"] = [deflated_sharpe(R[c].values, sr0) for c in R]
    trials.to_csv(DEST / "essais_pour_dsr.csv", index=False)

    years = R.groupby(R.index.year).sum()
    years.index.name = "Annee"
    perf.to_csv(DEST / "ablation_performance.csv", index=False)
    years.to_csv(DEST / "ablation_annees.csv")
    nested.to_csv(DEST / "choix_emboite.csv", index=False)
    (DEST / "choix.json").write_text(json.dumps({
        "prevision_mise_a_jour": winner, "R2_seconde_moitie": nested_means.to_dict(),
        "volatilite_mise_a_jour": "HAR", "N_essais_DSR": N, "SR0_annualise": sr0 * np.sqrt(52),
        **setup}, indent=1, ensure_ascii=False, default=float), encoding="utf-8")

    cols = ["Strategy", "Sharpe", "Sharpe_IC95_bas", "Sharpe_IC95_haut", "Ann_return", "Max_drawdown",
            "Weeks_traded", "Rang_parmi_1000_au_hasard", "Placebo_moyenne", "p_valeur", "DSR"]
    print("Validation emboîtée :", nested_means.round(5).to_dict(), "->", winner)
    print(nested.round(4).to_string(index=False))
    print(perf[cols].round(3).to_string(index=False))
    print(f"N essais = {N}, SR0 annualisé = {sr0 * np.sqrt(52):.3f}")
    print(years.round(4).to_string())

    fig_staircase(perf, winner)
    fig_nav(R)
    write_environment(DEST, prevision_mise_a_jour=winner, N_essais=N, B_boot=B_BOOT, block=BLOCK,
                      B_placebo=B_PLACEBO, seed=SEED)


## ---- figures ----
def fig_staircase(perf, winner):
    p = perf.set_index("Strategy")
    chains = [("Chaîne d'origine", ["A0 Naïf", "A1 origine (r̂ MIXT)", "A2 origine (r̂/σ̂ GARCH)",
                                    "A3 origine (+ méta)"], ["A0\nNaïf", "A1\n+ r̂ MIXT",
                                                             "A2\n+ σ̂ GARCH", "A3\n+ méta"]),
              ("Chaîne mise à jour (exploratoire)",
               ["A0 Naïf", f"A1 mise à jour (r̂ {winner})", "A2 mise à jour (r̂/σ̂ HAR)",
                "A3 mise à jour (+ méta)"], ["A0\nNaïf", f"A1\n+ r̂ {winner}", "A2\n+ σ̂ HAR",
                                             "A3\n+ méta"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, (title, names, labels) in zip(axes, chains):
        x = np.arange(len(names))
        s = p.loc[names]
        a3 = s.iloc[-1]
        ax.add_patch(plt.Rectangle((x[-1] - 0.3, a3.Placebo_q025), 0.6,
                                   a3.Placebo_q975 - a3.Placebo_q025, color=GRID_FILL, zorder=0,
                                   label="Placebo du filtre (95 %)"))
        ax.errorbar(x, s.Sharpe, yerr=[s.Sharpe - s.Sharpe_IC95_bas, s.Sharpe_IC95_haut - s.Sharpe],
                    fmt="o", color=SERIES[0], ms=6, capsize=3, lw=1.4, label="Sharpe et IC à 95 %")
        ax.plot(x, s.Sharpe, color=SERIES[0], lw=0.8, alpha=0.5)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Ratio de Sharpe annualisé (test)")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    save_fig(fig, DEST, "fig_escalier_ablation")


GRID_FILL = "#dcdad4"


def fig_nav(R):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    names = [c for c in R.columns if "mise à jour" not in c] + ["A3 mise à jour (+ méta)"]
    colors = [INK2, SERIES[0], SERIES[2], SERIES[1], SERIES[6]]
    for c, col in zip(names, colors):
        ax.plot(R.index, 100 * (np.exp(R[c].cumsum()) - 1), color=col, lw=1.4,
                ls="--" if "mise à jour" in c else "-", label=c)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("Rendement cumulé (%)")
    ax.set_xlabel("Date (période de test)")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save_fig(fig, DEST, "fig_valeur_cumulee")


if __name__ == "__main__":
    main()
