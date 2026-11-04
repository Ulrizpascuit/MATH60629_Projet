"""Stratégie long-short, meta-classifieurs logistiques, mesures de performance
et graphiques."""
import numpy as np
import pandas as pd

from Function.r_random import RRandom

META_COLS = ["Date", "Side", "Win", "PnL", "Abs_Forecast", "Gap", "Dispersion"]
META_X = ["Abs_Forecast", "Gap", "Dispersion"]


## ---- model_matrix ----
def model_matrix(models, element="predictions"):
    """Une colonne par paire, une ligne par semaine."""
    return pd.concat(
        {pair: pd.Series(m[element], index=m["dates"]) for pair, m in models.items()}, axis=1
    )


def _skip(y_hat, skip_all_positive):
    # Si toutes les prédictions sont négatives on ne fait rien
    # (et aussi si elles sont toutes positives quand skip_all_positive = True)
    return y_hat.max() <= 0 or (skip_all_positive and y_hat.min() >= 0)


## ---- strategy_returns ----
def strategy_returns(pred, actual, skip_all_positive=False):
    """Long la paire au plus haut Sharpe prévu, short celle au plus bas."""
    p, a = pred.values, actual.values
    out = np.zeros(len(p))
    for t in range(len(p)):
        if _skip(p[t], skip_all_positive):
            continue
        out[t] = a[t, np.argmax(p[t])] - a[t, np.argmin(p[t])]
    return pd.Series(out, index=pred.index)


## ---- strategy_legs ----
def strategy_legs(pred, actual, skip_all_positive=False):
    """Une ligne par jambe prise, avec les variables du classifieur :
    Abs_Forecast, Gap (écart avec la 2e meilleure / pire paire), Dispersion."""
    pairs = list(pred.columns)
    p, a = pred.values, actual.values
    rows = []
    for t in range(len(p)):
        y_hat, y_real = p[t], a[t]
        if _skip(y_hat, skip_all_positive):
            continue
        long, short = int(np.argmax(y_hat)), int(np.argmin(y_hat))
        s = np.sort(y_hat)
        disp = np.std(y_hat, ddof=1)
        rows.append((pred.index[t], pairs[long], "Long", abs(y_hat[long]), s[-1] - s[-2], disp,
                     y_real[long], int(y_real[long] > 0)))
        rows.append((pred.index[t], pairs[short], "Short", abs(y_hat[short]), s[1] - s[0], disp,
                     -y_real[short], int(-y_real[short] > 0)))
    return pd.DataFrame(rows, columns=["Date", "Pair", "Side", "Abs_Forecast", "Gap",
                                       "Dispersion", "PnL", "Win"])


## ---- logistic regression (glm binomial) ----
_DBL_EPS = np.finfo(float).eps
_THRESH = 30.0


def _linkinv(eta):
    tmp = np.where(eta < -_THRESH, _DBL_EPS,
                   np.where(eta > _THRESH, 1 / _DBL_EPS, np.exp(np.clip(eta, -_THRESH, _THRESH))))
    return tmp / (1 + tmp)


def _mu_eta(eta):
    e = np.exp(np.clip(eta, -_THRESH, _THRESH))
    return np.where(np.abs(eta) > _THRESH, _DBL_EPS, e / (1 + e) ** 2)


def _deviance(y, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(y == 1, np.log(1 / mu), np.log(1 / (1 - mu)))
    return 2 * d.sum()


def glm_logit(X, y, epsilon=1e-8, maxit=25):
    """glm(y ~ X, family = binomial()) : mêmes itérations IRLS que glm.fit de R."""
    A = np.column_stack([np.ones(len(y)), X])
    y = np.asarray(y, float)
    mu = (y + 0.5) / 2
    eta = np.log(mu / (1 - mu))
    mu = _linkinv(eta)
    dev_old = _deviance(y, mu)
    coef = None
    for _ in range(maxit):
        mu_eta = _mu_eta(eta)
        z = eta + (y - mu) / mu_eta
        w = np.sqrt(mu_eta ** 2 / (mu * (1 - mu)))
        coef = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)[0]
        eta = A @ coef
        mu = _linkinv(eta)
        dev = _deviance(y, mu)
        if abs(dev - dev_old) / (abs(dev) + 0.1) < epsilon:
            break
        dev_old = dev
    return coef


def glm_predict(coef, X):
    return _linkinv(coef[0] + np.atleast_2d(X) @ coef[1:])


## ---- expanding_meta_cv ----
def expanding_meta_cv(legs, initial_train_ratio=0.60):
    """Probabilité de gain de chaque jambe, classifieur ré-estimé sur les jambes passées."""
    out = []
    for side in ["Long", "Short"]:
        d = legs[legs["Side"] == side].sort_values("Date", kind="stable").reset_index(drop=True)
        n = len(d)
        X, y = d[META_X].values, d["Win"].values
        for i in range(int(np.floor(initial_train_ratio * n)), n):
            coef = glm_logit(X[:i], y[:i])
            out.append((d.at[i, "Date"], d.at[i, "Pair"], side,
                        float(glm_predict(coef, X[i])[0]), d.at[i, "Win"], d.at[i, "PnL"]))
    return pd.DataFrame(out, columns=["Date", "Pair", "Side", "Probability", "Win", "PnL"])


## ---- threshold_diagnostics ----
def threshold_diagnostics(meta_oof, thresholds=np.arange(0.50, 0.65 + 1e-9, 0.025)):
    rows = []
    for side in ["Long", "Short"]:
        d = meta_oof[meta_oof["Side"] == side]
        for th in thresholds:
            acc, rej = d[d["Probability"] >= th], d[d["Probability"] < th]
            rows.append({
                "Side": side, "Threshold": round(float(th), 10),
                "Accepted_N": len(acc),
                "Accepted_Hit": acc["Win"].mean() if len(acc) else np.nan,
                "Accepted_AvgPnL": acc["PnL"].mean() if len(acc) else np.nan,
                "Accepted_TotalPnL": acc["PnL"].sum(),
                "Rejected_N": len(rej),
                "Rejected_Hit": rej["Win"].mean() if len(rej) else np.nan,
                "Rejected_AvgPnL": rej["PnL"].mean() if len(rej) else np.nan,
                "Rejected_TotalPnL": rej["PnL"].sum(),
            })
    return pd.DataFrame(rows)


## ---- meta_expanding_filtered_returns ----
def meta_expanding_filtered_returns(pred, actual, validation_legs, long_threshold,
                                    short_threshold, funding_weekly, skip_all_positive=False):
    """Chaque jambe est gardée si sa probabilité de gain (classifieur ré-estimé
    sur la validation + les jambes OOS passées) atteint le seuil. Une seule
    jambe gardée : on paie (long) ou reçoit (short) le financement DFF."""
    legs = strategy_legs(pred, actual, skip_all_positive)
    legs["Funding_weekly"] = funding_weekly.reindex(legs["Date"]).values
    if legs["Funding_weekly"].isna().any():
        raise ValueError("Missing funding rate in OOS strategy data.")
    legs["Probability"] = np.nan
    legs["Accept"] = False
    returns = pd.Series(0.0, index=pred.index)
    thresholds = {"Long": long_threshold, "Short": short_threshold}

    for date_t in np.sort(legs["Date"].unique()):
        # Expanding : validation + OOS passé
        train = pd.concat([validation_legs[META_COLS], legs.loc[legs["Date"] < date_t, META_COLS]])
        pnl = {}
        for side in ["Long", "Short"]:
            tr = train[train["Side"] == side]
            coef = glm_logit(tr[META_X].values, tr["Win"].values)
            row = (legs["Date"] == date_t) & (legs["Side"] == side)
            if row.any():
                prob = glm_predict(coef, legs.loc[row, META_X].values)
                legs.loc[row, "Probability"] = prob
                legs.loc[row, "Accept"] = prob >= thresholds[side]
                if legs.loc[row, "Accept"].iloc[0]:
                    pnl[side] = (legs.loc[row, "PnL"].iloc[0], legs.loc[row, "Funding_weekly"].iloc[0])
        if "Long" in pnl and "Short" in pnl:  # deux jambes : autofinancé
            returns[date_t] = pnl["Long"][0] + pnl["Short"][0]
        elif "Long" in pnl:  # long seul : paie le funding
            returns[date_t] = pnl["Long"][0] - pnl["Long"][1]
        elif "Short" in pnl:  # short seul : reçoit le funding
            returns[date_t] = pnl["Short"][0] + pnl["Short"][1]
    return {"returns": returns, "decisions": legs}


## ---- performance_summary ----
def performance_summary(returns, periods_per_year=52):
    """returns : DataFrame de log-returns hebdomadaires, une colonne par stratégie."""
    rows = []
    for name in returns.columns:
        r = returns[name].values
        wealth = np.exp(np.cumsum(r))
        peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
        sd = np.std(r, ddof=1)
        rows.append({
            "Strategy": name,
            "Cumulative_return": np.exp(r.sum()) - 1,
            "Ann_return": r.mean() * periods_per_year,
            "Ann_vol": sd * np.sqrt(periods_per_year),
            "Sharpe": r.mean() / sd * np.sqrt(periods_per_year),
            "Max_drawdown": np.min(wealth / peak - 1),
            "Hit_rate": np.mean(r[r != 0] > 0),
            "Weeks_traded": np.mean(r != 0),
        })
    return pd.DataFrame(rows)


## ---- random_strategy_returns ----
def random_strategy_returns(actual, n_sim=1000, seed=123):
    """n_sim stratégies au hasard : long une paire, short une autre, tirées au
    sort chaque semaine (mêmes tirages que sample() de R après set.seed(seed))."""
    rng = RRandom(seed)
    a = actual.values
    n, k = a.shape
    sims = np.empty((n, n_sim))
    rows = np.arange(n)
    for s in range(n_sim):
        picks = np.array([rng.sample_int(k, 2) for _ in range(n)]) - 1
        sims[:, s] = a[rows, picks[:, 0]] - a[rows, picks[:, 1]]
    return pd.DataFrame(sims, index=actual.index)


## ---- leg_summary ----
def classifier_summary(decisions, side):
    d = decisions[decisions["Side"] == side]
    acc, rej = d[d["Accept"]], d[~d["Accept"]]
    return pd.DataFrame({
        "Group": ["Accepted", "Rejected"],
        "N": [len(acc), len(rej)],
        "Hit_Rate": [acc["Win"].mean(), rej["Win"].mean()],
        "Avg_PnL": [acc["PnL"].mean(), rej["PnL"].mean()],
        "Total_PnL": [acc["PnL"].sum(), rej["PnL"].sum()],
    })


## ---- plots ----
def plot_random_equity(random_returns, model_returns, title, colors, path):
    """Gauche : les courbes au hasard, leurs percentiles 5 / 50 / 95 % et les
    stratégies. Droite : distribution de la valeur finale au hasard."""
    import matplotlib.pyplot as plt

    dates = random_returns.index
    rand_eq = np.exp(np.cumsum(random_returns.values, axis=0))
    model_eq = np.exp(np.cumsum(model_returns.values, axis=0))
    bands = np.quantile(rand_eq, [0.05, 0.5, 0.95], axis=1).T
    lo, hi = min(rand_eq.min(), model_eq.min()), max(rand_eq.max(), model_eq.max())

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(10.6, 6.5), sharey=True,
                                  gridspec_kw={"width_ratios": [4, 1], "wspace": 0.03})
    ax.plot(dates, rand_eq, color="0.6", alpha=0.08, lw=0.6)
    ax.plot(dates, bands[:, 1], color="0.25", lw=1.2, label="Hasard : médiane")
    ax.plot(dates, bands[:, 0], color="0.25", lw=1.2, ls="--", label="Hasard : 5 % / 95 %")
    ax.plot(dates, bands[:, 2], color="0.25", lw=1.2, ls="--")
    for j, name in enumerate(model_returns.columns):
        ax.plot(dates, model_eq[:, j], color=colors[j], lw=2, label=name)
    ax.axhline(1, color="0.4", ls=":")
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur de 1 $ investi")
    handles, labels = ax.get_legend_handles_labels()
    order = list(range(2, len(labels))) + [0, 1]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
              frameon=False, fontsize=8)

    final = rand_eq[-1]
    axh.hist(final, bins=40, orientation="horizontal", color="0.8", edgecolor="0.6")
    for q, ls in zip(np.quantile(final, [0.05, 0.5, 0.95]), ["--", "-", "--"]):
        axh.axhline(q, color="0.25", ls=ls)
    for j in range(model_eq.shape[1]):
        axh.axhline(model_eq[-1, j], color=colors[j], lw=2)
    axh.axhline(1, color="0.4", ls=":")
    axh.set_title("Valeur finale")
    axh.set_xlabel("Fréquence")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_all_methods(all_returns, title, path):
    import matplotlib.pyplot as plt

    palette = ["grey", "steelblue", "forestgreen", "firebrick", "darkorange",
               "purple", "goldenrod", "deeppink", "saddlebrown"]
    eq = np.exp(all_returns.cumsum())
    fig, ax = plt.subplots(figsize=(10.6, 6.5))
    for j, name in enumerate(eq.columns):
        meta = name == "MIXT_Meta"
        ax.plot(eq.index, eq[name], color="black" if meta else palette[j % len(palette)],
                lw=2.5 if meta else 1.3, label=name)
    ax.axhline(1, color="0.4", ls=":")
    ax.set_title(title)
    ax.set_ylabel("Valeur de 1 $ investi")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_nav(returns, labels, colors, path, title=None):
    """NAV (base 100) de chaque stratégie, style du fact sheet."""
    import matplotlib.pyplot as plt

    primary, text_grey, rule_grey = "#0B2E59", "#58677A", "#D7DFE5"
    nav = 100 * np.exp(np.cumsum(returns.values, axis=0))
    nav = np.vstack([np.full(nav.shape[1], 100.0), nav])
    weeks = np.arange(len(nav))
    fig, ax = plt.subplots(figsize=(10.6, 6.5))
    for j, label in enumerate(labels):
        ax.plot(weeks, nav[:, j], color=colors[j], lw=1.6, label=label)
    ax.set_xlim(0, weeks[-1] * 1.01)
    ax.grid(color=rule_grey, lw=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(text_grey)
    ax.tick_params(colors=text_grey, labelsize=14)
    ax.set_ylabel("NAV", color=text_grey, fontsize=14)
    if title:
        ax.set_title(title, color=primary, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(labels), frameon=False,
              fontsize=14, labelcolor=text_grey)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
