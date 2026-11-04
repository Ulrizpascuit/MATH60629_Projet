"""Backtest de la stratégie sur les 260 semaines de test (à partir de
Data/Models/models.pkl) : tableaux et graphiques dans outputs/.

skip_all_positive = True donne la stratégie « corrigée » : on ne fait rien
non plus quand tous les Sharpe prévus sont positifs."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from Function.functions import MODELS, OUTPUTS, PROCESSED, load_frame
from Function.functions_strategy import (
    classifier_summary, meta_expanding_filtered_returns, model_matrix,
    performance_summary, plot_all_methods, plot_nav, plot_random_equity,
    random_strategy_returns, strategy_legs, strategy_returns,
)


def naive_last_vol_pred(models, fx_realized_vol):
    """Sharpe prévu du Naive : return de la semaine passée divisé par la
    volatilité réalisée de la semaine passée (il n'estime rien)."""
    out = {}
    for pair, m in models.items():
        last_vol = fx_realized_vol[pair].reindex(m["dates"]).values
        assert not np.isnan(last_vol).any() and (last_vol > 0).all()
        out[pair] = pd.Series(m["return_predictions"] / last_vol, index=m["dates"])
    return pd.concat(out, axis=1)


def run_backtest(skip_all_positive=False, out_dir=OUTPUTS, make_plots=True):
    with open(MODELS / "models.pkl", "rb") as f:
        models = pickle.load(f)
    funding = load_frame(PROCESSED / "funding.csv")
    fx_realized_vol = load_frame(PROCESSED / "realized_vol.csv")
    dff_weekly = funding["Funding_weekly"]
    sharpe_models = models["sharpe_models"]
    skip = skip_all_positive

    ## ---- actual / predictions ----
    actual = model_matrix(sharpe_models["MIXT"], "return_actual")
    assert len(actual) == 260
    mixt_pred = model_matrix(sharpe_models["MIXT"], "predictions")

    ## ---- mixt_without_classifier ----
    mixt_raw_returns = strategy_returns(mixt_pred, actual, skip).rename("MIXT_without_meta")

    ## ---- mixt_final ----
    validation = models["validation_sharpe_models"]["MIXT"]
    validation_legs = strategy_legs(
        model_matrix(validation, "predictions"), model_matrix(validation, "return_actual"), skip
    )
    mixt_meta = meta_expanding_filtered_returns(
        mixt_pred, actual, validation_legs, models["long_meta_threshold"],
        models["short_meta_threshold"], dff_weekly, skip,
    )
    mixt_meta_returns = mixt_meta["returns"].rename("MIXT_Meta")

    ## ---- naive_benchmark ----
    naive_pred = naive_last_vol_pred(sharpe_models["Naive"], fx_realized_vol)
    assert naive_pred.index.equals(actual.index)
    naive_returns = strategy_returns(naive_pred, actual, skip).rename("Naive (Benchmark)")

    ## ---- performance ----
    final_returns = pd.concat([mixt_meta_returns, naive_returns], axis=1)
    final_performance = performance_summary(final_returns)
    classifier_performance = performance_summary(
        pd.concat([mixt_raw_returns, mixt_meta_returns], axis=1)
    )
    long_summary = classifier_summary(mixt_meta["decisions"], "Long")
    short_summary = classifier_summary(mixt_meta["decisions"], "Short")

    ## ---- random_benchmark ----
    random_returns = random_strategy_returns(actual, n_sim=1000, seed=123)
    random_totals = random_returns.sum(axis=0).values
    random_comparison = pd.DataFrame({
        "Strategy": ["MIXT Meta", "Naive"],
        "Random_Percentile": [np.mean(random_totals < mixt_meta_returns.sum()),
                              np.mean(random_totals < naive_returns.sum())],
    })

    ## ---- all_methods ----
    all_returns = {}
    for method, m in sharpe_models.items():
        pred = naive_last_vol_pred(m, fx_realized_vol) if method == "Naive" else model_matrix(m)
        all_returns[method] = strategy_returns(pred, model_matrix(m, "return_actual"), skip)
    all_returns = pd.DataFrame(all_returns)
    all_returns["MIXT_Meta"] = mixt_meta_returns

    ## ---- one_pager_metrics ----
    perf = performance_summary(mixt_meta_returns.to_frame()).iloc[0]
    risk_free = dff_weekly.reindex(mixt_meta_returns.index).values
    assert not np.isnan(risk_free).any()
    excess = mixt_meta_returns.values - risk_free
    one_pager_metrics = pd.DataFrame({
        "Metric": ["Annualized return", "Annualized volatility", "Annualized excess return",
                   "Sharpe Ratio", "Sharpe Ratio (excess return)", "Max Drawdown", "Hit Rate",
                   "Cumulative return"],
        "Value": [f"{100 * perf.Ann_return:.2f}%", f"{100 * perf.Ann_vol:.2f}%",
                  f"{100 * excess.mean() * 52:.2f}%", f"{perf.Sharpe:.2f}",
                  f"{excess.mean() / excess.std(ddof=1) * np.sqrt(52):.2f}",
                  f"{100 * perf.Max_drawdown:.2f}%", f"{100 * perf.Hit_rate:.2f}%",
                  f"{100 * perf.Cumulative_return:+.1f}%"],
    })

    ## ---- yearly_returns ----
    # Le return daté du jeudi t est réalisé de t à t+1 : compté dans l'année du jeudi suivant
    realized_year = (mixt_meta_returns.index + pd.Timedelta(days=7)).year.astype(str)
    yearly = np.exp(mixt_meta_returns.groupby(realized_year).sum()) - 1
    yearly_returns = pd.DataFrame({"Year": yearly.index, "Return": yearly.values})

    ## ---- plots ----
    if make_plots:
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = " corrigée" if skip else ""
        plot_random_equity(random_returns, final_returns,
                           f"Strategy 1{suffix} - Expanding OOS (260 Weeks)",
                           ["navy", "darkorange"], out_dir / "equity_vs_random.png")
        plot_all_methods(all_returns, "Equity curves - Expanding window"
                         + (" (stratégie corrigée)" if skip else ""),
                         out_dir / "equity_all_methods.png")
        plot_nav(final_returns, ["GRUV Alpha", "Benchmark"], ["#0B2E59", "#16A89A"],
                 out_dir / "nav_one_pager.png")

    return {
        "actual": actual, "mixt_pred": mixt_pred, "mixt_meta": mixt_meta,
        "mixt_meta_returns": mixt_meta_returns, "mixt_raw_returns": mixt_raw_returns,
        "naive_returns": naive_returns, "final_returns": final_returns,
        "random_returns": random_returns, "validation_legs": validation_legs,
        "tables": {
            "performance_final": final_performance,
            "performance_classifier": classifier_performance,
            "classifier_long": long_summary,
            "classifier_short": short_summary,
            "random_percentile": random_comparison,
            "one_pager_metrics": one_pager_metrics,
            "yearly_returns": yearly_returns,
            "weekly_returns": final_returns.rename_axis("Date").reset_index(),
        },
    }


if __name__ == "__main__":
    res = run_backtest()
    for name, table in res["tables"].items():
        if name != "weekly_returns":
            print(f"\n{name}\n{table.to_string(index=False)}")
