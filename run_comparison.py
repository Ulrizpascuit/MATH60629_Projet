"""Compare les deux versions de la stratégie à partir des modèles déjà estimés
(Data/Models/models.pkl, créé par run_strategy.py) :
  actuelle : rien si tous les Sharpe prévus sont négatifs
  corrigée : rien non plus s'ils sont tous positifs
Les tableaux et graphiques de la stratégie corrigée vont dans outputs/corrected/
(mêmes noms de fichiers que outputs/), la comparaison dans outputs/."""
import numpy as np
import pandas as pd

from Code.backtest import run_backtest
from Function.functions import OUTPUTS
from Function.functions_strategy import (
    expanding_meta_cv, performance_summary, plot_nav, threshold_diagnostics,
)


def main():
    out_dir = OUTPUTS / "corrected"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stratégie actuelle : ses graphiques sont déjà dans outputs/
    current = run_backtest(skip_all_positive=False, make_plots=False)
    corrected = run_backtest(skip_all_positive=True, out_dir=out_dir)

    ## ---- corrected_tables ----
    tables = dict(corrected["tables"])
    tables["threshold_diagnostics"] = threshold_diagnostics(
        expanding_meta_cv(corrected["validation_legs"]),
        thresholds=np.arange(0.35, 0.65 + 1e-9, 0.025),
    )
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    ## ---- signal_weeks ----
    pred = current["mixt_pred"].values
    signal_weeks = pd.DataFrame({
        "Case": ["Tous negatifs", "Tous positifs", "Signes mixtes"],
        "Weeks": [int((pred.max(axis=1) <= 0).sum()), int((pred.min(axis=1) >= 0).sum()),
                  int(((pred.max(axis=1) > 0) & (pred.min(axis=1) < 0)).sum())],
    })

    ## ---- strategy_comparison ----
    comparison_returns = pd.concat({
        "MIXT_Meta_actuelle": current["mixt_meta_returns"],
        "MIXT_Meta_corrigee": corrected["mixt_meta_returns"],
        "MIXT_sans_meta_actuelle": current["mixt_raw_returns"],
        "MIXT_sans_meta_corrigee": corrected["mixt_raw_returns"],
        "Naive_actuelle": current["naive_returns"],
        "Naive_corrigee": corrected["naive_returns"],
    }, axis=1)
    comparison = performance_summary(comparison_returns)
    # Percentile par rapport aux 1000 stratégies au hasard (les mêmes dans les deux cas)
    random_totals = current["random_returns"].sum(axis=0).values
    comparison["Random_Percentile"] = [
        np.mean(random_totals < comparison_returns[c].sum()) for c in comparison_returns
    ]

    yearly = pd.DataFrame({
        "Year": current["tables"]["yearly_returns"]["Year"],
        "Actuelle": current["tables"]["yearly_returns"]["Return"].values,
        "Corrigee": corrected["tables"]["yearly_returns"]["Return"].values,
    })

    legs = []
    for name, run in [("Actuelle", current), ("Corrigee", corrected)]:
        d = run["mixt_meta"]["decisions"]
        acc = d[d["Accept"]]
        legs.append({"Strategy": name, "Legs": len(d), "Accepted": int(d["Accept"].sum()),
                     "Accepted_Hit_Rate": acc["Win"].mean(), "Accepted_Total_PnL": acc["PnL"].sum()})
    legs = pd.DataFrame(legs)

    for name, table in [("signal_weeks", signal_weeks), ("performance", comparison),
                        ("yearly_returns", yearly), ("legs", legs)]:
        print(f"\n{name}\n{table.to_string(index=False)}")
        table.to_csv(OUTPUTS / f"comparison_{name}.csv", index=False)

    plot_nav(comparison_returns[["MIXT_Meta_actuelle", "MIXT_Meta_corrigee"]],
             ["Actuelle", "Corrigée"], ["#0B2E59", "#16A89A"], OUTPUTS / "comparison_nav.png")
    print("Stratégie corrigée : outputs/corrected/ - comparaison : outputs/comparison_*")


if __name__ == "__main__":
    main()
