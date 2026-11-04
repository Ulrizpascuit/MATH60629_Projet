"""Lance la stratégie au complet à partir des données préparées par run_data.py :
  1. Code/build_model.py : estime les modèles en fenêtre croissante (en parallèle)
  2. Code/backtest.py    : backtest de la stratégie sur les 260 semaines de test
Tous les tableaux (.csv) et graphiques (.png) sont enregistrés dans outputs/.

Option --skip-models : refait seulement le backtest à partir de Data/Models/models.pkl."""
import sys

from Code import backtest, build_model
from Function.functions import OUTPUTS


def main(skip_models=False):
    OUTPUTS.mkdir(exist_ok=True)
    if skip_models:
        import pickle
        from Function.functions import MODELS
        with open(MODELS / "models.pkl", "rb") as f:
            models = pickle.load(f)
    else:
        models = build_model.main()
    result = backtest.run_backtest()

    validation_r2 = models["validation_r2"].rename_axis("Method").reset_index()
    tables = {
        # Modèles (build_model.py)
        "validation_r2": validation_r2,
        "mixt_selection": models["mixt_selection"],
        "return_rmse": models["rmse_comparison"],
        "sharpe_rmse": models["sharpe_rmse_comparison"],
        "sharpe_mae": models["sharpe_mae_comparison"],
        "sharpe_r2": models["sharpe_r2_comparison"],
        "volatility_metrics": models["vol_metrics"],
        "threshold_diagnostics": models["threshold_table"],
        # Backtest (backtest.py)
        **result["tables"],
    }
    for name, table in tables.items():
        table.to_csv(OUTPUTS / f"{name}.csv", index=False)
    print(f"Résultats enregistrés dans outputs/ : {len(tables)} tableaux et 3 graphiques.")
    print(result["tables"]["performance_final"].to_string(index=False))


if __name__ == "__main__":
    main(skip_models="--skip-models" in sys.argv)
