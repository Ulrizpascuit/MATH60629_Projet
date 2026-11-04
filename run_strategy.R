## ---- run_strategy ----
# Lance la stratégie au complet à partir des données préparées par run_data.R :
#   1. Code/build_model.R : estime les modèles en fenêtre croissante
#      (environ 5 minutes, calcul en parallèle)
#   2. Code/backtest.R    : backtest de la stratégie sur les 260 semaines de test
# Tous les tableaux (.csv) et graphiques (.png) sont enregistrés dans outputs/.
library("here")

dir.create(here("outputs"), showWarnings = FALSE)

## ---- models ----
source(here("Code", "build_model.R"))

## ---- backtest ----
# Les graphiques de backtest.R sont envoyés dans des fichiers PNG, un par
# graphique, dans l'ordre où ils sont tracés.
plot_names <- c(
  "equity_vs_random",
  "equity_all_methods",
  "nav_one_pager"
)

temp_plots <- here("outputs", "plot_%02d.png")
png(temp_plots, width = 1800, height = 1100, res = 170)
source(here("Code", "backtest.R"))
dev.off()

plot_files <- sprintf(temp_plots, seq_along(plot_names))
stopifnot(all(file.exists(plot_files)))
invisible(file.rename(plot_files, here("outputs", paste0(plot_names, ".png"))))

## ---- tables ----
# Rendements annuels de la stratégie finale, sous forme de tableau
yearly_table <- data.frame(
  Year = names(yearly_returns),
  Return = as.numeric(yearly_returns)
)

tables <- list(
  # Modèles (build_model.R)
  validation_r2          = data.frame(Method = rownames(expanding_validation_r2), expanding_validation_r2),
  mixt_selection         = expanding_mixt_selection,
  return_rmse            = expanding_rmse_comparison,
  sharpe_rmse            = expanding_sharpe_rmse_comparison,
  sharpe_mae             = expanding_sharpe_mae_comparison,
  sharpe_r2              = expanding_sharpe_r2_comparison,
  volatility_metrics     = expanding_vol_metrics,
  threshold_diagnostics  = threshold_table,
  # Backtest (backtest.R)
  performance_final      = final_performance,
  performance_classifier = classifier_performance,
  classifier_long        = long_classifier_summary,
  classifier_short       = short_classifier_summary,
  random_percentile      = random_comparison,
  one_pager_metrics      = one_pager_metrics,
  yearly_returns         = yearly_table,
  weekly_returns         = data.frame(Date = index(final_returns), coredata(final_returns))
)

for (name in names(tables)) {
  write.csv(
    tables[[name]],
    here("outputs", paste0(name, ".csv")),
    row.names = FALSE
  )
}

cat("Résultats enregistrés dans outputs/ :", length(tables), "tableaux et",
    length(plot_names), "graphiques.\n")
