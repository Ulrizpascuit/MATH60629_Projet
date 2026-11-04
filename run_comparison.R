## ---- run_comparison ----
# Compare les deux versions de la stratégie à partir des modèles déjà estimés
# (Data/Models/models.RData, créé par run_strategy.R) :
#   actuelle : Code/backtest.R           (rien si tous les Sharpe prévus sont négatifs)
#   corrigée : Code/backtest_corrected.R (rien non plus s'ils sont tous positifs)
# Les tableaux et graphiques de la stratégie corrigée vont dans outputs/corrected/
# (mêmes noms de fichiers que outputs/), la comparaison dans outputs/.
library("here")
library("xts")

out_dir <- here("outputs", "corrected")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

## ---- current ----
# Stratégie actuelle : ses graphiques sont déjà dans outputs/, on ne les refait pas
current <- new.env()
pdf(NULL)
source(here("Code", "backtest.R"), local = current)
dev.off()

## ---- corrected ----
plot_names <- c(
  "equity_vs_random",
  "equity_all_methods",
  "nav_one_pager"
)

corrected <- new.env()
temp_plots <- file.path(out_dir, "plot_%02d.png")
png(temp_plots, width = 1800, height = 1100, res = 170)
source(here("Code", "backtest_corrected.R"), local = corrected)
dev.off()

plot_files <- sprintf(temp_plots, seq_along(plot_names))
stopifnot(all(file.exists(plot_files)))
invisible(file.rename(plot_files, file.path(out_dir, paste0(plot_names, ".png"))))

## ---- corrected_tables ----
# Diagnostic des seuils refait sur les jambes de validation de la stratégie corrigée
corrected_threshold_table <- threshold_diagnostics(
  expanding_meta_cv(corrected$validation_legs),
  thresholds = seq(0.35, 0.65, by = 0.025)
)

corrected_tables <- list(
  threshold_diagnostics  = corrected_threshold_table,
  performance_final      = corrected$final_performance,
  performance_classifier = corrected$classifier_performance,
  classifier_long        = corrected$long_classifier_summary,
  classifier_short       = corrected$short_classifier_summary,
  random_percentile      = corrected$random_comparison,
  one_pager_metrics      = corrected$one_pager_metrics,
  yearly_returns         = data.frame(
    Year = names(corrected$yearly_returns),
    Return = as.numeric(corrected$yearly_returns)
  ),
  weekly_returns         = data.frame(
    Date = index(corrected$final_returns),
    coredata(corrected$final_returns)
  )
)

for (name in names(corrected_tables)) {
  write.csv(
    corrected_tables[[name]],
    file.path(out_dir, paste0(name, ".csv")),
    row.names = FALSE
  )
}

## ---- signal_weeks ----
# Semaines OOS selon le signe des Sharpe prévus du MIXT
pred <- coredata(current$mixt_pred)
signal_weeks <- data.frame(
  Case = c("Tous negatifs", "Tous positifs", "Signes mixtes"),
  Weeks = c(
    sum(apply(pred, 1, max) <= 0),
    sum(apply(pred, 1, min) >= 0),
    sum(apply(pred, 1, max) > 0 & apply(pred, 1, min) < 0)
  )
)
print(signal_weeks, row.names = FALSE)

## ---- strategy_comparison ----
comparison_returns <- merge(
  current$mixt_meta_returns,
  corrected$mixt_meta_returns,
  current$mixt_raw_returns,
  corrected$mixt_raw_returns,
  current$naive_returns,
  corrected$naive_returns
)
colnames(comparison_returns) <- c(
  "MIXT_Meta_actuelle", "MIXT_Meta_corrigee",
  "MIXT_sans_meta_actuelle", "MIXT_sans_meta_corrigee",
  "Naive_actuelle", "Naive_corrigee"
)

strategy_comparison <- performance_summary(comparison_returns)

# Percentile par rapport aux 1000 stratégies aléatoires (les mêmes dans les deux cas)
random_totals <- colSums(current$random_returns)
strategy_comparison$Random_Percentile <- sapply(
  colnames(comparison_returns),
  function(name) mean(random_totals < sum(comparison_returns[, name]))
)

print(strategy_comparison, digits = 4, row.names = FALSE)

# Rendements annuels de la stratégie finale (MIXT_Meta), côte à côte
yearly_comparison <- data.frame(
  Year = names(current$yearly_returns),
  Actuelle = as.numeric(current$yearly_returns),
  Corrigee = as.numeric(corrected$yearly_returns)
)
print(yearly_comparison, digits = 4, row.names = FALSE)

# Jambes gardées par le meta-classifier
legs_comparison <- do.call(rbind, lapply(
  list(Actuelle = current, Corrigee = corrected),
  function(run) {
    d <- run$mixt_meta$decisions
    data.frame(
      Legs = nrow(d),
      Accepted = sum(d$Accept),
      Accepted_Hit_Rate = mean(d$Win[d$Accept]),
      Accepted_Total_PnL = sum(d$PnL[d$Accept])
    )
  }
))
legs_comparison <- data.frame(Strategy = rownames(legs_comparison), legs_comparison, row.names = NULL)
print(legs_comparison, digits = 4, row.names = FALSE)

write.csv(signal_weeks, here("outputs", "comparison_signal_weeks.csv"), row.names = FALSE)
write.csv(strategy_comparison, here("outputs", "comparison_performance.csv"), row.names = FALSE)
write.csv(yearly_comparison, here("outputs", "comparison_yearly_returns.csv"), row.names = FALSE)
write.csv(legs_comparison, here("outputs", "comparison_legs.csv"), row.names = FALSE)

## ---- comparison_plot ----
png(here("outputs", "comparison_nav.png"), width = 1800, height = 1100, res = 170)
print(plot_nav(
  comparison_returns[, c("MIXT_Meta_actuelle", "MIXT_Meta_corrigee")],
  labels = c("Actuelle", "Corrigée"),
  colors = c("#0B2E59", "#16A89A")
))
dev.off()

cat("Stratégie corrigée : outputs/corrected/ - comparaison : outputs/comparison_*\n")
