library("here")
library("xts")

source(here("Function", "functions_model.R"))
source(here("Function", "functions_strategy.R"))

## ---- load_processed_data ----
load(here("Data", "Processed", "forex.RData"))
load(here("Data", "Processed", "realized_vol.RData"))
load(here("Data", "Features", "merged_features.RData"))
## ---- align_targets ----
fx_targets <- na.omit(xts::lag.xts(fx_returns, k = -1))

# Toutes les features couvrent la même période : on aligne returns et targets dessus.
features_list  <- list(eur_features, gbp_features, cad_features, chf_features, jpy_features)
features_start <- max(do.call(c, lapply(features_list, start)))
features_end   <- min(do.call(c, lapply(features_list, end)))
fx_returns <- fx_returns[paste0(features_start, "/", features_end)]
fx_targets <- fx_targets[paste0(features_start, "/", features_end)]

# Volatilité réalisée de la semaine t -> t+1, décalée comme fx_targets
fx_vol_targets <- na.omit(xts::lag.xts(fx_realized_vol, k = -1))
fx_vol_targets <- fx_vol_targets[paste0(features_start, "/", features_end)]

## ---- train_test_split ----
# Test final OOS = exactement 5 ans hebdomadaires
test_weeks <- 260

n_weeks <- nrow(na.omit(merge(
      fx_targets[, "EURUSD"],
      eur_features,
      join = "inner"))
)

stopifnot(
  n_weeks > test_weeks
)

# Le +0.5 garantit que floor(train_ratio * n_weeks)
# donne exactement n_weeks - 260
train_ratio <- (n_weeks - test_weeks + 0.5) / n_weeks

cat("Entraînement :",n_weeks - test_weeks,"semaines - Test :",test_weeks,"semaines\n")

## ---- walk_forward ----
# Chaque méthode est ré-estimée chaque semaine sur toutes les données
# disponibles (fenêtre croissante).
# Une tâche par méthode x paire, en parallèle (quelques minutes).
features_by_pair <- list(
  EURUSD = eur_features,
  GBPUSD = gbp_features,
  CADUSD = cad_features,
  CHFUSD = chf_features,
  JPYUSD = jpy_features
)

cluster <- parallel::makeCluster(max(1, parallel::detectCores() - 1))
invisible(parallel::clusterEvalQ(cluster, library("xts")))
invisible(parallel::clusterCall(cluster, source, here("Function", "functions_model.R")))

## ---- check_common_dates ----

pair_dates <- lapply(
  names(features_by_pair),
  function(pair) {
    aligned <- na.omit(
      merge(fx_targets[, pair],
        features_by_pair[[pair]],
        join = "inner")
    )
    as.Date(index(aligned))
  }
)

names(pair_dates) <- names(features_by_pair)
reference_dates <- pair_dates[[1]]

same_dates <- vapply( pair_dates,
  function(x) {
    identical(x,reference_dates)
  },
  logical(1)
)

print(same_dates)
stopifnot(all(same_dates))

## ---- expanding_models (test) ----
# Les 260 semaines de test
expanding_models <- build_walk_forward_models(
  features_by_pair, fx_targets, period = "test",
  train_ratio = train_ratio, cluster = cluster
)

## ---- expanding_validation_models ----
# Dans l'entraînement (après les premiers 60 %) :
# sert à choisir le modèle mixte et à entraîner la classification
expanding_validation_models <- build_walk_forward_models(
  features_by_pair, fx_targets, period = "validation",
  train_ratio = train_ratio, cluster = cluster
)

## ---- walk_forward_garch ----
# GARCH(1,1) ré-estimé chaque semaine, comme les modèles de return :
# même fenêtre croissante et mêmes semaines (test / validation)
expanding_garch_models <- build_walk_forward_garch_models(
  fx_targets, fx_vol_targets, expanding_models$Linear, cluster = cluster
)

expanding_validation_garch_models <- build_walk_forward_garch_models(
  fx_targets, fx_vol_targets, expanding_validation_models$Linear, cluster = cluster
)

parallel::stopCluster(cluster)

# Semaines où le GARCH n'a pas convergé (paramètres de la semaine d'avant)
print(sapply(
  list(
    Expanding = expanding_garch_models,
    Expanding_validation = expanding_validation_garch_models
  ),
  function(models) sapply(models, function(m) sum(m$refit_failed))
))

## ---- expanding_validation_r2 ----
# R2 de validation : une ligne par méthode, une colonne par paire
expanding_validation_r2 <- t(sapply(
  expanding_validation_models,
  function(models) sapply(models, function(m) m$r2)
))
print(round(expanding_validation_r2, 4))

expanding_best_models <- apply(expanding_validation_r2, 2, function(x) names(which.max(x)))

## ---- mixt_model expanding ----
# Modèle mixte : pour chaque paire, la méthode avec le meilleur R2 de validation,
# ré-estimée chaque semaine du test
expanding_models$MIXT <- build_mixt_models(expanding_models, expanding_best_models)

expanding_mixt_selection <- data.frame(
  Currency = names(expanding_best_models),
  Selected_Model = unname(expanding_best_models),
  Validation_R2 = apply(expanding_validation_r2, 2, max),
  row.names = NULL
)
print(expanding_mixt_selection)

## ---- sharpe_models expanding ----
# Forecast de volatilité : GARCH ré-estimé chaque semaine (fenêtre croissante)
expanding_sharpe_models <- lapply(expanding_models, build_sharpe_models, garch_models = expanding_garch_models)

## ---- classification_training expanding ----
# Un classifieur par méthode (MIXT inclus), entraîné sur ses prédictions de
# validation (fenêtre croissante), avec le GARCH ré-estimé chaque semaine
# (expanding_validation_garch_models).
expanding_validation_models$MIXT <- build_mixt_models(expanding_validation_models, expanding_best_models)

expanding_validation_sharpe_models <- lapply(
  expanding_validation_models, build_sharpe_models, garch_models = expanding_validation_garch_models
)

## ---- meta_threshold_diagnostics ----
# Diagnostic seulement : ce tableau montre ce que donneraient d'autres seuils
# sur la validation. Il ne sert pas à choisir les seuils (voir final_meta_rule).

validation_pred <- model_matrix(
  expanding_validation_sharpe_models$MIXT,
  element = "predictions"
)

validation_actual <- model_matrix(
  expanding_validation_sharpe_models$MIXT,
  element = "return_actual"
)

validation_legs <- strategy_legs(
  validation_pred,
  validation_actual
)

# Expanding meta-CV sur les jambes de validation
meta_oof <- expanding_meta_cv(validation_legs)

threshold_table <- threshold_diagnostics(
  meta_oof,
  thresholds = seq(0.35, 0.65, by = 0.025)
)

print(threshold_table, digits = 4, row.names = FALSE)

## ---- final_meta_rule ----
# Seuils fixés a priori à 50 % : une jambe est gardée si sa probabilité de gain
# estimée est d'au moins 50 %. Ils ne sont pas optimisés (trop peu de trades de
# validation pour choisir un seuil de façon fiable).
long_meta_threshold  <- 0.5
short_meta_threshold <- 0.5

## ---- compare_vol_metrics ----
# Erreur du forecast de volatilité (vs volatilité réalisée)
expanding_vol_metrics <- data.frame(
  Currency = names(expanding_garch_models),
  RMSE = sapply(expanding_garch_models, function(x) x$rmse),
  MAE = sapply(expanding_garch_models, function(x) x$mae),
  R2 = sapply(expanding_garch_models, function(x) x$r2),
  row.names = NULL
)
print(expanding_vol_metrics)

## ---- compare_expanding_metrics ----
# Tableaux d'erreur des forecasts de return et de Sharpe
expanding_rmse_comparison        <- metric_table(expanding_models, "rmse")
expanding_sharpe_rmse_comparison <- metric_table(expanding_sharpe_models, "rmse")
expanding_sharpe_mae_comparison  <- metric_table(expanding_sharpe_models, "mae")
expanding_sharpe_r2_comparison   <- metric_table(expanding_sharpe_models, "r2")
print(expanding_rmse_comparison)
print(expanding_sharpe_rmse_comparison)
print(expanding_sharpe_mae_comparison)
print(expanding_sharpe_r2_comparison)

## ---- save_models ----
# Les Sharpe remplacent les prédictions de return (gardées dans chaque
# élément : return_predictions / return_actual)
save(
  test_weeks,
  train_ratio,
  expanding_sharpe_models,
  expanding_garch_models,
  expanding_validation_models,
  expanding_validation_r2,
  expanding_mixt_selection,
  expanding_validation_sharpe_models,
  long_meta_threshold,
  short_meta_threshold,
  threshold_table,
  expanding_rmse_comparison,
  expanding_sharpe_rmse_comparison,
  expanding_sharpe_mae_comparison,
  expanding_sharpe_r2_comparison,
  expanding_vol_metrics,
  file = here("Data", "Models", "models.RData")
)
