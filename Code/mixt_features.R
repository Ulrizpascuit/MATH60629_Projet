library("here")
library("xts")

source(here("Function", "functions_model.R"))

## ---- load_data ----
load(here("Data", "Processed", "forex.RData"))
load(here("Data", "Features", "merged_features.RData"))
load(here("Data", "Models", "models.RData"))

## ---- align_targets ----
# Même alignement que build_model.R
fx_targets <- na.omit(xts::lag.xts(fx_returns, k = -1))

features_by_pair <- list(
  EURUSD = eur_features,
  GBPUSD = gbp_features,
  CADUSD = cad_features,
  CHFUSD = chf_features,
  JPYUSD = jpy_features
)

features_start <- max(do.call(c, lapply(features_by_pair, start)))
features_end   <- min(do.call(c, lapply(features_by_pair, end)))
fx_targets <- fx_targets[paste0(features_start, "/", features_end)]

## ---- used_features ----
# Features utilisées par une méthode estimée sur train (mêmes estimations que
# fit_predict_model) : toutes pour Linear et Ridge, coefficients non nuls pour
# Lasso et Elastic_Net, variables retenues par le BIC pour Stepwise.
used_features <- function(train, test, model_type, nfolds = 5) {

  candidates <- setdiff(colnames(train), "target")

  if (model_type == "Linear") {
    model <- lm(target ~ ., data = train)
    return(list(
      used = setNames(rep(TRUE, length(candidates)), candidates),
      prediction = as.numeric(predict(model, newdata = test))
    ))
  }

  if (model_type == "Stepwise") {
    fit <- select_stepwise_model(train, criterion = "BIC")
    return(list(
      used = setNames(candidates %in% fit$selected, candidates),
      prediction = as.numeric(predict(fit$model, newdata = test))
    ))
  }

  if (model_type %in% c("Lasso", "Ridge", "Elastic_Net")) {
    alpha <- switch(model_type, Lasso = 1, Ridge = 0, Elastic_Net = 0.5)
    set.seed(123)
    model <- glmnet::cv.glmnet(
      x = as.matrix(train[, candidates]),
      y = train$target,
      alpha = alpha,
      nfolds = nfolds,
      standardize = TRUE
    )
    beta <- as.matrix(coef(model, s = model$lambda.min))[candidates, 1]
    return(list(
      used = beta != 0,
      prediction = as.numeric(
        predict(model, newx = as.matrix(test[, candidates]), s = model$lambda.min)
      )
    ))
  }

  stop("Méthode sans features identifiables : ", model_type)
}

## ---- walk_forward_features ----
# Pour chaque paire : la méthode du modèle mixte est ré-estimée chaque semaine
# du test, et on note les features utilisées (une ligne par semaine).
mixt_methods <- setNames(
  expanding_mixt_selection$Selected_Model,
  expanding_mixt_selection$Currency
)

mixt_weekly_features <- lapply(names(mixt_methods), function(pair) {

  data <- na.omit(merge(fx_targets[, pair], features_by_pair[[pair]], join = "inner"))
  data_model <- data.frame(
    target = as.numeric(data[, 1]),
    as.data.frame(coredata(data[, -1]))
  )

  n <- nrow(data_model)
  rows <- (floor(train_ratio * n) + 1):n

  fits <- lapply(rows, function(i) {
    used_features(data_model[1:(i - 1), ], data_model[i, ], mixt_methods[[pair]])
  })

  # Mêmes prédictions que le modèle mixte sauvegardé
  saved <- expanding_sharpe_models$MIXT[[pair]]
  stopifnot(
    identical(index(data)[rows], saved$dates),
    isTRUE(all.equal(sapply(fits, function(f) f$prediction), saved$return_predictions))
  )

  used <- do.call(rbind, lapply(fits, function(f) f$used))
  rownames(used) <- as.character(index(data)[rows])
  used
})
names(mixt_weekly_features) <- names(mixt_methods)

## ---- mixt_features_table ----
# % des semaines de test où chaque feature est utilisée.
# Les spreads sont propres à chaque paire : on enlève le préfixe (eur_, gbp_, ...).
usage <- sapply(mixt_weekly_features, function(used) {
  out <- 100 * colMeans(used)
  names(out) <- sub("^[a-z]{3}_spread", "spread", names(out))
  out
})

mixt_features <- data.frame(
  Feature = rownames(usage),
  round(usage, 1),
  row.names = NULL
)

mixt_summary <- data.frame(
  Currency = names(mixt_methods),
  Selected_Model = unname(mixt_methods),
  Mean_Features = round(sapply(mixt_weekly_features, function(u) mean(rowSums(u))), 1),
  Min_Features = sapply(mixt_weekly_features, function(u) min(rowSums(u))),
  Max_Features = sapply(mixt_weekly_features, function(u) max(rowSums(u))),
  row.names = NULL
)

print(mixt_summary)
print(mixt_features)

write.csv(mixt_features, here("outputs", "mixt_features.csv"), row.names = FALSE)
write.csv(mixt_summary, here("outputs", "mixt_features_summary.csv"), row.names = FALSE)
