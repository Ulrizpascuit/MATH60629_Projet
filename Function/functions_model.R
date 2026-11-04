## ---- stepwise_formula ----
# Formule target ~ variables (target ~ 1 pour le modèle nul)
stepwise_formula <- function(vars) {
  if (length(vars) == 0) {
    return(target ~ 1)
  }
  reformulate(vars, response = "target")
}

## ---- forward_BIC_path ----
# Forward stepwise : part du modèle nul M0 et, à chaque étape k,
# ajoute la variable qui donne le plus petit RSS.
# Retourne les variables de M0, M1, ..., Mp (meilleur modèle pour chaque taille).
# .lm.fit donne le même RSS que lm, mais beaucoup plus vite.
forward_stepwise_path <- function(train) {
  
  candidates <- setdiff(colnames(train), "target")
  y <- train$target
  X <- as.matrix(train[, candidates])
  
  selected <- character(0)
  path <- list(selected)
  
  for (k in seq_along(candidates)) {
    remaining <- setdiff(candidates, selected)
    
    rss <- sapply(remaining, function(v) {
      fit <- .lm.fit(cbind(1, X[, c(selected, v), drop = FALSE]), y)
      sum(fit$residuals^2)
    })
    
    selected <- c(selected, names(which.min(rss)))
    path[[k + 1]] <- selected
  }
  
  path
}

## ---- select_BIC_model ----
# Sur un échantillon d'entraînement : meilleur modèle pour chaque taille (M0, ..., Mp),
# puis choix d'un seul modèle selon criterion :
#   "BIC", "AIC", "adjR2" : calculés sur l'entraînement
#   "validation" : le chemin est refait sur le début de l'entraînement et
#                  la taille k est choisie sur la fin (validation_ratio).
select_stepwise_model <- function(
    train,
    criterion = c("BIC", "AIC", "adjR2", "validation"),
    validation_ratio = 0.20
) {
  
  criterion <- match.arg(criterion)
  
  path <- forward_stepwise_path(train)
  
  best_by_size <- lapply(path, function(vars) {
    lm(stepwise_formula(vars), data = train)
  })
  names(best_by_size) <- paste0("M", seq_along(path) - 1)
  
  selection <- data.frame(
    k = seq_along(path) - 1,
    added = c("(aucune)", sapply(path[-1], function(vars) tail(vars, 1))),
    RSS = sapply(best_by_size, function(m) sum(residuals(m)^2)),
    R2 = sapply(best_by_size, function(m) summary(m)$r.squared),
    adjR2 = sapply(best_by_size, function(m) summary(m)$adj.r.squared),
    AIC = sapply(best_by_size, AIC),
    BIC = sapply(best_by_size, BIC)
  )
  
  if (criterion == "validation") {
    fit_size <- floor((1 - validation_ratio) * nrow(train))
    fit_part   <- train[1:fit_size, ]
    valid_part <- train[(fit_size + 1):nrow(train), ]
    
    valid_path <- forward_stepwise_path(fit_part)
    
    selection$validation_rmse <- sapply(valid_path, function(vars) {
      m <- lm(stepwise_formula(vars), data = fit_part)
      sqrt(mean((valid_part$target - predict(m, newdata = valid_part))^2))
    })
  }
  
  best <- switch(
    criterion,
    BIC        = which.min(selection$BIC),
    AIC        = which.min(selection$AIC),
    adjR2      = which.max(selection$adjR2),
    validation = which.min(selection$validation_rmse)
  )
  
  list(
    model = best_by_size[[best]],
    selected = path[[best]],
    selection = selection,
    best_by_size = best_by_size
  )
}

## ---- regression_metrics ----

regression_metrics <- function(actual, predicted) {
  
  actual <- as.numeric(actual)
  predicted <- as.numeric(predicted)
  
  err <- actual - predicted
  
  R2 <- 1 -
    sum(err^2) /
    sum((actual - mean(actual))^2)
  
  c(
    RMSE = sqrt(mean(err^2)),
    MAE = mean(abs(err)),
    R2 = R2
  )
}

## ---- build_walk_forward_garch_model ----
# on fixe à du GARCH(1,1) car après avoir tester sur des GARCH(p,q) pour p=q=12
# tout les GRCH retenu était du (1,1)
# sauf pour CHF (2,1) mais avec un delta de BIC entre (1,1) et (2,1) de 0.025 donc négligeable
# au moins on simplifie un peu
#
# Forecast de la volatilité : les paramètres du GARCH(1,1) sont ré-estimés
# chaque semaine sur tous les returns connus à la date du forecast (fenêtre
# croissante), comme les modèles de return en walk-forward.
# target et realized_vol sont DÉJÀ décalés (valeur à t = semaine t -> t+1) ;
# model_result donne les semaines à prédire (les mêmes que les modèles de return).
# Si une estimation ne converge pas, on garde les paramètres de la semaine d'avant.
build_walk_forward_garch_model <- function(
    target,
    realized_vol,
    model_result
) {

  forecast_dates <- model_result$dates

  returns <- na.omit(target)
  return_dates <- index(returns)
  returns <- as.numeric(returns) * 100

  # GARCH(1,1)
  spec <- rugarch::ugarchspec(
    variance.model = list(
      model = "sGARCH",
      garchOrder = c(1, 1)
    ),
    mean.model = list(
      armaOrder = c(0, 0),
      include.mean = TRUE
    ),
    distribution.model = "norm"
  )

  n <- length(forecast_dates)
  sigma_forecast <- numeric(n)
  refit_failed <- logical(n)
  params <- matrix(
    NA_real_, nrow = n, ncol = 4,
    dimnames = list(NULL, c("mu", "omega", "alpha", "beta"))
  )
  coef_fit <- NULL

  # Une estimation par semaine
  for (k in seq_len(n)) {

    # Returns connus à la date du forecast : target(t) = return(t -> t+1)
    # n'est connu qu'à t+1, donc on s'arrête strictement avant la date prédite
    last <- sum(return_dates < forecast_dates[k])
    train <- returns[1:last]

    fit <- tryCatch(
      rugarch::ugarchfit(spec = spec, data = train, solver = "hybrid"),
      error = function(e) NULL
    )

    if (!is.null(fit) && fit@fit$convergence == 0) {

      coef_fit <- rugarch::coef(fit)[c("mu", "omega", "alpha1", "beta1")]
      last_sigma2 <- tail(as.numeric(rugarch::sigma(fit))^2, 1)

    } else {

      if (is.null(coef_fit)) {
        stop("GARCH(1,1) did not converge.")
      }

      # Paramètres de la semaine d'avant : variance filtrée sur la fenêtre
      refit_failed[k] <- TRUE
      last_sigma2 <- var(train)
      for (r in train[-length(train)]) {
        last_sigma2 <-
          coef_fit[["omega"]] +
          coef_fit[["alpha1"]] * (r - coef_fit[["mu"]])^2 +
          coef_fit[["beta1"]] * last_sigma2
      }
    }

    last_residual <- train[length(train)] - coef_fit[["mu"]]

    next_sigma2 <-
      coef_fit[["omega"]] +
      coef_fit[["alpha1"]] * last_residual^2 +
      coef_fit[["beta1"]] * last_sigma2

    sigma_forecast[k] <- sqrt(next_sigma2) / 100
    params[k, ] <- coef_fit
  }

  # Volatilité réalisée des mêmes semaines
  actual_vol <- as.numeric(
    realized_vol[forecast_dates]
  )
  stopifnot(
    length(actual_vol) == length(sigma_forecast),
    !anyNA(actual_vol)
  )

  metrics <- regression_metrics(actual_vol, sigma_forecast)

  # Results (même format que les modèles de return)
  list(
    model = NULL,
    predictions = sigma_forecast,
    actual = actual_vol,
    dates = forecast_dates,
    rmse = unname(metrics["RMSE"]),
    mae = unname(metrics["MAE"]),
    r2 = unname(metrics["R2"]),
    params = params,
    persistence = params[, "alpha"] + params[, "beta"],
    refit_failed = refit_failed
  )
}

## ---- build_walk_forward_garch_models ----
# GARCH walk-forward pour toutes les paires.
# targets, realized_vols : une colonne par paire ; models : une liste par paire
# (donne les semaines à prédire).
# cluster (parallel::makeCluster) : une tâche par paire, en parallèle.
build_walk_forward_garch_models <- function(
    targets,
    realized_vols,
    models,
    cluster = NULL
) {

  # Évalués ici pour être envoyés aux processus du cluster
  force(targets)
  force(realized_vols)
  force(models)

  run_task <- function(currency) {
    build_walk_forward_garch_model(
      target = targets[, currency],
      realized_vol = realized_vols[, currency],
      model_result = models[[currency]]
    )
  }

  if (is.null(cluster)) {
    results <- lapply(names(models), run_task)
  } else {
    results <- parallel::parLapply(cluster, names(models), run_task)
  }

  names(results) <- names(models)
  results
}

## ---- build_sharpe_model ----
# Sharpe hebdomadaire d'une paire :
#   prévu = return prévu / volatilité prévue (GARCH)
#   réel  = return réalisé / volatilité réalisée
# Même format que les modèles de return : predictions et actual sont des Sharpe.
# Les returns et les volatilités sont gardés pour le backtest.
build_sharpe_model <- function(model_result, garch_model) {
  
  stopifnot(identical(model_result$dates, garch_model$dates))
  
  return_predictions <- as.numeric(model_result$predictions)
  return_actual <- as.numeric(model_result$actual)
  
  predictions <- return_predictions / garch_model$predictions
  actual <- return_actual / garch_model$actual
  
  metrics <- regression_metrics(actual, predictions)
  
  list(
    predictions = predictions,
    actual = actual,
    dates = model_result$dates,
    rmse = unname(metrics["RMSE"]),
    mae = unname(metrics["MAE"]),
    r2 = unname(metrics["R2"]),
    return_predictions = return_predictions,
    return_actual = return_actual,
    return_rmse = model_result$rmse,
    vol_predictions = garch_model$predictions,
    vol_actual = garch_model$actual
  )
}

## ---- build_sharpe_models ----
# Une méthode de forecast de return (une liste par paire) -> Sharpe de chaque paire
build_sharpe_models <- function(models, garch_models) {
  
  sharpe_models <- lapply(names(models), function(currency) {
    build_sharpe_model(models[[currency]], garch_models[[currency]])
  })
  
  names(sharpe_models) <- names(models)
  sharpe_models
}

## ---- metric_table ----
# Tableau d'une métrique d'erreur ("rmse", "mae", "r2") :
# une ligne par paire, une colonne par méthode.
# groups : liste nommée de méthodes, chacune étant une liste de modèles par paire.
metric_table <- function(groups, metric = "rmse") {
  
  out <- data.frame(
    Currency = names(groups[[1]]),
    lapply(groups, function(models) sapply(models, function(x) x[[metric]])),
    row.names = NULL
  )
  
  colnames(out)[-1] <- paste0(names(groups), "_", toupper(metric))
  out
}

## ---- fit_predict_model ----
# Estime une méthode sur train et prédit test (data.frames : target + features).
fit_predict_model <- function(train, test, model_type, nfolds = 5, num_trees = 500) {

  x_train <- train[, -1, drop = FALSE]
  x_test  <- test[, -1, drop = FALSE]

  # Naive : return de la semaine qui vient de se terminer
  if (model_type == "Naive") {
    return(test$lag_return)
  }

  if (model_type == "Linear") {
    model <- lm(target ~ ., data = train)
    return(as.numeric(predict(model, newdata = test)))
  }

  if (model_type == "Stepwise") {
    model <- select_stepwise_model(train, criterion = "BIC")$model
    return(as.numeric(predict(model, newdata = test)))
  }

  if (model_type %in% c("Lasso", "Ridge", "Elastic_Net")) {
    alpha <- switch(model_type, Lasso = 1, Ridge = 0, Elastic_Net = 0.5)
    set.seed(123)
    model <- glmnet::cv.glmnet(
      x = as.matrix(x_train),
      y = train$target,
      alpha = alpha,
      nfolds = nfolds,
      standardize = TRUE
    )
    return(as.numeric(predict(model, newx = as.matrix(x_test), s = model$lambda.min)))
  }

  if (model_type == "Random_Forest") {
    set.seed(123)
    model <- ranger::ranger(target ~ ., data = train, num.trees = num_trees)
    return(as.numeric(predict(model, data = test)$predictions))
  }

  if (model_type == "PCA") {
    pca <- prcomp(x_train, center = TRUE, scale. = TRUE)
    k <- max(1, sum(pca$sdev^2 > 1))
    pc_train <- predict(pca, newdata = x_train)[, 1:k, drop = FALSE]
    pc_test  <- predict(pca, newdata = x_test)[, 1:k, drop = FALSE]
    model <- lm(target ~ ., data = data.frame(target = train$target, pc_train))
    return(as.numeric(predict(model, newdata = as.data.frame(pc_test))))
  }

  stop("Méthode inconnue : ", model_type)
}

## ---- build_walk_forward_model ----
# Walk-forward (fenêtre croissante) : la méthode est ré-estimée chaque semaine
# sur les semaines 1 à i-1, puis prédit la semaine i.
#   period = "test"       : les semaines de test (après les premiers train_ratio)
#   period = "validation" : la fin de l'entraînement, après les premiers
#                           initial_train_ratio (le test n'est jamais utilisé)
# Le target doit être DÉJÀ décalé (target(t) = return entre t et t+1) :
# la dernière ligne d'entraînement (i-1) a pour target r(i), connu à la date i.
build_walk_forward_model <- function(
    features,
    target,
    model_type,
    period = c("test", "validation"),
    train_ratio = 0.80,
    initial_train_ratio = 0.60
) {

  period <- match.arg(period)

  data <- na.omit(
    merge(
      target,
      features,
      join = "inner"
    )
  )

  data_model <- data.frame(
    target = as.numeric(data[, 1]),
    as.data.frame(coredata(data[, -1]))
  )

  # --------------------------------------------------
  # Semaines à prédire
  # --------------------------------------------------
  n <- nrow(data_model)
  train_size <- floor(train_ratio * n)

  rows <- switch(
    period,
    test       = (train_size + 1):n,
    validation = (floor(initial_train_ratio * train_size) + 1):train_size
  )

  # --------------------------------------------------
  # Une estimation par semaine
  # --------------------------------------------------
  predictions <- numeric(length(rows))

  for (k in seq_along(rows)) {
    i <- rows[k]
    predictions[k] <- fit_predict_model(
      train = data_model[1:(i - 1), ],
      test = data_model[i, ],
      model_type = model_type
    )
  }

  actual <- data_model$target[rows]
  metrics <- regression_metrics(actual, predictions)

  # --------------------------------------------------
  # Results
  # --------------------------------------------------
  list(
    model = NULL,
    model_type = model_type,
    predictions = predictions,
    actual = actual,
    dates = index(data)[rows],
    rmse = unname(metrics["RMSE"]),
    mae = unname(metrics["MAE"]),
    r2 = unname(metrics["R2"])
  )
}

## ---- build_walk_forward_models ----
# Walk-forward pour toutes les méthodes et toutes les paires.
# features_list : features de chaque paire (liste nommée), targets : une colonne par paire.
# Retourne méthode -> paires.
# cluster (parallel::makeCluster) : une tâche par méthode x paire, en parallèle.
# Les résultats sont identiques avec ou sans cluster (graines fixées dans chaque fit).
build_walk_forward_models <- function(
    features_list,
    targets,
    period = c("test", "validation"),
    model_types = c(
      "Naive", "Linear", "Stepwise", "Lasso",
      "Ridge", "Elastic_Net", "Random_Forest", "PCA"
    ),
    train_ratio = 0.80,
    cluster = NULL
) {

  period <- match.arg(period)

  # Évalués ici pour être envoyés aux processus du cluster
  force(features_list)
  force(targets)
  force(train_ratio)

  tasks <- expand.grid(
    currency = names(features_list),
    model_type = model_types,
    stringsAsFactors = FALSE
  )

  run_task <- function(k) {
    build_walk_forward_model(
      features = features_list[[tasks$currency[k]]],
      target = targets[, tasks$currency[k]],
      model_type = tasks$model_type[k],
      period = period,
      train_ratio = train_ratio
    )
  }

  if (is.null(cluster)) {
    results <- lapply(seq_len(nrow(tasks)), run_task)
  } else {
    results <- parallel::parLapply(cluster, seq_len(nrow(tasks)), run_task)
  }

  models <- lapply(model_types, function(model_type) {
    out <- results[tasks$model_type == model_type]
    names(out) <- tasks$currency[tasks$model_type == model_type]
    out
  })

  names(models) <- model_types
  models
}

## ---- build_mixt_models ----
# Modèle mixte : pour chaque paire, on prend la méthode choisie (best_models,
# une méthode par paire) parmi models (méthode -> paires).
build_mixt_models <- function(models, best_models) {

  mixt_models <- lapply(names(best_models), function(currency) {
    selected <- best_models[[currency]]
    out <- models[[selected]][[currency]]
    out$selected_model <- selected
    out
  })

  names(mixt_models) <- names(best_models)
  mixt_models
}
