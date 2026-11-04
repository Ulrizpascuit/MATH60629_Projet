## ---- trading strat 1 (short pire pred et long best pred) ----
# skip_all_positive = TRUE : on ne fait rien non plus si toutes les prédictions
# sont positives (stratégie corrigée, voir Code/backtest_corrected.R).
strategy_returns <- function(pred, actual, skip_all_positive = FALSE) {

  strategy_ret <- rep(0, nrow(pred))

  for (t in 1:nrow(pred)) {

    y_hat <- as.numeric(pred[t, ])
    # Si toutes les prédictions sont négatives on fait rien
    if (max(y_hat) <= 0 || (skip_all_positive && min(y_hat) >= 0)) {
      strategy_ret[t] <- 0
      
    } else {
      # Long le y prédit max
      long <- which.max(y_hat)
      # Short le y prédit min
      short <- which.min(y_hat)
      
      # Return réellement réalisé
      strategy_ret[t] <-
        as.numeric(actual[t, long]) -
        as.numeric(actual[t, short])
    }
  }
  
  strategy_ret <- xts(
    strategy_ret,
    order.by = index(pred)
  )
  
  strategy_ret
}

## ---- model_matrix ----
# Une colonne par paire, une ligne par semaine de test.
# element = "predictions" ou "actual"
model_matrix <- function(models, element = "predictions") {
  
  out <- do.call(
    merge,
    lapply(models, function(m) xts(m[[element]], order.by = m$dates))
  )
  
  colnames(out) <- names(models)
  out
}

## ---- performance_summary ----
# returns : xts de log-returns hebdomadaires, une colonne par stratégie
performance_summary <- function(returns, periods_per_year = 52) {
  
  do.call(rbind, lapply(colnames(returns), function(name) {
    r <- as.numeric(returns[, name])
    wealth <- exp(cumsum(r))
    peak <- cummax(c(1, wealth))[-1]
    
    data.frame(
      Strategy = name,
      Cumulative_return = exp(sum(r)) - 1,
      Ann_return = mean(r) * periods_per_year,
      Ann_vol = sd(r) * sqrt(periods_per_year),
      Sharpe = mean(r) / sd(r) * sqrt(periods_per_year),
      Max_drawdown = min(wealth / peak - 1),
      Hit_rate = mean(r[r != 0] > 0),
      Weeks_traded = mean(r != 0)
    )
  }))
}

## ---- random_strategy_returns ----
# Log-returns hebdomadaires de n_sim stratégies au hasard :
# long une paire, short une autre, tirées au sort chaque semaine.
# Retourne un xts : une ligne par semaine, une colonne par simulation.
random_strategy_returns <- function(actual, n_sim = 1000, seed = 123) {

  set.seed(seed)
  a <- coredata(actual)
  n <- nrow(a)
  k <- ncol(a)

  sims <- replicate(n_sim, {
    picks <- t(replicate(n, sample(k, 2)))
    a[cbind(1:n, picks[, 1])] - a[cbind(1:n, picks[, 2])]
  })

  xts(sims, order.by = index(actual))
}

## ---- plot_random_equity ----
# Gauche : les n_sim courbes au hasard en gris, leurs percentiles 5 % / 50 % / 95 %,
#          et les stratégies des modèles par-dessus.
# Droite : distribution de la valeur finale des stratégies au hasard (même axe),
#          avec la valeur finale de chaque modèle.
plot_random_equity <- function(random_returns, model_returns, main, colors) {

  dates <- index(random_returns)
  random_equity <- exp(apply(coredata(random_returns), 2, cumsum))
  model_equity  <- exp(apply(coredata(model_returns), 2, cumsum))
  bands <- t(apply(random_equity, 1, quantile, probs = c(0.05, 0.5, 0.95)))
  ylim <- range(random_equity, model_equity)

  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  layout(matrix(1:2, nrow = 1), widths = c(4, 1))

  # ---- Courbes ----
  par(mar = c(4, 4, 3, 0.5))
  matplot(
    dates, random_equity,
    type = "l", lty = 1, col = adjustcolor("grey60", alpha.f = 0.08),
    ylim = ylim,
    main = main, xlab = "Date", ylab = "Valeur de 1 $ investi", xaxt = "n"
  )
  axis.Date(1, at = pretty(dates), format = "%Y-%m")
  matlines(dates, bands, lty = c(2, 1, 2), col = "grey25", lwd = 1.2)
  matlines(dates, model_equity, lty = 1, col = colors, lwd = 2)
  abline(h = 1, col = "grey40", lty = 3)

  legend(
    "topleft",
    legend = c(colnames(model_returns), "Hasard : médiane", "Hasard : 5 % / 95 %"),
    col = c(colors, "grey25", "grey25"),
    lty = c(rep(1, ncol(model_returns)), 1, 2),
    lwd = 2, bty = "n", cex = 0.85
  )

  # ---- Distribution des valeurs finales ----
  final_random <- random_equity[nrow(random_equity), ]
  final_model  <- model_equity[nrow(model_equity), ]
  h <- hist(final_random, breaks = 40, plot = FALSE)

  par(mar = c(4, 0.5, 3, 1))
  plot(
    NA, xlim = c(0, max(h$counts) * 1.05), ylim = ylim,
    yaxt = "n", xlab = "Fréquence", ylab = "", main = "Valeur finale"
  )
  rect(0, head(h$breaks, -1), h$counts, tail(h$breaks, -1),
       col = "grey80", border = "grey60")
  abline(h = quantile(final_random, c(0.05, 0.5, 0.95)), lty = c(2, 1, 2), col = "grey25")
  abline(h = final_model, col = colors, lwd = 2)
  abline(h = 1, col = "grey40", lty = 3)
}

## ---- strategy_legs ----
# Une ligne par jambe (long ou short) prise par la stratégie (strategy_returns),
# avec les variables du classifieur :
#   Abs_Forecast : |forecast| de la paire choisie
#   Gap          : écart avec la deuxième meilleure (long) ou pire (short) paire
#   Dispersion   : écart-type des forecasts de la semaine
# PnL = return réalisé de la jambe, Win = 1 si PnL > 0.
# La somme des PnL par semaine redonne strategy_returns.
strategy_legs <- function(pred, actual, skip_all_positive = FALSE) {

  pairs <- colnames(pred)
  rows <- list()

  for (t in 1:nrow(pred)) {

    y_hat <- as.numeric(pred[t, ])
    y_real <- as.numeric(actual[t, ])

    # Si toutes les prédictions sont négatives on fait rien
    # (et aussi si elles sont toutes positives quand skip_all_positive = TRUE)
    if (max(y_hat) <= 0 || (skip_all_positive && min(y_hat) >= 0)) {
      next
    }

    long  <- which.max(y_hat)
    short <- which.min(y_hat)
    sorted <- sort(y_hat)

    rows[[length(rows) + 1]] <- data.frame(
      Date = index(pred)[t],
      Pair = pairs[long],
      Side = "Long",
      Abs_Forecast = abs(y_hat[long]),
      Gap = sorted[length(sorted)] - sorted[length(sorted) - 1],
      Dispersion = sd(y_hat),
      PnL = y_real[long],
      Win = as.integer(y_real[long] > 0)
    )

    rows[[length(rows) + 1]] <- data.frame(
      Date = index(pred)[t],
      Pair = pairs[short],
      Side = "Short",
      Abs_Forecast = abs(y_hat[short]),
      Gap = sorted[2] - sorted[1],
      Dispersion = sd(y_hat),
      PnL = -y_real[short],
      Win = as.integer(-y_real[short] > 0)
    )
  }

  do.call(rbind, rows)
}

## ---- add_funding_weekly ----
# Ajoute le rendement hebdomadaire du DFF aux jambes OOS.
# Le funding n'entre pas dans le classifieur : il est appliqué uniquement
# au PnL lorsqu'une seule jambe est retenue.
add_funding_weekly <- function(legs, funding_weekly) {
  
  funding_map <- setNames(
    as.numeric(funding_weekly),
    as.character(as.Date(index(funding_weekly)))
  )
  
  legs$Funding_weekly <- funding_map[as.character(as.Date(legs$Date))]
  
  if (anyNA(legs$Funding_weekly)) {
    stop("Missing funding rate in OOS strategy data.")
  }
  
  legs
}

## ---- expanding_meta_cv ----
expanding_meta_cv <- function(legs,initial_train_ratio = 0.60) {
  
  results <- list()
  for (side in c("Long", "Short")) {
    data_side <- legs[legs$Side == side,
      ,drop = FALSE]
    
    data_side <- data_side[order(data_side$Date),
      ,drop = FALSE]
    
    n <- nrow(data_side)
    
    initial_size <- floor(initial_train_ratio * n)
    
    for (i in initial_size:(n - 1)) {
      
      train <- data_side[1:i,
        ,drop = FALSE]
      
      test <- data_side[i + 1,
        ,drop = FALSE]
      
      model <- glm(
        Win ~ Abs_Forecast + Gap + Dispersion,
        data = train,
        family = binomial()
      )
      
      probability <- as.numeric(
        predict(model,
          newdata = test,
          type = "response")
          )
      
      results[[length(results) + 1]] <-
        data.frame(
          Date = test$Date,
          Pair = test$Pair,
          Side = side,
          Probability = probability,
          Win = test$Win,
          PnL = test$PnL)
    }
  }
  do.call(rbind,results)
}

## ---- threshold_diagnostics ----
threshold_diagnostics <- function(
    meta_oof,
    thresholds = seq(
      0.50,
      0.65,
      by = 0.025
    )
) {
  
  results <- list()
  
  for (side in c("Long", "Short")) {
    
    data_side <- meta_oof[meta_oof$Side == side,
      ,drop = FALSE]
    
    for (th in thresholds) {
      
      accepted <- data_side[data_side$Probability >= th,
        ,drop = FALSE]
      
      rejected <- data_side[data_side$Probability < th,
        ,drop = FALSE]
      
      results[[length(results) + 1]] <-
        data.frame(
          Side = side,
          Threshold = th,
          
          Accepted_N = nrow(accepted),
          
          Accepted_Hit =
            if (nrow(accepted) > 0)
              mean(accepted$Win)
          else NA,
          
          Accepted_AvgPnL =
            if (nrow(accepted) > 0)
              mean(accepted$PnL)
          else NA,
          
          Accepted_TotalPnL = sum(accepted$PnL),
          
          Rejected_N = nrow(rejected),
          
          Rejected_Hit =
            if (nrow(rejected) > 0)
              mean(rejected$Win)
          else NA,
          
          Rejected_AvgPnL =
            if (nrow(rejected) > 0)
              mean(rejected$PnL)
          else NA,
          
          Rejected_TotalPnL = sum(rejected$PnL))
    }
  }
  
  do.call(rbind,results)
}

## ---- meta_expanding_filtered_returns ----
meta_expanding_filtered_returns <- function(pred, actual, validation_legs,
                                            long_threshold, short_threshold,
                                            funding_weekly,
                                            skip_all_positive = FALSE) {

  oos_legs <- strategy_legs(pred, actual, skip_all_positive = skip_all_positive)
  oos_legs <- add_funding_weekly(oos_legs, funding_weekly)
  
  oos_legs$Probability <- NA_real_
  oos_legs$Accept <- FALSE
  strategy_ret <- xts(rep(0, nrow(pred)), order.by = index(pred))
  
  trade_dates <- sort(unique(oos_legs$Date))
  meta_cols <- c("Date", "Side", "Win", "PnL", "Abs_Forecast",
                 "Gap", "Dispersion")
  
  for (date_t in trade_dates) {
    
    date_t <- as.Date(date_t, origin = "1970-01-01")
    
    # Expanding : validation + OOS passé
    past_oos <- oos_legs[oos_legs$Date < date_t, , drop = FALSE]
    train_legs <- rbind(
      validation_legs[, meta_cols, drop = FALSE],
      past_oos[, meta_cols, drop = FALSE]
    )
    
    long_train <- train_legs[train_legs$Side == "Long", , drop = FALSE]
    short_train <- train_legs[train_legs$Side == "Short", , drop = FALSE]
    
    long_model <- glm(Win ~ Abs_Forecast + Gap + Dispersion,
                      data = long_train, family = binomial())
    short_model <- glm(Win ~ Abs_Forecast + Gap + Dispersion,
                       data = short_train, family = binomial())
    
    current_rows <- oos_legs$Date == date_t
    long_row <- current_rows & oos_legs$Side == "Long"
    short_row <- current_rows & oos_legs$Side == "Short"
    
    if (any(long_row)) {
      oos_legs$Probability[long_row] <- predict(
        long_model, newdata = oos_legs[long_row, , drop = FALSE], type = "response"
      )
      oos_legs$Accept[long_row] <- oos_legs$Probability[long_row] >= long_threshold
    }
    
    if (any(short_row)) {
      oos_legs$Probability[short_row] <- predict(
        short_model, newdata = oos_legs[short_row, , drop = FALSE], type = "response"
      )
      oos_legs$Accept[short_row] <- oos_legs$Probability[short_row] >= short_threshold
    }
    
    long_leg <- oos_legs[long_row, , drop = FALSE]
    short_leg <- oos_legs[short_row, , drop = FALSE]
    
    long_ok <- nrow(long_leg) > 0 && long_leg$Accept
    short_ok <- nrow(short_leg) > 0 && short_leg$Accept
    
    # Deux jambes : autofinancé
    if (long_ok && short_ok) {
      strategy_ret[as.character(date_t)] <- long_leg$PnL + short_leg$PnL
      
      # Long seul : paie le funding
    } else if (long_ok) {
      strategy_ret[as.character(date_t)] <-
        long_leg$PnL - long_leg$Funding_weekly
      
      # Short seul : reçoit le funding
    } else if (short_ok) {
      strategy_ret[as.character(date_t)] <-
        short_leg$PnL + short_leg$Funding_weekly
    }
  }
  
  list(returns = strategy_ret, decisions = oos_legs)
}

## ---- plot_nav ----
# NAV (base 100) de chaque stratégie, style du fact sheet.
#   returns : xts de returns hebdomadaires, une colonne par stratégie
#   labels  : nom affiché de chaque colonne
#   colors  : couleur de chaque colonne, dans le même ordre
plot_nav <- function(returns, labels, colors, title = NULL) {

  Primary  <- "#0B2E59"
  TextGrey <- "#58677A"
  RuleGrey <- "#D7DFE5"

  equity_data <- as.data.frame(returns)
  colnames(equity_data) <- labels

  equity_data <- equity_data %>%
    mutate(Week = row_number()) %>%
    pivot_longer(
      cols = -Week,
      names_to = "Strategy",
      values_to = "Return"
    ) %>%
    group_by(Strategy) %>%
    arrange(Week, .by_group = TRUE) %>%
    # Log-returns : la valeur cumulée est exp(somme), comme performance_summary
    mutate(NAV = 100 * exp(cumsum(Return))) %>%
    ungroup()

  # Semaine 0 : toutes les stratégies partent de 100
  equity_data <- bind_rows(
    tibble(
      Week = 0,
      Strategy = labels,
      Return = NA_real_,
      NAV = 100
    ),
    equity_data
  )

  ggplot(
    equity_data,
    aes(x = Week, y = NAV, color = Strategy)
  ) +
    # Courbe réelle, sans lissage
    geom_line(linewidth = 0.7) +

    scale_color_manual(
      values = setNames(colors, labels)
    ) +

    scale_x_continuous(
      name = NULL,
      expand = expansion(mult = c(0, 0.01))
    ) +

    scale_y_continuous(
      name = "NAV",
      expand = expansion(mult = c(0.02, 0.04))
    ) +

    labs(
      title = title,
      color = NULL
    ) +

    theme_minimal(base_size = 10) +

    theme(
      # Fond
      plot.background = element_rect(
        fill = "white", color = NA
      ),
      panel.background = element_rect(
        fill = "white", color = NA
      ),

      # Grille
      panel.grid.major = element_line(
        color = RuleGrey, linewidth = 0.3
      ),
      panel.grid.minor = element_blank(),

      # Axes
      axis.line = element_line(
        color = TextGrey, linewidth = 0.3
      ),
      axis.text = element_text(
        color = TextGrey, size = 20
      ),
      axis.title.y = element_text(
        color = TextGrey, size = 20
      ),

      # Titre
      plot.title = element_text(
        color = Primary,
        face = "bold",
        size = 12,
        margin = margin(b = 8)
      ),

      # Légende
      legend.position = "bottom",
      legend.text = element_text(
        color = TextGrey, size = 20
      ),
      legend.key.width = grid::unit(1.2, "cm"),
      legend.background = element_blank(),

      # Marges
      plot.margin = margin(8, 8, 4, 4)
    )
}