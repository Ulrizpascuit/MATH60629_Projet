library("here")
library("xts")
library("ggplot2")
library("dplyr")
library("tidyr")
library("tibble")

source(here("Function", "functions_strategy.R"))

## ---- load_models ----
load(here("Data", "Models", "models.RData"))
load(here("Data", "Processed", "funding.RData"))

## ---- actual_returns ----
# Rendements réalisés sur les 260 semaines OOS du MIXT Expanding
actual <- model_matrix(
  expanding_sharpe_models$MIXT,
  element = "return_actual"
)

stopifnot(nrow(actual) == 260)

## ---- mixt_predictions ----
# Prévisions de Sharpe du modèle MIXT Expanding
mixt_pred <- model_matrix(
  expanding_sharpe_models$MIXT,
  element = "predictions"
)

stopifnot(identical(index(mixt_pred), index(actual)))

## ---- mixt_without_classifier ----
# Strategy 1 sans meta-classifier
# Sert uniquement à mesurer la valeur ajoutée du meta-classifier
mixt_raw_returns <- strategy_returns(
  pred = mixt_pred,
  actual = actual
)

colnames(mixt_raw_returns) <- "MIXT_without_meta"

## ---- meta_validation_data ----
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


## ---- mixt_final ----
# Meta-classifier expanding : une jambe est gardée si P(Win) >= 50 %
# (long_meta_threshold et short_meta_threshold, fixés dans build_model.R)

mixt_meta <- meta_expanding_filtered_returns(
  pred = mixt_pred,
  actual = actual,
  validation_legs = validation_legs,
  long_threshold = long_meta_threshold,
  short_threshold = short_meta_threshold,
  funding_weekly = dff_weekly
)

mixt_meta_returns <- mixt_meta$returns
colnames(mixt_meta_returns) <- "MIXT_Meta"

## ---- naive_benchmark ----
# Benchmark Naive utilisant la même Strategy 1 mais sans meta-classifier
# Le Naive n'estime rien : son Sharpe prévu est le return de la semaine passée
# divisé par la volatilité réalisée de la semaine passée (au lieu du GARCH).
# fx_realized_vol au jeudi t mesure la semaine t-1 -> t, comme le return prévu.
load(here("Data", "Processed", "realized_vol.RData"))

naive_last_vol_pred <- function(models) {

  pred <- do.call(
    merge,
    lapply(names(models), function(currency) {
      m <- models[[currency]]
      last_vol <- as.numeric(fx_realized_vol[m$dates, currency])
      stopifnot(
        length(last_vol) == length(m$dates),
        !anyNA(last_vol),
        all(last_vol > 0)
      )
      xts(m$return_predictions / last_vol, order.by = m$dates)
    })
  )

  colnames(pred) <- names(models)
  pred
}

naive_pred <- naive_last_vol_pred(expanding_sharpe_models$Naive)

stopifnot(identical(index(naive_pred), index(actual)))

naive_returns <- strategy_returns(
  pred = naive_pred,
  actual = actual
)

colnames(naive_returns) <- "Naive (Benchmark)"

## ---- final_performance ----
# Comparaison principale : MIXT final vs Naive
final_returns <- merge(
  mixt_meta_returns,
  naive_returns,
  join = "inner"
)

final_performance <- performance_summary(final_returns)

print(final_performance,digits = 4,row.names = FALSE)

## ---- classifier_gain ----
# Diagnostic : impact du meta-classifier sur la performance du MIXT
classifier_comparison <- merge(
  mixt_raw_returns,
  mixt_meta_returns,
  join = "inner"
)

classifier_performance <- performance_summary(classifier_comparison)

print(classifier_performance,digits = 4,row.names = FALSE)

## ---- short_classifier_oos ----
# Diagnostic OOS du classifier Short
short_decisions <- mixt_meta$decisions[mixt_meta$decisions$Side == "Short",
  ,drop = FALSE]

short_classifier_summary <- data.frame(
  Group = c("Accepted", "Rejected"),
  N = c(
    sum(short_decisions$Accept),
    sum(!short_decisions$Accept)
  ),
  Hit_Rate = c(
    mean(short_decisions$Win[short_decisions$Accept]),
    mean(short_decisions$Win[!short_decisions$Accept])
  ),
  Avg_PnL = c(
    mean(short_decisions$PnL[short_decisions$Accept]),
    mean(short_decisions$PnL[!short_decisions$Accept])
  ),
  Total_PnL = c(
    sum(short_decisions$PnL[short_decisions$Accept]),
    sum(short_decisions$PnL[!short_decisions$Accept])
  )
)

print(short_classifier_summary,digits = 4,row.names = FALSE)

## ---- long_classifier_oos ----
# Diagnostic OOS du classifier Long
long_decisions <- mixt_meta$decisions[
  mixt_meta$decisions$Side == "Long",
  , drop = FALSE
]

long_classifier_summary <- data.frame(
  Group = c("Accepted", "Rejected"),
  N = c(
    sum(long_decisions$Accept),
    sum(!long_decisions$Accept)
  ),
  Hit_Rate = c(
    mean(long_decisions$Win[long_decisions$Accept]),
    mean(long_decisions$Win[!long_decisions$Accept])
  ),
  Avg_PnL = c(
    mean(long_decisions$PnL[long_decisions$Accept]),
    mean(long_decisions$PnL[!long_decisions$Accept])
  ),
  Total_PnL = c(
    sum(long_decisions$PnL[long_decisions$Accept]),
    sum(long_decisions$PnL[!long_decisions$Accept])
  )
)

print(
  long_classifier_summary,
  digits = 4,
  row.names = FALSE
)

## ---- random_benchmark ----
# 1000 stratégies aléatoires sur les mêmes 260 semaines OOS
random_returns <- random_strategy_returns(
  actual = actual,
  n_sim = 1000,
  seed = 123
)

# Performance cumulée finale de chaque simulation aléatoire
random_totals <- colSums(random_returns)

mixt_total <- sum(mixt_meta_returns)
naive_total <- sum(naive_returns)

# Percentile de MIXT et Naive par rapport aux stratégies Random
random_comparison <- data.frame(
  Strategy = c("MIXT Meta", "Naive"),
  Random_Percentile = c(
    mean(random_totals < mixt_total),
    mean(random_totals < naive_total)
  )
)

print(random_comparison,digits = 4,row.names = FALSE)

## ---- equity_plot ----
# Equity curves finales avec distribution du benchmark Random
plot_random_equity(
  random_returns = random_returns,
  model_returns = final_returns,
  main = "Strategy 1 - Expanding OOS (260 Weeks)",
  colors = c("navy", "darkorange")
)

## ---- equity_all_methods ----
# Equity curves de toutes les méthodes de forecast (Strategy 1, sans
# meta-classifier). Le MIXT final (avec meta-classifier et financement) est
# ajouté en noir.

# Log-returns hebdomadaires de la stratégie : une colonne par méthode
all_returns <- do.call(
  merge,
  lapply(names(expanding_sharpe_models), function(method) {
    m <- expanding_sharpe_models[[method]]
    # Naive : volatilité de la semaine passée, comme le benchmark plus haut
    pred <- if (method == "Naive") {
      naive_last_vol_pred(m)
    } else {
      model_matrix(m, element = "predictions")
    }
    strategy_returns(
      pred = pred,
      actual = model_matrix(m, element = "return_actual")
    )
  })
)
colnames(all_returns) <- names(expanding_sharpe_models)

all_returns$MIXT_Meta <- mixt_meta_returns

all_equity <- exp(cumsum(all_returns))
equity_dates <- index(all_equity)
methods <- colnames(all_equity)

# Une couleur par méthode
method_colors <- setNames(
  c("grey50", "steelblue", "forestgreen", "firebrick", "darkorange",
    "purple", "goldenrod", "deeppink", "saddlebrown")[seq_along(methods)],
  methods
)
method_colors["MIXT_Meta"] <- "black"

old_par <- par(no.readonly = TRUE)
# Marge de droite élargie pour la légende
par(mfrow = c(1, 1), mar = c(3, 4, 2.5, 10), xpd = FALSE)

matplot(
  equity_dates, coredata(all_equity),
  type = "l", lty = 1,
  col = method_colors[methods],
  lwd = ifelse(methods == "MIXT_Meta", 2.5, 1.3),
  main = "Equity curves - Expanding window",
  xlab = "", ylab = "Valeur de 1 $ investi", xaxt = "n"
)

# Légende à droite, en dehors du graphique (xpd = NA : dessin dans la marge).
# graphics::legend : PerformanceAnalytics remplace legend() par sa propre
# version, qui ne dessine pas en dehors du graphique.
plot_area <- par("usr")
graphics::legend(
  x = plot_area[2] + 0.02 * (plot_area[2] - plot_area[1]),
  y = plot_area[4],
  legend = methods,
  col = method_colors[methods],
  lty = 1,
  lwd = ifelse(methods == "MIXT_Meta", 2.5, 1.3),
  bty = "n", cex = 0.8, xpd = NA
)

axis.Date(1, at = pretty(equity_dates), format = "%Y-%m")
abline(h = 1, col = "grey40", lty = 3)

par(old_par)

## ---- ggplot_equity ----
# NAV (base 100), même style que le one-pager
Primary  <- "#0B2E59"
Accent   <- "#16A89A"

p <- plot_nav(
  final_returns,
  labels = c("GRUV Alpha", "Benchmark"),
  colors = c(Primary, Accent)
)

print(p)

## ---- one_pager_metrics ----
# Métriques du one-pager pour la stratégie finale (MIXT_Meta), mêmes
# conventions que performance_summary (log-returns hebdomadaires).
final_perf <- performance_summary(mixt_meta_returns)

# Taux sans risque : Fed Funds hebdomadaire des mêmes semaines
risk_free <- as.numeric(dff_weekly)[
  match(as.Date(index(mixt_meta_returns)), as.Date(index(dff_weekly)))
]
stopifnot(!anyNA(risk_free))

excess_returns <- as.numeric(mixt_meta_returns) - risk_free

one_pager_metrics <- data.frame(
  Metric = c(
    "Annualized return",
    "Annualized volatility",
    "Annualized excess return",
    "Sharpe Ratio",
    "Sharpe Ratio (excess return)",
    "Max Drawdown",
    "Hit Rate",
    "Cumulative return"
  ),
  Value = c(
    sprintf("%.2f%%", 100 * final_perf$Ann_return),
    sprintf("%.2f%%", 100 * final_perf$Ann_vol),
    sprintf("%.2f%%", 100 * mean(excess_returns) * 52),
    sprintf("%.2f", final_perf$Sharpe),
    sprintf("%.2f", mean(excess_returns) / sd(excess_returns) * sqrt(52)),
    sprintf("%.2f%%", 100 * final_perf$Max_drawdown),
    sprintf("%.2f%%", 100 * final_perf$Hit_rate),
    sprintf("%+.1f%%", 100 * final_perf$Cumulative_return)
  )
)

print(one_pager_metrics, row.names = FALSE)

# Rendement par année civile : le return daté du jeudi t est réalisé de t à
# t+1, il est donc compté dans l'année du jeudi suivant.
realized_year <- format(index(mixt_meta_returns) + 7, "%Y")

yearly_returns <- exp(tapply(as.numeric(mixt_meta_returns), realized_year, sum)) - 1

print(sprintf("%s : %+.2f%%", names(yearly_returns), 100 * yearly_returns))
