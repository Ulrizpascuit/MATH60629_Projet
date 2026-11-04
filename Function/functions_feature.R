## ---- build_rate_features ----
build_rate_features <- function(currency, rates) {
  spread2 <- rates[, paste0(currency, "2")] - rates[, "US2"]
  spread10 <- rates[, paste0(currency, "10")] - rates[, "US10"]
  spread2_diff <- diff(spread2)
  spread10_diff <- diff(spread10)
  dgs2_diff <- diff(rates[, "US2"])
  dgs10_diff <- diff(rates[, "US10"])
  
  features <- merge(
    spread2,
    spread10,
    spread2_diff,
    spread10_diff,
    dgs2_diff,
    dgs10_diff
  )
  
  colnames(features) <- c(
    paste0(tolower(currency), "_spread2"),
    paste0(tolower(currency), "_spread10"),
    paste0(tolower(currency), "_spread2_diff"),
    paste0(tolower(currency), "_spread10_diff"),
    "dgs2_diff",
    "dgs10_diff"
  )
  
  na.omit(features)
}


## ---- build_technical_indicator_features ----
build_ti_features <- function(
    currency,
    fx_prices,
    fx_returns,
    vol_windows = c(4, 12),
    rsi_n = 14,
    macd_fast = 12,
    macd_slow = 26,
    macd_sig = 9,
    bb_n = 20,
    bb_sd = 2
) {
  
  pair <- paste0(currency, "USD")
  
  price <- fx_prices[, pair]
  
  # Remplit uniquement les NA internes avec la dernière observation connue
  # Les NA au début restent NA pour éviter tout look-ahead
  price <- na.locf(
    price,
    na.rm = FALSE
  )
  
  returns <- fx_returns[, pair]
  
  #lag
  lag_return <- xts::lag.xts(
    fx_returns[, pair],
    k = 0
  )
  
  #vol
  vol_4 <- TTR::runSD(
    returns,
    n = vol_windows[1]
  )
  
  vol_12 <- TTR::runSD(
    returns,
    n = vol_windows[2]
  )
  #RSI
  rsi <- TTR::RSI(
    price,
    n = rsi_n
  )
  # MACD
  macd <- TTR::MACD(
    price,
    nFast = macd_fast,
    nSlow = macd_slow,
    nSig = macd_sig
  )[, "macd"]
  # Bollinger Bands %B
  bb_mid <- TTR::SMA(
    price,
    n = bb_n
  )
  
  bb_sdv <- TTR::runSD(
    price,
    n = bb_n
  )
  
  bb_up <- bb_mid + bb_sd * bb_sdv
  bb_lo <- bb_mid - bb_sd * bb_sdv
  
  bb_pctB <- (
    price - bb_lo
  ) / (
    bb_up - bb_lo
  )
  # Merge
  features <- merge(
    lag_return,
    vol_4,
    vol_12,
    rsi,
    macd,
    bb_pctB
  )
  
  colnames(features) <- c(
    "lag_return",
    "vol_4",
    "vol_12",
    "rsi",
    "macd",
    "bb_pctB"
  )
  
  features
}


## ---- na_summary ----
# Tableau des NA : nombre de lignes, lignes avec au moins un NA, et NA par colonne.
na_summary <- function(features_list) {
  do.call(rbind, lapply(names(features_list), function(name) {
    x <- features_list[[name]]
    na_rows <- sum(!complete.cases(x))
    data.frame(
      table = name,
      debut = start(x),
      fin = end(x),
      n_lignes = nrow(x),
      lignes_avec_na = na_rows,
      pct_na = round(100 * na_rows / nrow(x), 1),
      colonnes_avec_na = sum(colSums(is.na(x)) > 0),
      n_colonnes = ncol(x)
    )
  }))
}


