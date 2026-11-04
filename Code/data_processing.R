library("here")
library("xts")
library("quantmod")
library("PerformanceAnalytics")
# Taux de la zone euro (BCE) et du Royaume-Uni (Banque d'Angleterre)
library("readecb")
library("boe")

source(here("Function", "functions.R"))

## ---- get_data ----
# Toutes les données téléchargées sont conservées telles quelles dans Data/Raw
# (raw_xts / raw_table) : aux exécutions suivantes, elles sont relues dans ces
# fichiers au lieu d'être téléchargées de nouveau.
fx_pairs <- c(
  "EURUSD" = "EURUSD=X",
  "GBPUSD" = "GBPUSD=X",
  "CADUSD" = "CADUSD=X",
  "CHFUSD" = "CHFUSD=X",
  "JPYUSD" = "JPYUSD=X"
)

# Prix quotidiens : gardés pour la volatilité réalisée de chaque semaine
fx_daily_list <- lapply(names(fx_pairs), function(pair) {
  ticker <- fx_pairs[[pair]]
  x <- raw_xts(paste0("yahoo_", pair), function() {
    getSymbols(ticker, src = "yahoo", from = "2004-01-01", auto.assign = FALSE)
  })
  na.omit(x[, paste0(ticker, ".Adjusted")])
})
fx_prices <- do.call(merge, lapply(fx_daily_list, to_thursday))
colnames(fx_prices) <- names(fx_pairs)

## ---- calculate_returns ----
fx_returns <- na.omit(Return.calculate(fx_prices, method = "log"))

## ---- calculate_realized_vol ----
# Volatilité réalisée hebdomadaire (log-returns quotidiens), mêmes jeudis que fx_returns
fx_realized_vol <- do.call(merge, lapply(fx_daily_list, realized_volatility))
colnames(fx_realized_vol) <- names(fx_pairs)

## ---- get_macro_data ----
macro_tickers <- c(
  "gold"  = "GC=F",
  "sp500" = "^GSPC",
  "oil"   = "CL=F"
)

macro_prices_list <- lapply(names(macro_tickers), function(name) {

  ticker <- macro_tickers[[name]]
  x <- raw_xts(paste0("yahoo_", name), function() {
    getSymbols(ticker, src = "yahoo", from = "2004-01-01", auto.assign = FALSE)
  })
  adjusted_col <- grep("\\.Adjusted$", colnames(x), value = TRUE)
  x <- na.omit(x[, adjusted_col])
  to_thursday(x)
})
macro_prices <- do.call(merge, macro_prices_list)
colnames(macro_prices) <- names(macro_tickers)

## ---- calculate_macro_returns ----
macro_returns <- na.omit(Return.calculate(macro_prices, method = "log"))
colnames(macro_returns) <- paste0(colnames(macro_returns), "_return")

## ---- get_rates_data ----
# US - FRED
us_rates <- lapply(c("DGS2", "DGS10"), function(ticker) {
  x <- raw_xts(paste0("fred_", ticker), function() {
    getSymbols(ticker, src = "FRED", auto.assign = FALSE)
  })
  to_thursday(na.omit(x["2004-01-01/"]))
})
names(us_rates) <- c("US2", "US10")

# EUR - ECB
eur_rates <- lapply(c("2Y", "10Y"), function(tenor) {
  x <- raw_table(paste0("ecb_yield_", tenor), function() {
    ecb_yield_curve(tenor = tenor, from = "2004-01-01")
  })
  x <- xts(x$value, order.by = x$date)
  to_thursday(na.omit(x))
})
names(eur_rates) <- c("EUR2", "EUR10")

# GBP - BoE
gbp_data <- raw_table("boe_yield_curve", function() {
  boe_curve_panel(curve = "nominal", measure = "spot", maturities = c(2, 10), from = "2004-01-01")
})

gbp_rates <- list(GBP2  = to_thursday(na.omit(xts(gbp_data$m2, order.by = as.Date(gbp_data$date))))
                  ,
                  GBP10 = to_thursday(na.omit(xts(gbp_data$m10, order.by = as.Date(gbp_data$date)))))

# CAD
cad_data <- read.csv(here("Data", "Raw", "cad_rates.csv"))

cad_rates <- list(CAD2 = to_thursday(na.omit(xts(cad_data$VALUE[
        grepl("2 year", cad_data$Financial.market.statistics)
      ],
      order.by = as.Date(cad_data$REF_DATE[
        grepl("2 year", cad_data$Financial.market.statistics)
      ])
    )
  )),
  CAD10 = to_thursday(na.omit(
    xts(
      cad_data$VALUE[
        grepl("10 year", cad_data$Financial.market.statistics)
      ],
      order.by = as.Date(cad_data$REF_DATE[
        grepl("10 year", cad_data$Financial.market.statistics)
      ])
    )
  ))
)


# CHF
chf_data <- read.csv(
  here("Data", "Raw", "chf_rates.csv"),
  check.names = FALSE
)

chf_rates <- list(CHF2 = to_thursday(na.omit(xts(
                          chf_data[["GSWISS02 Index"]],
                          order.by = as.Date(chf_data$Dates))))
                  ,
                  CHF10 = to_thursday(na.omit(xts(
                          chf_data[["GSWISS10 Index"]],
                          order.by = as.Date(chf_data$Dates)))))

# JPY
jpy_data <- read.csv(
  here("Data", "Raw", "jpy_rates.csv"),
  check.names = FALSE
)

jpy_rates <- list(JPY2 = to_thursday(na.omit(xts(
                         jpy_data[["GJGB2 Index"]],
                         order.by = as.Date(jpy_data$Dates))))
                  ,
                  JPY10 = to_thursday(na.omit(xts(
                         jpy_data[["GJGB10 Index"]],
                         order.by = as.Date(jpy_data$Dates)))))

## ---- combine_rates ----
rates_2y <- do.call(merge,c(list(
      us_rates$US2,
      eur_rates$EUR2,
      cad_rates$CAD2,
      gbp_rates$GBP2,
      chf_rates$CHF2,
      jpy_rates$JPY2)))

colnames(rates_2y) <- c("US2", "EUR2", "CAD2", "GBP2", "CHF2", "JPY2")

rates_10y <- do.call(merge,list(
    us_rates$US10,
    eur_rates$EUR10,
    cad_rates$CAD10,
    gbp_rates$GBP10,
    chf_rates$CHF10,
    jpy_rates$JPY10))

colnames(rates_10y) <- c("US10", "EUR10", "CAD10", "GBP10", "CHF10", "JPY10")

rates <- merge(rates_2y, rates_10y)

## ---- funding_rate ----
DFF <- raw_xts("fred_DFF", function() {
  getSymbols("DFF", src = "FRED", auto.assign = FALSE)
})

# Taux Fed Funds observé chaque semaine
dff_rate <- to_thursday(DFF)
dff_rate <- zoo::na.locf(dff_rate, na.rm = FALSE)
colnames(dff_rate) <- "Funding"

# Taux annualisé (%) -> coût hebdomadaire
dff_weekly <- log1p((dff_rate / 100) * 7 / 360)
colnames(dff_weekly) <- "Funding_weekly"

## ---- save_data ----
save(fx_prices, fx_returns, file = here("Data", "Processed", "forex.RData"))
save(fx_realized_vol, file = here("Data", "Processed", "realized_vol.RData"))
save(macro_prices, macro_returns, file = here("Data", "Processed", "macro.RData"))
save(rates,file = here("Data", "Processed", "rates.RData"))
save(dff_rate,dff_weekly,file = here("Data", "Processed", "funding.RData"))

