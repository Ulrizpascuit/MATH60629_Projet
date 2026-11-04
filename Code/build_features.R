library("here")
library("xts")

source(here("Function", "functions_feature.R"))

## ---- load_processed_data ----
load(here("Data", "Processed", "forex.RData"))
load(here("Data", "Processed", "rates.RData"))
load(here("Data", "Processed", "macro.RData"))

## ---- sample_start ----
start_date <- "2004-01-01"

## ---- sample_end ----
end_date <- "2026-08-27"
stopifnot(end(fx_prices) > as.Date(end_date))

sample_range <- paste0(start_date, "/", end_date)
fx_prices     <- fx_prices[sample_range]
rates         <- rates[sample_range]
macro_prices  <- macro_prices[sample_range]
macro_returns <- macro_returns[sample_range]

## ---- fill_fx_gap ----
# fx_prices a des NA le 2008-08-14 et le 2008-08-21 (Yahoo sans prix > 5 jours).
# Les fonctions TTR refusent un NA au milieu d'une série : on reprend le
# dernier prix connu (aucun look-ahead)
fx_prices <- na.locf(fx_prices, na.rm = FALSE)

## ---- calculate_rate_features ----
eur_features <- build_rate_features("EUR", rates)
cad_features <- build_rate_features("CAD", rates)
gbp_features <- build_rate_features("GBP", rates)
chf_features <- build_rate_features("CHF", rates)
jpy_features <- build_rate_features("JPY", rates)

## ---- merge_all_features ----
rate_features <- list(
  EUR = eur_features,
  CAD = cad_features,
  GBP = gbp_features,
  CHF = chf_features,
  JPY = jpy_features
)

## ---- calculate_ti_features ----
ti_features <- list(
  EUR = build_ti_features("EUR", fx_prices, fx_returns),
  GBP = build_ti_features("GBP", fx_prices, fx_returns),
  CAD = build_ti_features("CAD", fx_prices, fx_returns),
  CHF = build_ti_features("CHF", fx_prices, fx_returns),
  JPY = build_ti_features("JPY", fx_prices, fx_returns)
)

merged_features <- lapply(
  names(rate_features),
  function(currency) {
    rate <- rate_features[[currency]]
    ti  <- ti_features[[currency]]
    features <- merge(rate, macro_returns, join = "inner")
    features <- merge(features, ti, join = "inner")
    na.omit(features)
    })
names(merged_features) <- names(rate_features)

## ---- common_date_range ----
# Les taux GBP (BoE) et CAD sont publiés avec du retard : on coupe toutes
# les devises sur la même période (dernier début, première fin).
common_start <- max(do.call(c, lapply(merged_features, start)))
common_end   <- min(do.call(c, lapply(merged_features, end)))
merged_features <- lapply(
  merged_features,
  function(x) x[paste0(common_start, "/", common_end)]
)

## ---- rename_objects ----
eur_features <- merged_features$EUR
cad_features <- merged_features$CAD
gbp_features <- merged_features$GBP
chf_features <- merged_features$CHF
jpy_features <- merged_features$JPY

## ---- save_features ----
save(
  eur_features,
  cad_features,
  gbp_features,
  chf_features,
  jpy_features,
  file = here("Data", "Features", "merged_features.RData")
)