## ---- raw_xts ----
# Série brute (xts) conservée dans Data/Raw : lue dans <name>.csv si le
# fichier existe, sinon téléchargée avec download() puis enregistrée.
# Pour mettre une série à jour, supprimer son fichier et relancer.
raw_xts <- function(name, download) {
  file <- here("Data", "Raw", paste0(name, ".csv"))
  if (file.exists(file)) {
    return(as.xts(read.zoo(
      file, header = TRUE, sep = ",", FUN = as.Date,
      check.names = FALSE, drop = FALSE
    )))
  }
  x <- download()
  write.zoo(x, file, sep = ",")
  x
}

## ---- raw_table ----
# Même chose pour un tableau brut (data.frame avec une colonne date)
raw_table <- function(name, download) {
  file <- here("Data", "Raw", paste0(name, ".csv"))
  if (file.exists(file)) {
    x <- read.csv(file)
  } else {
    x <- as.data.frame(download())
    write.csv(x, file, row.names = FALSE)
  }
  x$date <- as.Date(x$date)
  x
}

## ---- to_thursday ----
# Pour chaque jeudi : dernière observation connue à ce jeudi ou avant,
# si elle date de max_days jours au plus (sinon NA).
to_thursday <- function(x, from = "2004-01-01", max_days = 5) {
  x <- na.omit(x)
  first_thu <- as.Date(from) + (4 - as.POSIXlt(as.Date(from))$wday) %% 7
  thursdays <- seq(first_thu, max(index(x)), by = "week")
  pos <- findInterval(as.numeric(thursdays), as.numeric(index(x)))
  out <- xts(matrix(NA_real_, length(thursdays), ncol(x)), order.by = thursdays)
  ok <- pos > 0
  ok[ok] <- as.numeric(thursdays[ok] - index(x)[pos[ok]]) <= max_days
  out[ok, ] <- coredata(x)[pos[ok], ]
  colnames(out) <- colnames(x)
  out
}

## ---- realized_volatility ----
# Volatilité réalisée de la semaine qui se termine à chaque jeudi :
# racine de la somme des log-returns quotidiens au carré, du jeudi précédent
# (exclu) à ce jeudi (inclus). Même convention de date que fx_returns :
# la valeur au jeudi t mesure la semaine t-1 -> t (NA si aucun prix cette semaine).
realized_volatility <- function(x, from = "2004-01-01") {
  x <- na.omit(x)
  daily_returns <- na.omit(diff(log(x)))
  first_thu <- as.Date(from) + (4 - as.POSIXlt(as.Date(from))$wday) %% 7
  thursdays <- seq(first_thu, max(index(x)), by = "week")
  # Semaine de chaque jour : premier jeudi >= ce jour
  week <- findInterval(as.numeric(index(daily_returns)) - 1, as.numeric(thursdays)) + 1
  ok <- week <= length(thursdays)
  sum_sq <- tapply(as.numeric(daily_returns)[ok]^2, week[ok], sum)
  out <- xts(rep(NA_real_, length(thursdays)), order.by = thursdays)
  out[as.integer(names(sum_sq))] <- sqrt(sum_sq)
  out
}
