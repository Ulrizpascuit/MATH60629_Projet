## ---- run_install_packages ----
# Installe les packages nécessaires au projet (seulement ceux qui manquent).
# À lancer une fois avant run_data.R et run_strategy.R.
# parallel et grid sont livrés avec R : rien à installer.
required_packages <- c(
  # Chemins et affichage
  "here", "fs",
  # Séries temporelles et données de marché
  "zoo", "xts", "quantmod", "TTR", "PerformanceAnalytics",
  # Taux : Banque centrale européenne et Banque d'Angleterre
  "readecb", "boe",
  # Modèles
  "glmnet", "ranger", "rugarch",
  # Graphiques et manipulation de tableaux
  "ggplot2", "dplyr", "tidyr", "tibble"
)

missing_packages <- setdiff(required_packages, rownames(installed.packages()))

if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

# Vérification : arrête avec un message clair si un package manque encore
still_missing <- setdiff(required_packages, rownames(installed.packages()))
if (length(still_missing) > 0) {
  stop("Packages non installés : ", paste(still_missing, collapse = ", "))
}

cat("Tous les packages sont installés.\n")

## ---- print_structure ----
# Arborescence du projet
fs::dir_tree(here::here())
