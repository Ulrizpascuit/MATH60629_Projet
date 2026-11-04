## ---- run_data ----
# Prépare toutes les données du projet, dans l'ordre :
#   1. Code/data_processing.R : télécharge les données du web (Yahoo, FRED,
#      BCE, Banque d'Angleterre), lit les fichiers bruts de Data/Raw et
#      enregistre les séries hebdomadaires dans Data/Processed
#   2. Code/build_features.R  : construit les features de chaque devise et
#      les enregistre dans Data/Features
# Une connexion internet est nécessaire. Voir Data/README.md pour les sources.
library("here")

source(here("Code", "data_processing.R"))
source(here("Code", "build_features.R"))

cat("Données prêtes : Data/Processed et Data/Features.\n")
