"""Prépare toutes les données du projet, dans l'ordre :
  1. Code/data_processing.py : lit les fichiers bruts de Data/Raw et enregistre
     les séries hebdomadaires dans Data/Processed
  2. Code/build_features.py  : construit les features de chaque devise et les
     enregistre dans Data/Features
Voir Data/README.md pour les sources."""
from Code import build_features, data_processing

if __name__ == "__main__":
    data_processing.main()
    build_features.main()
    print("Données prêtes : Data/Processed et Data/Features.")
