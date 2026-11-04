---
editor_options: 
  markdown: 
    wrap: 72
---

# Données

Toutes les séries commencent le 2004-01-01. L'échantillon utilisé par
les modèles s'arrête le 2026-08-27 (fixé dans `Code/build_features.R`).

## Organisation du dossier

| Dossier | Contenu | Créé par |
|----|----|----|
| `Raw/` | Fichiers bruts téléchargés à la main (taux CAD, CHF, JPY) | À la main, voir plus bas |
| `Processed/` | Séries hebdomadaires (jeudi) : prix, rendements, volatilité réalisée, taux, financement | `run_data.R` |
| `Features/` | Features de chaque devise | `run_data.R` |
| `Models/` | Prévisions des modèles | `run_strategy.R` |

## Données récupérées du web par `run_data.R`

Aucune étape manuelle : `Code/data_processing.R` les télécharge à chaque
exécution. Une connexion internet est nécessaire.

| Donnée | Source | Identifiants | Fonction R |
|----|----|----|----|
| Taux de change (prix quotidiens) | Yahoo Finance | `EURUSD=X`, `GBPUSD=X`, `CADUSD=X`, `CHFUSD=X`, `JPYUSD=X` | `quantmod::getSymbols(src = "yahoo")` |
| Or, S&P 500, pétrole | Yahoo Finance | `GC=F`, `^GSPC`, `CL=F` | `quantmod::getSymbols(src = "yahoo")` |
| Taux US 2 ans et 10 ans | FRED | `DGS2`, `DGS10` | `quantmod::getSymbols(src = "FRED")` |
| Taux de financement (Fed Funds effectif) | FRED | `DFF` | `quantmod::getSymbols(src = "FRED")` |
| Taux zone euro 2 ans et 10 ans | Banque centrale européenne (courbe des taux) | maturités `2Y`, `10Y` | `readecb::ecb_yield_curve()` |
| Taux Royaume-Uni 2 ans et 10 ans | Banque d'Angleterre (courbe nominale, taux spot) | maturités 2 et 10 | `boe::boe_curve_panel()` |

## Fichiers bruts de `Raw/`

Ces trois fichiers ne peuvent pas être téléchargés par le code : ils ont
été exportés à la main.

### `cad_rates.csv` : taux du Canada, 2 ans et 10 ans

-   Source : Statistique Canada, statistiques du marché financier
    (données de la Banque du Canada), fréquence quotidienne.
-   Séries : « Government of Canada benchmark bond yields, 2 year »
    (vecteur `v39051`) et « ..., 10 year ».

### `chf_rates.csv` : taux de la Suisse, 2 ans et 10 ans

-   Source : Bloomberg.
-   Séries : `GSWISS02 Index` (2 ans) et `GSWISS10 Index` (10 ans),
    quotidien.

### `jpy_rates.csv` : taux du Japon, 2 ans et 10 ans

-   Source : Bloomberg.
-   Séries : `GJGB2 Index` (2 ans) et `GJGB10 Index` (10 ans),
    quotidien.

## Traitement

`Code/data_processing.R` ramène chaque série à une observation par
semaine : pour chaque jeudi, la dernière valeur connue ce jeudi ou
avant, si elle date de 5 jours au plus (fonction `to_thursday`, dans
`Function/functions.R`). `Code/build_features.R` construit ensuite les
features (écarts de taux avec les États-Unis, rendements de l'or, du S&P
500 et du pétrole, indicateurs techniques).
