# Synthèse des résultats — pour Ulrich et Rami

Ce document résume, en quelques pages, ce qui a été fait et trouvé avant la rédaction du rapport en
équipe. **Ce n'est pas le rapport :** il sert à ce qu'on se mette d'accord sur les résultats et sur les
décisions à prendre ensemble. Le détail complet (choix, protocole, résultats, littérature, références
vérifiées) est dans [`docs/JOURNAL_PROJET.md`](JOURNAL_PROJET.md). Les figures (PDF) et les tableaux
(LaTeX) prêts pour Overleaf sont dans [`docs/paquet_equipe/`](paquet_equipe/).

> **Version 1, à relire pour la méthode et non pour les résultats.** Le barème ne note pas la
> performance obtenue : il note la question, la littérature, la conception des expériences et la
> présentation. En relisant cette V1, concentrez-vous sur la **logique des tests** : chaque test
> répond-il à la question ? Les choix sont-ils faits sans regarder le test ? Les comparaisons sont-elles
> équitables ? Les chiffres sont inclus pour que vous ayez tout, mais ils ne sont pas l'objet de la
> relecture. La section 8 propose une revue critique de la méthode et des pistes pour la rendre plus
> rigoureuse.

---

## 1. Question et approche

**Question validée par le prof :** des modèles d'apprentissage automatique, combinés à une prévision de
volatilité et à un filtre de classification, peuvent-ils produire un signal ajusté au risque assez
fiable pour bâtir une stratégie long/short hebdomadaire rentable sur cinq grandes devises ?

**Approche : une étude d'ablation.** La question contient trois blocs ML. On évalue chacun seul, avec des
métriques ML, puis on les ajoute un à un à la stratégie pour mesurer ce que chacun apporte. Conformément
aux consignes du prof, la finance ne sert que d'évaluation finale.

![Pipeline](paquet_equipe/figures/fig02_pipeline.png)

## 2. Protocole

Trois règles, appliquées à toutes les expériences :
1. tous les choix (modèle, réglage, seuil, poids) se font **sur la validation seulement** ;
2. chaque variante est évaluée **une seule fois** sur le test ;
3. **tous les résultats sont rapportés**, même mauvais.

Les données de l'ancien projet sont figées, les versions des paquets aussi (`requirements-lock.txt`), et
le pipeline se reproduit exactement, sauf une petite différence de la forêt aléatoire due à la machine
(journal, étape 0).

![Protocole](paquet_equipe/figures/fig03_protocole.png)

## 3. Résultats clés

**Bloc 1 — Prévoir les rendements.**
- Après correction de Holm pour 40 tests, **aucun modèle ne bat la moyenne historique** sur le test
  (11 tests significatifs à 10 % sans correction, 0 avec) — tableau 1, figure 4.
- Choisir un modèle par devise (MIXT) est **instable** : rééchantillonner la validation ne fait rechoisir
  le même modèle que 35 % à 80 % du temps (figure 9). Sur le test, la **moyenne simple** des 7 modèles
  fait mieux que MIXT ($R^2_{OS}$ de −0,32 % contre −1,80 %) — tableau 3, figure 7.
- **Régler la forêt** sur la validation fait passer son $R^2_{OS}$ de test de −5,23 % à −0,15 % (figure 6).
- La validation croisée par plis aléatoires est gardée selon la règle ; c'est pourtant le
  *forward-chaining* qui fait le mieux sur le test (tableau 2, figure 5).
- Les variables qui comptent un peu sont le rendement de la semaine passée et la volatilité sur
  4 semaines ; les modèles ne s'accordent pas d'une devise à l'autre (figures 10 et 11).

**Bloc 2 — Prévoir la volatilité.**
- Le **HAR** prévoit un peu mieux que le GARCH (QLIKE 0,401 contre 0,415 sur le test), sans écart
  significatif après Holm ; les deux battent nettement les références naïves — tableau 4, figure 12.
- **Diviser la prévision de rendement par la volatilité n'améliore pas le classement des devises**
  (figure 14).

**Bloc 3 — Le méta-classifieur** (l'élément central de la nouveauté).
- Il sépare à peine les bonnes jambes des mauvaises : AUC de 0,53 (achats) et 0,55 (ventes) ;
  permutation des étiquettes $p$ = 0,14 et 0,06 — tableau 5, figures 15 et 17.
- **Placebo :** 12 % des filtres au hasard qui gardent le même nombre de jambes font au moins aussi bien
  que lui ($p$ = 0,118) — figure 16.
- La **force du signal du premier modèle** ($|\hat s|$) classe les jambes aussi bien que lui — tableau 6,
  figure 18.

**Ablation de bout en bout** (tableau 7, figures 19 et 20).

| | Naïf | + prévision ML | + volatilité | + méta-classifieur |
|---|---|---|---|---|
| Chaîne d'origine (Sharpe, test) | 0,15 | 0,54 | 0,45 | **0,88** |
| Chaîne mise à jour, exploratoire | 0,15 | 0,57 | 0,52 | 0,74 |

- Le Sharpe de 0,88 est le seul dont l'intervalle de confiance à 95 % exclut zéro ([0,09 ; 1,83]), mais
  il ne résiste ni au placebo ($p$ = 0,117), ni à la correction pour les 29 stratégies essayées dans le
  projet (Deflated Sharpe de 0,77, sous le seuil de 0,95).
- **Effet mécanique découvert :** le filtre garde surtout des semaines avec la vente seule, qui encaissent
  le taux de financement (DFF). Sans ce financement, le Sharpe tombe de 0,88 à 0,73 ; dans la chaîne mise
  à jour, le gain du filtre disparaît entièrement.

**Complément LSTM** (tableau 8, figure 21) : un LSTM à 5 sorties, réglé sur la validation et moyenné sur
5 graines, ne classe pas mieux les devises qu'une Ridge réestimée aux mêmes dates. Son léger gain de
précision vient de la moyenne des graines.

## 4. Réponse à la question

Sur cinq devises à l'horizon hebdomadaire, la combinaison proposée **ne produit pas un signal dont on
puisse établir statistiquement la fiabilité**. La performance observée (Sharpe de 0,88) est compatible
avec la chance une fois contrôlés le nombre d'essais, la sélection des jambes et le financement. C'est
un résultat négatif, mais démontré avec rigueur, et le barème note la qualité de la démonstration, pas
la performance.

## 5. Limites à discuter dans le rapport

- **Puissance statistique faible :** 260 semaines de test donnent un intervalle de confiance d'environ
  ±0,9 sur un Sharpe annualisé.
- **Une seule période de test** (2021–2026), marquée par la hausse des taux ; aucune autre période
  hors échantillon n'est disponible.
- **Coûts de transaction et portage (*carry*) ignorés**, comme dans l'ancien projet.
- **Résultats de test déjà vus** avant ces expériences (ancien projet) : c'est pourquoi la chaîne
  d'origine reste la réponse principale, et la chaîne mise à jour est présentée comme exploratoire.
- **Méta-classifieur entraîné sur des jambes de validation optimistes** (MIXT choisi sur ces mêmes
  semaines) — point P2 du journal.
- **Courbes risque-couverture :** diagnostic qui classe toutes les jambes de test à la fois, pas une
  règle utilisable en temps réel.
- **Reproductibilité de la forêt d'une machine à l'autre** : petites différences pour CAD et CHF.

## 6. Pistes pour les travaux futurs

- Entraîner le modèle directement pour la décision de classement plutôt que pour l'erreur quadratique
  (apprentissage guidé par la décision, *learning to rank* sur le change).
- Plus de devises et plus de prédicteurs, ou un horizon mensuel, où la littérature trouve plus de
  prévisibilité.
- Le débat sur la « vertu de la complexité » appliqué au change.
- Intégrer les coûts de transaction et le portage dans l'évaluation.

## 7. Décisions à prendre en équipe

1. **Style de citation** du rapport : auteur-année (comme dans le journal) ou numérique.
2. **Partie financière :** la raccourcir (Moreira et Muir, Menkhoff et al.) pour laisser la place au ML,
   comme le demande le prof ?
3. **Répartition des 13 pages :** quelles figures dans le texte, lesquelles en annexe (21 figures et
   8 tableaux sont disponibles).
4. **Langue et ton du rapport** (français, comme le plan d'étude ?).

**Point P3, résolu :** la règle de la stratégie « corrigée » (ne rien faire quand tous les Sharpe prévus
sont positifs) était déjà dans l'ancien projet et a été décidée avant de voir le test. C'est une variante
légitime, comptée parmi les 29 essais du Deflated Sharpe.

Les DOI de Joubert (2022) et de Meyer et al. (2023) sont maintenant confirmés (journal, section 11).

## 8. Revue critique de la méthode (V1)

Cette revue juge la conception des tests, pas leurs résultats. Elle reprend les critères du barème et
les conseils du cours : ne pas régler les paramètres en trichant, ne pas comparer injustement, et
vérifier que tout fonctionne encore quand on retire une partie des données ou qu'on ajoute du bruit.

### 8.1 Ce qui est solide

- **Une seule question, découpée en trois blocs ML évalués chacun avec sa propre métrique**
  ($R^2_{OS}$, QLIKE, AUC), puis ensemble par ablation. La finance ne sert qu'à l'évaluation finale.
- **Le protocole :** les choix se font sur la validation, chaque variante n'est évaluée qu'une fois sur
  le test, et tout est rapporté. Le registre des expériences recense chaque variante essayée.
- **Les comparaisons sont équitables :** mêmes dates de ré-estimation (la Ridge est réestimée au même
  rythme que le LSTM, par exemple), et chaque réglage est fait sur la validation.
- **Les tests multiples sont contrôlés :** Holm sur 40 tests, Deflated Sharpe avec tous les essais du
  projet, validation emboîtée pour choisir la combinaison.
- **Des tests témoins :** placebo du filtre, permutation des étiquettes, 1 000 stratégies au hasard.
- **La reproductibilité :** versions figées, journaux d'exécution, un script par expérience, et des
  écarts de reproduction expliqués.

### 8.2 Les faiblesses de rigueur, par ordre d'importance

1. **Aucun test qui prouve que le pipeline peut trouver un signal quand il y en a un.** Tous nos
   résultats sont négatifs. Un lecteur critique peut donc se demander si un bogue ne détruirait pas
   tout signal. Un résultat négatif n'est convaincant que si l'on montre que la méthode réussit sur un
   cas où la bonne réponse est connue.
2. **La puissance statistique n'est pas chiffrée.** Avec 260 semaines de test, quelle est la plus
   petite amélioration ($R^2_{OS}$, AUC, Sharpe) qu'on aurait pu détecter ? Sans ce chiffre, on ne peut
   pas distinguer « il n'y a pas de signal » de « le test est trop petit pour le voir ».
3. **La robustesse n'est pas testée.** La liste de contrôle du journal (section 12) la prévoit
   (sous-périodes, sensibilité aux choix fixés *a priori*), mais elle n'a pas été faite. C'est
   exactement le type de test que le prof demande en plus.
4. **Le test n'est pas vierge.** L'ancien projet l'avait déjà regardé. Nous l'avons signalé et traité
   (chaîne d'origine comme réponse principale, chaîne mise à jour comme exploratoire), mais il n'existe
   aucune période ni aucune devise jamais vue.
5. **Le pré-enregistrement ne peut pas être prouvé.** Les règles ont été écrites avant les calculs,
   mais elles ont été commitées en même temps que les résultats (commit 902c24c). L'historique git ne
   permet donc pas de le démontrer.
6. **L'ablation est un escalier dans un ordre fixe** (ML, puis volatilité, puis filtre). On ne sait
   pas ce que vaut le filtre sans la volatilité, ni la volatilité sans la prévision ML : les
   interactions entre les blocs ne sont pas mesurées.
7. **Le méta-classifieur est en partie circulaire.** Ses trois variables viennent toutes du signal
   lui-même. Il ne peut donc que reclasser les jambes selon la force du signal, ce qui explique en partie
   que $|\hat s|$ seul fasse aussi bien. De plus, il est entraîné sur des jambes de validation
   optimistes (point P2).
8. **Des incohérences entre le texte et ce qui a été calculé :**
   - le *Model Confidence Set* est présenté dans le journal, mais n'a pas été appliqué ;
   - en change, la référence classique est la marche aléatoire, soit une prévision nulle (Meese et
     Rogoff, 1983), alors que notre $R^2_{OS}$ est mesuré contre la moyenne historique ;
   - le test de Clark-West est conçu pour des modèles emboîtés, alors que la forêt et MIXT ne sont pas
     des extensions linéaires de la moyenne historique.
9. **Des choix fixés sans analyse de sensibilité :** blocs de 8 semaines pour le bootstrap, seuil de
   0,5 pour le filtre, 60 % d'entraînement initial. Le Deflated Sharpe traite aussi les 29 essais comme
   indépendants, alors qu'ils sont très corrélés.
10. **Aucun test unitaire du code.** Les fonctions statistiques (Clark-West, Deflated Sharpe, bootstrap,
    plis avec tampon) ne sont vérifiées que par des `assert` ponctuels.

### 8.3 Pistes d'amélioration réalistes, classées par priorité

Toutes ces pistes respectent le protocole : elles ne modifient aucun choix et ne servent qu'à vérifier
la méthode. Pour chacune, les règles seraient **commitées avant le calcul**, ce qui règle la faiblesse 5
pour la suite. Chaque nouvel essai sur le test serait ajouté au registre et compté dans le Deflated
Sharpe.

**Priorité 1 : peu coûteux, gain de rigueur important.**

| Piste | Ce qu'elle démontre | Faiblesse traitée |
|---|---|---|
| **Signal planté :** remplacer la cible par des données simulées contenant un signal connu, et vérifier que les 3 blocs le retrouvent | que le pipeline fonctionne et peut détecter un vrai signal | 1 |
| **Cible permutée :** mélanger la cible du bloc 1 et vérifier que le $R^2_{OS}$ tombe sous zéro | l'absence de fuite d'information | 1 |
| **Effet minimal détectable** pour le $R^2_{OS}$, l'AUC et le Sharpe avec 260 semaines | la portée exacte du résultat négatif | 2 |
| **Robustesse :** test coupé en deux sous-périodes ; entraînement initial de 50 %, 60 % ou 70 % ; blocs bootstrap de 4, 8 ou 16 semaines ; seuils de 0,45 à 0,55 | que les conclusions ne tiennent pas à un choix arbitraire | 3, 9 |
| **Tests unitaires** (`pytest`) des fonctions statistiques, comparées à des valeurs de référence | que le code calcule ce qu'il prétend calculer | 10 |
| **Mettre le texte en accord avec le code :** appliquer le MCS (ou le retirer du texte), ajouter la marche aléatoire comme seconde référence, et signaler la limite de Clark-West | la cohérence entre la méthode décrite et la méthode calculée | 8 |

**Priorité 2 : coût moyen.**

| Piste | Ce qu'elle démontre | Faiblesse traitée |
|---|---|---|
| **Ablation factorielle :** les 8 combinaisons des 3 blocs (avec ou sans chacun) | l'apport propre de chaque bloc et leurs interactions | 6 |
| **Validation coupée en deux :** MIXT choisi sur la première moitié, filtre entraîné sur la seconde | l'ampleur du biais P2 | 7 |
| **Deflated Sharpe avec un nombre effectif d'essais** (essais regroupés selon leur corrélation), en plus de N = 29 | la sensibilité de la correction au compte des essais | 9 |

**Priorité 3 : facultatif, à présenter comme exploratoire ou comme travaux futurs.**

- Ajouter au méta-classifieur des variables indépendantes du signal (régime de volatilité du marché,
  écarts de taux), choisies sur la validation (faiblesse 7).
- Une évaluation sur des devises jamais vues (AUD, NZD, SEK, NOK), seul vrai test vierge possible, si
  les données de taux existent (faiblesse 4).
- Une sensibilité aux coûts de transaction, en points de base par jambe.

---

## 9. Contenu du paquet

| Fichier | Contenu |
|---|---|
| `paquet_equipe/figures/fig01` à `fig21` (PDF) | Figures numérotées dans l'ordre logique : découpage, pipeline, protocole, puis blocs 1, 2, 3, ablation et LSTM |
| `paquet_equipe/figures/fig02`, `fig03` (PNG aussi) | Schémas du pipeline et du protocole |
| `paquet_equipe/tableaux/tab01` à `tab08` (`.tex` + `.csv`) | Tableaux principaux, générés automatiquement à partir des résultats ; à inclure avec `\input{}`, nécessitent `\usepackage{booktabs}` |
| `Code/experiments/` | Un script par expérience, plus `e7_paquet_equipe.py` qui régénère ce paquet |
| `experiments/` | Résultats complets de chaque expérience (CSV, figures, environnement) |

Tout se régénère avec les commandes de l'étape 0 du journal, puis les scripts de `Code/experiments/`.
