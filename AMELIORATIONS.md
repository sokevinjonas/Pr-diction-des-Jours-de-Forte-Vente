# Propositions d'amelioration du modele

## Etat actuel

Le modele predit avec une precision moyenne de **78%** (MAPE ~22%) sur des donnees reelles.
C'est suffisant pour donner des alertes utiles, mais pas assez pour des recommandations
quantitatives precises (ex: "commandez exactement 47 cartons de riz").

Objectif : passer a **85-88% de precision** (MAPE 12-15%).

---

## 1. Ensemble de modeles (XGBoost + LightGBM)

**Quoi** : Au lieu d'un seul modele, entrainer 2 modeles differents et faire la moyenne
de leurs predictions.

**Pourquoi** : Chaque modele fait des erreurs differentes. XGBoost peut se tromper sur un
jour ou LightGBM a raison, et inversement. En moyennant, les erreurs individuelles se
compensent. C'est le meme principe que demander l'avis a 2 experts au lieu d'un seul.

**Impact estime** : -1 a -3% MAPE

**Effort** : Faible (30 min) — LightGBM a la meme API que XGBoost.

---

## 2. Lags supplementaires (J-28, meme semaine annee passee)

**Quoi** : Ajouter comme features :
- `ventes_j_28` : ventes il y a exactement 4 semaines
- `ventes_meme_semaine_an_passe` : meme semaine l'annee derniere
- `ratio_j7_vs_moy` : ventes de la semaine derniere / moyenne historique

**Pourquoi** : Le commerce a des cycles mensuels (paie tous les 30 jours) et annuels
(Tabaski revient chaque annee). Le modele actuel ne "voit" que 30 jours en arriere.
En lui donnant l'annee passee, il peut mieux anticiper les fetes recurentes.

**Impact estime** : -1 a -2% MAPE

**Effort** : Faible (10 min) — juste des shift() supplementaires.

---

## 3. Optimisation des hyperparametres avec Optuna

**Quoi** : Au lieu de choisir manuellement les parametres du modele (nombre d'arbres,
profondeur, learning rate), laisser un algorithme tester des centaines de combinaisons
et garder la meilleure.

**Pourquoi** : Les hyperparametres actuels sont "raisonnables" mais pas optimaux pour
chaque dataset. Un supermarche avec des ventes stables beneficie d'un modele different
d'un commerce Mobile Money avec des pics imprevisibles. Optuna trouve la meilleure
combinaison en 50-200 essais.

**Exemple** : Peut-etre que pour le grossiste, `max_depth=4` avec `learning_rate=0.08`
marcherait mieux que les valeurs par defaut.

**Impact estime** : -2 a -4% MAPE

**Effort** : Moyen (1h de calcul automatique, pas d'intervention humaine).

---

## 4. Features meteo (API open-meteo.com)

**Quoi** : Ajouter la temperature et la pluie comme variables predictives.

**Pourquoi** : En Afrique de l'Ouest, la saison des pluies (hivernage) impacte fortement
le commerce :
- Pluie forte = moins de clients dans les marches ouverts
- Chaleur extreme = plus de boissons vendues
- Debut d'hivernage = hausse des ventes de parapluies, baches, etc.

Le modele actuel ne sait pas qu'il pleut demain. Avec cette info, il peut ajuster.

L'API open-meteo.com est **gratuite** et fournit l'historique et les previsions a 14 jours.

**Impact estime** : -2 a -4% MAPE (surtout pour les marches ouverts)

**Effort** : Faible (1h) — API simple, pas de cle necessaire.

---

## 5. Modele specialise pour la detection des pics

**Quoi** : En plus du modele de regression (predit un montant), entrainer un modele de
classification binaire : "ce jour sera-t-il un pic oui ou non ?"

**Pourquoi** : Le modele actuel a un recall de pics de seulement 6-33%. Il detecte mal
les jours exceptionnels. Un classificateur entraine specifiquement sur les pics
(classe 1 = ventes > +30% vs normale) sera meilleur pour declencher des alertes.

**En pratique** : Le modele de regression donne le montant, le classificateur confirme
ou infirme l'alerte. Si les deux sont d'accord → alerte fiable.

**Impact estime** : +20 a +30% de recall sur les pics

**Effort** : Moyen (ajouter un second modele XGBClassifier).

---

## 6. Plus de donnees historiques

**Quoi** : Passer de 1-2 ans a 3+ ans d'historique.

**Pourquoi** : Le modele apprend des patterns annuels (fetes, saisons). Avec 1 an, il ne
voit chaque fete qu'une seule fois — il ne peut pas distinguer si l'effet Tabaski est
de +50% ou +100%. Avec 3 ans, il a 3 exemples de chaque fete et peut moyenner.

**Regle empirique** : en prevision de series temporelles, la precision s'ameliore
de ~2-3% MAPE par annee supplementaire de donnees, jusqu'a un plateau a 3-5 ans.

**Impact estime** : -3 a -5% MAPE

**Effort** : Depend du client (faut que les donnees existent).

---

## 7. Features de contexte externe

**Quoi** : Ajouter des variables comme :
- Jour de marche hebdomadaire (lundi = marche a Sandaga, mercredi = marche HLM)
- Coupure d'electricite prevue
- Match de football important (CAN, Coupe du Monde)
- Jour de greve ou manifestation

**Pourquoi** : Ces evenements impactent fortement les ventes mais sont invisibles dans
les donnees numeriques. Un match du Senegal un dimanche soir → les bars et restaurants
explosent. Une coupure d'electricite → les gens achètent des bougies et glacieres.

**Impact estime** : -2 a -3% MAPE (fort sur certains jours specifiques)

**Effort** : Eleve (collecte manuelle des donnees, pas d'API fiable).

---

## Ordre de priorite recommande

| Priorite | Amelioration | Effort | Impact |
|----------|-------------|--------|--------|
| 1 | Ensemble XGBoost + LightGBM | 30 min | -2% MAPE |
| 2 | Lags supplementaires | 10 min | -1.5% MAPE |
| 3 | Optuna (hyperparametres) | 1h calcul | -3% MAPE |
| 4 | Features meteo | 1h code | -3% MAPE |
| 5 | Classificateur de pics | 2h code | +25% recall |
| 6 | Plus de donnees | Variable | -4% MAPE |
| 7 | Contexte externe | Long | -2% MAPE |

**Total estime si on implemente 1-5 : passer de 78% a 85-88% de precision.**
