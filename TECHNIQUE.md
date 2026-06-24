# Documentation Technique — Prediction des Jours de Forte Vente

## 1. Vue d'ensemble du pipeline

```
DONNEES BRUTES          FEATURE ENGINEERING         MODELE              PREDICTION
(CSV/Excel/JSON)  -->   (27 variables)        -->  (XGBoost)     -->   (J+1 a J+30)
     |                       |                        |                     |
     v                       v                        v                     v
 Dates + Montants     Temporelles + Lags +      600 arbres de        Montant prevu
                      Evenements                 decision en          + Niveau alerte
                                                 cascade              + Recommandations
```

## 2. Flux des donnees

### 2.1 Entree (data_loader.py)

Le systeme accepte n'importe quel fichier contenant au minimum :
- Une colonne de **dates**
- Une colonne de **montants/ventes**

Detection automatique :
1. Lecture du fichier (CSV, Excel, JSON)
2. Recherche de la colonne date parmi : `date`, `jour`, `ds`, `fecha`, `created_at`...
3. Recherche de la colonne montant parmi : `sales`, `amount`, `revenue`, `total`, `ca`...
4. Si plusieurs lignes par jour (transactionnel) → agregation automatique par jour
5. Resultat : DataFrame avec `date`, `montant_total`, `nb_transactions`

Si la detection echoue → interface de mapping manuel (l'utilisateur associe ses colonnes).

### 2.2 Feature Engineering (feature_engineering.py)

A partir d'une serie de dates et montants, on cree **27 variables explicatives** :

#### Features temporelles (14 variables)
```
jour_semaine        : 0 (Lundi) a 6 (Dimanche)
jour_mois           : 1 a 31
semaine_mois        : 1 a 5
mois                : 1 a 12
trimestre           : 1 a 4
est_weekend         : 1 si samedi ou dimanche
est_debut_mois      : 1 si jour 1-5 (effet paie)
est_fin_mois        : 1 si jour 26-31
jour_annee          : 1 a 365
jour_semaine_sin    : sin(2*pi*jour/7) — encodage cyclique
jour_semaine_cos    : cos(2*pi*jour/7)
mois_sin            : sin(2*pi*mois/12)
mois_cos            : cos(2*pi*mois/12)
```

Pourquoi l'encodage cyclique ? Sans lui, le modele pense que Dimanche (6) est loin de
Lundi (0). Avec sin/cos, la distance est correctement representee.

#### Features evenementielles (4 variables)
```
est_jour_ferie      : 1 si fete (Tabaski, Korite, Noel, etc.)
veille_ferie        : 1 si lendemain est ferie
nb_jours_avant_fete : distance en jours avant la prochaine fete
nb_jours_apres_fete : distance en jours apres la derniere fete
```

#### Features de lag — valeurs passees (9 variables)
```
ventes_j_1          : montant d'hier
ventes_j_2          : montant d'avant-hier
ventes_j_7          : meme jour la semaine derniere
ventes_j_14         : meme jour il y a 2 semaines
ventes_moy_7j       : moyenne des 7 derniers jours
ventes_moy_14j      : moyenne des 14 derniers jours
ventes_moy_30j      : moyenne des 30 derniers jours
ventes_std_7j       : ecart-type sur 7 jours (volatilite)
tendance_7j         : pente de regression sur 7 jours (monte ou descend ?)
ratio_vs_moy_30j    : ventes hier / moyenne 30 jours (anomalie ?)
```

IMPORTANT : tous les lags sont decales de 1 jour (shift). On n'utilise JAMAIS
la valeur du jour qu'on predit. C'est ce qui evite le "data leakage".

### 2.3 Modele (model.py)

#### Algorithme : XGBoost (eXtreme Gradient Boosting)

Principe : construire des arbres de decision en **cascade**. Chaque arbre corrige
les erreurs du precedent.

```
Arbre 1 : predit grossierement (erreur = 500 000 FCFA)
Arbre 2 : corrige l'erreur de l'arbre 1 (erreur residuelle = 200 000)
Arbre 3 : corrige l'erreur de l'arbre 2 (erreur residuelle = 80 000)
...
Arbre 600 : affine la prediction (erreur finale = ~240 000 FCFA)
```

#### Hyperparametres

```python
n_estimators    = 600       # nombre d'arbres
learning_rate   = 0.04      # pas d'apprentissage (petit = plus precis, plus lent)
max_depth       = 6         # profondeur max de chaque arbre
subsample       = 0.8       # 80% des donnees par arbre (evite l'overfitting)
colsample       = 0.8       # 80% des features par arbre
min_child_weight = 5        # minimum d'exemples dans une feuille
reg_alpha       = 0.1       # regularisation L1 (sparsity)
reg_lambda      = 1.0       # regularisation L2 (lissage)
```

Ces parametres s'adaptent automatiquement a la taille des donnees :
- < 100 jours : modele simple (100 arbres, profondeur 3)
- 100-500 jours : modele moyen (300 arbres, profondeur 5)
- > 500 jours : modele complet (600 arbres, profondeur 6)

#### Validation

Split temporel (pas de melange passe/futur) :
```
|---- 80% entrainement ----|---- 20% test ----|
|    2013 ---- 2016        |     2017         |
```

Validation croisee temporelle (TimeSeriesSplit) en interne pour estimer la variance.

#### Auto-adaptation

Le modele utilise un "fingerprint" des donnees (hash de : nb lignes, moyenne, ecart-type,
dates min/max). Si les donnees changent → reentrainement automatique.

### 2.4 Prediction (predict.py)

Pour predire J+1 a J+N :

```
1. Charger l'historique
2. Pour chaque jour futur :
   a. Construire les features (temporelles + evenements)
   b. Calculer les lags a partir de l'historique connu
   c. Appliquer le modele → montant prevu
   d. Calculer la variation vs moyenne 30 jours
   e. Determiner le niveau d'alerte (NORMAL / ORANGE / ROUGE)
   f. Generer les recommandations metier
   g. Ajouter la prediction a l'historique (pour les lags du jour suivant)
3. Retourner le tableau de predictions
```

Seuils d'alerte :
- Variation >= +30% → ORANGE (attention)
- Variation >= +60% → ROUGE (critique)

### 2.5 Sortie

```json
{
  "date": "2024-07-05",
  "jour": "Vendredi",
  "prediction": 2100000,
  "variation": "+68%",
  "niveau_alerte": "ROUGE",
  "message": "Pic de ventes attendu (+68%). Renforcer le stock et le personnel.",
  "recommandations": {
    "stock_supplementaire": true,
    "personnel_extra": 3,
    "caisses_recommandees": 8
  }
}
```

## 3. Metriques d'evaluation

| Metrique | Definition | Objectif |
|----------|-----------|----------|
| MAE | Erreur moyenne absolue (en FCFA) | Le plus bas possible |
| MAPE | Erreur moyenne en % | < 15% = bon |
| RMSE | Erreur quadratique (penalise les grosses erreurs) | < MAE x 1.5 |
| Recall pics | % de vrais pics detectes (sensibilite) | > 65% |
| Precision pics | % d'alertes qui sont de vrais pics | > 70% |

## 4. Importance des features

Le modele attribue un score d'importance a chaque variable. Plus le score est eleve,
plus la variable influence la prediction.

Resultats typiques :
```
1. veille_ferie         0.187  ← la plus importante
2. nb_jours_avant_fete  0.141
3. est_debut_mois       0.140
4. est_weekend          0.129
5. jour_mois            0.051
```

Interpretation : les ventes sont principalement determinees par la proximite d'une fete
et le moment du mois (paie), plus que par le jour de la semaine seul.

## 5. Limites et axes d'amelioration

| Limite | Impact | Solution |
|--------|--------|----------|
| Donnees synthetiques = patterns codes a la main | Le modele "retrouve" ce qu'on a mis | Utiliser des donnees reelles |
| Pas de features meteo | Ignore l'effet pluie/chaleur | Ajouter API open-meteo |
| Un seul modele pour tout | Patterns differents par magasin | Un modele par commerce |
| Pas de detection d'anomalies | Ne voit pas les evenements imprevisibles | Ajouter un module de detection |
| Calendrier fetes statique | Ne couvre pas toutes les annees futures | Generer dynamiquement les dates |

## 6. Stack technique

| Composant | Technologie | Pourquoi |
|-----------|-------------|----------|
| ML | XGBoost | Rapide, performant, interpretable |
| Features | pandas + numpy | Standard, efficace |
| Dashboard | Streamlit + Plotly | Rapide a developper, interactif |
| Export | openpyxl (Excel), fpdf2 (PDF) | Formats demandes par les PME |
| Tests | pytest | 16 tests couvrant le pipeline |
| Serialisation | joblib | Sauvegarde/chargement rapide du modele |
