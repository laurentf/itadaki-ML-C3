# 🍜 Itadaki ML - Recipe Calorie Prediction System

_"Itadaki" signifie "bon appétit" en japonais_

## 🎓 Projet de certification DEV IA - Alyra

Ce projet constitue le **projet final** de la certification **Développeur Intelligence Artificielle** d'Alyra. Il implémente un système de prédiction du niveau calorique des recettes utilisant l'apprentissage automatique.

**Compétences évaluées (Bloc 3 : Appliquer des techniques d'analyse IA via des algorithmes d'apprentissage automatique)** :

- **C1** : sélectionner l'algorithme d'apprentissage le plus adapté en comparant les performances et les caractéristiques des différentes familles d'algorithmes afin d'apporter une réponse pertinente à la problématique métier rencontrée
- **C2** : préparer et transformer des données en utilisant des techniques de prétraitement (preprocessing) pour les adapter aux spécificités du modèle d'apprentissage automatique choisi
- **C3** : entraîner un modèle d'apprentissage automatique en optimisant une loss function (fonction de coût) à partir des données d'entraînement afin de permettre à l'algorithme d'effectuer le moins d'erreurs possibles selon des indicateurs de succès clairement définis

## 🎯 Vue d'ensemble

Système d'intelligence artificielle qui **prédit les calories des recettes** à partir de leurs ingrédients. Ce projet explore une **approche méthodique** d'amélioration itérative en testant différents algorithmes et optimisations de preprocessing.

## 🚀 Choix de l'approche : Machine Learning avant Deep Learning

### 🎯 Stratégie progressive validée

**Pourquoi commencer par ML traditionnel ?**

1. **🔍 Compréhension du problème** : ML classique permet d'identifier les patterns nutritionnels
2. **⚡ Rapidité de développement** : itérations rapides vs complexité DL
3. **🎪 Interprétabilité** : features importantes compréhensibles par les nutritionnistes
4. **📊 Baseline solide** : établir un benchmark avant de complexifier
5. **🎯 Efficacité démontrée** : ML peut suffire pour ce type de problème

**Résultat :** ML traditionnel s'avère **satisfaisant** (53% accuracy sur 3 classes), validant cette approche qui répond au besoin fonctionnel exprimé par l'équipe.

## 📈 Évolution progressive - 3 versions itératives

Ce projet documente une **approche méthodique** avec 3 versions d'amélioration continue :

### 🔸 Version 1 (v1) : XGBoost + NLP Basique ✅

**Approche :** Première implémentation avec feature engineering classique

**🏗️ Architecture :**

```
Ingrédients → TF-IDF (500 features) + Features numériques (15) → XGBoost
```

**🔧 Composants :**

- **TF-IDF** : vectorisation basique des ingrédients (ngram_range=(1,2))
- **Feature Engineering** : compteurs et ratios par catégorie nutritionnelle
- **XGBoost** : paramètres par défaut avec optimisation légère

**📊 Features numériques (15) :**

- **Compteurs** : nb_fat, nb_sugar, nb_protein, nb_vegetable, nb_grain, nb_drink, nb_spice
- **Ratios** : fat_ratio, sugar_ratio, protein_ratio, etc. (pour normaliser par nb d'ingrédients)
- **Total** : n_ingredients

**⚠️ Découverte clé :** **TF-IDF seul ne suffit pas !** Les features numériques sont essentielles.

**🎯 Résultats v1 :**

- **Test accuracy :** ~50-51%
- **Apprentissage :** l'approche est viable mais perfectible

### 🔸 Version 2 (v2) : XGBoost + NLP optimisé ✅

**Approche :** optimisation du preprocessing NLP et normalisation des ingrédients

**🏗️ Architecture améliorée :**

```
Ingrédients → Normalisation → TF-IDF (300 features, ngram=(1,1)) + Features numériques → XGBoost optimisé
```

**🔧 Améliorations majeures :**

1. **🎯 Normalisation intelligente des ingrédients :**

   ```python
   ingredient_normalization = {
       'brown_sugar': 'sugar',
       'white_sugar': 'sugar',
       'granulated_sugar': 'sugar',
       'extra_virgin_olive_oil': 'olive_oil',
       'unsalted_butter': 'butter',
       # ... +200 normalisations
   }
   ```

2. **📝 NGrams optimisés :**

   - **ngram_range=(1,1)** au lieu de (1,2)
   - **Logique :** avec underscores (`olive_oil`), les ingrédients sont déjà des tokens uniques
   - **Résultat :** meilleur regroupement, moins de bruit

3. **🎪 Catégories d'ingrédients étendues :**

   ```python
   fat_ingredients = {'butter', 'olive_oil', 'vegetable_oil', 'coconut_oil', ...}
   protein_ingredients = {'chicken', 'beef', 'fish', 'eggs', 'tofu', ...}
   sugar_ingredients = {'sugar', 'honey', 'syrup', 'brown_sugar', ...}
   # + vegetable, grain, spice, drink ingredients
   ```

4. **⚙️ XGBoost optimisé :**
   ```python
   XGBClassifier(
       n_estimators=200,
       max_depth=6,
       learning_rate=0.1,
       subsample=0.8,
       colsample_bytree=0.8,
       reg_alpha=0.2,
       reg_lambda=1.2
   )
   ```

**🎯 Résultats v2 :**

- **Test accuracy :** **53.49%** (+2-3% vs v1)
- **Train accuracy :** 61.09%
- **Gap d'overfitting :** 7.6% (bien maîtrisé)
- **Performance par classe :**
  - BAS : 57% precision, 67% recall ✅
  - MOYEN : 57% precision, 51% recall ✅
  - HAUT : 46% precision, 42% recall (plus difficile)

**✅ Amélioration confirmée :** le preprocessing intelligent améliore significativement les résultats des tests (notamment sur le cas "moyen") !

### 🔸 Version 3 (v3) : Random Forest - test de robustesse ⚡

**Approche :** tester si un modèle plus "simple" et robuste peut égaler XGBoost

**🌲 Pourquoi Random Forest ?**

- **Réputation de robustesse** : Moins d'overfitting naturellement
- **class_weight='balanced'** : Gestion automatique des classes déséquilibrées
- **Simplicité** : Moins d'hyperparamètres à optimiser que XGBoost
- **Parallélisation** : Excellente sur CPU multi-core

**🏗️ Architecture v3 :**

```
Ingrédients → Normalisation (v2) → TF-IDF (300 features, ngram=(1,1)) + Features numériques → Random Forest
```

**🔧 Configuration Random Forest :**

```python
RandomForestClassifier(
    n_estimators=200,           # Nombre d'arbres
    max_depth=7,                # Profondeur limitée (anti-overfitting)
    min_samples_split=10,       # Min échantillons pour diviser
    min_samples_leaf=4,         # Min échantillons par feuille
    max_features=0.3,           # 30% des features par split
    bootstrap=True,             # Échantillonnage avec remise (force de Random Forest)
    criterion='entropy',        # Critère de division (on a testé gini également mais entropy meilleur résultat donc optimisation pour ne laisser que entropy)
    class_weight='balanced'     # Équilibrage automatique
)
```

**⚖️ Paramètres anti-overfitting :**

- **max_depth ≤ 7** : évite les arbres trop profonds
- **min_samples_leaf ≥ 4** : force la généralisation
- **max_features = 0.3** : aléatoire contrôlé pour diversité

**🎯 Résultats v3 :**

- **Test accuracy :** 52.94% (légèrement inférieur à XGBoost v2)
- **Train accuracy :** 96.60% (⚠️ overfitting important si mal configuré)
- **Avantage :** class_weight='balanced' améliore la classe HAUT
- **Temps :** plus long que XGBoost (~20-25 min vs 10 min et SHAP explainer très long)

## 📊 Comparaison des 3 versions

| Version | Algorithme    | NLP      | Test Accuracy | Train Accuracy | Gap   | Points forts           |
| ------- | ------------- | -------- | ------------- | -------------- | ----- | ---------------------- |
| **v1**  | XGBoost       | Basique  | ~50-51%       | ~58%           | ~7%   | Baseline rapide        |
| **v2**  | XGBoost       | Optimisé | **53.30%** ✅ | 58.96%         | 5.66% | **Meilleur équilibre** |
| **v3**  | Random Forest | Optimisé | 52.15%        | 62.08%         | 9.93% | Robustesse théorique   |

**🏆 Gagnant : Version 2 (XGBoost optimisé)**

### 🔍 Analyse détaillée XGBoost v2 vs Random Forest v3

**XGBoost v2 (meilleur) :**

- **Test accuracy :** 53.30%
- **Train accuracy :** 58.96%
- **Gap overfitting :** 5.66% (excellent contrôle)
- **Performance par classe :**
  - BAS : 56% precision, 68% recall
  - MOYEN : 57% precision, 50% recall
  - HAUT : 46% precision, 42% recall

**Random Forest v3 :**

- **Test accuracy :** 52.15%
- **Train accuracy :** 62.08%
- **Gap overfitting :** 9.93% (surapprentissage plus important)
- **Performance par classe :**
  - BAS : 56% precision, 67% recall
  - MOYEN : 56% precision, 50% recall
  - HAUT : 44% precision, 40% recall

### ⚠️ Test anti-overfitting Random Forest (échec)

**Configuration testée :**

```python
# Paramètres très restrictifs pour éviter l'overfitting
param_rf_restrictive = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [3, 5, 7],  # Très limité
    'min_samples_leaf': [4, 8],  # Très restrictif
    'min_samples_split': [20],  # Très restrictif
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
    'criterion': ['entropy']
}
```

**Résultat :** **0.49 d'accuracy** (chute drastique de 3% vs configuration équilibrée)

**Pourquoi l'échec :**

- **max_depth ≤ 7** : trop restrictif, modèle sous-apprend
- **min_samples_leaf ≥ 4** : force la généralisation mais perd en précision
- **min_samples_split = 20** : arbres trop simples pour capturer les patterns nutritionnels

**Leçon apprise :** L'équilibre entre contrôle de l'overfitting et capacité d'apprentissage est crucial. Des paramètres trop restrictifs peuvent dégrader significativement les performances.

## 🎯 Architecture finale optimale (v2)

### 📝 Pipeline de preprocessing

```python
# 1. Nettoyage et normalisation
ingredients → clean_text() → sort_ingredients() → normalize_ingredients()

# 2. Vectorisation TF-IDF
tfidf = TfidfVectorizer(
    max_features=300,
    min_df=100,
    max_df=0.95,
    ngram_range=(1, 1),
    stop_words=None
)

# 3. Features numériques (15)
[n_ingredients, nb_fat, nb_sugar, nb_protein, nb_vegetable, nb_grain, nb_drink, nb_spice,
 fat_ratio, sugar_ratio, protein_ratio, vegetable_ratio, grain_ratio, drink_ratio, spice_ratio]

# 4. Combinaison
X_combined = hstack([tfidf_features, numeric_features])
```

### 🎪 Modèle final

```python
XGBClassifier(
    objective='multi:softprob',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.2,
    reg_lambda=1.2,
    random_state=42
)
```

## 🔬 Insights techniques majeurs

### 🎯 Découvertes de preprocessing

1. **Normalisation critique :** 200+ mappings d'ingrédients similaires améliorent légèrement l'accuracy et donnent de meilleurs résultats de tests
2. **NGrams (1,1) optimal :** avec underscores, les bigrammes ajoutent plus de bruit que de signal
3. **Features numériques essentielles :** TF-IDF seul = ~40%, TF-IDF + numeric = 53%

### 📊 Features les plus importantes

**Top 5 TF-IDF (ingrédients) :**

1. **butter** - indicateur de richesse calorique
2. **sugar** - sucrant direct
3. **olive_oil** - matière grasse saine
4. **garlic** - aromate de base vs sophistiqué
5. **lettuce** - indicateur de légèreté

**Top 5 Numériques :**

1. **fat_ratio** - proportion de matières grasses
2. **sugar_ratio** - proportion de sucrants
3. **n_ingredients** - complexité de la recette
4. **protein_ratio** - proportion de protéines
5. **vegetable_ratio** - proportion de légumes

## 🚀 Perspectives d'amélioration

### 🔮 Version 4 (future) : Feature Engineering avancé

**Nouvelle feature catégorielle envisagée (à calculer grâce aux features numériques):**

```python
recipe_type = [
    'DESSERT',
    'SNACK',
    'APPETIZER',
    'MAIN_DISH',
    'DRINK',
]
```

**Impact attendu :** +2-5% accuracy par contextualisation culinaire

### 🎯 Autres améliorations possibles

1. **Instructions NLP** : utiliser les instructions qui contiennent les quantités, méthodes de cuisson etc mais cela nécessite un traitement NLP complexe
2. **Dataset plus conséquent** : utiliser un dataset plus riche permettrait d'obtenir certainement de meilleurs résultats (le ML a besoin d'une quantité importante de données)

## 📥 Dataset

**Food Ingredients Dataset**

- **228,430 recettes uniques** avec ingrédients et instructions
- **Classification calorique** : BAS (<215 cal), MOYEN (215-430 cal), HAUT (>430 cal)
- **Ingrédients détaillés** en format texte
- **Distribution équilibrée** des classes (33% chacune)

**🔗 Source :** [Food Ingredients and Recipe Dataset with Raw Text](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-raw-text) sur Kaggle

**📁 Fichier utilisé :** `RAW_recipes.csv`

## 📚 Structure du projet

```
itadaki_ML/
├── 📓 NOTEBOOKS
│   ├── calorie_prediction_notebook_xgboost_v1.ipynb    # v1 - Baseline XGBoost
│   ├── calorie_prediction_notebook_xgboost_v2.ipynb    # v2 - XGBoost optimisé ✅
│   └── calorie_prediction_notebook_random_forest_v3.ipynb # v3 - Random Forest
├── 📁 DONNÉES
│   └── data/
│       └── RAW_recipes.csv                             # Dataset original
├── 📁 MODÈLES
│   └── models/                                         # Modèles sauvegardés
└── 📁 REPORTS
    ├── top_1000_ingredients.txt                        # Analyse fréquence ingrédients
    └── différents notebooks exportés
```

## 🎯 Résultats finaux et métriques

### 🏆 Performance optimale (v2)

**Métriques globales :**

- **Test accuracy :** **53.30%** (vs 33% aléatoire = +60% d'amélioration)
- **Train accuracy :** 58.96%
- **Gap d'overfitting :** 5.66% (excellent contrôle)

**Performance par classe :**

```
              precision    recall  f1-score   support
         bas       0.56      0.68      0.61     15208  ✅ Très bonne
       moyen       0.57      0.50      0.53     15032  ✅ Satisfaisante
        haut       0.46      0.42      0.44     15446  ⚖️ Acceptable

    accuracy                           0.53     45686
   macro avg       0.53      0.53      0.53     45686
weighted avg       0.53      0.53      0.53     45686
```

### 🎯 Contexte d'évaluation

**Pourquoi 53% est très bon au vu de l'algorithme et des données :**

1. **🥄 Quantités manquantes** : "flour, sugar, butter" → pancake ou gâteau au chocolat ?
2. **🔥 Méthodes de cuisson** : grillé vs frit change drastiquement les calories
3. **📏 Portions variables** : même recette, tailles différentes
4. **🌍 Variabilité culturelle** : "rice" peut être 100-400 calories selon préparation

## 📚 Technologies utilisées

- **Scikit-learn** : TF-IDF, preprocessing, métriques
- **XGBoost** : algorithme principal (meilleurs résultats)
- **Random Forest** : alternative testée pour robustesse
- **Pandas/NumPy** : manipulation de données
- **Matplotlib/Seaborn** : visualisations et analyses
- **SHAP** : interprétabilité des modèles
- **Jupyter** : environnement de développement

## 🎯 Conclusions

### **🏆 Apprentissages majeurs**

1. **ML traditionnel suffisant** : pas besoin de DL pour ce problème (car le besoin était minimaliste) mais on irait bien plus loin avec du DL
2. **Preprocessing critique** : la normalisation des ingrédients apporte +3% de performance
3. **Features hybrides** : TF-IDF + features numériques = combinaison gagnante
4. **XGBoost optimal** : surpasse Random Forest sur ce type de données sparses
5. **53% = benchmark très correct** : performance remarquable pour la prédiction nutritionnelle (avec 2 classes "pas trop calorique" | "calorique", on obtiendrait certainement une très bonne accuracy)

### **🚀 Recommandations**

1. **Déployer v2** : XGBoost v2 ready peut être déployé (cela ne pourra que s'améliorer et cela permet de créer la partie MLOPS)
2. **Enrichir dataset** : quantités et méthodes de cuisson
3. **Applications métier** : apps nutrition, recommandations alimentaires

---

**Projet réalisé dans le cadre de la certification Développeur IA - Alyra**

_L'approche progressive v1 → v2 → v3 démontre l'importance du preprocessing intelligent et valide XGBoost comme choix optimal pour la classification nutritionnelle._

**🏆 BENCHMARK ÉTABLI : 53.30% d'accuracy en classification calorique avec XGBoost optimisé**
