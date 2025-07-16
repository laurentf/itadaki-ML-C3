# 🍜 Itadaki ML - Recipe Calorie Prediction System

_"Itadaki" signifie "bon appétit" en japonais_

## 🎓 Projet de certification DEV IA - Alyra

Ce projet constitue le **projet final** de la certification **Développeur Intelligence Artificielle** d'Alyra. Il implémente un système de prédiction du niveau calorique des recettes utilisant l'apprentissage automatique.

**Compétences évaluées (Bloc 3 : Appliquer des techniques d'analyse IA via des algorithmes d'apprentissage automatique)** :

- **C1** : Sélectionner l'algorithme d'apprentissage le plus adapté en comparant les performances et les caractéristiques des différentes familles d'algorithmes afin d'apporter une réponse pertinente à la problématique métier rencontrée
- **C2** : Préparer et transformer des données en utilisant des techniques de prétraitement (preprocessing) pour les adapter aux spécificités du modèle d'apprentissage automatique choisi
- **C3** : Entraîner un modèle d'apprentissage automatique en optimisant une loss function (fonction de coût) à partir des données d'entraînement afin de permettre à l'algorithme d'effectuer le moins d'erreurs possibles selon des indicateurs de succès clairement définis

## 🎯 Vue d'ensemble

Système d'intelligence artificielle qui **prédit les calories des recettes** à partir de leurs ingrédients et instructions. Ce projet explore **deux approches complémentaires** : classification par catégories et régression pour prédiction directe des valeurs caloriques.

## 📈 Évolution du projet - De la Classification à la Régression

Ce projet documente une **approche itérative** avec deux phases principales d'expérimentation.

### 🔸 Phase 1 : Approche Classification (Réalisée)

**Transformation en problème de classification**

- **Approche** : Conversion des calories (variable continue) en **3 catégories** : BAS/MOYEN/HAUT
- **Méthode de catégorisation** : Par percentiles (33% chacune)
  - BAS : < 215 calories
  - MOYEN : 215-430 calories
  - HAUT : > 430 calories
- **Architecture** : `TF-IDF (500 features) + Features nutritionnelles (15) → XGBoost`

**Résultats obtenus**

- **Performance** : 43% d'accuracy (vs 33% aléatoire)
- **Amélioration** : +30% par rapport au hasard
- **Temps d'entraînement** : ~15 minutes

**✅ Avantages observés**

- Interprétabilité élevée des catégories
- Logique nutritionnelle claire
- XGBoost performant sur features sparses (TF-IDF)

**❌ Limites identifiées**

- **Perte d'information** : La discrétisation efface les nuances entre 214 et 216 calories
- **Frontières arbitraires** : Une recette à 214 cal (BAS) vs 216 cal (MOYEN) sont très proches
- **Granularité insuffisante** : Impossible de distinguer 200 cal vs 400 cal dans la même catégorie
- **Performance plafonnée** : 43% semble être le maximum atteignable avec cette approche

### 🔸 Phase 2 : Approche Régression (À explorer)

**Retour à la prédiction continue**

- **Approche** : Prédiction directe des **valeurs caloriques continues**
- **Preprocessing** : Normalisation `log1p(calories)` pour gérer la distribution asymétrique
- **Architecture** : `TF-IDF + Features nutritionnelles → Régresseurs ML`

**Régresseurs candidats à tester**

- **XGBoost Regressor** : Extension naturelle de notre approche classification
- **Random Forest Regressor** : Robuste, parallélisable, moins d'overfitting
- **SVR (Support Vector Regression)** : Efficace sur features sparses

**Avantages attendus**

- **Précision granulaire** : Distinction fine entre 200 et 210 calories
- **Métriques continues** : MAE, RMSE plus intuitives que l'accuracy
- **Pas de perte d'information** : Conservation de toute la richesse des données
- **Applications pratiques** : Prédictions exploitables directement

**Défis à relever**

- **Distribution asymétrique** : Transformation log1p nécessaire
- **Outliers caloriques** : Gestion des recettes extrêmes (>2000 cal)
- **Évaluation complexe** : Définir une erreur acceptable (±50 cal ?)

## 🔬 Comparaison Classification vs Régression

| Aspect                     | Classification (Phase 1)    | Régression (Phase 2)         |
| -------------------------- | --------------------------- | ---------------------------- |
| **Variable cible**         | 3 catégories (BAS/MOY/HAUT) | Calories continues           |
| **Preprocessing cible**    | Percentiles 33%             | log1p(calories)              |
| **Métrique principale**    | Accuracy (43%)              | MAE/RMSE (à déterminer)      |
| **Interprétabilité**       | Excellente                  | Bonne                        |
| **Précision**              | Grossière (3 niveaux)       | Fine (valeur exacte)         |
| **Perte d'information**    | ⚠️ Élevée                   | ✅ Minimale                  |
| **Applications pratiques** | Recommandations générales   | Calculs nutritionnels précis |
| **Complexité évaluation**  | Simple                      | Modérée                      |

## 🎯 Pourquoi ce changement d'approche ?

### **Limitations de la classification observées**

1. **Frontières arbitraires** : 214 cal (BAS) vs 216 cal (MOYEN) - différence non significative
2. **Granularité insuffisante** : 200 cal et 400 cal dans la même catégorie "BAS"
3. **Performance plafond** : 43% semble être le maximum avec la discrétisation
4. **Usage limité** : Applications pratiques nécessitent des valeurs précises

### **Avantages attendus de la régression**

1. **Précision métier** : Prédiction directe utilisable en nutrition
2. **Pas de perte d'information** : Conservation de toute la richesse des données
3. **Métriques intuitives** : "Erreur moyenne de ±45 calories" vs "43% d'accuracy"
4. **Flexibilité d'usage** : Peut être convertie en catégories a posteriori si besoin

## 📊 Stratégie d'évaluation pour la régression

### **Métriques principales**

- **MAE (Mean Absolute Error)** : Erreur moyenne en calories
- **RMSE (Root Mean Square Error)** : Pénalise plus les grosses erreurs
- **MAPE (Mean Absolute Percentage Error)** : Erreur relative

### **Seuils d'acceptabilité**

- **Excellent** : MAE < 50 calories (±50 cal)
- **Bon** : MAE < 100 calories (±100 cal)
- **Acceptable** : MAE < 150 calories (±150 cal)
- **Baseline** : MAE moyenne = 200-300 calories

### **Preprocessing de la cible**

```python
# Transformation log pour normaliser la distribution
y_log = np.log1p(df['calories'])  # log(1 + calories)

# Après prédiction, retour aux valeurs originales
calories_pred = np.expm1(y_pred)  # exp(pred) - 1
```

## 🛠️ Pipeline de développement prévu

### **Phase 2A : baseline Régression**

1. **Linear Regression (Ridge)** : Baseline rapide
2. **Random Forest Regressor** : Référence robuste
3. **Analyse des résidus** : Identification des patterns d'erreur

### **Phase 2B : optimisation Avancée**

1. **XGBoost Regressor** : Fine-tuning hyperparamètres
2. **Feature Engineering** : Ratios nutritionnels optimisés
3. **Ensemble Methods** : Combinaison des meilleurs modèles

## 📚 Structure du projet évolutive

```
itadaki_ML/
├── 📓 NOTEBOOKS
│   ├── calorie_prediction_classification_xgboost.ipynb  # ✅ Phase 1 - Classification
│   ├── calorie_prediction_regression_baseline.ipynb    # 🔄 Phase 2A - Baseline régression
│   └── calorie_prediction_regression_advanced.ipynb    # 📋 Phase 2B - Optimisation
├── 📁 DONNÉES
│   └── data/
│       └── RAW_recipes.csv                             # Dataset original
├── 📁 MODÈLES
    └── models/                                         # Modèles de classification et regression

```

## 📥 Dataset

**Food Ingredients Dataset** (source Kaggle ou équivalent)

- **228,430 recettes uniques** avec ingrédients et instructions
- **Classification calorique** : BAS (<215 cal), MOYEN (215-430 cal), HAUT (>430 cal) (via percentile)
- **Ingrédients détaillés** en format texte
- **Instructions complètes** de préparation
- **Distribution équilibrée** des classes

## 🏗️ Choix d'architecture : pourquoi XGBoost + TF-IDF ?

### 🎯 Comparaison des approches ML

| Approche                   | Avantages                    | Inconvénients            | Performance | Temps    |
| -------------------------- | ---------------------------- | ------------------------ | ----------- | -------- |
| **XGBoost + TF-IDF** ✅    | Équilibre performance/temps  | Hyperparamètres nombreux | 43-48%      | 15-20min |
| **Random Forest + TF-IDF** | Robuste, moins d'overfitting | Plus lent, moins précis  | 40-45%      | 20-25min |

### 🔍 Justification du choix XGBoost + TF-IDF

#### ✅ **Avantages décisifs**

1. **🎯 Adapté aux features sparses** : TF-IDF génère des matrices très sparses, XGBoost les gère efficacement

2. **🚀 Performance/temps optimal** :

   - **Performance** : 43-48% (correct pour ce problème difficile)
   - **Temps** : 15-20 minutes (acceptable pour l'itération)
   - **Interprétabilité** : Feature importance directement disponible

3. **🔧 Flexibilité** : Hyperparamètres nombreux pour fine-tuning
   - **learning_rate** : Contrôle de l'overfitting
   - **max_depth** : Complexité des arbres
   - **n_estimators** : Équilibre performance/temps

#### ❌ **Pourquoi pas les autres (en cours de test) ?**

##### **Random Forest** - Performance insuffisante

- **🎯 Performance** : 2-3% de moins que XGBoost
- **⏱️ Plus lent** : Pas de parallélisation GPU
- **🔧 Moins de contrôle** : Hyperparamètres moins nombreux

### 🎯 Pourquoi ce problème est difficile ?

#### **Limites intrinsèques du dataset**

1. **🥄 Quantités manquantes** : "flour, sugar, butter" → pancake ou gâteau ?
2. **🔥 Méthodes de cuisson** : grillé vs frit change tout
3. **📏 Portions variables** : même recette, calories différentes
4. **🌍 Variabilité culturelle** : "rice" peut être 100-300 calories

#### **Contexte réaliste d'accuracy**

- **🎲 Baseline aléatoire** : 33% (3 classes équilibrées)
- **👨‍🍳 Performance humaine estimée** : 60-70% (chef expérimenté)
- **🤖 Notre modèle** : 43-48% → **amélioration de 30-45%** vs aléatoire
- **🎯 Performance raisonnable** pour un problème si complexe

## 📈 Résultats et insights

### Principales découvertes

1. **📊 Features TF-IDF dominantes** : Les ingrédients individuels (butter, sugar, lettuce) sont plus discriminants que les features nutritionnelles
2. **🔍 Contextualité importante** : "lettuce" dans burger vs salade change tout
3. **⚖️ Équilibre ML/Expert** : Approche hybride nécessaire pour performance optimale
4. **🎯 Hard cases révélateurs** : Le modèle excelle sur les cas "évidents" (salade vs gâteau)

### Métriques par classe

```
              precision    recall  f1-score   support
         bas       0.48      0.54      0.51     15208
       moyen       0.43      0.31      0.36     15032  # Classe la plus difficile
        haut       0.38      0.44      0.41     15446
```

**Insight clé** : La classe MOYEN est la plus difficile (frontière floue entre BAS et HAUT)

### Features les plus discriminantes

1. **yellow_onion** (21.4%) - Ingrédient de base vs sophistiqué
2. **garlic** (11.0%) - Aromate courant
3. **dried_oregano** (10.3%) - Épice méditerranéenne
4. **butter** (8.2%) - Matière grasse évidente
5. **sugar** (7.1%) - Sucrant direct

## 🎓 Apprentissages et limites

### ✅ Réussites

- **Baseline solide** : 43% sur problème complexe
- **Interprétabilité** : Top features nutritionnellement logiques
- **Temps raisonnable** : 15-20min d'entraînement
- **Approche progressive** : Du simple au complexe

### ⚠️ Limites identifiées

- **Quantités manquantes** : Impact majeur sur la précision
- **Ambiguïté MOYEN** : Classe intermédiaire difficile à discriminer
- **Biais contextuel** : Certains ingrédients mal associés
- **Dataset size** : Plus de données améliorerait les performances

### 🚀 Améliorations futures

1. **🔢 Quantités** : Dataset avec quantités précises
2. **📸 Images** : Ajout de computer vision pour validation
3. **🍽️ Contexte** : Utiliser les instructions à la place des ingrédients pour informations de cuisson et de quantité (demande traitement NLP plus lourd)

## 📚 Technologies utilisées

- **Scikit-learn** : TF-IDF, preprocessing, métriques
- **XGBoost** : Algorithme de classification principal
- **Pandas/NumPy** : Manipulation de données
- **Matplotlib/Seaborn** : Visualisations et analyses
- **SHAP** : Interprétabilité du modèle
- **Jupyter** : Environnement de développement

## 🎯 Conclusions et perspectives

### **Apprentissages de la Phase 1 (Classification)**

- L'**approche XGBoost + TF-IDF** fonctionne bien pour la classification
- **43% d'accuracy** représente une amélioration significative (+30% vs aléatoire)
- Les **features nutritionnelles engineerées** apportent de la valeur
- **Limitation fondamentale** : La discrétisation fait perdre trop d'information

### **Hypothèses pour la Phase 2 (Régression)**

- La **prédiction continue** devrait capturer plus de nuances
- **Transformation log1p** devrait normaliser la distribution des calories
- **Métriques MAE/RMSE** seront plus parlantes que l'accuracy
- **Applications pratiques** directement exploitables

### **Impact métier attendu**

- **Précision nutritionnelle** : Calculs caloriques utilisables en pratique
- **Flexibilité d'usage** : Conversion possible en catégories selon le besoin
- **Évolutivité** : Base solide pour d'autres prédictions nutritionnelles (protéines, lipides...)

---

**Projet réalisé dans le cadre de la certification Développeur IA - Alyra**

_La progression Classification → Régression illustre parfaitement l'approche itérative et l'amélioration continue en Machine Learning._
