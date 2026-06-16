# Parkinson's Disease Progression - Predictive Modeling

> **Objectif :** Développer une architecture prédictive robuste pour anticiper l'évolution des scores moteurs cliniques (MDS-UPDRS) chez les patients atteints de la maladie de Parkinson. 
> *Projet réalisé dans le cadre du Challenge Data de l'ENS et de l'Institut du Cerveau.*

## 🧬 Contexte & Défis Techniques
L'analyse de données longitudinales en neurologie se heurte à la réalité du terrain clinique :
* **Sparsité des données (Le « Gruyère ») :** Les examens médicaux génèrent une forte proportion de valeurs manquantes.
* **Hétérogénéité temporelle :** Les intervalles entre les visites médicales sont hautement irréguliers, rendant les approches classiques par séries temporelles (ARIMA, LSTMs basiques) inopérantes.
* **Asymétrie clinique :** La progression de la maladie varie drastiquement d'un patient à l'autre.

## 🛠️ Architecture de la Solution
Ce projet s'écarte des notebooks linéaires académiques pour adopter une architecture logicielle modulaire, respectant le principe de séparation des responsabilités (SoC).

### 1. Ingénierie des Caractéristiques (Feature Engineering)
* **Modélisation PK/PD (Pharmacocinétique/Pharmacodynamique) :** Implémentation mathématique de la courbe de décroissance de l'efficacité de la Lévodopa (équations de Dietz et al.) pour estimer la concentration effective lors des phases "ON" et "OFF".
* **Transduction temporelle :** Extraction des pentes de dégradation individuelles par régression polynomiale et calcul des vitesses de dégradation inter-visites.

### 2. Modélisation Ensembliste
Conception d'une architecture prédictive robuste face à la variance clinique, encapsulée dans un `VotingRegressor` :
* **HistGradientBoosting :** Robustesse naturelle aux valeurs aberrantes grâce au binning continu.
* **LightGBM :** Détection d'interactions complexes via une croissance des arbres par les feuilles (leaf-wise).
* **XGBoost :** Élagage agressif et optimisation de second ordre pour limiter le surapprentissage.

### 3. Validation Stratégique
* Évaluation stricte via **GroupKFold** (isolant les trajectoires des patients) pour bloquer toute fuite de données temporelle (Data Leakage).
* Transformation logarithmique (`log1p`) de la cible pour stabiliser l'apprentissage sur les distributions asymétriques.

## 📊 Performances & Audit Clinique
L'évaluation ne se limite pas à la RMSE, mais inclut un audit clinique traduisant l'erreur en points moteurs sur l'échelle MDS-UPDRS.

* **RMSE (Root Mean Square Error) :** [Insère ton score ici]
* **MAE (Mean Absolute Error) :** [Insère ton score ici] points moteurs.
* **Fiabilité Clinique :** [Insère ton score ici] % des prédictions tombent dans la marge d'acceptabilité d'un neurologue (±5 points).

## 📂 Structure du Dépôt
```text
├── data/                  # Données brutes (ignorées par Git)
├── notebooks/
│   └── 01_exploration_et_modelisation.ipynb  # Orchestrateur visuel & EDA
├── src/                   # Package Python (Logique métier)
│   ├── config.py          # Constantes et hyperparamètres
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── metrics.py         # Fonctions d'audit clinique et sMAPE
│   ├── models.py          # Factory Pattern des algorithmes
│   └── pipeline.py        # Moteur d'exécution central
└── README.md
