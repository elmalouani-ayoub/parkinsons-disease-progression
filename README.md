Parkinson's Disease Progression - Predictive Modeling
Objectif
Prédiction de l'évolution des scores moteurs cliniques (MDS-UPDRS) chez les patients atteints de la maladie de Parkinson. Projet réalisé dans le cadre du Challenge Data de l'ENS et de l'Institut du Cerveau.

Défi Technique
Les données cliniques présentent une forte proportion de valeurs manquantes et une haute irrégularité dans les intervalles de visites médicales, rendant les approches de séries temporelles classiques obsolètes.

Architecture de la Solution
Prétraitement : Imputation et alignement des trajectoires hétérogènes des patients.

Modélisation : Conception d'une architecture prédictive basée sur des méthodes ensemblistes intégrant LightGBM, RandomForestRegressor, et ExtraTreesRegressor.

Optimisation : Réglage des hyperparamètres et aggrégation via un VotingRegressor pour minimiser la variance.

Performances
Métrique d'évaluation : RMSE (Root Mean Square Error).

[Ajoute ici ton score RMSE final ou ton classement si pertinent]
