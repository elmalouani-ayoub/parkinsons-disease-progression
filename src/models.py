"""
models.py — Architecture algorithmique.

Ce module instancie les modèles avec leurs hyperparamètres.
Il n'accède jamais aux données : c'est une pure fabrique d'objets
(Factory Pattern), ce qui facilite le tuning et les tests unitaires.
"""

from __future__ import annotations

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor

from src.config import MODEL_PARAMS


# ---------------------------------------------------------------------------
# 1. Estimateurs de base
# ---------------------------------------------------------------------------

def get_base_estimators() -> dict[str, object]:
    """
    Instancie les trois algorithmes de Gradient Boosting avec leurs
    hyperparamètres définis dans ``config.MODEL_PARAMS``.

    Chaque appel crée de **nouvelles** instances indépendantes — nécessaire
    pour la validation croisée GroupKFold où chaque fold doit entraîner
    des objets vierges.

    Choix des algorithmes :
    - **HistGBM** (Scikit-Learn) : binning en 256 intervalles → régularisation
      naturelle, robustesse aux valeurs aberrantes cliniques.
    - **LightGBM** (Microsoft) : croissance leaf-wise → détection d'interactions
      fines entre variables cliniques et génétiques.
    - **XGBoost** : Taylor du 2ᵉ ordre + pénalités L1/L2 → modèle le plus
      prudent, élagage agressif des branches peu informatives.

    Returns
    -------
    dict
        ``{"hgb": <HistGBM>, "lgb": <LGBM>, "xgb": <XGB>}``
    """
    return {
        "hgb": HistGradientBoostingRegressor(**MODEL_PARAMS["hgb"]),
        "lgb": lgb.LGBMRegressor(**MODEL_PARAMS["lgb"]),
        "xgb": xgb.XGBRegressor(**MODEL_PARAMS["xgb"]),
    }


# ---------------------------------------------------------------------------
# 2. Modèle ensembliste
# ---------------------------------------------------------------------------

def build_ensemble_model() -> VotingRegressor:
    """
    Construit le ``VotingRegressor`` encapsulant les trois estimateurs.

    Usage :
    - **Fit final** (sans early stopping) sur la totalité du jeu train.
    - Exposition publique du modèle entraîné pour l'interprétabilité (XAI).

    Note : dans la boucle de cross-validation de ``pipeline.py``, on
    utilise directement :func:`get_base_estimators` pour conserver le
    contrôle sur les callbacks LightGBM/XGBoost (``early_stopping``).
    Le ``VotingRegressor`` ne supporte pas nativement ces callbacks.

    Returns
    -------
    VotingRegressor
        Ensemble non ajusté (appeler ``.fit(X, y)`` pour l'entraîner).
    """
    estimators = get_base_estimators()
    return VotingRegressor(
        estimators=[
            ("hgb", estimators["hgb"]),
            ("lgb", estimators["lgb"]),
            ("xgb", estimators["xgb"]),
        ]
    )
