"""
metrics.py — Isolement du calcul de l'erreur.

Toutes les fonctions de ce module sont **sans état** (stateless) :
elles acceptent des tableaux et retournent des scalaires ou des dicts.
Elles ne touchent jamais aux données ou aux modèles.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# Type alias
ArrayLike = Union[np.ndarray, pd.Series, list]


# ---------------------------------------------------------------------------
# Métrique principale : sMAPE
# ---------------------------------------------------------------------------

def calculate_smape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    eps: float = 1e-8,
) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) vectorisé.

    Formule :
        sMAPE = (2/n) × Σ [ |y_true − y_pred| / (|y_true| + |y_pred| + ε) ] × 100

    L'epsilon ``eps`` évite la division par zéro quand les deux termes sont nuls
    simultanément (ex. : patient avec score moteur = 0 prédit à 0).

    Parameters
    ----------
    y_true : valeurs observées
    y_pred : valeurs prédites par le modèle
    eps    : régularisateur anti-division-par-zéro (défaut : 1e-8)

    Returns
    -------
    float
        sMAPE en pourcentage (de 0 à 200).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    numerator   = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + eps

    return float(np.mean(2.0 * numerator / denominator) * 100)


# ---------------------------------------------------------------------------
# Audit clinique complet
# ---------------------------------------------------------------------------

def clinical_audit(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    margin: float = 5.0,
) -> dict[str, float]:
    """
    Calcule l'ensemble des métriques d'audit clinique.

    En plus des métriques mathématiques classiques (MAE, RMSE, sMAPE),
    retourne le **taux de fiabilité clinique** : pourcentage de prédictions
    tombant dans la marge d'acceptabilité naturelle d'un neurologue (±5 points
    sur l'échelle motrice).

    Parameters
    ----------
    y_true  : valeurs observées
    y_pred  : valeurs prédites
    margin  : marge d'acceptabilité clinique en points moteurs (défaut : 5.0)

    Returns
    -------
    dict avec les clés :
        ``mae``             : Erreur Absolue Moyenne (points moteurs)
        ``rmse``            : Root Mean Squared Error
        ``smape``           : sMAPE (%)
        ``taux_fiabilite``  : % de prédictions dans la marge ±``margin``
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    abs_errors = np.abs(y_true - y_pred)

    return {
        "mae":           float(mean_absolute_error(y_true, y_pred)),
        "rmse":          float(root_mean_squared_error(y_true, y_pred)),
        "smape":         calculate_smape(y_true, y_pred),
        "taux_fiabilite": float(np.mean(abs_errors <= margin) * 100),
    }
