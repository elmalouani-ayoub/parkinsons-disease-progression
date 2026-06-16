"""
data_preprocessing.py — Ingestion et alignement des données brutes.

Responsabilité unique : passer des fichiers CSV bruts à une table
relationnelle unifiée et propre, sans aucun feature engineering.
"""

import numpy as np
import pandas as pd

from src.config import DATA_PATHS, TARGET


# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------

def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les trois fichiers CSV de données brutes depuis les chemins définis
    dans ``config.DATA_PATHS``.

    Returns
    -------
    X_train : pd.DataFrame
        Matrice des features d'entraînement.
    y_train : pd.DataFrame
        Vecteur des cibles d'entraînement (contient ``Index`` + ``target``).
    X_test : pd.DataFrame
        Matrice des features du jeu de test.

    Raises
    ------
    FileNotFoundError
        Si l'un des fichiers est absent du répertoire de travail.
    """
    X_train = pd.read_csv(DATA_PATHS["X_train"])
    y_train = pd.read_csv(DATA_PATHS["y_train"])
    X_test  = pd.read_csv(DATA_PATHS["X_test"])

    print(
        f"[load_datasets] Train : {len(X_train):,} visites | "
        f"Test : {len(X_test):,} visites | "
        f"Patients train : {X_train['patient_id'].nunique()}"
    )
    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# 2. Fusion et alignement
# ---------------------------------------------------------------------------

def merge_molecular_data(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Fusionne les données cliniques train/test en une unique table longitudinale.

    Stratégie :
    - Jointure interne ``X_train ⋈ y_train`` sur la clé ``Index``.
    - Concaténation verticale avec ``X_test`` (la cible est ``NaN`` côté test).
    - Marqueur ``is_test`` (0 = train, 1 = test) pour distinguer les splits
      sans jamais perdre d'information.

    Parameters
    ----------
    X_train : pd.DataFrame
        Issu de :func:`load_datasets`.
    y_train : pd.DataFrame
        Issu de :func:`load_datasets`.
    X_test : pd.DataFrame
        Issu de :func:`load_datasets`.

    Returns
    -------
    df_all : pd.DataFrame
        Table unifiée (train + test) avec la colonne ``is_test``.
    """
    # Jointure train
    df_train = pd.merge(X_train, y_train, on="Index", how="inner")
    df_train["is_test"] = 0

    # Côté test : la cible est inconnue
    df_test = X_test.copy()
    df_test["is_test"] = 1
    df_test[TARGET] = np.nan

    df_all = pd.concat([df_train, df_test], ignore_index=True)

    print(
        f"[merge_molecular_data] Table unifiée : {len(df_all):,} lignes — "
        f"{df_train['patient_id'].nunique()} patients train, "
        f"{df_test['patient_id'].nunique()} patients test."
    )
    return df_all
