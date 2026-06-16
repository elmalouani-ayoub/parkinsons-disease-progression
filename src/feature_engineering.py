"""
feature_engineering.py — Transformation de la donnée propre en espace vectoriel.

Ce module implémente trois niveaux de vectorisation :
  1. Imputation de base (médiane, encodage catégoriel) → ``impute_missing_values``
  2. Ingénierie des biomarqueurs (PK/PD, pentes, statistiques, lags/leads)
     → ``aggregate_biomarkers``
  3. Construction finale des matrices (X_tr, X_te, y) → ``build_training_matrix``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import COLS_DROP, GENE_MAP, PKPD


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _compute_pkpd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modèle PK/PD de la Lévodopa — variables de concentration effective.

    Équations (Dietz et al., 2001) :
        Teq    = max(5, 90 - 3.8 × disease_duration)   [minutes]
        ke0    = ln(2) / Teq                             [min⁻¹]
        Ce_on  = (ledd / 400) × exp(−ke0 × t_on × 60)
        Ce_off = (ledd / 400) × exp(−ke0 × t_off × 60)
    """
    Teq = (
        PKPD["teq_base"] - PKPD["teq_decay"] * df["disease_duration"]
    ).clip(PKPD["teq_min"], PKPD["teq_max"])

    ke0   = np.log(2) / Teq
    t_on  = df["time_since_intake_on"].fillna(PKPD["default_t_on"])
    t_off = df["time_since_intake_off"].fillna(PKPD["default_t_off"])

    df = df.copy()
    df["pkpd_Ce_on"]   = (df["ledd_f"] / PKPD["ledd_norm"]) * np.exp(-ke0 * t_on  * 60).clip(0)
    df["pkpd_Ce_off"]  = (df["ledd_f"] / PKPD["ledd_norm"]) * np.exp(-ke0 * t_off * 60).clip(0)
    df["pkpd_Teq"]     = Teq
    df["pkpd_storage"] = Teq / PKPD["teq_base"]
    return df


def _fit_patient_slopes(
    df: pd.DataFrame,
    score_col: str,
) -> tuple[dict, dict]:
    """
    Régression polynomiale de degré 1 (pente de dégradation) par patient.

    Parameters
    ----------
    df : DataFrame unifié avec ``disease_duration`` et ``score_col``.
    score_col : colonne clinique à modéliser (``"on"`` ou ``"off"``).

    Returns
    -------
    slopes, intercepts : dicts {patient_id → float}
    """
    slopes, intercepts = {}, {}
    fallback_col = "on" if score_col == "off" else None

    for pid, grp in df.groupby("patient_id", sort=False):
        sub = grp.dropna(subset=[score_col])

        if len(sub) >= 2:
            try:
                x   = sub["disease_duration"].values.astype(float)
                y   = sub[score_col].values.astype(float)
                x_c = x - x.mean()
                c   = np.polyfit(x_c, y, 1)
                slopes[pid]     = c[0]
                intercepts[pid] = c[1] + c[0] * (-x.mean())
            except Exception:
                slopes[pid]     = 0.0
                intercepts[pid] = float(sub[score_col].mean())

        elif len(sub) == 1:
            slopes[pid]     = 0.0
            intercepts[pid] = float(sub[score_col].iloc[0])

        else:
            # Aucune donnée : repli sur la colonne alternative si disponible
            alt = (
                float(grp[fallback_col].mean())
                if fallback_col and grp[fallback_col].notna().any()
                else np.nan
            )
            slopes[pid]     = 0.0
            intercepts[pid] = alt

    return slopes, intercepts


# ---------------------------------------------------------------------------
# 1. Imputation des valeurs manquantes de base
# ---------------------------------------------------------------------------

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation de premier niveau sur la table unifiée brute.

    Opérations :
    - Médiane globale pour ``age_at_diagnosis`` et ``ledd``.
    - Calcul de ``disease_duration`` (âge − âge au diagnostic, ≥ 0).
    - Compteurs longitudinaux : ``visit_number``, ``n_visits_patient``.
    - Encodage catégoriel : ``gene`` → ``gene_enc``, ``cohort`` → ``cohort_enc``.

    Parameters
    ----------
    df : pd.DataFrame
        Table unifiée issue de :func:`~data_preprocessing.merge_molecular_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les champs de base imputés et encodés.
    """
    df = df.copy()

    # Médiane globale (calculée sur l'ensemble train+test pour stabilité)
    med_age_diag = df["age_at_diagnosis"].median()
    med_ledd     = df["ledd"].median()

    df["age_at_diagnosis_f"] = df["age_at_diagnosis"].fillna(med_age_diag)
    df["disease_duration"]   = (df["age"] - df["age_at_diagnosis_f"]).clip(lower=0)
    df["ledd_f"]             = df["ledd"].fillna(med_ledd)

    # Compteurs de visites par patient
    df["visit_number"]     = df.groupby("patient_id").cumcount() + 1
    df["n_visits_patient"] = df.groupby("patient_id")["age"].transform("count")

    # Encodage catégoriel des variables non-numériques
    df["gene_enc"]   = df["gene"].fillna("Unknown").map(GENE_MAP).fillna(-1)
    df["cohort_enc"] = (df["cohort"] == "B").astype(int)

    print(
        f"[impute_missing_values] disease_duration : "
        f"{df['disease_duration'].mean():.1f} ans (moy) | "
        f"ledd_f : {df['ledd_f'].mean():.0f} mg (moy)"
    )
    return df


# ---------------------------------------------------------------------------
# 2. Vectorisation des biomarqueurs
# ---------------------------------------------------------------------------

def aggregate_biomarkers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorisation complète des signaux cliniques pour chaque visite.

    Étapes (dans l'ordre logique d'application) :
    1. **PK/PD** : variables de concentration Lévodopa (``pkpd_Ce_on``, etc.).
    2. **Interpolation intra-patient** du score ``off`` (réparation temporelle).
    3. **Pentes de dégradation** (``p_off_slope``, ``p_on_slope``) par
       régression polynomiale de degré 1 sur chaque trajectoire individuelle.
    4. **Statistiques agrégées** (mean/std/min/max) pour ``on``, ``off``,
       ``off_final`` par patient.
    5. **Ratio pharmacologique** ``ledd_per_year``.
    6. **Lags / Leads** (approche transductive) : fenêtre temporelle ±2 visites.
    7. **Vitesses inter-visites** : delta et dérivée temporelle des scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame issu de :func:`impute_missing_values`.

    Returns
    -------
    pd.DataFrame
        DataFrame enrichi de l'ensemble des features vectorisées.
    """
    df = df.copy()

    # ---- 1. Modèle PK/PD ---------------------------------------------------
    df = _compute_pkpd(df)

    # ---- 2. Interpolation linéaire intra-patient (score 'off') -------------
    df["off_interp"] = (
        df.groupby("patient_id", sort=False)["off"]
        .transform(lambda x: x.interpolate(method="linear", limit_direction="both"))
    )

    # ---- 3. Pentes de dégradation par patient (transductif) ----------------
    for score_col in ["off", "on"]:
        slopes, intercepts = _fit_patient_slopes(df, score_col)
        df[f"p_{score_col}_slope"]    = df["patient_id"].map(slopes)
        df[f"{score_col}_trend_pred"] = (
            df[f"p_{score_col}_slope"] * df["disease_duration"]
            + df["patient_id"].map(intercepts)
        )

    # Consolidation de la variable 'off' : mesurée > interpolée > extrapolée
    df["off_final"] = (
        df["off"]
        .fillna(df["off_interp"])
        .fillna(df["off_trend_pred"])
    )

    # ---- 4. Statistiques agrégées par patient ------------------------------
    for col in ["on", "off", "off_final"]:
        grp = df.groupby("patient_id")[col]
        df[f"p_mean_{col}"] = grp.transform("mean")
        df[f"p_std_{col}"]  = grp.transform("std").fillna(0)
        df[f"p_min_{col}"]  = grp.transform("min")
        df[f"p_max_{col}"]  = grp.transform("max")

    # ---- 5. Ratio pharmacologique ------------------------------------------
    df["ledd_per_year"] = df["ledd_f"] / (df["disease_duration"] + 0.5)

    # ---- 6. Lags & Leads (fenêtre transductive ±2 visites) -----------------
    for col in ["on", "off_final"]:
        for lag in [1, 2]:
            df[f"{col}_lag{lag}"]  = df.groupby("patient_id")[col].shift(lag)
            df[f"{col}_lead{lag}"] = df.groupby("patient_id")[col].shift(-lag)

    # ---- 7. Vitesses de dégradation inter-visites --------------------------
    lag_age = df.groupby("patient_id")["age"].shift(1)
    df["time_since_last"] = (df["age"] - lag_age).clip(0).fillna(0)
    df["on_delta1"]       = df["on"]        - df["on_lag1"]
    df["off_delta1"]      = df["off_final"] - df["off_final_lag1"]
    df["on_speed"]        = df["on_delta1"]  / (df["time_since_last"] + 0.01)
    df["off_speed"]       = df["off_delta1"] / (df["time_since_last"] + 0.01)

    # Première visite : aucun historique disponible → vitesse = 0
    mask_v1 = df["visit_number"] == 1
    df.loc[mask_v1, ["on_delta1", "off_delta1", "on_speed", "off_speed", "time_since_last"]] = 0

    n_features = len([c for c in df.columns if c not in COLS_DROP])
    print(f"[aggregate_biomarkers] {n_features} features générées.")
    return df


# ---------------------------------------------------------------------------
# 3. Construction finale de la matrice (X, y)
# ---------------------------------------------------------------------------

def build_training_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index]:
    """
    Extrait la matrice explicative X et le vecteur cible y depuis le DataFrame
    enrichi.

    Encodage résiduel :
    Les colonnes catégorielles restantes (type ``object`` ou ``category``) sont
    converties en codes entiers pour la compatibilité avec les arbres de décision.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame issu de :func:`aggregate_biomarkers`.

    Returns
    -------
    X_tr : pd.DataFrame
        Matrice des features (jeu train).
    X_te : pd.DataFrame
        Matrice des features (jeu test).
    y_tr_raw : pd.Series
        Vecteur cible brut (échelle originale).
    groups : pd.Series
        Identifiants patients pour :class:`~sklearn.model_selection.GroupKFold`.
    idx_te : pd.Index
        Index du jeu de test (pour la soumission).
    """
    feature_cols = [c for c in df.columns if c not in COLS_DROP]

    df_tr = df[df["is_test"] == 0].copy()
    df_te = df[df["is_test"] == 1].copy()

    X_tr = df_tr[feature_cols].copy()
    X_te = df_te[feature_cols].copy()

    # Encodage numérique des colonnes catégorielles résiduelles
    for col in X_tr.select_dtypes(["category", "object"]).columns:
        X_tr[col] = X_tr[col].astype("category").cat.codes
        X_te[col] = X_te[col].astype("category").cat.codes

    y_tr_raw = df_tr["target"]
    groups   = df_tr["patient_id"]
    idx_te   = df_te["Index"]

    print(
        f"[build_training_matrix] X_tr : {X_tr.shape} | "
        f"X_te : {X_te.shape} | "
        f"Cible train — min={y_tr_raw.min():.1f}, max={y_tr_raw.max():.1f}, "
        f"moy={y_tr_raw.mean():.1f}"
    )
    return X_tr, X_te, y_tr_raw, groups, idx_te
