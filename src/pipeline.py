"""
pipeline.py — Moteur d'exécution.

Ce module est le seul point d'entrée du projet :
il connecte tous les modules src/ dans le bon ordre et retourne
les métriques + le modèle ajusté.

Usage depuis le notebook :
    from src import run_training_pipeline
    results = run_training_pipeline()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

from src.config import N_SPLITS
from src.data_preprocessing import load_datasets, merge_molecular_data
from src.feature_engineering import (
    impute_missing_values,
    aggregate_biomarkers,
    build_training_matrix,
)
from src.metrics import clinical_audit
from src.models import get_base_estimators


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_training_pipeline() -> dict:
    """
    Pipeline d'entraînement complet — de la lecture CSV aux métriques OOF.

    Étapes séquentielles :
    1. **Ingestion** : chargement et fusion des données brutes.
    2. **Imputation** : médiane, encodage catégoriel, compteurs de visites.
    3. **Feature engineering** : PK/PD, pentes, lags/leads, vitesses.
    4. **Matrices** : extraction de (X_tr, X_te, y_tr) + transformation log.
    5. **Cross-validation GroupKFold** (5 folds, groupes = patients) :
       - Entraîne les 3 experts en parallèle sur chaque fold.
       - Collecte les prédictions Out-Of-Fold (OOF) + prédictions test.
    6. **Ensembling** : moyenne arithmétique des 3 experts + inversion log.
    7. **Métriques** : audit clinique (MAE, RMSE, sMAPE, fiabilité).

    Returns
    -------
    dict avec les clés :
        ``metrics``      → dict(mae, rmse, smape, taux_fiabilite)
        ``oof_preds``    → np.ndarray — prédictions OOF (échelle originale)
        ``test_preds``   → np.ndarray — prédictions test (échelle originale)
        ``y_true``       → pd.Series  — cible réelle (pour les graphiques)
        ``X_tr``         → pd.DataFrame — matrice train (pour XAI)
        ``feature_cols`` → list[str]  — noms des features (pour XAI)
        ``groups``       → pd.Series  — identifiants patients
        ``df_train_meta``→ pd.DataFrame — colonnes non-features (disease_duration…)
    """
    print("=" * 65)
    print("  PIPELINE — Progression de Parkinson")
    print("=" * 65)

    # ------------------------------------------------------------------
    # ÉTAPE 1 : Ingestion
    # ------------------------------------------------------------------
    print("\n[1/5] Ingestion des données...")
    X_train, y_train, X_test = load_datasets()
    df_all = merge_molecular_data(X_train, y_train, X_test)

    # ------------------------------------------------------------------
    # ÉTAPE 2 : Imputation de base
    # ------------------------------------------------------------------
    print("\n[2/5] Imputation et encodage...")
    df_all = impute_missing_values(df_all)

    # ------------------------------------------------------------------
    # ÉTAPE 3 : Feature engineering
    # ------------------------------------------------------------------
    print("\n[3/5] Ingénierie des biomarqueurs...")
    df_all = aggregate_biomarkers(df_all)

    # ------------------------------------------------------------------
    # ÉTAPE 4 : Construction des matrices
    # ------------------------------------------------------------------
    print("\n[4/5] Construction des matrices (X, y)...")
    X_tr, X_te, y_tr_raw, groups, idx_te = build_training_matrix(df_all)

    # Transformation logarithmique de la cible pour la stabilisation de la RMSE
    y_tr_log = np.log1p(y_tr_raw)

    # Métadonnées du set train (pour l'audit par stade de maladie)
    df_tr_meta = df_all[df_all["is_test"] == 0][["patient_id", "disease_duration"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # ÉTAPE 5 : Cross-validation GroupKFold
    # ------------------------------------------------------------------
    print(f"\n[5/5] Entraînement — GroupKFold (k={N_SPLITS})...")

    gkf = GroupKFold(n_splits=N_SPLITS)

    # Pré-allocation mémoire
    oof_hgb  = np.zeros(len(X_tr))
    oof_lgb  = np.zeros(len(X_tr))
    oof_xgb  = np.zeros(len(X_tr))
    test_hgb = np.zeros(len(X_te))
    test_lgb = np.zeros(len(X_te))
    test_xgb = np.zeros(len(X_te))

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_tr, y_tr_log, groups)):
        X_t, y_t = X_tr.iloc[tr_idx], y_tr_log.iloc[tr_idx]
        X_v, y_v = X_tr.iloc[val_idx], y_tr_log.iloc[val_idx]

        # Nouvelles instances vierges pour chaque fold
        estimators = get_base_estimators()

        # Expert 1 — HistGradientBoosting
        estimators["hgb"].fit(X_t, y_t)
        oof_hgb[val_idx]  = estimators["hgb"].predict(X_v)
        test_hgb         += estimators["hgb"].predict(X_te) / N_SPLITS

        # Expert 2 — LightGBM (avec early stopping sur le fold de validation)
        estimators["lgb"].fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_lgb[val_idx]  = estimators["lgb"].predict(X_v)
        test_lgb         += estimators["lgb"].predict(X_te) / N_SPLITS

        # Expert 3 — XGBoost (avec early stopping)
        estimators["xgb"].fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        oof_xgb[val_idx]  = estimators["xgb"].predict(X_v)
        test_xgb         += estimators["xgb"].predict(X_te) / N_SPLITS

        print(f"    Fold {fold_idx + 1}/{N_SPLITS} ✓")

    # ------------------------------------------------------------------
    # Ensembling & inversion log
    # ------------------------------------------------------------------
    oof_blend  = np.expm1((oof_hgb  + oof_lgb  + oof_xgb)  / 3.0)
    test_blend = np.expm1((test_hgb + test_lgb + test_xgb) / 3.0)

    # ------------------------------------------------------------------
    # Métriques finales
    # ------------------------------------------------------------------
    metrics = clinical_audit(y_tr_raw, oof_blend)

    print("\n" + "=" * 65)
    print(f"  RMSE OOF      : {metrics['rmse']:.4f}")
    print(f"  MAE  OOF      : {metrics['mae']:.4f}")
    print(f"  sMAPE         : {metrics['smape']:.2f}%")
    print(f"  Fiabilité (±{5:.0f}pt) : {metrics['taux_fiabilite']:.1f}%")
    print("=" * 65)

    return {
        "metrics":       metrics,
        "oof_preds":     oof_blend,
        "test_preds":    test_blend,
        "y_true":        y_tr_raw,
        "X_tr":          X_tr,
        "feature_cols":  X_tr.columns.tolist(),
        "groups":        groups,
        "df_train_meta": df_tr_meta,
    }
