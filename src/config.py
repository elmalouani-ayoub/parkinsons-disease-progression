"""
config.py — Référentiel central du projet.

Toutes les constantes, chemins et hyperparamètres figés sont isolés ici.
Aucune logique métier : ce fichier ne fait qu'*être lu*, jamais exécuté.
"""

# ---------------------------------------------------------------------------
# Chemins relatifs vers les fichiers de données brutes
# ---------------------------------------------------------------------------
DATA_PATHS: dict[str, str] = {
    "X_train": "data/X_train_6ZIKlTY.csv",
    "y_train": "data/y_train_lXj6X5y.csv",
    "X_test":  "data/X_test_oiZ2ukx.csv",
}

# ---------------------------------------------------------------------------
# Variable(s) cible(s)
# ---------------------------------------------------------------------------
TARGET: str = "target"

# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Validation croisée
# ---------------------------------------------------------------------------
N_SPLITS: int = 5

# ---------------------------------------------------------------------------
# Constantes du modèle PK/PD de la Lévodopa
# Référence : Dietz et al. (2001) — Parkinson's levodopa PK/PD modeling
# ---------------------------------------------------------------------------
PKPD: dict = {
    "teq_base":    90.0,   # Capacité de stockage initiale (minutes)
    "teq_decay":    3.8,   # Perte par année de maladie (minutes/an)
    "teq_min":      5.0,   # Valeur plancher de Teq
    "teq_max":    150.0,   # Valeur plafond de Teq
    "ledd_norm":  400.0,   # Dose normalisatrice (mg)
    "default_t_on":  2.0,  # Heures depuis prise ON (valeur clinique par défaut)
    "default_t_off": 12.0, # Heures depuis prise OFF (valeur clinique par défaut)
}

# ---------------------------------------------------------------------------
# Encodage de la mutation génétique
# ---------------------------------------------------------------------------
GENE_MAP: dict[str, int] = {
    "LRRK2+":     1,
    "GBA+":        2,
    "OTHER+":      3,
    "No Mutation": 0,
    "Unknown":    -1,
}

# ---------------------------------------------------------------------------
# Colonnes exclues de la matrice des features
# ---------------------------------------------------------------------------
COLS_DROP: list[str] = [
    "Index", "patient_id", "target", "is_test",
    "cohort", "gene", "age_at_diagnosis", "age_at_diagnosis_f",
]

# ---------------------------------------------------------------------------
# Hyperparamètres des modèles (figés après tuning)
# ---------------------------------------------------------------------------
MODEL_PARAMS: dict[str, dict] = {
    "hgb": {
        "max_iter":        1500,
        "max_leaf_nodes":  63,
        "min_samples_leaf": 20,
        "learning_rate":   0.03,
        "random_state":    RANDOM_STATE,
    },
    "lgb": {
        "n_estimators":      1500,
        "learning_rate":     0.03,
        "num_leaves":        63,
        "min_child_samples": 20,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "n_jobs":           -1,
        "random_state":      RANDOM_STATE,
        "verbose":          -1,
    },
    "xgb": {
        "n_estimators":     1500,
        "learning_rate":    0.03,
        "max_depth":        6,
        "min_child_weight": 20,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "n_jobs":          -1,
        "random_state":     RANDOM_STATE,
    },
}

# ---------------------------------------------------------------------------
# Audit clinique
# ---------------------------------------------------------------------------
CLINICAL_MARGIN: float = 5.0  # Marge d'acceptabilité humaine (±5 points moteurs)
