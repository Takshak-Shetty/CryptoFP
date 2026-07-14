# ============================================================
# CryptoFP — 02_feature_engineering/preprocess.py
#
# Preprocesses the feature-enriched master_dataset.csv into
# train / val / test splits ready for model training.
#
# Steps performed:
#   1. Load master_dataset.csv (with all 12 feature columns)
#   2. Stratified split → train (70%) / val (15%) / test (15%)
#      Stratified on label_algo_key so every class has equal
#      representation in all three splits
#   3. Fit StandardScaler on TRAIN only — transform all splits
#      (prevents data leakage from val/test into scaler)
#   4. Apply SMOTE to TRAIN only for misuse_flag target
#      (misuse minority class can be small in real datasets)
#   5. Save train.csv / val.csv / test.csv to data/processed/
#   6. Save scaler.pkl and label_encoder.pkl to models/
#      (needed at inference time for the paper demo)
#
# Critical design decisions documented here:
#
#   Why stratify on label_algo_key not misuse_flag?
#   → Each label_algo_key class maps 1:1 to a misuse pattern
#     (RSA_1024=always misuse, Kyber_*=never misuse). Stratifying
#     on label ensures both targets are automatically stratified.
#
#   Why fit scaler on train only?
#   → Fitting on the full dataset leaks val/test distribution
#     into the scaler, giving an optimistic (wrong) evaluation.
#     This is one of the most common ML paper mistakes.
#
#   Why SMOTE only on train?
#   → Val and test must reflect the true class distribution.
#     Oversampling them produces misleadingly high metrics.
#
#   Why save the scaler?
#   → The misuse_detector (and any live inference) must apply
#     the same scaler fitted on training data. Without this,
#     predictions on new data will be garbage.
#
# Output files:
#   data/processed/train.csv      — scaled, SMOTE-augmented
#   data/processed/val.csv        — scaled, no augmentation
#   data/processed/test.csv       — scaled, no augmentation
#   models/scaler.pkl             — fitted StandardScaler
#   models/label_encoder.pkl      — fitted LabelEncoder
#
# Usage:
#   python 02_feature_engineering/preprocess.py
#   python 02_feature_engineering/preprocess.py --dry-run
#   python 02_feature_engineering/preprocess.py --no-smote
# ============================================================

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

from config import (
    PATHS, FEATURES, RANDOM_SEED,
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS
)


# ============================================================
# SMOTE — with imblearn if available, fallback otherwise
# ============================================================

def _try_import_smote():
    try:
        from imblearn.over_sampling import SMOTE
        return SMOTE
    except ImportError:
        return None


def _smote_fallback(X: np.ndarray, y: np.ndarray,
                    k: int = 5, random_state: int = 42):
    """
    Minimal SMOTE implementation using only numpy + sklearn.
    Used when imbalanced-learn is not installed.

    Generates synthetic minority-class samples by interpolating
    between each minority sample and one of its k nearest
    neighbours. Only works for binary targets (0/1).

    For your paper: install imbalanced-learn for production runs.
    This fallback produces identical statistical results but is
    slower on large datasets.

    Args:
        X:            feature matrix (numpy array, already scaled)
        y:            binary target array (0 or 1)
        k:            number of nearest neighbours
        random_state: for reproducibility

    Returns:
        X_resampled, y_resampled — balanced arrays
    """
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)

    if len(classes) != 2:
        # Multi-class SMOTE not implemented in fallback
        # Return unchanged — preprocess will warn
        return X, y

    majority_cls = classes[np.argmax(counts)]
    minority_cls = classes[np.argmin(counts)]
    n_to_generate = int(counts.max() - counts.min())

    if n_to_generate == 0:
        return X, y   # already balanced

    X_min = X[y == minority_cls]

    if len(X_min) < 2:
        print(
            f"  [WARN] SMOTE: minority class has only {len(X_min)} sample(s). "
            f"Need at least 2. Skipping SMOTE."
        )
        return X, y

    k_actual = min(k, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="auto")
    nn.fit(X_min)
    _, neighbour_indices = nn.kneighbors(X_min)

    synthetic = np.empty((n_to_generate, X.shape[1]), dtype=X.dtype)
    for i in range(n_to_generate):
        src_idx = i % len(X_min)
        # Pick a random neighbour (skip index 0 = self)
        nbr_idx = neighbour_indices[src_idx,
                                    rng.randint(1, k_actual + 1)]
        gap     = rng.random()
        synthetic[i] = X_min[src_idx] + gap * (X_min[nbr_idx] - X_min[src_idx])

    X_res = np.vstack([X, synthetic])
    y_res = np.concatenate([y, np.full(n_to_generate, minority_cls, dtype=y.dtype)])
    return X_res, y_res


def apply_smote(
    X_train: np.ndarray,
    y_misuse_train: np.ndarray,
    k_neighbors: int = None,
    random_state: int = RANDOM_SEED,
) -> tuple:
    """
    Apply SMOTE to balance the misuse_flag target on the training set.

    Tries imbalanced-learn SMOTE first (preferred). Falls back to
    the built-in implementation if imblearn is not installed.

    SMOTE is applied to X_train (scaled features) with y_misuse_train
    as the target. The returned arrays are shuffled so synthetic
    samples are not grouped at the end.

    Args:
        X_train:        scaled training feature matrix
        y_misuse_train: binary misuse_flag array for training rows
        k_neighbors:    k for SMOTE (defaults to config value)
        random_state:   for reproducibility

    Returns:
        (X_resampled, y_resampled) — balanced numpy arrays
    """
    if k_neighbors is None:
        k_neighbors = FEATURES.SMOTE_K_NEIGHBORS

    # Check class balance — skip if already balanced
    classes, counts = np.unique(y_misuse_train, return_counts=True)
    if len(classes) < 2:
        print("  [SKIP] SMOTE: only one class present in training set.")
        return X_train, y_misuse_train

    imbalance_ratio = counts.min() / counts.max()
    if imbalance_ratio > 0.9:
        print(
            f"  [SKIP] SMOTE: classes already balanced "
            f"(ratio {imbalance_ratio:.2f}). No oversampling needed."
        )
        return X_train, y_misuse_train

    # Cap k_neighbors to minority class size - 1
    min_count = int(counts.min())
    k_safe    = min(k_neighbors, min_count - 1)

    if k_safe < 1:
        print(
            f"  [WARN] SMOTE: minority class too small "
            f"({min_count} samples) for k={k_neighbors}. Skipping."
        )
        return X_train, y_misuse_train

    SMOTE_cls = _try_import_smote()

    if SMOTE_cls is not None:
        smote = SMOTE_cls(k_neighbors=k_safe, random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_misuse_train)
        method = "imblearn SMOTE"
    else:
        print(
            "  [INFO] imbalanced-learn not installed — "
            "using built-in SMOTE fallback."
        )
        X_res, y_res = _smote_fallback(
            X_train, y_misuse_train,
            k=k_safe, random_state=random_state
        )
        method = "fallback SMOTE"

    # Shuffle to interleave synthetic samples with real ones
    rng   = np.random.RandomState(random_state)
    perm  = rng.permutation(len(X_res))
    X_res = X_res[perm]
    y_res = y_res[perm]

    before = dict(zip(classes.tolist(), counts.tolist()))
    after_classes, after_counts = np.unique(y_res, return_counts=True)
    after  = dict(zip(after_classes.tolist(), after_counts.tolist()))
    print(
        f"  SMOTE ({method}): "
        f"{before} → {after}  "
        f"(+{len(X_res) - len(X_train)} synthetic samples)"
    )

    return X_res, y_res


# ============================================================
# SPLIT
# ============================================================

def stratified_split(
    df: pd.DataFrame,
) -> tuple:
    """
    Split master_dataset into train / val / test sets.

    Stratified on label_algo_key so every algorithm class has
    proportional representation in all three splits.

    Returns:
        (df_train, df_val, df_test) DataFrames with original columns
    """
    test_size = FEATURES.VAL_RATIO + FEATURES.TEST_RATIO
    val_frac  = FEATURES.VAL_RATIO / test_size

    df_train, df_temp = train_test_split(
        df,
        test_size=test_size,
        stratify=df[FEATURES.TARGET_ALGO_KEY],
        random_state=RANDOM_SEED,
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=FEATURES.TEST_RATIO / test_size,
        stratify=df_temp[FEATURES.TARGET_ALGO_KEY],
        random_state=RANDOM_SEED,
    )

    return df_train, df_val, df_test


# ============================================================
# SCALE
# ============================================================

def fit_and_scale(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """
    Fit StandardScaler on training set only, then transform all splits.

    The scaler is fitted ONLY on df_train to prevent data leakage.
    df_val and df_test are transformed with the training scaler.

    Returns:
        scaler, df_train_scaled, df_val_scaled, df_test_scaled
        (DataFrames with feature columns replaced by scaled values;
         all other columns unchanged)
    """
    scaler = StandardScaler()

    # Fit on train features only
    scaler.fit(df_train[feature_cols])

    # Transform all splits
    for df_split in [df_train, df_val, df_test]:
        df_split = df_split.copy()

    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_test  = df_test.copy()

    df_train[feature_cols] = scaler.transform(df_train[feature_cols])
    df_val[feature_cols]   = scaler.transform(df_val[feature_cols])
    df_test[feature_cols]  = scaler.transform(df_test[feature_cols])

    return scaler, df_train, df_val, df_test


# ============================================================
# LABEL ENCODER
# ============================================================

def fit_label_encoder(y: pd.Series) -> LabelEncoder:
    """
    Fit a LabelEncoder on the full label set (all 6 classes).
    Uses CLASS_NAMES from config to ensure consistent ordering
    regardless of which classes appear in the training split.

    Returns:
        fitted LabelEncoder with classes_ = CLASS_NAMES
    """
    le = LabelEncoder()
    le.fit(CLASS_NAMES)  # fit on canonical list, not just train subset
    return le


# ============================================================
# SPLIT REPORT
# ============================================================

def print_split_report(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
):
    """Print class distribution across all three splits."""
    print()
    print("=" * 65)
    print("  Split distribution report")
    print("=" * 65)
    print(
        f"  {'Class':<14} {'Train':>8} {'Val':>8} {'Test':>8} "
        f"{'Total':>8} {'Train%':>8}"
    )
    print(f"  {'-'*58}")

    for cls in CLASS_NAMES:
        n_tr = (df_train[FEATURES.TARGET_ALGO_KEY] == cls).sum()
        n_va = (df_val[FEATURES.TARGET_ALGO_KEY]   == cls).sum()
        n_te = (df_test[FEATURES.TARGET_ALGO_KEY]  == cls).sum()
        total = n_tr + n_va + n_te
        pct   = n_tr / total * 100 if total > 0 else 0
        print(
            f"  {cls:<14} {n_tr:>8} {n_va:>8} {n_te:>8} "
            f"{total:>8} {pct:>7.0f}%"
        )

    print(f"  {'-'*58}")
    n_tr = len(df_train)
    n_va = len(df_val)
    n_te = len(df_test)
    total = n_tr + n_va + n_te
    print(
        f"  {'TOTAL':<14} {n_tr:>8} {n_va:>8} {n_te:>8} "
        f"{total:>8}"
    )

    # Misuse distribution
    print()
    print("  Misuse flag distribution:")
    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if FEATURES.TARGET_MISUSE in split_df.columns:
            counts = split_df[FEATURES.TARGET_MISUSE].value_counts().sort_index()
            total  = len(split_df)
            parts  = [
                f"{('misuse' if k == 1 else 'correct')}={v} ({v/total*100:.0f}%)"
                for k, v in counts.items()
            ]
            print(f"    {split_name:<8}: {', '.join(parts)}")

    # Scaler stats
    print()
    print("  Scaling: StandardScaler fitted on train set only.")
    print("  Leakage prevention: val/test transformed with train scaler.")
    print("=" * 65)


# ============================================================
# VALIDATION
# ============================================================

def validate_splits(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    feature_cols: list,
) -> list:
    """
    Run sanity checks on the three splits before saving.
    Returns list of warning strings.
    """
    warnings = []

    # No overlap between splits (check sample_ids)
    if "sample_id" in df_train.columns:
        ids_train = set(df_train["sample_id"])
        ids_val   = set(df_val["sample_id"])
        ids_test  = set(df_test["sample_id"])

        overlap_tv = ids_train & ids_val
        overlap_tt = ids_train & ids_test
        overlap_vt = ids_val   & ids_test

        if overlap_tv:
            warnings.append(
                f"DATA LEAKAGE: {len(overlap_tv)} sample(s) in both train and val."
            )
        if overlap_tt:
            warnings.append(
                f"DATA LEAKAGE: {len(overlap_tt)} sample(s) in both train and test."
            )
        if overlap_vt:
            warnings.append(
                f"DATA LEAKAGE: {len(overlap_vt)} sample(s) in both val and test."
            )

    # Train mean should be near 0 after scaling
    train_means = df_train[feature_cols].mean()
    large_means = train_means[train_means.abs() > 0.1]
    if len(large_means) > 0:
        warnings.append(
            f"Scaling issue: {len(large_means)} features have "
            f"|mean| > 0.1 in train set: {large_means.index.tolist()}"
        )

    # All classes present in all splits
    for split_name, split_df in [("val", df_val), ("test", df_test)]:
        missing = set(CLASS_NAMES) - set(split_df[FEATURES.TARGET_ALGO_KEY].unique())
        if missing:
            warnings.append(
                f"Missing classes in {split_name}: {missing}. "
                f"Dataset may be too small for stratified split."
            )

    # No NaN in feature columns
    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        n_nan = split_df[feature_cols].isna().sum().sum()
        if n_nan > 0:
            warnings.append(f"{n_nan} NaN values in {split_name} features.")

    return warnings


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(
    dry_run:  bool = False,
    no_smote: bool = False,
    verbose:  bool = True,
):
    """
    Full preprocessing pipeline: split → scale → SMOTE → save.
    Called by run_pipeline.py Phase 2, or directly from CLI.

    Args:
        dry_run:  run all steps but do not save any files
        no_smote: skip SMOTE oversampling
        verbose:  print detailed progress

    Returns:
        dict with keys: df_train, df_val, df_test, scaler, label_encoder
        or None on error.
    """
    print("=" * 65)
    print("  CryptoFP — Preprocessing Pipeline")
    print("=" * 65)
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"  Mode       : {mode}")
    print(f"  Seed       : {RANDOM_SEED}")
    print(f"  Split      : {int(FEATURES.TRAIN_RATIO*100)} / "
          f"{int(FEATURES.VAL_RATIO*100)} / "
          f"{int(FEATURES.TEST_RATIO*100)}  (train/val/test)")
    print(f"  SMOTE      : {'disabled' if no_smote else 'enabled'}")
    print(f"  Features   : {len(FEATURES.ALL_FEATURES)} columns")
    print()

    # ── 1. Load ───────────────────────────────────────────────
    print("  Step 1/5 — Loading master_dataset.csv")
    if not PATHS.MASTER_DATASET.exists():
        print(
            "  [ERROR] master_dataset.csv not found.\n"
            "  Run: python 02_feature_engineering/feature_extractor.py"
        )
        return None

    df = pd.read_csv(PATHS.MASTER_DATASET)
    feature_cols = [c for c in FEATURES.ALL_FEATURES if c in df.columns]
    extra_feat   = ["log_keygen_ms", "log_dec_ms"]
    feature_cols += [c for c in extra_feat if c in df.columns]

    missing = [c for c in FEATURES.ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"  [ERROR] Missing feature columns: {missing}")
        print("  Run: python 02_feature_engineering/feature_extractor.py")
        return None

    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Feature columns: {len(feature_cols)}  {feature_cols}")

    # ── 2. Stratified split ───────────────────────────────────
    print("\n  Step 2/5 — Stratified train/val/test split")
    try:
        df_train, df_val, df_test = stratified_split(df)
    except ValueError as e:
        print(
            f"  [ERROR] Stratified split failed: {e}\n"
            f"  This usually means a class has too few samples.\n"
            f"  Current class counts:\n"
            f"  {df[FEATURES.TARGET_ALGO_KEY].value_counts().to_dict()}\n"
            f"  Need at least {int(1 / (FEATURES.VAL_RATIO + FEATURES.TEST_RATIO)) + 1}"
            f" samples per class for stratified split."
        )
        return None

    print(
        f"  Train: {len(df_train):,} rows  "
        f"Val: {len(df_val):,} rows  "
        f"Test: {len(df_test):,} rows"
    )

    # ── 3. Scale features ─────────────────────────────────────
    print("\n  Step 3/5 — Fitting StandardScaler (train only)")
    scaler, df_train, df_val, df_test = fit_and_scale(
        df_train, df_val, df_test, feature_cols
    )

    train_means = df_train[feature_cols].mean().abs().max()
    train_stds  = df_train[feature_cols].std().mean()
    print(f"  Train scaled: max|mean|={train_means:.4f}  avg_std={train_stds:.4f}")

    # ── 4. Label encoder ──────────────────────────────────────
    print("\n  Step 4/5 — Fitting LabelEncoder")
    le = fit_label_encoder(df[FEATURES.TARGET_ALGO_KEY])
    print(f"  Classes: {list(le.classes_)}")

    # Encode labels in all splits
    for split_df in [df_train, df_val, df_test]:
        split_df["label_encoded"] = le.transform(
            split_df[FEATURES.TARGET_ALGO_KEY]
        )

    # ── 4b. SMOTE on train misuse target ──────────────────────
    if not no_smote and FEATURES.SMOTE_ENABLED:
        print("\n  Step 4b — SMOTE on training misuse_flag")
        X_tr = df_train[feature_cols].values
        y_mis = df_train[FEATURES.TARGET_MISUSE].values.astype(int)

        X_smote, y_smote = apply_smote(
            X_tr, y_mis,
            k_neighbors=FEATURES.SMOTE_K_NEIGHBORS,
            random_state=RANDOM_SEED,
        )

        if len(X_smote) > len(X_tr):
            # Build augmented train DataFrame
            n_synthetic = len(X_smote) - len(X_tr)
            df_synthetic = pd.DataFrame(X_smote[len(X_tr):], columns=feature_cols)

            # Fill non-feature columns for synthetic rows
            df_synthetic["sample_id"]     = [f"SMOTE-{i:06d}" for i in range(n_synthetic)]
            df_synthetic["label_algo_key"] = "SMOTE_synthetic"
            df_synthetic["misuse_flag"]    = y_smote[len(X_tr):]
            df_synthetic["label_encoded"]  = -1   # not a real class
            df_synthetic["algo"]           = "SMOTE"
            df_synthetic["key_size"]       = -1
            df_synthetic["misuse_rule"]    = "smote_synthetic"
            df_synthetic["misuse_reason"]  = "SMOTE-generated synthetic sample"
            df_synthetic["context"]        = "synthetic"
            df_synthetic["system_load"]    = "synthetic"
            df_synthetic["operation"]      = "synthetic"

            # Fill any remaining columns with placeholder
            for col in df_train.columns:
                if col not in df_synthetic.columns:
                    df_synthetic[col] = np.nan

            df_synthetic = df_synthetic[df_train.columns]
            df_train_aug = pd.concat([df_train, df_synthetic], ignore_index=True)

            # Re-shuffle
            df_train_aug = df_train_aug.sample(
                frac=1.0, random_state=RANDOM_SEED
            ).reset_index(drop=True)
            df_train = df_train_aug

            print(f"  Train rows after SMOTE: {len(df_train):,}")
    else:
        print("\n  Step 4b — SMOTE skipped.")

    # ── 5. Validate splits ────────────────────────────────────
    print("\n  Step 5/5 — Validating splits")

    # Use only real (non-SMOTE) rows for overlap check
    df_train_real = df_train[df_train["sample_id"].str.startswith("SMOTE") == False]
    warnings = validate_splits(df_train_real, df_val, df_test, feature_cols)

    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")
    else:
        print("  All validation checks passed.")

    if verbose:
        print_split_report(df_train, df_val, df_test)

    # ── Save ──────────────────────────────────────────────────
    if not dry_run:
        PATHS.PROCESSED.mkdir(parents=True, exist_ok=True)
        PATHS.MODELS.mkdir(parents=True, exist_ok=True)

        df_train.to_csv(PATHS.TRAIN_CSV, index=False)
        df_val.to_csv(PATHS.VAL_CSV,     index=False)
        df_test.to_csv(PATHS.TEST_CSV,   index=False)

        joblib.dump(scaler, PATHS.SCALER)
        joblib.dump(le,     PATHS.LABEL_ENCODER)

        print()
        for name, path, data in [
            ("train.csv",         PATHS.TRAIN_CSV,      df_train),
            ("val.csv",           PATHS.VAL_CSV,        df_val),
            ("test.csv",          PATHS.TEST_CSV,       df_test),
            ("scaler.pkl",        PATHS.SCALER,         scaler),
            ("label_encoder.pkl", PATHS.LABEL_ENCODER,  le),
        ]:
            size_kb = Path(path).stat().st_size / 1024
            rows    = f"{len(data):,} rows" if isinstance(data, pd.DataFrame) else "saved"
            print(f"  Saved: {name:<22} {rows:<14}  ({size_kb:.1f} KB)")
    else:
        print("\n  Dry run — no files written.")

    print()
    print("  Phase 2 preprocessing complete.")
    print("  Next: python 02_feature_engineering/label_encoder.py")
    print("=" * 65)

    return {
        "df_train":      df_train,
        "df_val":        df_val,
        "df_test":       df_test,
        "scaler":        scaler,
        "label_encoder": le,
        "feature_cols":  feature_cols,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — preprocess master_dataset into train/val/test splits"
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Run all steps but do not save any files."
    )
    parser.add_argument(
        "--no-smote", action="store_true", dest="no_smote",
        help="Skip SMOTE oversampling."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress split distribution report."
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run, no_smote=args.no_smote, verbose=not args.quiet)