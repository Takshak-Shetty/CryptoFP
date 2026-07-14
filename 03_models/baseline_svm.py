# ============================================================
# CryptoFP — 03_models/baseline_svm.py
#
# Trains an SVM classifier (RBF kernel) on the cryptographic
# benchmark dataset and saves the best model.
#
# Role in the paper:
#   Baseline 2 of 4.  SVM complements Random Forest:
#     - Kernel trick captures non-linear feature boundaries
#       between RSA timing distributions (skewed) and
#       Kyber (near-constant) without explicit feature
#       engineering for non-linearity
#     - Provides a second independent baseline so your
#       Table 3 shows RF vs SVM vs CNN vs LSTM — four
#       points of comparison is the IEEE expectation
#     - SVM is the strongest classical ML baseline for
#       side-channel timing analysis in the literature
#
# Key difference from RF:
#   SVM has no built-in feature_importances_. This script
#   uses sklearn permutation_importance as a model-agnostic
#   alternative — valid for any estimator and produces
#   comparable results to SHAP for the paper.
#
# What this script does:
#   1. Loads train / val / test splits (same as RF)
#   2. Runs GridSearchCV (5-fold, f1_macro) to find best SVM
#   3. Evaluates best model on val and test sets
#   4. Computes permutation feature importance on val set
#   5. Computes all paper metrics: F1, Acc, ROC-AUC, CM
#   6. Saves model to models/svm_model.pkl
#   7. Logs results to results/svm_results.json
#
# Usage:
#   python 03_models/baseline_svm.py
#   python 03_models/baseline_svm.py --quick
#   python 03_models/baseline_svm.py --eval-only
# ============================================================

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "02_feature_engineering"))

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from config import (
    PATHS, FEATURES, SVM, EVAL,
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, RANDOM_SEED,
)
from label_encoder import CryptoFPLabelEncoder


# ============================================================
# RESULTS DIRECTORY
# ============================================================

RESULTS_DIR    = ROOT / "results"
SVM_RESULTS_JSON = RESULTS_DIR / "svm_results.json"


# ============================================================
# DATA LOADING — shared with baseline_rf.py
# ============================================================

def load_splits(verbose: bool = True) -> tuple:
    """
    Load train / val / test splits. Identical to RF version —
    both baselines use the same preprocessed data.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test,
        feat_cols, encoder
    """
    for path in [PATHS.TRAIN_CSV, PATHS.VAL_CSV, PATHS.TEST_CSV]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found.\n"
                "Run Phase 2 first:\n"
                "  python 02_feature_engineering/preprocess.py\n"
                "  python 02_feature_engineering/label_encoder.py"
            )

    train = pd.read_csv(PATHS.TRAIN_CSV)
    val   = pd.read_csv(PATHS.VAL_CSV)
    test  = pd.read_csv(PATHS.TEST_CSV)

    base_feat = FEATURES.ALL_FEATURES + ["log_keygen_ms", "log_dec_ms"]
    feat_cols = [c for c in base_feat if c in train.columns]

    # Exclude SMOTE synthetic rows from train
    real_mask   = train["label_encoded"] >= 0
    train_real  = train[real_mask]

    X_train = train_real[feat_cols].values
    y_train = train_real["label_encoded"].values.astype(int)
    X_val   = val[feat_cols].values
    y_val   = val["label_encoded"].values.astype(int)
    X_test  = test[feat_cols].values
    y_test  = test["label_encoded"].values.astype(int)

    enc = CryptoFPLabelEncoder.load()

    if verbose:
        print(f"  Features   : {len(feat_cols)}  {feat_cols}")
        print(f"  Train rows : {len(X_train)}")
        print(f"  Val rows   : {len(X_val)}")
        print(f"  Test rows  : {len(X_test)}")
        print(f"  Classes    : {CLASS_NAMES}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feat_cols, enc


# ============================================================
# GRID SEARCH
# ============================================================

def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> GridSearchCV:
    """
    Run 5-fold stratified GridSearchCV over SVM hyperparameters.

    Parameter grid from config.py SVM.PARAM_GRID:
        C:      [0.1, 1, 10, 100]  — regularisation strength
        gamma:  ['scale', 'auto']  — RBF kernel width
        kernel: ['rbf']            — always RBF for this task

    Why RBF kernel:
        RSA timing distributions are heavily skewed (keygen
        can vary 10-100× due to prime search). The RBF kernel
        maps features into a high-dimensional space where these
        non-linear patterns become linearly separable, without
        needing explicit polynomial feature construction.

    probability=True is required for ROC-AUC computation. It
    uses Platt scaling (5-fold internal CV) which adds some
    training time but is essential for the paper's AUC metric.

    Args:
        X_train: scaled feature matrix
        y_train: encoded label vector
        verbose: print progress

    Returns:
        Fitted GridSearchCV object
    """
    cv = StratifiedKFold(
        n_splits=SVM.CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    svm_base = SVC(
        probability=True,               # required for predict_proba / ROC-AUC
        decision_function_shape="ovr",  # one-vs-rest for multiclass
        class_weight="balanced",        # handles class imbalance
        random_state=RANDOM_SEED,
        max_iter=SVM.MAX_ITER,
        cache_size=500,                 # MB — speeds up kernel computation
    )

    grid = GridSearchCV(
        estimator=svm_base,
        param_grid=SVM.PARAM_GRID,
        scoring=SVM.SCORING,
        cv=cv,
        n_jobs=-1,
        verbose=1 if verbose else 0,
        refit=True,
        return_train_score=True,
    )

    if verbose:
        n_combos = 1
        for v in SVM.PARAM_GRID.values():
            n_combos *= len(v)
        print(
            f"  Grid search: {n_combos} param combos × "
            f"{SVM.CV_FOLDS} folds = {n_combos * SVM.CV_FOLDS} fits"
        )
        print(
            f"  Note: probability=True uses Platt scaling "
            f"(internal 5-fold CV per fit) — slower than RF grid search."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*The least populated.*")
        warnings.filterwarnings("ignore", message=".*max_iter.*")
        t0 = time.time()
        grid.fit(X_train, y_train)
        elapsed = time.time() - t0

    if verbose:
        print(f"  Grid search complete in {elapsed:.1f}s")
        print(f"  Best params : {grid.best_params_}")
        print(f"  Best CV F1  : {grid.best_score_:.4f}")

    return grid


# ============================================================
# QUICK TRAIN
# ============================================================

def train_quick(
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> SVC:
    """
    Train SVM with C=10, gamma='scale' — sensible defaults
    for normalised timing features. No grid search.
    Use for rapid iteration / pipeline testing only.
    """
    svm = SVC(
        C=10,
        gamma="scale",
        kernel="rbf",
        probability=True,
        decision_function_shape="ovr",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=SVM.MAX_ITER,
        cache_size=500,
    )
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*max_iter.*")
        svm.fit(X_train, y_train)
    if verbose:
        n_sv = sum(svm.n_support_)
        print(
            f"  Quick train: C=10, gamma=scale  "
            f"in {time.time()-t0:.2f}s  "
            f"support vectors={n_sv}"
        )
    return svm


# ============================================================
# PERMUTATION FEATURE IMPORTANCE
# ============================================================

def compute_permutation_importance(
    model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    feat_cols: list,
    n_repeats: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Compute permutation importance for the SVM on the validation set.

    Unlike RF's built-in feature_importances_ (which uses impurity
    reduction on training data), permutation importance works by
    randomly shuffling one feature at a time and measuring the drop
    in F1. This is model-agnostic and measures impact on held-out
    data — more honest than training-set impurity.

    For the paper:
        Use this alongside RF's SHAP values as a cross-check.
        If both methods agree on the top features, that's a
        stronger finding than either alone.

    Args:
        model:     trained SVC
        X:         feature matrix (val set recommended)
        y:         true labels
        feat_cols: feature names in column order
        n_repeats: number of shuffle repeats per feature
        verbose:   print top features

    Returns:
        dict mapping feature name → mean importance score
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=RANDOM_SEED,
            scoring="f1_macro",
        )

    importance_dict = dict(
        sorted(
            zip(feat_cols, result.importances_mean.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    if verbose:
        print(f"\n  Permutation feature importance (val set, {n_repeats} repeats):")
        print(f"  (shuffling a feature and measuring F1 drop — higher = more important)")
        for i, (feat, imp) in enumerate(list(importance_dict.items())[:8]):
            bar = "█" * max(0, int(imp * 100))
            sign = "+" if imp > 0 else ""
            print(f"    {i+1}. {feat:<24} {sign}{imp:.4f}  {bar}")
        print()
        print(
            "  Note: negative importance means the feature adds noise.\n"
            "  Features with importance near 0 are not contributing.\n"
            "  Compare top features here with RF SHAP values."
        )

    return importance_dict


# ============================================================
# EVALUATION — shared structure with baseline_rf.py
# ============================================================

def evaluate(
    model: SVC,
    X: np.ndarray,
    y_true: np.ndarray,
    split_name: str,
    feat_cols: list,
    enc: CryptoFPLabelEncoder,
    perm_importance: dict = None,
    verbose: bool = True,
) -> dict:
    """
    Compute all evaluation metrics for one data split.

    Same metrics as RF for direct comparison in Table 3:
        accuracy, f1_macro, f1_weighted,
        precision_macro, recall_macro,
        roc_auc_ovr, per_class_f1, confusion_matrix

    Additionally records:
        n_support_vectors — SVM-specific diagnostic
        decision_function_shape

    Args:
        model:           trained SVC
        X:               feature matrix
        y_true:          true labels (canonical int encoding)
        split_name:      "val" or "test"
        feat_cols:       feature column names
        enc:             CryptoFPLabelEncoder
        perm_importance: pre-computed permutation importance dict
        verbose:         print metrics

    Returns:
        dict of all metrics (JSON-serialisable)
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    acc       = accuracy_score(y_true, y_pred)
    f1_mac    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wei    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_mac  = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    rec_mac   = recall_score(y_true,  y_pred, average="macro",    zero_division=0)

    # ROC-AUC — only over classes present in y_true
    try:
        present   = sorted(np.unique(y_true).tolist())
        y_prob_pr = y_prob[:, present]
        roc_auc   = roc_auc_score(
            y_true, y_prob_pr,
            multi_class=EVAL.ROC_MULTICLASS,
            average=EVAL.AVERAGE,
            labels=present,
        )
    except Exception as e:
        roc_auc = -1.0
        if verbose:
            print(f"  [WARN] ROC-AUC failed: {e}")

    # Per-class F1
    per_class_f1_arr = f1_score(
        y_true, y_pred,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    per_class_dict = {
        CLASS_NAMES[i]: round(float(per_class_f1_arr[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    # Confusion matrix
    cm = confusion_matrix(
        y_true, y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    # SVM-specific diagnostics
    n_support = int(sum(model.n_support_)) if hasattr(model, "n_support_") else -1

    results = {
        "split":                split_name,
        "n_samples":            len(y_true),
        "accuracy":             round(float(acc),      4),
        "f1_macro":             round(float(f1_mac),   4),
        "f1_weighted":          round(float(f1_wei),   4),
        "precision_macro":      round(float(prec_mac), 4),
        "recall_macro":         round(float(rec_mac),  4),
        "roc_auc_ovr":          round(float(roc_auc),  4),
        "per_class_f1":         per_class_dict,
        "confusion_matrix":     cm.tolist(),
        "n_support_vectors":    n_support,
        "permutation_importance": perm_importance or {},
    }

    if verbose:
        print(f"\n  === {split_name.upper()} SET RESULTS ===")
        print(f"  Accuracy        : {acc:.4f}")
        print(f"  F1 (macro)      : {f1_mac:.4f}  "
              f"{'OK' if f1_mac >= EVAL.MIN_F1_THRESHOLD else 'BELOW TARGET'}")
        print(f"  F1 (weighted)   : {f1_wei:.4f}")
        print(f"  Precision (mac) : {prec_mac:.4f}")
        print(f"  Recall (macro)  : {rec_mac:.4f}")
        print(f"  ROC-AUC (OVR)   : {roc_auc:.4f}")
        print(f"  Support vectors : {n_support}")
        print()
        print("  Per-class F1:")
        for cls, f1_val in per_class_dict.items():
            bar = "█" * int(f1_val * 20)
            print(f"    {cls:<14} {f1_val:.4f}  {bar}")
        print()
        report = classification_report(
            y_true, y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
        print("  Classification report:")
        for line in report.split("\n"):
            print(f"    {line}")
        print()
        print("  Confusion matrix (rows=true, cols=pred):")
        header = "  " + "".join(f"{c[:8]:>10}" for c in CLASS_NAMES)
        print(header)
        for i, row in enumerate(cm):
            row_str = (
                f"  {CLASS_NAMES[i][:8]:<10}"
                + "".join(f"{v:>10}" for v in row)
            )
            print(row_str)

    return results


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(
    model: SVC,
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    5-fold stratified CV to report mean ± std F1.
    Required for IEEE table — proves result is not a lucky split.

    Uses the best SVM estimator (already fitted by GridSearchCV).
    """
    cv = StratifiedKFold(
        n_splits=EVAL.CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )

    result = {
        "cv_f1_scores": [round(float(s), 4) for s in scores],
        "cv_f1_mean":   round(float(scores.mean()), 4),
        "cv_f1_std":    round(float(scores.std()),  4),
    }

    if verbose:
        print(f"  Cross-validation ({EVAL.CV_FOLDS}-fold):")
        print(f"    Scores : {result['cv_f1_scores']}")
        print(f"    Mean   : {result['cv_f1_mean']:.4f}")
        print(f"    Std    : {result['cv_f1_std']:.4f}")
        print(
            f"    Report : F1 = {result['cv_f1_mean']:.3f} "
            f"± {result['cv_f1_std']:.3f}  (cite in Table 3)"
        )

    return result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results: dict, verbose: bool = True):
    """Save all metrics to JSON for paper Table 3."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SVM_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    size_kb = SVM_RESULTS_JSON.stat().st_size / 1024
    if verbose:
        print(f"  Saved: {SVM_RESULTS_JSON.relative_to(ROOT)}  ({size_kb:.1f} KB)")


# ============================================================
# MAIN
# ============================================================

def main(
    quick:     bool = False,
    eval_only: bool = False,
    verbose:   bool = True,
) -> dict:
    """
    Full SVM training and evaluation pipeline.
    Called by run_pipeline.py Phase 3, or directly from CLI.

    Args:
        quick:     skip grid search, use default C=10
        eval_only: load existing model, re-evaluate only
        verbose:   print detailed metrics

    Returns:
        dict with model, val_results, test_results, cv_results
    """
    print("=" * 65)
    print("  CryptoFP — SVM Baseline (RBF kernel)")
    print("=" * 65)
    mode_str = (
        "EVAL ONLY" if eval_only
        else "QUICK (C=10, gamma=scale)" if quick
        else "FULL (with grid search)"
    )
    print(f"  Mode     : {mode_str}")
    print(f"  Kernel   : RBF")
    print(f"  Seed     : {RANDOM_SEED}")
    print(f"  CV folds : {SVM.CV_FOLDS}")
    print(f"  Max iter : {SVM.MAX_ITER}")
    print(f"  Min F1   : {EVAL.MIN_F1_THRESHOLD}")
    print()

    # ── 1. Load data ─────────────────────────────────────────
    print("  Step 1/6 — Loading splits")
    try:
        (X_train, y_train, X_val, y_val,
         X_test,  y_test,  feat_cols, enc) = load_splits(verbose=verbose)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return None

    # ── 2. Train or load ──────────────────────────────────────
    if eval_only:
        print("\n  Step 2/6 — Loading existing model")
        if not PATHS.SVM_MODEL.exists():
            print(f"  [ERROR] {PATHS.SVM_MODEL} not found. Run without --eval-only.")
            return None
        model = joblib.load(PATHS.SVM_MODEL)
        best_params = model.get_params()
        print(f"  Loaded: {PATHS.SVM_MODEL.name}")
        print(
            f"  C={best_params.get('C')}  "
            f"gamma={best_params.get('gamma')}  "
            f"kernel={best_params.get('kernel')}"
        )
        grid = None
    elif quick:
        print("\n  Step 2/6 — Quick training (no grid search)")
        model = train_quick(X_train, y_train, verbose=verbose)
        best_params = model.get_params()
        grid = None
    else:
        print("\n  Step 2/6 — Grid search training")
        grid = run_grid_search(X_train, y_train, verbose=verbose)
        model = grid.best_estimator_
        best_params = grid.best_params_

    # ── 3. Permutation importance (val set) ───────────────────
    print("\n  Step 3/6 — Permutation feature importance")
    perm_imp = compute_permutation_importance(
        model, X_val, y_val, feat_cols,
        n_repeats=10,
        verbose=verbose,
    )

    # ── 4. Cross-validation ───────────────────────────────────
    print("\n  Step 4/6 — Cross-validation")
    cv_results = run_cross_validation(model, X_train, y_train, verbose=verbose)

    # ── 5. Evaluate ───────────────────────────────────────────
    print("\n  Step 5/6 — Evaluation")
    val_results = evaluate(
        model, X_val, y_val, "val", feat_cols, enc,
        perm_importance=perm_imp, verbose=verbose,
    )
    test_results = evaluate(
        model, X_test, y_test, "test", feat_cols, enc,
        perm_importance=None, verbose=verbose,
    )

    # Paper-readiness check
    test_f1 = test_results["f1_macro"]
    if test_f1 < EVAL.MIN_F1_THRESHOLD:
        print(
            f"\n  [WARN] Test F1={test_f1:.4f} below "
            f"target {EVAL.MIN_F1_THRESHOLD}.\n"
            f"  Expected with full 1800-sample dataset."
        )
    else:
        print(f"\n  Test F1={test_f1:.4f} >= target {EVAL.MIN_F1_THRESHOLD} OK")

    # ── RF comparison hint ────────────────────────────────────
    rf_results_path = RESULTS_DIR / "rf_results.json"
    if rf_results_path.exists():
        with open(rf_results_path) as f:
            rf_res = json.load(f)
        rf_f1  = rf_res.get("test", {}).get("f1_macro", None)
        svm_f1 = test_results["f1_macro"]
        if rf_f1 is not None:
            diff = svm_f1 - rf_f1
            sign = "+" if diff >= 0 else ""
            print(
                f"\n  SVM vs RF comparison (test F1):\n"
                f"    SVM : {svm_f1:.4f}\n"
                f"    RF  : {rf_f1:.4f}\n"
                f"    Diff: {sign}{diff:.4f}  "
                f"({'SVM better' if diff > 0 else 'RF better' if diff < 0 else 'tied'})"
            )

    # ── 6. Save ───────────────────────────────────────────────
    print("\n  Step 6/6 — Saving")
    PATHS.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PATHS.SVM_MODEL)
    size_kb = PATHS.SVM_MODEL.stat().st_size / 1024
    print(f"  Saved model : {PATHS.SVM_MODEL.name}  ({size_kb:.1f} KB)")

    full_results = {
        "model":        "SVM_RBF",
        "mode":         mode_str,
        "best_params":  {k: str(v) for k, v in best_params.items()},
        "n_train":      len(X_train),
        "n_val":        len(X_val),
        "n_test":       len(X_test),
        "n_features":   len(feat_cols),
        "feature_cols": feat_cols,
        "cv":           cv_results,
        "val":          val_results,
        "test":         test_results,
        "paper_cite":   (
            f"SVM(RBF): F1={test_results['f1_macro']:.3f} "
            f"± {cv_results['cv_f1_std']:.3f}, "
            f"Acc={test_results['accuracy']:.3f}, "
            f"AUC={test_results['roc_auc_ovr']:.3f}"
        ),
    }
    save_results(full_results, verbose=verbose)

    # ── Final summary ─────────────────────────────────────────
    print()
    print("=" * 65)
    print("  SVM (RBF) — FINAL SUMMARY")
    print("=" * 65)
    print(f"  Test accuracy   : {test_results['accuracy']:.4f}")
    print(f"  Test F1 (macro) : {test_results['f1_macro']:.4f}")
    print(f"  Test ROC-AUC    : {test_results['roc_auc_ovr']:.4f}")
    print(f"  CV F1           : {cv_results['cv_f1_mean']:.3f} ± {cv_results['cv_f1_std']:.3f}")
    print(f"  Support vectors : {test_results['n_support_vectors']}")
    print()
    print(f"  Top feature (permutation): {list(perm_imp.keys())[0]}")
    print()
    print(f"  Paper cite (Table 3):")
    print(f"    {full_results['paper_cite']}")
    print()
    print("  Model saved to: models/svm_model.pkl")
    print("  Results saved to: results/svm_results.json")
    print()
    print("  Next: python 03_models/cnn_1d.py")
    print("=" * 65)

    return {
        "model":        model,
        "val_results":  val_results,
        "test_results": test_results,
        "cv_results":   cv_results,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — train and evaluate SVM (RBF) baseline"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip grid search — use C=10, gamma=scale."
    )
    parser.add_argument(
        "--eval-only", action="store_true", dest="eval_only",
        help="Load existing svm_model.pkl and re-evaluate."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed per-class metrics."
    )
    args = parser.parse_args()

    main(quick=args.quick, eval_only=args.eval_only, verbose=not args.quiet)