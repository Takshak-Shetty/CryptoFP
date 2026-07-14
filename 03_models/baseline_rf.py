# ============================================================
# CryptoFP — 03_models/baseline_rf.py
#
# Trains a Random Forest classifier on the cryptographic
# benchmark dataset and saves the best model.
#
# Role in the paper:
#   Baseline 1 of 4.  Random Forest is the first model you
#   build because it:
#     - Trains in seconds (no GPU needed)
#     - Gives immediate signal on whether features are useful
#     - Provides SHAP feature importance (via shap_analysis.py)
#     - Sets the accuracy floor your LSTM must beat
#
#   IEEE reviewers expect a strong baseline. RF at >95% F1
#   shows your features are genuinely discriminating, which
#   makes the LSTM result credible.
#
# What this script does:
#   1. Loads train / val / test splits
#   2. Runs GridSearchCV (5-fold, f1_macro) to find best RF
#   3. Evaluates best model on val and test sets
#   4. Computes all paper metrics: F1, accuracy, precision,
#      recall, ROC-AUC, confusion matrix
#   5. Saves model to models/rf_model.pkl
#   6. Logs results to results/rf_results.json (paper Table 3)
#   7. Warns if F1 is below EVAL.MIN_F1_THRESHOLD
#
# Usage:
#   python 03_models/baseline_rf.py
#   python 03_models/baseline_rf.py --quick   # no grid search
#   python 03_models/baseline_rf.py --eval-only  # skip training
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from config import (
    PATHS, FEATURES, RF, EVAL,
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, RANDOM_SEED,
)
from label_encoder import CryptoFPLabelEncoder


# ============================================================
# RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = ROOT / "results"
RF_RESULTS_JSON = RESULTS_DIR / "rf_results.json"


# ============================================================
# DATA LOADING
# ============================================================

def load_splits(verbose: bool = True) -> tuple:
    """
    Load train / val / test splits and extract feature matrix
    and label vectors.

    Filters out SMOTE synthetic rows from train (label_encoded=-1)
    since they should not be in the test-time evaluation.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_cols, encoder
    """
    for path in [PATHS.TRAIN_CSV, PATHS.VAL_CSV, PATHS.TEST_CSV]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found. "
                "Run Phase 2 first:\n"
                "  python 02_feature_engineering/preprocess.py\n"
                "  python 02_feature_engineering/label_encoder.py"
            )

    train = pd.read_csv(PATHS.TRAIN_CSV)
    val   = pd.read_csv(PATHS.VAL_CSV)
    test  = pd.read_csv(PATHS.TEST_CSV)

    # Feature columns — use all available (raw + derived + log)
    base_feat = FEATURES.ALL_FEATURES + ["log_keygen_ms", "log_dec_ms"]
    feat_cols = [c for c in base_feat if c in train.columns]

    # Filter SMOTE synthetic rows from train
    real_mask = train["label_encoded"] >= 0
    train_real = train[real_mask]

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
    Run 5-fold stratified GridSearchCV over RF hyperparameters.

    Parameter grid is defined in config.py RF.PARAM_GRID.
    Scoring metric is RF.SCORING (f1_macro).

    With small datasets (< 200 rows), GridSearchCV may warn
    about class imbalance per fold — this is expected and safe.
    The warning disappears with the full 1800-sample dataset.

    Args:
        X_train: scaled training feature matrix
        y_train: encoded label vector
        verbose: print progress

    Returns:
        Fitted GridSearchCV object
    """
    cv = StratifiedKFold(
        n_splits=RF.CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    rf_base = RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=RF.N_JOBS,
        class_weight="balanced",   # handles any remaining imbalance
    )

    grid = GridSearchCV(
        estimator=rf_base,
        param_grid=RF.PARAM_GRID,
        scoring=RF.SCORING,
        cv=cv,
        n_jobs=RF.N_JOBS,
        verbose=1 if verbose else 0,
        refit=True,                # refit best params on full train set
        return_train_score=True,
    )

    if verbose:
        n_combos = 1
        for v in RF.PARAM_GRID.values():
            n_combos *= len(v)
        print(
            f"  Grid search: {n_combos} param combos × "
            f"{RF.CV_FOLDS} folds = {n_combos * RF.CV_FOLDS} fits"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*The least populated class in y has only.*",
        )
        t0 = time.time()
        grid.fit(X_train, y_train)
        elapsed = time.time() - t0

    if verbose:
        print(f"  Grid search complete in {elapsed:.1f}s")
        print(f"  Best params : {grid.best_params_}")
        print(f"  Best CV F1  : {grid.best_score_:.4f}")

    return grid


# ============================================================
# QUICK TRAIN (no grid search)
# ============================================================

def train_quick(
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> RandomForestClassifier:
    """
    Train RF with sensible defaults — no grid search.
    Use for rapid iteration / pipeline testing.
    Results should NOT be reported in the paper.
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=RF.N_JOBS,
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    if verbose:
        print(f"  Quick train: 200 trees in {time.time()-t0:.2f}s")
    return rf


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    split_name: str,
    feat_cols: list,
    enc: CryptoFPLabelEncoder,
    verbose: bool = True,
) -> dict:
    """
    Compute all evaluation metrics for one data split.

    Metrics computed:
        accuracy      — overall % correct
        f1_macro      — unweighted mean F1 across all classes
        f1_weighted   — class-frequency-weighted mean F1
        precision_macro, recall_macro
        roc_auc_ovr   — one-vs-rest ROC-AUC (multi-class)
        per_class_f1  — F1 score for each of the 6 classes
        confusion_matrix — 6×6 matrix

    These map directly to the paper's Table 3 (model comparison).

    Args:
        model:      trained sklearn estimator
        X:          feature matrix
        y_true:     true labels (canonical integer encoding)
        split_name: "val" or "test" — used in output labels
        feat_cols:  feature column names (for feature importance)
        enc:        CryptoFPLabelEncoder instance
        verbose:    print classification report

    Returns:
        dict of all metrics (serialisable to JSON)
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    acc            = accuracy_score(y_true, y_pred)
    f1_mac         = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wei         = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_mac       = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    rec_mac        = recall_score(y_true,  y_pred, average="macro",    zero_division=0)

    # ROC-AUC: needs probability scores for all classes present in y_true
    try:
        # Only pass probability columns for classes present in y_true
        present_classes = sorted(np.unique(y_true).tolist())
        y_prob_present  = y_prob[:, present_classes]
        roc_auc = roc_auc_score(
            y_true, y_prob_present,
            multi_class=EVAL.ROC_MULTICLASS,
            average=EVAL.AVERAGE,
            labels=present_classes,
        )
    except Exception as e:
        roc_auc = -1.0
        if verbose:
            print(f"  [WARN] ROC-AUC computation failed: {e}")

    # Per-class F1
    per_class_f1 = f1_score(
        y_true, y_pred,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    per_class_dict = {
        CLASS_NAMES[i]: round(float(per_class_f1[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    # Confusion matrix
    cm = confusion_matrix(
        y_true, y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    # Feature importances (RF-specific)
    feat_imp = {}
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        feat_imp = dict(
            sorted(
                zip(feat_cols, imp.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    results = {
        "split":           split_name,
        "n_samples":       len(y_true),
        "accuracy":        round(float(acc),      4),
        "f1_macro":        round(float(f1_mac),   4),
        "f1_weighted":     round(float(f1_wei),   4),
        "precision_macro": round(float(prec_mac), 4),
        "recall_macro":    round(float(rec_mac),  4),
        "roc_auc_ovr":     round(float(roc_auc),  4),
        "per_class_f1":    per_class_dict,
        "confusion_matrix": cm.tolist(),
        "feature_importances": feat_imp,
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
        print()
        print("  Per-class F1:")
        for cls, f1 in per_class_dict.items():
            bar = "█" * int(f1 * 20)
            print(f"    {cls:<14} {f1:.4f}  {bar}")
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
            row_str = f"  {CLASS_NAMES[i][:8]:<10}" + "".join(f"{v:>10}" for v in row)
            print(row_str)

        if feat_imp:
            print("\n  Top-5 feature importances:")
            for i, (feat, imp) in enumerate(list(feat_imp.items())[:5]):
                bar = "█" * int(imp * 200)
                print(f"    {i+1}. {feat:<24} {imp:.4f}  {bar}")

    return results


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Run 5-fold stratified CV on the best model to report
    mean ± std F1. Required for IEEE papers — proves
    the result is not a lucky split.

    Returns:
        dict with cv_f1_mean and cv_f1_std
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
            n_jobs=RF.N_JOBS,
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
            f"± {result['cv_f1_std']:.3f}  "
            f"(cite this in paper Table 3)"
        )

    return result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results: dict, verbose: bool = True):
    """Save evaluation results to JSON for paper Table 3."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RF_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    size_kb = RF_RESULTS_JSON.stat().st_size / 1024
    if verbose:
        print(f"  Saved: {RF_RESULTS_JSON.relative_to(ROOT)}  ({size_kb:.1f} KB)")


# ============================================================
# MAIN
# ============================================================

def main(
    quick:     bool = False,
    eval_only: bool = False,
    verbose:   bool = True,
) -> dict:
    """
    Full RF training and evaluation pipeline.
    Called by run_pipeline.py Phase 3 or directly from CLI.

    Args:
        quick:     skip grid search, use default hyperparams
        eval_only: load existing model and re-evaluate only
        verbose:   print detailed metrics

    Returns:
        dict with model, val_results, test_results, cv_results
    """
    print("=" * 65)
    print("  CryptoFP — Random Forest Baseline")
    print("=" * 65)
    mode_str = (
        "EVAL ONLY" if eval_only
        else "QUICK (no grid search)" if quick
        else "FULL (with grid search)"
    )
    print(f"  Mode     : {mode_str}")
    print(f"  Seed     : {RANDOM_SEED}")
    print(f"  CV folds : {RF.CV_FOLDS}")
    print(f"  Min F1   : {EVAL.MIN_F1_THRESHOLD}")
    print()

    # ── 1. Load data ─────────────────────────────────────────
    print("  Step 1/5 — Loading splits")
    try:
        (X_train, y_train, X_val, y_val,
         X_test,  y_test,  feat_cols, enc) = load_splits(verbose=verbose)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return None

    # ── 2. Train or load model ────────────────────────────────
    if eval_only:
        print("\n  Step 2/5 — Loading existing model")
        if not PATHS.RF_MODEL.exists():
            print(f"  [ERROR] {PATHS.RF_MODEL} not found. Run without --eval-only.")
            return None
        model = joblib.load(PATHS.RF_MODEL)
        best_params = getattr(model, "get_params", lambda: {})()
        print(f"  Loaded: {PATHS.RF_MODEL.name}")
        print(f"  Params : {best_params}")
        grid = None
    elif quick:
        print("\n  Step 2/5 — Quick training (no grid search)")
        model = train_quick(X_train, y_train, verbose=verbose)
        best_params = model.get_params()
        grid = None
    else:
        print("\n  Step 2/5 — Grid search training")
        grid = run_grid_search(X_train, y_train, verbose=verbose)
        model = grid.best_estimator_
        best_params = grid.best_params_

    # ── 3. Cross-validation ───────────────────────────────────
    print("\n  Step 3/5 — Cross-validation")
    cv_results = run_cross_validation(model, X_train, y_train, verbose=verbose)

    # ── 4. Evaluate ───────────────────────────────────────────
    print("\n  Step 4/5 — Evaluation")
    val_results  = evaluate(model, X_val,  y_val,  "val",  feat_cols, enc, verbose)
    test_results = evaluate(model, X_test, y_test, "test", feat_cols, enc, verbose)

    # Paper-readiness check
    test_f1 = test_results["f1_macro"]
    if test_f1 < EVAL.MIN_F1_THRESHOLD:
        print(
            f"\n  [WARN] Test F1={test_f1:.4f} is below "
            f"target {EVAL.MIN_F1_THRESHOLD}.\n"
            f"  With only {len(X_train)} train rows this is expected.\n"
            f"  Target will be met with the full 1800-sample dataset."
        )
    else:
        print(f"\n  Test F1={test_f1:.4f} >= target {EVAL.MIN_F1_THRESHOLD} OK")

    # ── 5. Save ───────────────────────────────────────────────
    print("\n  Step 5/5 — Saving")
    PATHS.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PATHS.RF_MODEL)
    size_kb = PATHS.RF_MODEL.stat().st_size / 1024
    print(f"  Saved model : {PATHS.RF_MODEL.name}  ({size_kb:.1f} KB)")

    # Compile full results dict
    full_results = {
        "model":        "RandomForest",
        "mode":         mode_str,
        "best_params":  best_params,
        "n_train":      len(X_train),
        "n_val":        len(X_val),
        "n_test":       len(X_test),
        "n_features":   len(feat_cols),
        "feature_cols": feat_cols,
        "cv":           cv_results,
        "val":          val_results,
        "test":         test_results,
        "paper_cite":   (
            f"RF: F1={test_results['f1_macro']:.3f} ± {cv_results['cv_f1_std']:.3f}, "
            f"Acc={test_results['accuracy']:.3f}, "
            f"AUC={test_results['roc_auc_ovr']:.3f}"
        ),
    }
    save_results(full_results, verbose=verbose)

    # Final summary
    print()
    print("=" * 65)
    print("  RANDOM FOREST — FINAL SUMMARY")
    print("=" * 65)
    print(f"  Test accuracy   : {test_results['accuracy']:.4f}")
    print(f"  Test F1 (macro) : {test_results['f1_macro']:.4f}")
    print(f"  Test ROC-AUC    : {test_results['roc_auc_ovr']:.4f}")
    print(f"  CV F1           : {cv_results['cv_f1_mean']:.3f} ± {cv_results['cv_f1_std']:.3f}")
    print()
    print(f"  Paper cite (Table 3):")
    print(f"    {full_results['paper_cite']}")
    print()
    print(f"  Model saved to: models/rf_model.pkl")
    print(f"  Results saved to: results/rf_results.json")
    print()
    print("  Next: python 03_models/baseline_svm.py")
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
        description="CryptoFP — train and evaluate Random Forest baseline"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip grid search — use default hyperparams. Faster but not paper-ready."
    )
    parser.add_argument(
        "--eval-only", action="store_true", dest="eval_only",
        help="Load existing rf_model.pkl and re-evaluate without retraining."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed per-class metrics."
    )
    args = parser.parse_args()

    main(quick=args.quick, eval_only=args.eval_only, verbose=not args.quiet)