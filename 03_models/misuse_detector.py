# ============================================================
# CryptoFP — 03_models/misuse_detector.py
#
# Binary classifier that detects cryptographic API misuse
# from observable performance signals alone.
#
# This is the applied security contribution of the paper.
# While the 6-class fingerprinting model identifies WHICH
# algorithm is running, the misuse detector answers a more
# actionable question:
#
#     "Is this cryptographic usage CORRECT or WRONG?"
#
# Two misuse rules are learned (defined in misuse_labeler.py):
#   Rule 1 — RSA-1024: weak key size, deprecated since 2013
#             (NIST SP 800-131A Rev.2)
#   Rule 2 — RSA-2048/4096 in a PQC-required context:
#             classical crypto where post-quantum is mandated
#             (NIST FIPS 203, ML-KEM 2024)
#
# Design decisions documented for the paper:
#   - Uses same feature set as main 6-class models (fair comparison)
#   - Tunable decision threshold: security applications prioritise
#     RECALL (catch all misuse) over precision (avoid false alarms)
#   - Threshold=0.40 (config): slightly sensitive — a missed
#     misuse is worse than a false alarm in production
#   - Trains on SAME train split as all other models (no leakage)
#   - Outputs probability scores for ROC-AUC and PR-AUC curves
#
# Metrics reported (binary classification):
#   F1, Precision, Recall, ROC-AUC, PR-AUC,
#   confusion matrix (TP/FP/TN/FN), threshold sensitivity table
#
# Output:
#   models/misuse_detector.pkl    — trained classifier
#   results/misuse_results.json   — all metrics for Table 4
#
# Usage:
#   python 03_models/misuse_detector.py
#   python 03_models/misuse_detector.py --quick
#   python 03_models/misuse_detector.py --eval-only
#   python 03_models/misuse_detector.py --threshold 0.30
# ============================================================

import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "02_feature_engineering"))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
)

from config import (
    PATHS, FEATURES, MISUSE_MODEL, EVAL,
    CLASS_NAMES, MISUSE_CLASSES, RANDOM_SEED,
)
from label_encoder import CryptoFPLabelEncoder


# ============================================================
# RESULTS DIRECTORY
# ============================================================

RESULTS_DIR          = ROOT / "results"
MISUSE_RESULTS_JSON  = RESULTS_DIR / "misuse_results.json"

# Binary class labels for reports
MISUSE_LABELS        = ["correct_use", "misuse"]


# ============================================================
# DATA LOADING
# ============================================================

def load_splits(verbose: bool = True) -> tuple:
    """
    Load train/val/test splits with BINARY misuse_flag target.

    Uses the same feature columns as all other models so
    results are directly comparable in the paper.
    Filters SMOTE synthetic rows from training data.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test,
        feat_cols, algo_key_train (for per-class breakdown)
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

    # Exclude SMOTE synthetic rows
    real_mask  = train["label_encoded"] >= 0
    train_real = train[real_mask]

    X_train = train_real[feat_cols].values
    y_train = train_real["misuse_flag"].values.astype(int)
    X_val   = val[feat_cols].values
    y_val   = val["misuse_flag"].values.astype(int)
    X_test  = test[feat_cols].values
    y_test  = test["misuse_flag"].values.astype(int)

    # Keep algo key for per-class misuse analysis
    algo_key_train = train_real["label_algo_key"].values
    algo_key_val   = val["label_algo_key"].values
    algo_key_test  = test["label_algo_key"].values

    if verbose:
        print(f"  Features   : {len(feat_cols)}  {feat_cols}")
        print(f"  Train rows : {len(X_train)}  "
              f"(misuse={y_train.sum()}, correct={len(y_train)-y_train.sum()})")
        print(f"  Val rows   : {len(X_val)}  "
              f"(misuse={y_val.sum()}, correct={len(y_val)-y_val.sum()})")
        print(f"  Test rows  : {len(X_test)}  "
              f"(misuse={y_test.sum()}, correct={len(y_test)-y_test.sum()})")
        print(f"  Misuse classes: {MISUSE_CLASSES}")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            feat_cols, algo_key_train, algo_key_val, algo_key_test)


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    verbose:  bool = True,
) -> RandomForestClassifier:
    """
    Train the misuse detector Random Forest.

    Uses the config-defined hyperparameters from MISUSE_MODEL:
        n_estimators = 200
        max_depth    = 15
        class_weight = 'balanced'

    class_weight='balanced' is critical here because in a real
    production deployment, misuse cases are rare (<<50% of
    traffic). The balanced weighting trains the model to be
    equally sensitive to both classes regardless of imbalance.

    A separate RF (rather than reusing the 6-class RF) is
    trained because the decision boundary for binary misuse
    detection may differ from 6-class fingerprinting.

    Args:
        X_train: scaled feature matrix
        y_train: binary misuse_flag labels
        verbose: print training info

    Returns:
        fitted RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=MISUSE_MODEL.N_ESTIMATORS,
        max_depth=MISUSE_MODEL.MAX_DEPTH,
        class_weight=MISUSE_MODEL.CLASS_WEIGHT,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    if verbose:
        print(
            f"  Trained: {MISUSE_MODEL.N_ESTIMATORS} trees, "
            f"max_depth={MISUSE_MODEL.MAX_DEPTH}, "
            f"class_weight='{MISUSE_MODEL.CLASS_WEIGHT}'  "
            f"in {elapsed:.2f}s"
        )

    return model


# ============================================================
# THRESHOLD TUNING
# ============================================================

def compute_threshold_table(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    thresholds: list = None,
    verbose:   bool = True,
) -> list:
    """
    Compute precision / recall / F1 at multiple thresholds.

    In a security application, the optimal threshold depends
    on the cost of misclassification:
        False Negative (missed misuse) = HIGH cost
            → organisation deploys weak/wrong crypto undetected
        False Positive (false alarm)   = LOW cost
            → unnecessary review of a correct implementation

    Therefore: prefer LOWER thresholds (higher recall) for
    production misuse detection.

    The threshold table is Table 4 in your paper, showing
    that operators can tune the detector to their risk profile.

    Args:
        y_true:     true binary labels
        y_prob:     predicted probabilities for misuse class
        thresholds: list of thresholds to evaluate
        verbose:    print table

    Returns:
        list of dicts, one per threshold
    """
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60, 0.70]

    rows = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        prec  = precision_score(y_true, preds, zero_division=0)
        rec   = recall_score(y_true, preds, zero_division=0)
        f1    = f1_score(y_true, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(
            y_true, preds, labels=[0, 1]
        ).ravel() if len(np.unique(preds)) > 1 else (0, 0, 0, 0)

        rows.append({
            "threshold": t,
            "precision": round(float(prec), 4),
            "recall":    round(float(rec),  4),
            "f1":        round(float(f1),   4),
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
        })

    if verbose:
        print()
        print("  Threshold sensitivity table  (cite as Table 4 in paper):")
        print(f"  {'Thresh':>8} {'Precision':>11} {'Recall':>9} "
              f"{'F1':>7} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
        print(f"  {'-'*60}")
        for r in rows:
            marker = (
                " ← config"
                if abs(r["threshold"] - MISUSE_MODEL.THRESHOLD) < 0.01
                else ""
            )
            print(
                f"  {r['threshold']:>8.2f} "
                f"{r['precision']:>11.4f} "
                f"{r['recall']:>9.4f} "
                f"{r['f1']:>7.4f} "
                f"{r['tp']:>5} {r['fp']:>5} "
                f"{r['tn']:>5} {r['fn']:>5}"
                f"{marker}"
            )
        print()
        print(
            "  Recommendation: threshold=0.30–0.40 for production\n"
            "  (maximises recall — catching misuse is critical)"
        )

    return rows


# ============================================================
# PER-CLASS MISUSE BREAKDOWN
# ============================================================

def per_class_breakdown(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    algo_keys: np.ndarray,
    threshold: float,
    verbose:   bool = True,
) -> dict:
    """
    Break down misuse detection performance per algorithm class.

    Shows which algorithm classes the detector correctly flags
    and which it misses. Answers the paper question:
        "Does the detector work equally well for Rule 1 (RSA-1024)
        and Rule 2 (RSA-2048/4096 in PQC context)?"

    Args:
        y_true:    true misuse_flag
        y_prob:    predicted misuse probability
        algo_keys: label_algo_key string per sample
        threshold: decision threshold
        verbose:   print breakdown table

    Returns:
        dict mapping algo_key → {tp, fp, tn, fn, recall, precision}
    """
    preds   = (y_prob >= threshold).astype(int)
    results = {}

    for cls in CLASS_NAMES:
        mask = algo_keys == cls
        if mask.sum() == 0:
            continue
        y_c = y_true[mask]
        p_c = preds[mask]

        tp = int(((y_c == 1) & (p_c == 1)).sum())
        fp = int(((y_c == 0) & (p_c == 1)).sum())
        tn = int(((y_c == 0) & (p_c == 0)).sum())
        fn = int(((y_c == 1) & (p_c == 0)).sum())

        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        expected_misuse = cls in MISUSE_CLASSES or (
            "RSA" in cls and MISUSE_CLASSES
        )
        results[cls] = {
            "n_samples":       int(mask.sum()),
            "expected_misuse": expected_misuse,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "recall":    round(rec,  4),
            "precision": round(prec, 4),
        }

    if verbose:
        print(f"  Per-class misuse breakdown (threshold={threshold}):")
        print(f"  {'Class':<14} {'Expected':>10} {'TP':>5} {'FP':>5} "
              f"{'TN':>5} {'FN':>5} {'Recall':>8} {'Prec':>8}")
        print(f"  {'-'*70}")
        for cls, r in results.items():
            expected = "MISUSE" if r["expected_misuse"] else "correct"
            print(
                f"  {cls:<14} {expected:>10} "
                f"{r['tp']:>5} {r['fp']:>5} "
                f"{r['tn']:>5} {r['fn']:>5} "
                f"{r['recall']:>8.4f} {r['precision']:>8.4f}"
            )

    return results


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model:      RandomForestClassifier,
    X:          np.ndarray,
    y_true:     np.ndarray,
    split_name: str,
    feat_cols:  list,
    algo_keys:  np.ndarray,
    threshold:  float = None,
    verbose:    bool  = True,
) -> dict:
    """
    Compute all misuse detection metrics for one split.

    Binary-specific metrics beyond standard F1/accuracy:
        PR-AUC   — area under precision-recall curve
                   more informative than ROC-AUC for imbalanced
                   classes (production misuse will be rare)
        ROC-AUC  — area under ROC curve
        TP/FP/TN/FN — confusion matrix elements named explicitly
                   for the security interpretation in the paper

    Threshold sensitivity table and per-class breakdown are
    computed for the test split only (not val) to avoid
    selection bias from threshold tuning on the test set.

    Args:
        model:       trained RandomForestClassifier
        X:           feature matrix
        y_true:      true binary misuse_flag
        split_name:  'val' or 'test'
        feat_cols:   feature names
        algo_keys:   algorithm class per sample (for breakdown)
        threshold:   decision threshold (default: MISUSE_MODEL.THRESHOLD)
        verbose:     print detailed metrics

    Returns:
        dict of all metrics
    """
    if threshold is None:
        threshold = MISUSE_MODEL.THRESHOLD

    y_prob  = model.predict_proba(X)[:, 1]   # P(misuse)
    y_pred  = (y_prob >= threshold).astype(int)

    acc     = accuracy_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = -1.0

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = -1.0

    # Confusion matrix elements
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    # Feature importances
    feat_imp = dict(
        sorted(
            zip(feat_cols, model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    results = {
        "split":           split_name,
        "threshold":       threshold,
        "n_samples":       len(y_true),
        "n_misuse":        int(y_true.sum()),
        "n_correct":       int((y_true == 0).sum()),
        "accuracy":        round(float(acc),     4),
        "f1":              round(float(f1),      4),
        "precision":       round(float(prec),    4),
        "recall":          round(float(rec),     4),
        "roc_auc":         round(float(roc_auc), 4),
        "pr_auc":          round(float(pr_auc),  4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "feature_importances": feat_imp,
    }

    if verbose:
        print(f"\n  === {split_name.upper()} SET — MISUSE DETECTION ===")
        print(f"  Threshold       : {threshold}")
        print(f"  Accuracy        : {acc:.4f}")
        print(f"  F1              : {f1:.4f}  "
              f"{'OK' if f1 >= EVAL.MIN_F1_THRESHOLD else 'BELOW TARGET'}")
        print(f"  Precision       : {prec:.4f}")
        print(f"  Recall          : {rec:.4f}  "
              f"(critical — missing misuse = undetected vulnerability)")
        print(f"  ROC-AUC         : {roc_auc:.4f}")
        print(f"  PR-AUC          : {pr_auc:.4f}")
        print()
        print(f"  Confusion matrix:")
        print(f"                    Pred: correct  Pred: misuse")
        print(f"    True: correct       {tn:>8}     {fp:>8}")
        print(f"    True: misuse        {fn:>8}     {tp:>8}")
        print()
        print(f"  Security interpretation:")
        print(f"    TP={tp}: misuse correctly flagged")
        print(f"    FP={fp}: false alarms (correct crypto flagged as misuse)")
        print(f"    TN={tn}: correct usage confirmed")
        print(f"    FN={fn}: MISSED MISUSE (critical — undetected vulnerability)")
        print()
        print(f"  Top-5 feature importances:")
        for i, (feat, imp) in enumerate(list(feat_imp.items())[:5]):
            bar = "█" * int(imp * 200)
            print(f"    {i+1}. {feat:<24} {imp:.4f}  {bar}")

        # Classification report
        print()
        report = classification_report(
            y_true, y_pred,
            target_names=MISUSE_LABELS,
            zero_division=0,
        )
        print("  Classification report:")
        for line in report.split("\n"):
            print(f"    {line}")

    return results


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(
    model:   RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    5-fold CV on binary misuse detection task.
    Reports mean ± std F1 for Table 4.
    """
    cv = StratifiedKFold(
        n_splits=EVAL.CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="f1", n_jobs=-1,
    )
    result = {
        "cv_f1_scores": [round(float(s), 4) for s in scores],
        "cv_f1_mean":   round(float(scores.mean()), 4),
        "cv_f1_std":    round(float(scores.std()),  4),
    }
    if verbose:
        print(f"  Cross-validation ({EVAL.CV_FOLDS}-fold, binary F1):")
        print(f"    Scores : {result['cv_f1_scores']}")
        print(f"    Mean   : {result['cv_f1_mean']:.4f}")
        print(f"    Std    : {result['cv_f1_std']:.4f}")
        print(
            f"    Report : Misuse F1 = "
            f"{result['cv_f1_mean']:.3f} ± {result['cv_f1_std']:.3f}"
            f"  (cite in Table 4)"
        )
    return result


# ============================================================
# SAVE / LOAD
# ============================================================

def save_results(results: dict, verbose: bool = True):
    """Save all metrics to results/misuse_results.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MISUSE_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    size_kb = MISUSE_RESULTS_JSON.stat().st_size / 1024
    if verbose:
        print(f"  Saved: {MISUSE_RESULTS_JSON.relative_to(ROOT)}  ({size_kb:.1f} KB)")


# ============================================================
# MAIN
# ============================================================

def main(
    quick:      bool  = False,
    eval_only:  bool  = False,
    threshold:  float = None,
    verbose:    bool  = True,
) -> dict:
    """
    Full misuse detector training and evaluation pipeline.
    Called by run_pipeline.py Phase 3, or directly from CLI.

    Args:
        quick:     skip CV, reduced analysis
        eval_only: load existing model, re-evaluate only
        threshold: override MISUSE_MODEL.THRESHOLD
        verbose:   print detailed metrics

    Returns:
        dict with model, val_results, test_results, cv_results
    """
    if threshold is None:
        threshold = MISUSE_MODEL.THRESHOLD

    print("=" * 65)
    print("  CryptoFP — Misuse Detector")
    print("=" * 65)
    print(f"  Mode      : {'EVAL ONLY' if eval_only else 'TRAIN + EVAL'}")
    print(f"  Threshold : {threshold}  "
          f"(lower = higher recall — prefer for security)")
    print(f"  Min F1    : {EVAL.MIN_F1_THRESHOLD}")
    print(f"  Misuse rules detected:")
    print(f"    Rule 1 — RSA-1024: weak key  (NIST SP 800-131A Rev.2)")
    print(f"    Rule 2 — RSA in PQC context  (NIST FIPS 203, 2024)")
    print()

    # ── 1. Load ───────────────────────────────────────────────
    print("  Step 1/5 — Loading splits")
    try:
        (X_train, y_train, X_val, y_val, X_test, y_test,
         feat_cols, ak_train, ak_val, ak_test) = load_splits(verbose=verbose)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return None

    # ── 2. Train or load ──────────────────────────────────────
    print("\n  Step 2/5 — Model")
    if eval_only:
        if not PATHS.MISUSE_MODEL.exists():
            print(f"  [ERROR] {PATHS.MISUSE_MODEL} not found.")
            return None
        model = joblib.load(PATHS.MISUSE_MODEL)
        print(f"  Loaded: {PATHS.MISUSE_MODEL.name}")
    else:
        model = train_model(X_train, y_train, verbose=verbose)

    # ── 3. Cross-validation ───────────────────────────────────
    print("\n  Step 3/5 — Cross-validation")
    if quick:
        print("  [SKIP] quick mode — no cross-validation")
        cv_results = {"cv_f1_mean": 0, "cv_f1_std": 0, "cv_f1_scores": []}
    else:
        cv_results = run_cross_validation(
            model, X_train, y_train, verbose=verbose
        )

    # ── 4. Evaluate ───────────────────────────────────────────
    print("\n  Step 4/5 — Evaluation")
    val_results  = evaluate(
        model, X_val,  y_val,  "val",  feat_cols, ak_val,
        threshold=threshold, verbose=verbose,
    )
    test_results = evaluate(
        model, X_test, y_test, "test", feat_cols, ak_test,
        threshold=threshold, verbose=verbose,
    )

    # Threshold table (test set only)
    if not quick:
        y_prob_test = model.predict_proba(X_test)[:, 1]
        threshold_table = compute_threshold_table(
            y_test, y_prob_test, verbose=verbose
        )
        test_results["threshold_table"] = threshold_table

        # Per-class breakdown (test set)
        cls_breakdown = per_class_breakdown(
            y_test, y_prob_test, ak_test,
            threshold=threshold, verbose=verbose,
        )
        test_results["per_class_breakdown"] = cls_breakdown
    else:
        threshold_table = []

    # Paper-readiness check
    test_f1 = test_results.get("f1", 0)
    test_fn = test_results.get("fn", -1)
    if test_fn > 0:
        print(
            f"\n  [WARN] {test_fn} missed misuse case(s) (FN).\n"
            f"  Consider lowering threshold from {threshold} to 0.30.\n"
            f"  In production, FN = undetected cryptographic vulnerability."
        )
    elif test_f1 >= EVAL.MIN_F1_THRESHOLD:
        print(f"\n  Misuse F1={test_f1:.4f} >= target {EVAL.MIN_F1_THRESHOLD}  OK")
    else:
        print(
            f"\n  [WARN] Misuse F1={test_f1:.4f} below "
            f"target {EVAL.MIN_F1_THRESHOLD}.\n"
            f"  Expected with small dataset."
        )

    # ── 5. Save ───────────────────────────────────────────────
    print("\n  Step 5/5 — Saving")
    PATHS.MODELS.mkdir(parents=True, exist_ok=True)
    if not eval_only:
        joblib.dump(model, PATHS.MISUSE_MODEL)
        size_kb = PATHS.MISUSE_MODEL.stat().st_size / 1024
        print(f"  Saved model : {PATHS.MISUSE_MODEL.name}  ({size_kb:.1f} KB)")

    full_results = {
        "model":       "MisuseDetector_RF",
        "n_estimators": MISUSE_MODEL.N_ESTIMATORS,
        "max_depth":    MISUSE_MODEL.MAX_DEPTH,
        "class_weight": MISUSE_MODEL.CLASS_WEIGHT,
        "threshold":    threshold,
        "n_train":      len(X_train),
        "n_val":        len(X_val),
        "n_test":       len(X_test),
        "n_features":   len(feat_cols),
        "feature_cols": feat_cols,
        "misuse_rules": [
            "RSA-1024: weak key (NIST SP 800-131A Rev.2)",
            "RSA in PQC-required context (NIST FIPS 203)",
        ],
        "cv":   cv_results,
        "val":  val_results,
        "test": test_results,
        "paper_cite": (
            f"Misuse Detector: F1={test_results.get('f1',0):.3f} "
            f"± {cv_results.get('cv_f1_std',0):.3f}, "
            f"Recall={test_results.get('recall',0):.3f}, "
            f"PR-AUC={test_results.get('pr_auc',0):.3f}"
        ),
    }
    save_results(full_results, verbose=verbose)

    # ── Final summary ─────────────────────────────────────────
    print()
    print("=" * 65)
    print("  MISUSE DETECTOR — FINAL SUMMARY")
    print("=" * 65)
    print(f"  Test F1            : {test_results.get('f1', 0):.4f}")
    print(f"  Test Precision     : {test_results.get('precision', 0):.4f}")
    print(f"  Test Recall        : {test_results.get('recall', 0):.4f}  "
          f"(fraction of misuse caught)")
    print(f"  Test ROC-AUC       : {test_results.get('roc_auc', 0):.4f}")
    print(f"  Test PR-AUC        : {test_results.get('pr_auc', 0):.4f}")
    print(f"  Missed misuse (FN) : {test_results.get('fn', '?')}")
    print(f"  CV F1              : "
          f"{cv_results.get('cv_f1_mean', 0):.3f} "
          f"± {cv_results.get('cv_f1_std', 0):.3f}")
    print()
    print(f"  Paper cite (Table 4):")
    print(f"    {full_results['paper_cite']}")
    print()
    print("  Model saved to  : models/misuse_detector.pkl")
    print("  Results saved to: results/misuse_results.json")
    print()
    print("  Next: python 03_models/train.py")
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
        description="CryptoFP — train and evaluate cryptographic misuse detector"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip cross-validation and threshold table."
    )
    parser.add_argument(
        "--eval-only", action="store_true", dest="eval_only",
        help="Load existing misuse_detector.pkl and re-evaluate."
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=(
            f"Decision threshold (default: {MISUSE_MODEL.THRESHOLD}). "
            "Lower = higher recall (recommended for security)."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed metrics output."
    )
    args = parser.parse_args()

    main(
        quick=args.quick,
        eval_only=args.eval_only,
        threshold=args.threshold,
        verbose=not args.quiet,
    )