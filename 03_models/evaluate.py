# ============================================================
# CryptoFP — 03_models/evaluate.py
#
# Unified evaluation script. Loads every trained model,
# re-evaluates on the held-out test set, and produces the
# complete set of paper tables and figures data.
#
# What this script produces:
#
#   Console output:
#     Table 3 — Model comparison: RF / SVM / CNN / LSTM
#     Table 4 — Misuse detection results
#     Per-class F1 breakdown across all models
#     McNemar significance test between best two models
#
#   Saved files:
#     results/evaluation_report.json  — full metrics (all models)
#     results/comparison_table.csv    — Table 3 as CSV
#     results/per_class_f1.csv        — per-class F1 matrix
#     results/mcnemar_test.json       — statistical significance
#
# Why a separate evaluate.py:
#   Each model script (baseline_rf.py etc.) evaluates only
#   its own model. This script loads ALL models simultaneously
#   so you can:
#     - Compare on the identical test set (same X_test, y_test)
#     - Produce a single comparison table in one pass
#     - Run statistical significance tests between models
#     - Generate the CSV tables that go directly into LaTeX
#
# Usage:
#   python 03_models/evaluate.py
#   python 03_models/evaluate.py --models rf svm
#   python 03_models/evaluate.py --split val
#   python 03_models/evaluate.py --table-only
# ============================================================

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "02_feature_engineering"))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

from config import (
    PATHS, FEATURES, EVAL,
    CLASS_NAMES, MISUSE_CLASSES, RANDOM_SEED,
)
from label_encoder import CryptoFPLabelEncoder

# ── optional PyTorch for deep models ─────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR          = ROOT / "results"
EVAL_REPORT_JSON     = RESULTS_DIR / "evaluation_report.json"
COMPARISON_TABLE_CSV = RESULTS_DIR / "comparison_table.csv"
PER_CLASS_F1_CSV     = RESULTS_DIR / "per_class_f1.csv"
MCNEMAR_JSON         = RESULTS_DIR / "mcnemar_test.json"

# Model registry — name → (model_path, model_type)
MODEL_REGISTRY = {
    "rf":    (PATHS.RF_MODEL,    "sklearn"),
    "svm":   (PATHS.SVM_MODEL,   "sklearn"),
    "cnn":   (PATHS.CNN_MODEL,   "pytorch"),
    "lstm":  (PATHS.LSTM_MODEL,  "pytorch"),
}


# ============================================================
# DATA LOADING
# ============================================================

def load_test_data(split: str = "test", verbose: bool = True) -> tuple:
    """
    Load the evaluation split and return feature/label arrays.

    Args:
        split:   'test' (paper results) or 'val' (development)
        verbose: print data summary

    Returns:
        X, y_algo, y_misuse, feat_cols, algo_keys, enc
    """
    path_map = {
        "test": PATHS.TEST_CSV,
        "val":  PATHS.VAL_CSV,
        "train": PATHS.TRAIN_CSV,
    }
    csv_path = path_map.get(split, PATHS.TEST_CSV)
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run Phase 2 preprocessing first."
        )

    df = pd.read_csv(csv_path)

    # Feature columns
    base_feat = FEATURES.ALL_FEATURES + ["log_keygen_ms", "log_dec_ms"]
    feat_cols = [c for c in base_feat if c in df.columns]

    # Filter synthetic rows if evaluating train split
    if split == "train":
        df = df[df["label_encoded"] >= 0]

    X         = df[feat_cols].values
    y_algo    = df["label_encoded"].values.astype(int)
    y_misuse  = df["misuse_flag"].values.astype(int)
    algo_keys = df["label_algo_key"].values

    enc = CryptoFPLabelEncoder.load()

    if verbose:
        print(f"  Split      : {split}")
        print(f"  Rows       : {len(df)}")
        print(f"  Features   : {len(feat_cols)}")
        print(f"  Classes    : {CLASS_NAMES}")
        print(
            f"  Label dist : "
            + "  ".join(
                f"{CLASS_NAMES[i]}={int((y_algo==i).sum())}"
                for i in range(len(CLASS_NAMES))
            )
        )

    return X, y_algo, y_misuse, feat_cols, algo_keys, enc


# ============================================================
# MODEL LOADING
# ============================================================

def load_sklearn_model(path: Path, name: str):
    """Load a joblib-serialised sklearn model."""
    if not path.exists():
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        print(f"  [WARN] Could not load {name} from {path.name}: {e}")
        return None


def load_pytorch_model(path: Path, name: str, in_length: int):
    """
    Load a PyTorch model (.pt weights file).
    Returns None if PyTorch unavailable or file missing.
    """
    if not TORCH_AVAILABLE:
        return None
    if not path.exists():
        return None
    try:
        sys.path.insert(0, str(ROOT / "03_models"))
        if name == "cnn":
            from cnn_1d import CryptoFP_CNN1D
            model = CryptoFP_CNN1D(in_length=in_length)
        elif name == "lstm":
            from lstm_attention import CryptoFP_LSTM_Attention
            model = CryptoFP_LSTM_Attention(
                input_size=1, seq_len=in_length
            )
        else:
            return None
        model.load_state_dict(
            torch.load(path, map_location="cpu")
        )
        model.eval()
        return model
    except Exception as e:
        print(f"  [WARN] Could not load {name}: {e}")
        return None


def load_all_models(feat_cols: list, verbose: bool = True) -> dict:
    """
    Load all available trained models.
    Gracefully skips models whose files do not exist yet
    (e.g. CNN/LSTM if PyTorch is not installed).

    Returns:
        dict mapping model name → loaded model (or None if unavailable)
    """
    in_length = len(feat_cols)
    models    = {}

    for name, (path, mtype) in MODEL_REGISTRY.items():
        path = Path(path)
        if mtype == "sklearn":
            m = load_sklearn_model(path, name)
        else:
            m = load_pytorch_model(path, name, in_length)

        status = "loaded" if m is not None else "NOT FOUND / unavailable"
        if verbose:
            size = f"  ({path.stat().st_size/1024:.1f} KB)" if path.exists() else ""
            print(f"  {name.upper():<8} {status}{size}")
        models[name] = m

    # Misuse detector is separate
    m_mis = load_sklearn_model(Path(PATHS.MISUSE_MODEL), "misuse")
    if verbose:
        size = f"  ({Path(PATHS.MISUSE_MODEL).stat().st_size/1024:.1f} KB)" \
               if Path(PATHS.MISUSE_MODEL).exists() else ""
        status = "loaded" if m_mis is not None else "NOT FOUND"
        print(f"  {'MISUSE':<8} {status}{size}")
    models["misuse"] = m_mis

    available = [k for k, v in models.items() if v is not None]
    if verbose:
        print(f"\n  Available: {available}")

    return models


# ============================================================
# PREDICTION HELPERS
# ============================================================

def predict_sklearn(model, X: np.ndarray) -> tuple:
    """Return (predictions, probabilities) for sklearn model."""
    preds = model.predict(X)
    probs = model.predict_proba(X)
    return preds, probs


def predict_pytorch(
    model,
    X:        np.ndarray,
    is_lstm:  bool = False,
) -> tuple:
    """Return (predictions, probabilities) for PyTorch model."""
    if not TORCH_AVAILABLE:
        return None, None
    model.eval()
    # CNN input: (B, 1, L); LSTM input: (B, L, 1)
    if is_lstm:
        X_t = torch.from_numpy(X.astype(np.float32))[:, :, np.newaxis]
    else:
        X_t = torch.from_numpy(X.astype(np.float32))[:, np.newaxis, :]
    with torch.no_grad():
        if is_lstm:
            logits, _ = model(X_t)
        else:
            logits = model(X_t)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy()
    return preds, probs


# ============================================================
# CORE METRICS
# ============================================================

def compute_metrics(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    y_prob:     np.ndarray,
    model_name: str,
    task:       str = "multiclass",
) -> dict:
    """
    Compute the full metric set for one model on one split.

    For multiclass (6-class fingerprinting):
        accuracy, f1_macro, f1_weighted, precision_macro,
        recall_macro, roc_auc_ovr, per_class_f1

    For binary (misuse detection):
        accuracy, f1, precision, recall, roc_auc, pr_auc,
        tp, fp, tn, fn

    Args:
        y_true:     ground-truth integer labels
        y_pred:     predicted integer labels
        y_prob:     predicted probabilities (n_samples, n_classes)
        model_name: label for the model
        task:       'multiclass' or 'binary'

    Returns:
        dict of all metrics
    """
    metrics = {"model": model_name, "task": task}

    if task == "multiclass":
        metrics["accuracy"]        = round(float(accuracy_score(y_true, y_pred)), 4)
        metrics["f1_macro"]        = round(float(f1_score(y_true, y_pred,
                                       average="macro", zero_division=0)), 4)
        metrics["f1_weighted"]     = round(float(f1_score(y_true, y_pred,
                                       average="weighted", zero_division=0)), 4)
        metrics["precision_macro"] = round(float(precision_score(y_true, y_pred,
                                       average="macro", zero_division=0)), 4)
        metrics["recall_macro"]    = round(float(recall_score(y_true, y_pred,
                                       average="macro", zero_division=0)), 4)

        # ROC-AUC — only over classes present in y_true
        try:
            present = sorted(np.unique(y_true).tolist())
            metrics["roc_auc_ovr"] = round(float(roc_auc_score(
                y_true, y_prob[:, present],
                multi_class=EVAL.ROC_MULTICLASS,
                average=EVAL.AVERAGE,
                labels=present,
            )), 4)
        except Exception:
            metrics["roc_auc_ovr"] = -1.0

        # Per-class F1
        per_class = f1_score(
            y_true, y_pred,
            average=None,
            labels=list(range(len(CLASS_NAMES))),
            zero_division=0,
        )
        metrics["per_class_f1"] = {
            CLASS_NAMES[i]: round(float(per_class[i]), 4)
            for i in range(len(CLASS_NAMES))
        }

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(
            y_true, y_pred,
            labels=list(range(len(CLASS_NAMES))),
        ).tolist()

    elif task == "binary":
        metrics["accuracy"]  = round(float(accuracy_score(y_true, y_pred)), 4)
        metrics["f1"]        = round(float(f1_score(y_true, y_pred,
                                   zero_division=0)), 4)
        metrics["precision"] = round(float(precision_score(y_true, y_pred,
                                   zero_division=0)), 4)
        metrics["recall"]    = round(float(recall_score(y_true, y_pred,
                                   zero_division=0)), 4)

        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true,
                                   y_prob[:, 1])), 4)
        except Exception:
            metrics["roc_auc"] = -1.0

        try:
            metrics["pr_auc"] = round(float(average_precision_score(y_true,
                                  y_prob[:, 1])), 4)
        except Exception:
            metrics["pr_auc"] = -1.0

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                "tp": int(tp), "fp": int(fp),
                "tn": int(tn), "fn": int(fn),
            })

    return metrics


# ============================================================
# McNEMAR SIGNIFICANCE TEST
# ============================================================

def mcnemar_test(
    y_true:    np.ndarray,
    preds_a:   np.ndarray,
    preds_b:   np.ndarray,
    name_a:    str,
    name_b:    str,
    verbose:   bool = True,
) -> dict:
    """
    McNemar test for statistical significance between two classifiers.

    Tests H0: the two classifiers make errors on the same samples.
    A significant result (p < 0.05) means the difference in
    performance is not due to chance — required by IEEE reviewers
    when comparing models.

    Contingency table:
        b = model_a correct, model_b wrong
        c = model_a wrong,   model_b correct

    Test statistic: χ² = (|b - c| - 1)² / (b + c)
    Uses continuity correction (Yates) as recommended for
    small sample sizes (< 25 discordant pairs).

    Args:
        y_true:  ground truth labels
        preds_a: predictions from model A
        preds_b: predictions from model B
        name_a:  model A label
        name_b:  model B label
        verbose: print results

    Returns:
        dict with b, c, chi2, p_value, significant
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    b = int(( correct_a & ~correct_b).sum())   # A right, B wrong
    c = int((~correct_a &  correct_b).sum())   # A wrong, B right

    result = {
        "model_a": name_a,
        "model_b": name_b,
        "n_samples": len(y_true),
        "b_a_right_b_wrong": b,
        "c_a_wrong_b_right": c,
        "discordant_pairs": b + c,
    }

    if b + c == 0:
        result.update({
            "chi2":        0.0,
            "p_value":     1.0,
            "significant": False,
            "note": "Models make identical errors — McNemar not applicable.",
        })
    else:
        # Yates continuity correction
        stat  = float((abs(b - c) - 1) ** 2 / (b + c)) if b + c >= 1 else 0.0
        p_val = float(1 - chi2.cdf(stat, df=1))
        result.update({
            "chi2":        round(stat,  4),
            "p_value":     round(p_val, 4),
            "significant": p_val < 0.05,
            "note": (
                f"p={p_val:.4f} — "
                + ("significant difference (p<0.05)"
                   if p_val < 0.05
                   else "no significant difference (p≥0.05)")
            ),
        })

    if verbose:
        print(f"  McNemar test: {name_a} vs {name_b}")
        print(f"    b (A✓ B✗): {b}   c (A✗ B✓): {c}")
        print(f"    χ²={result.get('chi2', 0):.4f}  "
              f"p={result.get('p_value', 1):.4f}  "
              f"{'SIGNIFICANT' if result.get('significant') else 'not significant'}")
        print(f"    {result.get('note','')}")

    return result


# ============================================================
# COMPARISON TABLE
# ============================================================

def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """
    Build the paper Table 3 DataFrame from all model results.
    Includes: Accuracy, F1 (macro), Precision, Recall, ROC-AUC.
    Marks the best value in each column with *.

    Returns:
        pd.DataFrame with models as rows, metrics as columns
    """
    rows   = []
    mc_cols = [
        "accuracy", "f1_macro", "precision_macro",
        "recall_macro", "roc_auc_ovr",
    ]
    display_names = {
        "rf":   "Random Forest",
        "svm":  "SVM (RBF)",
        "cnn":  "CNN-1D",
        "lstm": "LSTM+Attention",
    }

    for name in ["rf", "svm", "cnn", "lstm"]:
        r = all_results.get(name)
        if r is None:
            continue
        row = {"Model": display_names.get(name, name)}
        for col in mc_cols:
            row[col] = r.get(col, float("nan"))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Model")

    # Mark best in each column
    for col in df.columns:
        best = df[col].max()
        df[col + "_best"] = df[col] == best

    return df


def print_comparison_table(df: pd.DataFrame, verbose: bool = True):
    """Print Table 3 to console in paper-ready format."""
    if df.empty:
        print("  No models available for comparison.")
        return

    metric_labels = {
        "accuracy":        "Accuracy",
        "f1_macro":        "F1 (macro)",
        "precision_macro": "Precision",
        "recall_macro":    "Recall",
        "roc_auc_ovr":     "ROC-AUC",
    }
    base_cols = list(metric_labels.keys())
    base_cols = [c for c in base_cols if c in df.columns]

    print()
    print("=" * 70)
    print("  Table 3 — Model Comparison (test set)")
    print("=" * 70)
    header = f"  {'Model':<20}" + "".join(
        f"{metric_labels[c]:>12}" for c in base_cols
    )
    print(header)
    print(f"  {'-'*65}")

    for model_name, row in df.iterrows():
        line = f"  {model_name:<20}"
        for col in base_cols:
            val  = row[col]
            best = row.get(col + "_best", False)
            if np.isnan(val):
                line += f"{'N/A':>12}"
            else:
                marker = "*" if best else " "
                line += f"{val:>11.4f}{marker}"
        print(line)

    print(f"  {'-'*65}")
    print("  (* = best in column)")
    print()

    # F1 target check
    for model_name, row in df.iterrows():
        f1 = row.get("f1_macro", 0)
        status = "OK" if f1 >= EVAL.MIN_F1_THRESHOLD else "BELOW TARGET"
        print(f"  {model_name:<20} F1={f1:.4f}  [{status}]")
    print()
    print("  Note: With full 1800-sample dataset all models")
    print("  are expected to exceed F1 target of "
          f"{EVAL.MIN_F1_THRESHOLD}.")
    print("=" * 70)


def print_per_class_table(all_results: dict):
    """Print per-class F1 matrix across all models."""
    print()
    print("=" * 75)
    print("  Per-class F1 breakdown (test set)")
    print("=" * 75)

    model_order = ["rf", "svm", "cnn", "lstm"]
    avail = [n for n in model_order if n in all_results and
             "per_class_f1" in (all_results.get(n) or {})]

    if not avail:
        print("  No multiclass results available.")
        return

    header = f"  {'Class':<14}" + "".join(f"{n.upper():>12}" for n in avail)
    print(header)
    print(f"  {'-'*60}")

    for cls in CLASS_NAMES:
        row = f"  {cls:<14}"
        best_val = max(
            all_results[n]["per_class_f1"].get(cls, 0)
            for n in avail
        )
        for n in avail:
            val = all_results[n]["per_class_f1"].get(cls, float("nan"))
            if np.isnan(val):
                row += f"{'N/A':>12}"
            else:
                marker = "*" if abs(val - best_val) < 1e-4 else " "
                row += f"{val:>11.4f}{marker}"
        print(row)

    print("  (* = best for that class)")
    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

def main(
    models_to_eval: list = None,
    split:          str  = "test",
    table_only:     bool = False,
    verbose:        bool = True,
) -> dict:
    """
    Load all models, evaluate on test set, print comparison tables.
    Called by run_pipeline.py Phase 4, or directly from CLI.

    Args:
        models_to_eval: list of model names to evaluate
                        (default: all available)
        split:          data split to evaluate on ('test' or 'val')
        table_only:     print tables from existing JSON, no recompute
        verbose:        print detailed metrics per model

    Returns:
        dict mapping model name → metrics dict
        (also returned to run_pipeline.py for threshold checking)
    """
    print("=" * 65)
    print("  CryptoFP — Unified Evaluation")
    print("=" * 65)
    print(f"  Split    : {split}")
    print(f"  Min F1   : {EVAL.MIN_F1_THRESHOLD}")
    print()

    # ── table-only mode ───────────────────────────────────────
    if table_only:
        if not EVAL_REPORT_JSON.exists():
            print("  evaluation_report.json not found. "
                  "Run without --table-only first.")
            return {}
        with open(EVAL_REPORT_JSON) as f:
            report = json.load(f)
        all_results = report.get("models", {})
        df_cmp = build_comparison_table(all_results)
        print_comparison_table(df_cmp, verbose=verbose)
        print_per_class_table(all_results)
        return all_results

    # ── 1. Load data ─────────────────────────────────────────
    print("  Step 1/5 — Loading test data")
    try:
        X, y_algo, y_misuse, feat_cols, algo_keys, enc = load_test_data(
            split=split, verbose=verbose
        )
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return {}

    # ── 2. Load models ────────────────────────────────────────
    print("\n  Step 2/5 — Loading models")
    all_models = load_all_models(feat_cols, verbose=verbose)

    if models_to_eval:
        all_models = {k: v for k, v in all_models.items()
                      if k in models_to_eval or k == "misuse"}

    # ── 3. Compute metrics ────────────────────────────────────
    print("\n  Step 3/5 — Computing metrics")
    all_results  = {}
    all_preds    = {}

    for name in ["rf", "svm", "cnn", "lstm"]:
        model = all_models.get(name)
        if model is None:
            print(f"  [SKIP] {name.upper()} — not available")
            continue

        print(f"\n  --- {name.upper()} ---")
        try:
            if name in ("rf", "svm"):
                preds, probs = predict_sklearn(model, X)
            elif name == "cnn":
                preds, probs = predict_pytorch(model, X, is_lstm=False)
            elif name == "lstm":
                preds, probs = predict_pytorch(model, X, is_lstm=True)

            if preds is None:
                print(f"  [SKIP] {name.upper()} — prediction failed")
                continue

            metrics = compute_metrics(
                y_algo, preds, probs, name, task="multiclass"
            )
            all_results[name] = metrics
            all_preds[name]   = preds

            if verbose:
                print(
                    f"  Accuracy: {metrics['accuracy']:.4f}  "
                    f"F1: {metrics['f1_macro']:.4f}  "
                    f"AUC: {metrics['roc_auc_ovr']:.4f}"
                )
                print(f"  Per-class F1: "
                      + "  ".join(
                          f"{k}={v:.3f}"
                          for k, v in metrics["per_class_f1"].items()
                      ))

        except Exception as e:
            print(f"  [ERROR] {name.upper()}: {e}")

    # ── Misuse detector ───────────────────────────────────────
    mis_model = all_models.get("misuse")
    mis_results = {}
    mis_preds   = None
    if mis_model is not None:
        print(f"\n  --- MISUSE DETECTOR ---")
        try:
            mis_pred, mis_prob = predict_sklearn(mis_model, X)
            mis_results = compute_metrics(
                y_misuse, mis_pred, mis_prob, "misuse", task="binary"
            )
            mis_preds = mis_pred
            if verbose:
                print(
                    f"  F1: {mis_results.get('f1',0):.4f}  "
                    f"Recall: {mis_results.get('recall',0):.4f}  "
                    f"PR-AUC: {mis_results.get('pr_auc',0):.4f}  "
                    f"FN: {mis_results.get('fn',0)}"
                )
        except Exception as e:
            print(f"  [ERROR] misuse detector: {e}")

    # ── 4. Statistical significance ───────────────────────────
    print("\n  Step 4/5 — Statistical significance (McNemar test)")
    mcnemar_results = []

    avail_pairs = [
        (a, b) for a in list(all_preds.keys())
        for b in list(all_preds.keys())
        if a < b
    ]

    for name_a, name_b in avail_pairs:
        result = mcnemar_test(
            y_algo,
            all_preds[name_a],
            all_preds[name_b],
            name_a.upper(),
            name_b.upper(),
            verbose=verbose,
        )
        mcnemar_results.append(result)

    if not avail_pairs:
        print("  Need at least 2 models for significance testing.")

    # ── 5. Save outputs ───────────────────────────────────────
    print("\n  Step 5/5 — Saving outputs")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Full evaluation report JSON
    report = {
        "split":    split,
        "n_test":   len(X),
        "models":   all_results,
        "misuse":   mis_results,
        "mcnemar":  mcnemar_results,
    }
    with open(EVAL_REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    size_kb = EVAL_REPORT_JSON.stat().st_size / 1024
    print(f"  Saved: {EVAL_REPORT_JSON.relative_to(ROOT)}  ({size_kb:.1f} KB)")

    # Comparison table CSV (Table 3)
    df_cmp = build_comparison_table(all_results)
    if not df_cmp.empty:
        base_cols = ["accuracy", "f1_macro", "precision_macro",
                     "recall_macro", "roc_auc_ovr"]
        csv_cols = [c for c in base_cols if c in df_cmp.columns]
        df_cmp[csv_cols].to_csv(COMPARISON_TABLE_CSV)
        size_kb = COMPARISON_TABLE_CSV.stat().st_size / 1024
        print(f"  Saved: {COMPARISON_TABLE_CSV.relative_to(ROOT)}  ({size_kb:.1f} KB)")

    # Per-class F1 CSV
    pcf1_rows = {}
    for name, r in all_results.items():
        if "per_class_f1" in r:
            pcf1_rows[name] = r["per_class_f1"]
    if pcf1_rows:
        df_pcf1 = pd.DataFrame(pcf1_rows).T
        df_pcf1.index.name = "model"
        df_pcf1.to_csv(PER_CLASS_F1_CSV)
        size_kb = PER_CLASS_F1_CSV.stat().st_size / 1024
        print(f"  Saved: {PER_CLASS_F1_CSV.relative_to(ROOT)}  ({size_kb:.1f} KB)")

    # McNemar JSON
    with open(MCNEMAR_JSON, "w") as f:
        json.dump(mcnemar_results, f, indent=2)
    size_kb = MCNEMAR_JSON.stat().st_size / 1024
    print(f"  Saved: {MCNEMAR_JSON.relative_to(ROOT)}  ({size_kb:.1f} KB)")

    # ── Print tables ──────────────────────────────────────────
    print_comparison_table(df_cmp, verbose=verbose)
    print_per_class_table(all_results)

    # ── Misuse summary ────────────────────────────────────────
    if mis_results:
        print()
        print("=" * 65)
        print("  Table 4 — Misuse Detection Results")
        print("=" * 65)
        for metric in ["f1", "precision", "recall", "roc_auc", "pr_auc"]:
            val = mis_results.get(metric, float("nan"))
            print(f"  {metric:<20} {val:.4f}")
        tp = mis_results.get("tp", "?")
        fp = mis_results.get("fp", "?")
        fn = mis_results.get("fn", "?")
        tn = mis_results.get("tn", "?")
        print(f"  {'TP / FP / FN / TN':<20} "
              f"{tp} / {fp} / {fn} / {tn}")
        if isinstance(fn, int) and fn > 0:
            print(f"\n  [WARN] {fn} missed misuse case(s).")
            print("  Consider lowering threshold in misuse_detector.py.")
        print("=" * 65)

    # ── Final paper-cite lines ────────────────────────────────
    print()
    print("  Paper cite lines (Table 3 + Table 4):")
    for name, r in all_results.items():
        cv_path = RESULTS_DIR / f"{name}_results.json"
        cv_std  = 0.0
        if cv_path.exists():
            with open(cv_path) as f:
                prev = json.load(f)
            cv_std = prev.get("cv", {}).get("cv_f1_std", 0.0)
        print(
            f"    {name.upper():<8}: "
            f"F1={r.get('f1_macro',0):.3f} ± {cv_std:.3f}, "
            f"Acc={r.get('accuracy',0):.3f}, "
            f"AUC={r.get('roc_auc_ovr',0):.3f}"
        )
    if mis_results:
        print(
            f"    {'MISUSE':<8}: "
            f"F1={mis_results.get('f1',0):.3f}, "
            f"Recall={mis_results.get('recall',0):.3f}, "
            f"PR-AUC={mis_results.get('pr_auc',0):.3f}"
        )

    print()
    print("  Outputs written to results/")
    print("  Next: python 04_explainability/shap_analysis.py")

    return all_results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — unified model evaluation and comparison"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["rf", "svm", "cnn", "lstm"],
        default=None,
        help="Evaluate specific models only (default: all available)."
    )
    parser.add_argument(
        "--split", choices=["test", "val"], default="test",
        help="Data split to evaluate on (default: test)."
    )
    parser.add_argument(
        "--table-only", action="store_true", dest="table_only",
        help="Print tables from existing evaluation_report.json only."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-model detailed output."
    )
    args = parser.parse_args()

    main(
        models_to_eval=args.models,
        split=args.split,
        table_only=args.table_only,
        verbose=not args.quiet,
    )