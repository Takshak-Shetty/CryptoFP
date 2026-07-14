# ============================================================
# CryptoFP — 03_models/cnn_1d.py
#
# 1D Convolutional Neural Network for cryptographic algorithm
# fingerprinting. Deep learning baseline 1 of 2.
#
# Role in the paper:
#   CNN baseline 3 of 4. Shows that a deep learning model
#   can learn local patterns across the feature dimension —
#   e.g. the relationship between adjacent timing features —
#   without explicit feature engineering. Bridges the gap
#   between classical ML (RF, SVM) and the novelty model
#   (LSTM + attention). IEEE reviewers expect at least one
#   deep learning baseline before the novelty architecture.
#
# Architecture (tabular mode — used by default):
#   Input:    (batch, 1, n_features)  — 1 channel, signal over features
#   Conv1:    kernel=3, pad=1 → 64  filters → BN → ReLU → MaxPool(2)
#   Conv2:    kernel=3, pad=1 → 128 filters → BN → ReLU → MaxPool(2)
#   Conv3:    kernel=3, pad=1 → 256 filters → BN → ReLU → AdaptiveAvgPool(1)
#   Flatten:  → (batch, 256)
#   FC1:      256 → 128 → Dropout(0.3) → ReLU
#   FC2:      128 → n_classes  (softmax at inference)
#
# Two input modes:
#   tabular  (default) — each row is treated as a 1D signal of
#                        length n_features. No sequence data needed.
#                        Works with the 60-sample test dataset now.
#   sequence           — uses data/sequences/*.npy files built from
#                        50 consecutive operation timing windows.
#                        Requires merge_dataset.py to have run with
#                        the full 1800-sample dataset first.
#                        Enable with --mode sequence.
#
# Usage:
#   python 03_models/cnn_1d.py
#   python 03_models/cnn_1d.py --quick
#   python 03_models/cnn_1d.py --eval-only
#   python 03_models/cnn_1d.py --mode sequence
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

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from config import (
    PATHS, FEATURES, CNN, SEQUENCES, EVAL,
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, RANDOM_SEED,
)
from label_encoder import CryptoFPLabelEncoder


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_SEED):
    """Fix all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ============================================================
# RESULTS DIRECTORY
# ============================================================

RESULTS_DIR      = ROOT / "results"
CNN_RESULTS_JSON = RESULTS_DIR / "cnn_results.json"


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

if TORCH_AVAILABLE:

    class CryptoFP_CNN1D(nn.Module):
        """
        1D CNN for cryptographic algorithm fingerprinting.

        Treats each benchmark sample as a 1D signal of length
        n_features (tabular mode) or SEQ_LENGTH (sequence mode).
        Three convolutional blocks with batch normalisation and
        dropout extract local feature patterns, followed by two
        fully-connected layers for classification.

        Architecture decisions documented for the paper:
          - BatchNorm after each Conv: stabilises training on
            small datasets where gradients can be noisy
          - MaxPool(2) × 2 + AdaptiveAvgPool(1): progressively
            compresses the signal, AdaptiveAvgPool handles any
            input length gracefully
          - Dropout(0.3) before FC2: primary regularisation —
            prevents overfitting on small training sets
          - class_weight in loss: mirrors the balanced setting
            used in RF and SVM baselines for fair comparison

        Args:
            in_length:  length of input signal (n_features or SEQ_LENGTH)
            n_classes:  number of output classes (6)
            num_filters: list of filter counts per conv block
            kernel_size: convolution kernel width
            dropout:    dropout rate before final FC layer
        """

        def __init__(
            self,
            in_length:   int,
            n_classes:   int   = CNN.NUM_CLASSES,
            num_filters: list  = None,
            kernel_size: int   = CNN.KERNEL_SIZE,
            dropout:     float = CNN.DROPOUT,
        ):
            super().__init__()
            if num_filters is None:
                num_filters = CNN.NUM_FILTERS   # [64, 128, 256]

            self.in_length  = in_length
            self.n_classes  = n_classes

            # ── Convolutional blocks ──────────────────────────
            # Block 1: 1 → 64 channels
            self.conv1 = nn.Sequential(
                nn.Conv1d(1, num_filters[0], kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(num_filters[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )
            # Block 2: 64 → 128 channels
            self.conv2 = nn.Sequential(
                nn.Conv1d(num_filters[0], num_filters[1], kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(num_filters[1]),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )
            # Block 3: 128 → 256 channels + global average pool
            self.conv3 = nn.Sequential(
                nn.Conv1d(num_filters[1], num_filters[2], kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(num_filters[2]),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),   # → (batch, 256, 1)
            )

            # ── Classifier head ───────────────────────────────
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_filters[2], 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Forward pass.
            x shape: (batch, 1, signal_length)
            Returns logits shape: (batch, n_classes)
            """
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.classifier(x)

        def predict_proba(
            self, x: "torch.Tensor", device: str = "cpu"
        ) -> np.ndarray:
            """Softmax probabilities for sklearn-compatible evaluation."""
            self.eval()
            with torch.no_grad():
                x = x.to(device)
                logits = self.forward(x)
                probs  = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def __repr__(self):
            return (
                f"CryptoFP_CNN1D("
                f"in_length={self.in_length}, "
                f"n_classes={self.n_classes}, "
                f"params={self.count_parameters():,})"
            )


# ============================================================
# DATA LOADING
# ============================================================

def load_tabular(verbose: bool = True) -> tuple:
    """
    Load train/val/test from CSV splits.
    Input shape: (n_samples, 1, n_features).

    This is the default mode — no sequence files needed.
    Works with any dataset size including the 60-sample test set.
    """
    for path in [PATHS.TRAIN_CSV, PATHS.VAL_CSV, PATHS.TEST_CSV]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found. Run Phase 2 preprocessing first."
            )

    train = pd.read_csv(PATHS.TRAIN_CSV)
    val   = pd.read_csv(PATHS.VAL_CSV)
    test  = pd.read_csv(PATHS.TEST_CSV)

    base_feat = FEATURES.ALL_FEATURES + ["log_keygen_ms", "log_dec_ms"]
    feat_cols = [c for c in base_feat if c in train.columns]

    # Filter SMOTE rows from train
    real_mask  = train["label_encoded"] >= 0
    train_real = train[real_mask]

    def to_tensor(df, fc):
        X = df[fc].values.astype(np.float32)
        # Add channel dimension: (n, feat) → (n, 1, feat)
        return torch.from_numpy(X[:, np.newaxis, :])

    def labels(df):
        return torch.from_numpy(
            df["label_encoded"].values.astype(np.int64)
        )

    X_train = to_tensor(train_real, feat_cols)
    y_train = labels(train_real)
    X_val   = to_tensor(val,   feat_cols)
    y_val   = labels(val)
    X_test  = to_tensor(test,  feat_cols)
    y_test  = labels(test)

    in_length = len(feat_cols)

    if verbose:
        print(f"  Mode       : tabular  (each row = 1D signal over features)")
        print(f"  Features   : {len(feat_cols)}  {feat_cols}")
        print(f"  X_train    : {tuple(X_train.shape)}")
        print(f"  X_val      : {tuple(X_val.shape)}")
        print(f"  X_test     : {tuple(X_test.shape)}")
        print(f"  Signal len : {in_length}")

    enc = CryptoFPLabelEncoder.load()
    return X_train, y_train, X_val, y_val, X_test, y_test, feat_cols, in_length, enc


def load_sequences(verbose: bool = True) -> tuple:
    """
    Load pre-built sequence arrays from data/sequences/*.npy.
    Input shape: (n_samples, 1, SEQ_LENGTH).

    Requires the full 1800-sample dataset and sequence-building
    step in merge_dataset.py / feature_extractor.py.
    Falls back to tabular mode if sequence files are missing.
    """
    seq_files = [PATHS.SEQ_TRAIN, PATHS.SEQ_VAL, PATHS.SEQ_TEST,
                 PATHS.SEQ_LABELS_TRAIN, PATHS.SEQ_LABELS_VAL, PATHS.SEQ_LABELS_TEST]

    missing = [str(f) for f in seq_files if not Path(f).exists()]
    if missing:
        print(
            f"  [INFO] Sequence files not found: {missing}\n"
            f"  Falling back to tabular mode.\n"
            f"  Build sequences with the full 1800-sample dataset."
        )
        return load_tabular(verbose=verbose)

    X_train = torch.from_numpy(
        np.load(PATHS.SEQ_TRAIN).astype(np.float32)[:, np.newaxis, :]
    )
    X_val   = torch.from_numpy(
        np.load(PATHS.SEQ_VAL).astype(np.float32)[:, np.newaxis, :]
    )
    X_test  = torch.from_numpy(
        np.load(PATHS.SEQ_TEST).astype(np.float32)[:, np.newaxis, :]
    )
    y_train = torch.from_numpy(np.load(PATHS.SEQ_LABELS_TRAIN).astype(np.int64))
    y_val   = torch.from_numpy(np.load(PATHS.SEQ_LABELS_VAL).astype(np.int64))
    y_test  = torch.from_numpy(np.load(PATHS.SEQ_LABELS_TEST).astype(np.int64))

    in_length = X_train.shape[2]
    feat_cols = SEQUENCES.SEQ_FEATURES

    if verbose:
        print(f"  Mode       : sequence  (50-step timing windows)")
        print(f"  X_train    : {tuple(X_train.shape)}")
        print(f"  X_val      : {tuple(X_val.shape)}")
        print(f"  Signal len : {in_length}")

    enc = CryptoFPLabelEncoder.load()
    return X_train, y_train, X_val, y_val, X_test, y_test, feat_cols, in_length, enc


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(y_train: "torch.Tensor", device: str) -> "torch.Tensor":
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    Mirrors the class_weight='balanced' setting in RF and SVM.
    """
    y_np     = y_train.numpy()
    classes  = np.arange(CNN.NUM_CLASSES)
    counts   = np.array(
        [(y_np == c).sum() for c in classes], dtype=np.float32
    )
    counts   = np.where(counts == 0, 1, counts)   # avoid div/0
    weights  = 1.0 / counts
    weights  = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(
    model:     "CryptoFP_CNN1D",
    X_train:   "torch.Tensor",
    y_train:   "torch.Tensor",
    X_val:     "torch.Tensor",
    y_val:     "torch.Tensor",
    device:    str,
    quick:     bool = False,
    verbose:   bool = True,
) -> tuple:
    """
    Train the CNN with early stopping.

    Training details (documented for the paper):
      - Loss:      CrossEntropyLoss with class weights
      - Optimiser: Adam (lr=5e-4, weight_decay=1e-4)
      - Schedule:  ReduceLROnPlateau (patience=5, factor=0.5)
      - Stopping:  Patience=10 epochs on val F1 (macro)
      - Best model: saved in memory, restored after training

    Args:
        model:   CryptoFP_CNN1D instance (on CPU or CUDA)
        X_train: training tensors (batch, 1, L)
        y_train: training labels
        X_val:   validation tensors
        y_val:   validation labels
        device:  'cuda' or 'cpu'
        quick:   reduced epochs for testing
        verbose: print per-epoch progress

    Returns:
        trained model, training history dict
    """
    epochs       = 10 if quick else CNN.EPOCHS
    batch_size   = min(CNN.BATCH_SIZE, len(X_train))
    patience     = 3  if quick else CNN.PATIENCE

    model = model.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Class-weighted loss
    class_weights = compute_class_weights(y_train, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=CNN.LEARNING_RATE,
        weight_decay=CNN.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5,
        factor=0.5, min_lr=1e-6,
    )

    dataset    = TensorDataset(X_train.to(device), y_train.to(device))
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, drop_last=False,
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [],   "val_f1": [],
    }

    best_val_f1    = -1.0
    best_weights   = None
    patience_count = 0

    if verbose:
        print(f"  Epochs: {epochs}  Batch: {batch_size}  "
              f"LR: {CNN.LEARNING_RATE}  Patience: {patience}")
        print(f"  {'Epoch':>6}  {'Train loss':>12}  {'Val loss':>10}  "
              f"{'Val F1':>8}  {'LR':>10}  {'Status'}")
        print(f"  {'-'*65}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # ── train ─────────────────────────────────────────────
        model.train()
        train_losses = []
        all_preds, all_labels = [], []

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

        train_loss = float(np.mean(train_losses))
        train_f1   = f1_score(
            all_labels, all_preds,
            average="macro", zero_division=0,
        )

        # ── validate ──────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss   = criterion(val_logits, y_val).item()
            val_preds  = val_logits.argmax(dim=1).cpu().numpy()

        val_f1 = f1_score(
            y_val.cpu().numpy(), val_preds,
            average="macro", zero_division=0,
        )

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(float(val_loss), 4))
        history["train_f1"].append(round(float(train_f1), 4))
        history["val_f1"].append(round(float(val_f1), 4))

        # ── early stopping ────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
            status = "BEST"
        else:
            patience_count += 1
            status = f"wait {patience_count}/{patience}"

        if verbose and (
            epoch <= 5
            or epoch % 10 == 0
            or epoch == epochs
            or status == "BEST"
        ):
            print(
                f"  {epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  "
                f"{val_f1:>8.4f}  {current_lr:>10.6f}  {status}"
            )

        if patience_count >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val F1: {best_val_f1:.4f})")
            break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n  Training complete: {elapsed:.1f}s  "
              f"Best val F1: {best_val_f1:.4f}")

    history["best_val_f1"]  = round(best_val_f1, 4)
    history["elapsed_sec"]  = round(elapsed, 1)
    history["epochs_run"]   = epoch

    return model, history


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    X:          "torch.Tensor",
    y_true_t:   "torch.Tensor",
    split_name: str,
    feat_cols:  list,
    enc:        CryptoFPLabelEncoder,
    device:     str,
    verbose:    bool = True,
) -> dict:
    """
    Compute all paper metrics for one data split.
    Same metric set as RF and SVM for Table 3 comparison.
    """
    if not TORCH_AVAILABLE:
        return {}

    model.eval()
    model = model.to(device)
    X     = X.to(device)

    with torch.no_grad():
        logits = model(X)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

    y_true = y_true_t.numpy()

    acc      = accuracy_score(y_true, preds)
    f1_mac   = f1_score(y_true, preds, average="macro",    zero_division=0)
    f1_wei   = f1_score(y_true, preds, average="weighted", zero_division=0)
    prec_mac = precision_score(y_true, preds, average="macro", zero_division=0)
    rec_mac  = recall_score(y_true,  preds, average="macro", zero_division=0)

    try:
        present      = sorted(np.unique(y_true).tolist())
        y_prob_pr    = probs[:, present]
        roc_auc      = roc_auc_score(
            y_true, y_prob_pr,
            multi_class=EVAL.ROC_MULTICLASS,
            average=EVAL.AVERAGE,
            labels=present,
        )
    except Exception as e:
        roc_auc = -1.0
        if verbose:
            print(f"  [WARN] ROC-AUC: {e}")

    per_class_f1_arr = f1_score(
        y_true, preds,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    per_class_dict = {
        CLASS_NAMES[i]: round(float(per_class_f1_arr[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    cm = confusion_matrix(
        y_true, preds,
        labels=list(range(len(CLASS_NAMES))),
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
        for cls, f1_val in per_class_dict.items():
            bar = "█" * int(f1_val * 20)
            print(f"    {cls:<14} {f1_val:.4f}  {bar}")
        print()
        report = classification_report(
            y_true, preds,
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
# SAVE / LOAD
# ============================================================

def save_model(model: "CryptoFP_CNN1D", verbose: bool = True):
    """Save model weights to models/cnn1d_best.pt."""
    if not TORCH_AVAILABLE:
        return
    PATHS.MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), PATHS.CNN_MODEL)
    size_kb = PATHS.CNN_MODEL.stat().st_size / 1024
    if verbose:
        print(f"  Saved model : {PATHS.CNN_MODEL.name}  ({size_kb:.1f} KB)")


def load_model(in_length: int, device: str, verbose: bool = True) -> "CryptoFP_CNN1D":
    """Load model weights from models/cnn1d_best.pt."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")
    if not PATHS.CNN_MODEL.exists():
        raise FileNotFoundError(
            f"{PATHS.CNN_MODEL} not found. Run without --eval-only first."
        )
    model = CryptoFP_CNN1D(in_length=in_length)
    model.load_state_dict(torch.load(PATHS.CNN_MODEL, map_location=device))
    model = model.to(device)
    model.eval()
    if verbose:
        print(f"  Loaded: {PATHS.CNN_MODEL.name}  {repr(model)}")
    return model


def save_results(results: dict, verbose: bool = True):
    """Save all metrics to results/cnn_results.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CNN_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    size_kb = CNN_RESULTS_JSON.stat().st_size / 1024
    if verbose:
        print(f"  Saved: {CNN_RESULTS_JSON.relative_to(ROOT)}  ({size_kb:.1f} KB)")


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison(test_results: dict):
    """Print RF / SVM / CNN side-by-side for paper Table 3."""
    metrics = ["accuracy", "f1_macro", "roc_auc_ovr"]
    rows = {}

    for name, path in [("RF",  RESULTS_DIR / "rf_results.json"),
                        ("SVM", RESULTS_DIR / "svm_results.json")]:
        if path.exists():
            with open(path) as f:
                rows[name] = json.load(f).get("test", {})

    rows["CNN-1D"] = test_results

    print()
    print("  Comparison — RF / SVM / CNN-1D  (test set)")
    print(f"  {'Metric':<22} {'RF':>10} {'SVM':>10} {'CNN-1D':>10}")
    print(f"  {'-'*55}")
    for m in metrics:
        vals = [rows.get(n, {}).get(m, float("nan")) for n in ["RF", "SVM", "CNN-1D"]]
        best = max((v for v in vals if not np.isnan(v)), default=0)
        row  = f"  {m:<22}"
        for v in vals:
            flag = " *" if abs(v - best) < 1e-4 else ""
            row += f" {v:>9.4f}{flag}"
        print(row)
    print("  (* = best)")


# ============================================================
# MAIN
# ============================================================

def main(
    quick:     bool = False,
    eval_only: bool = False,
    mode:      str  = "tabular",
    verbose:   bool = True,
) -> dict:
    """
    Full CNN training and evaluation pipeline.

    Args:
        quick:     reduced epochs (10) for fast testing
        eval_only: load existing model and re-evaluate
        mode:      'tabular' (default) or 'sequence'
        verbose:   print detailed metrics

    Returns:
        dict with model, val_results, test_results, history
    """
    if not TORCH_AVAILABLE:
        print(
            "[ERROR] PyTorch not found.\n"
            "Install with: pip install torch\n"
            "See requirements.txt for the correct version."
        )
        return None

    set_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("  CryptoFP — 1D CNN")
    print("=" * 65)
    mode_str = (
        "EVAL ONLY" if eval_only
        else "QUICK (10 epochs)" if quick
        else f"FULL ({CNN.EPOCHS} epochs)"
    )
    print(f"  Mode     : {mode_str}  [{mode}]")
    print(f"  Device   : {device}")
    print(f"  Seed     : {RANDOM_SEED}")
    print(f"  Filters  : {CNN.NUM_FILTERS}")
    print(f"  Kernel   : {CNN.KERNEL_SIZE}")
    print(f"  Dropout  : {CNN.DROPOUT}")
    print(f"  Min F1   : {EVAL.MIN_F1_THRESHOLD}")
    print()

    # ── 1. Load data ─────────────────────────────────────────
    print("  Step 1/5 — Loading data")
    try:
        loader = load_sequences if mode == "sequence" else load_tabular
        (X_train, y_train, X_val, y_val,
         X_test,  y_test,  feat_cols, in_length, enc) = loader(verbose=verbose)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return None

    # ── 2. Build or load model ────────────────────────────────
    print(f"\n  Step 2/5 — {'Loading' if eval_only else 'Building'} model")
    if eval_only:
        try:
            model = load_model(in_length, device, verbose=verbose)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [ERROR] {e}")
            return None
        history = {}
    else:
        model = CryptoFP_CNN1D(in_length=in_length)
        print(f"  {repr(model)}")
        print(f"  Parameters: {model.count_parameters():,}")

        # ── 3. Train ──────────────────────────────────────────
        print(f"\n  Step 3/5 — Training")
        model, history = train_model(
            model, X_train, y_train, X_val, y_val,
            device=device, quick=quick, verbose=verbose,
        )

    # ── 4. Evaluate ───────────────────────────────────────────
    print(f"\n  Step {'4' if eval_only else '4'}/5 — Evaluation")
    val_results  = evaluate(model, X_val,  y_val,  "val",  feat_cols, enc, device, verbose)
    test_results = evaluate(model, X_test, y_test, "test", feat_cols, enc, device, verbose)

    test_f1 = test_results.get("f1_macro", 0)
    if test_f1 < EVAL.MIN_F1_THRESHOLD:
        print(
            f"\n  [WARN] Test F1={test_f1:.4f} below "
            f"target {EVAL.MIN_F1_THRESHOLD}.\n"
            f"  Expected with small dataset — will improve at full scale."
        )
    else:
        print(f"\n  Test F1={test_f1:.4f} >= {EVAL.MIN_F1_THRESHOLD}  OK")

    # ── 5. Save ───────────────────────────────────────────────
    print(f"\n  Step 5/5 — Saving")
    if not eval_only:
        save_model(model, verbose=verbose)

    cv_std_note = history.get("best_val_f1", test_f1)
    full_results = {
        "model":        "CNN_1D",
        "mode":         mode_str,
        "input_mode":   mode,
        "in_length":    in_length,
        "n_features":   len(feat_cols),
        "feature_cols": feat_cols,
        "architecture": {
            "filters":     CNN.NUM_FILTERS,
            "kernel_size": CNN.KERNEL_SIZE,
            "dropout":     CNN.DROPOUT,
            "n_classes":   CNN.NUM_CLASSES,
        },
        "training":    history,
        "val":         val_results,
        "test":        test_results,
        "paper_cite":  (
            f"CNN-1D: F1={test_results.get('f1_macro',0):.3f}, "
            f"Acc={test_results.get('accuracy',0):.3f}, "
            f"AUC={test_results.get('roc_auc_ovr',0):.3f}"
        ),
    }
    save_results(full_results, verbose=verbose)
    print_comparison(test_results)

    print()
    print("=" * 65)
    print("  CNN-1D — FINAL SUMMARY")
    print("=" * 65)
    print(f"  Test accuracy   : {test_results.get('accuracy', 0):.4f}")
    print(f"  Test F1 (macro) : {test_results.get('f1_macro', 0):.4f}")
    print(f"  Test ROC-AUC    : {test_results.get('roc_auc_ovr', 0):.4f}")
    if history:
        print(f"  Best val F1     : {history.get('best_val_f1', 0):.4f}")
        print(f"  Epochs run      : {history.get('epochs_run', 0)}")
        print(f"  Training time   : {history.get('elapsed_sec', 0):.1f}s")
    print()
    print(f"  Paper cite (Table 3):")
    print(f"    {full_results['paper_cite']}")
    print()
    print("  Model saved to: models/cnn1d_best.pt")
    print("  Results saved to: results/cnn_results.json")
    print()
    print("  Next: python 03_models/lstm_attention.py")
    print("=" * 65)

    return {
        "model":        model,
        "val_results":  val_results,
        "test_results": test_results,
        "history":      history,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — train and evaluate 1D CNN"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Train for 10 epochs only — fast test."
    )
    parser.add_argument(
        "--eval-only", action="store_true", dest="eval_only",
        help="Load existing cnn1d_best.pt and re-evaluate."
    )
    parser.add_argument(
        "--mode", choices=["tabular", "sequence"], default="tabular",
        help=(
            "Input mode: 'tabular' (default, no sequence files needed) "
            "or 'sequence' (uses data/sequences/*.npy)."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed per-class metrics."
    )
    args = parser.parse_args()

    main(
        quick=args.quick,
        eval_only=args.eval_only,
        mode=args.mode,
        verbose=not args.quiet,
    )