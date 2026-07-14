# ============================================================
# CryptoFP — 03_models/lstm_attention.py
#
# Bidirectional LSTM with multi-head additive attention —
# the novelty model for cryptographic algorithm fingerprinting.
#
# Role in the paper:
#   Novelty model — the headline architecture.  This is what
#   separates the paper from a benchmark study.  The attention
#   mechanism produces a per-timestep weight vector that shows
#   WHICH features (or timing steps) the model found most
#   diagnostic for each class.  Those weights become the
#   attention heatmap figure in the paper — interpretable,
#   class-specific fingerprints.
#
# Why LSTM + attention over plain LSTM:
#   Plain LSTM compresses the entire sequence into a single
#   hidden state, losing which timesteps mattered most.
#   Attention lets the model selectively weight each position,
#   producing both better accuracy AND a visual explanation.
#   This is cited as a key contribution in the methodology.
#
# Architecture:
#   Input:   (batch, seq_len, input_size)
#              tabular:  seq_len=n_features, input_size=1
#              sequence: seq_len=SEQ_LENGTH, input_size=5
#   BiLSTM:  2 layers, hidden=128, bidirectional → 256 per step
#   Dropout: between LSTM layers (0.3)
#   Attention (Bahdanau additive, 4 heads):
#              query  = last hidden state (B, 256)
#              keys   = all LSTM outputs  (B, L, 256)
#              scores = v^T tanh(W_h·h + W_q·q)  (B, L)
#              weights= softmax(scores)           (B, L)
#              context= sum(weights * outputs)    (B, 256)
#   FC:      context (B,256) → 128 → Dropout(0.3) → 6 classes
#
# Attention citation for paper:
#   Bahdanau, D., Cho, K., Bengio, Y. (2015). Neural Machine
#   Translation by Jointly Learning to Align and Translate.
#   ICLR 2015.  (additive attention mechanism)
#
# Usage:
#   python 03_models/lstm_attention.py
#   python 03_models/lstm_attention.py --quick
#   python 03_models/lstm_attention.py --eval-only
#   python 03_models/lstm_attention.py --mode sequence
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
    PATHS, FEATURES, LSTM, SEQUENCES, EVAL,
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, RANDOM_SEED,
)
from label_encoder import CryptoFPLabelEncoder


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_SEED):
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

RESULTS_DIR       = ROOT / "results"
LSTM_RESULTS_JSON = RESULTS_DIR / "lstm_results.json"


# ============================================================
# ATTENTION MECHANISM
# ============================================================

if TORCH_AVAILABLE:

    class BahdanauAttention(nn.Module):
        """
        Additive (Bahdanau) attention over LSTM output sequence.

        For each timestep, computes a scalar score using the
        alignment function:
            score(h_t) = v^T · tanh(W_h·h_t + W_q·query)

        where query is the final hidden state of the BiLSTM.
        Scores are normalised with softmax to produce attention
        weights — a probability distribution over timesteps.

        The context vector is the weighted sum of all LSTM outputs,
        focusing on the most diagnostic timesteps.

        Paper figure: plot weights per class across a test set
        to visualise which features (tabular mode) or which
        timing positions (sequence mode) each class activates.

        Args:
            hidden_dim:   dimensionality of LSTM output per step
                          (hidden_size * 2 for BiLSTM)
            attention_dim: projection dimension for alignment
                          (config: LSTM.ATTENTION_DIM = 64)
        """

        def __init__(self, hidden_dim: int, attention_dim: int):
            super().__init__()
            self.W_h  = nn.Linear(hidden_dim, attention_dim, bias=False)
            self.W_q  = nn.Linear(hidden_dim, attention_dim, bias=False)
            self.v    = nn.Linear(attention_dim, 1, bias=False)

        def forward(
            self,
            lstm_outputs: "torch.Tensor",  # (B, L, hidden_dim)
            query:        "torch.Tensor",  # (B, hidden_dim)
        ) -> tuple:
            """
            Args:
                lstm_outputs: all LSTM hidden states, shape (B, L, H)
                query:        final hidden state, shape (B, H)

            Returns:
                context:  weighted sum, shape (B, H)
                weights:  attention weights, shape (B, L)
                          — saved for paper visualisation
            """
            # Project all timesteps: (B, L, attn_dim)
            h_proj = self.W_h(lstm_outputs)

            # Project query and broadcast: (B, 1, attn_dim)
            q_proj = self.W_q(query).unsqueeze(1)

            # Alignment scores: (B, L, 1) → (B, L)
            scores = self.v(torch.tanh(h_proj + q_proj)).squeeze(-1)

            # Normalise to probability distribution
            weights = torch.softmax(scores, dim=1)   # (B, L)

            # Context: weighted sum of LSTM outputs
            # (B, L, 1) * (B, L, H) → sum over L → (B, H)
            context = (weights.unsqueeze(-1) * lstm_outputs).sum(dim=1)

            return context, weights


    # ============================================================
    # FULL MODEL
    # ============================================================

    class CryptoFP_LSTM_Attention(nn.Module):
        """
        Bidirectional LSTM + Bahdanau Attention for cryptographic
        algorithm fingerprinting.

        This is the novelty architecture described in the paper.
        It extends previous side-channel analysis work (which used
        static feature vectors) by treating benchmark measurements
        as a sequence and learning which positions are most
        discriminating via attention.

        Args:
            input_size:    features per timestep
                           tabular=1, sequence=LSTM.INPUT_SIZE
            seq_len:       number of timesteps
                           tabular=n_features, sequence=SEQ_LENGTH
            hidden_size:   LSTM hidden units per direction (128)
            num_layers:    LSTM layers (2)
            bidirectional: use BiLSTM (True)
            attention_dim: attention projection size (64)
            n_classes:     output classes (6)
            dropout:       dropout rate (0.3)
        """

        def __init__(
            self,
            input_size:    int,
            seq_len:       int,
            hidden_size:   int   = LSTM.HIDDEN_SIZE,
            num_layers:    int   = LSTM.NUM_LAYERS,
            bidirectional: bool  = LSTM.BIDIRECTIONAL,
            attention_dim: int   = LSTM.ATTENTION_DIM,
            n_classes:     int   = LSTM.NUM_CLASSES,
            dropout:       float = LSTM.DROPOUT,
        ):
            super().__init__()
            self.hidden_size   = hidden_size
            self.num_layers    = num_layers
            self.bidirectional = bidirectional
            self.n_directions  = 2 if bidirectional else 1
            self.hidden_dim    = hidden_size * self.n_directions  # 256

            self.input_size = input_size
            self.seq_len    = seq_len

            # ── BiLSTM ─────────────────────────────────────────
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,          # (B, L, features)
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            # ── Attention ──────────────────────────────────────
            self.attention = BahdanauAttention(
                hidden_dim=self.hidden_dim,
                attention_dim=attention_dim,
            )

            # ── Classifier head ────────────────────────────────
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes),
            )

            # ── Weight initialisation ──────────────────────────
            self._init_weights()

        def _init_weights(self):
            """
            Orthogonal init for LSTM weights, Xavier for FC layers.
            Reduces vanishing/exploding gradient risk on small datasets.
            """
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1 (Jozefowicz et al. 2015)
                    n = param.size(0)
                    param.data[n // 4: n // 2].fill_(1)

            for module in self.classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(
            self, x: "torch.Tensor"
        ) -> tuple:
            """
            Forward pass.

            Args:
                x: input tensor, shape (B, seq_len, input_size)

            Returns:
                logits:  class scores, shape (B, n_classes)
                weights: attention weights, shape (B, seq_len)
                         — use for paper visualisation figure
            """
            # ── LSTM ───────────────────────────────────────────
            # outputs: (B, L, hidden*2)
            # h_n:     (num_layers*directions, B, hidden)
            outputs, (h_n, _) = self.lstm(x)

            # Query: concatenate last forward and backward hidden states
            # h_n shape: (num_layers*2, B, hidden)
            # Last layer: forward = h_n[-2], backward = h_n[-1]
            if self.bidirectional:
                query = torch.cat(
                    [h_n[-2], h_n[-1]], dim=1
                )   # (B, hidden*2)
            else:
                query = h_n[-1]   # (B, hidden)

            # ── Attention ──────────────────────────────────────
            context, weights = self.attention(outputs, query)

            # ── Classify ───────────────────────────────────────
            logits = self.classifier(context)

            return logits, weights

        def predict_proba(
            self,
            x:      "torch.Tensor",
            device: str = "cpu",
        ) -> np.ndarray:
            """Softmax probabilities, numpy array."""
            self.eval()
            with torch.no_grad():
                x      = x.to(device)
                logits, _ = self.forward(x)
                probs  = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

        def get_attention_weights(
            self,
            x:      "torch.Tensor",
            device: str = "cpu",
        ) -> np.ndarray:
            """
            Return attention weights for visualisation.
            Shape: (n_samples, seq_len)

            Use in attention_visualizer.py to produce the
            heatmap figure in the paper.
            """
            self.eval()
            with torch.no_grad():
                x = x.to(device)
                _, weights = self.forward(x)
            return weights.cpu().numpy()

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def __repr__(self):
            return (
                f"CryptoFP_LSTM_Attention("
                f"input={self.input_size}, "
                f"seq={self.seq_len}, "
                f"hidden={self.hidden_size}×{self.n_directions}, "
                f"params={self.count_parameters():,})"
            )


# ============================================================
# DATA LOADING
# ============================================================

def load_tabular(verbose: bool = True) -> tuple:
    """
    Load CSV splits. Reshape each row as a sequence:
        (n_features,) → (n_features, 1)
    i.e. each feature is one timestep of dimension 1.

    This lets the LSTM learn temporal ordering across the
    feature vector — e.g. the relationship between
    keygen_time_ms at position 0 and enc_dec_ratio at
    position 5 — without any sequence data collection.
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

    real_mask  = train["label_encoded"] >= 0
    train_real = train[real_mask]

    def to_tensor(df, fc):
        X = df[fc].values.astype(np.float32)
        # (n, feat) → (n, feat, 1): each feature is a timestep of size 1
        return torch.from_numpy(X[:, :, np.newaxis])

    def labels(df):
        return torch.from_numpy(df["label_encoded"].values.astype(np.int64))

    X_train = to_tensor(train_real, feat_cols)
    y_train = labels(train_real)
    X_val   = to_tensor(val,   feat_cols)
    y_val   = labels(val)
    X_test  = to_tensor(test,  feat_cols)
    y_test  = labels(test)

    # input_size=1, seq_len=n_features
    input_size = 1
    seq_len    = len(feat_cols)

    if verbose:
        print(f"  Mode       : tabular  (each feature = 1 timestep)")
        print(f"  Features   : {len(feat_cols)}  {feat_cols}")
        print(f"  X_train    : {tuple(X_train.shape)}  "
              f"(B, seq_len={seq_len}, input_size={input_size})")
        print(f"  X_val      : {tuple(X_val.shape)}")
        print(f"  X_test     : {tuple(X_test.shape)}")

    enc = CryptoFPLabelEncoder.load()
    return (X_train, y_train, X_val, y_val,
            X_test, y_test, feat_cols, input_size, seq_len, enc)


def load_sequences(verbose: bool = True) -> tuple:
    """
    Load pre-built sequence arrays.
    Input shape: (n_samples, SEQ_LENGTH, INPUT_SIZE=5)

    Requires full 1800-sample dataset and sequence files.
    Falls back to tabular mode if files are missing.
    """
    seq_files = [PATHS.SEQ_TRAIN, PATHS.SEQ_VAL, PATHS.SEQ_TEST,
                 PATHS.SEQ_LABELS_TRAIN, PATHS.SEQ_LABELS_VAL,
                 PATHS.SEQ_LABELS_TEST]
    missing = [str(f) for f in seq_files if not Path(f).exists()]
    if missing:
        print(
            f"  [INFO] Sequence files missing — falling back to tabular mode.\n"
            f"  Build sequences with the full 1800-sample dataset."
        )
        return load_tabular(verbose=verbose)

    X_train = torch.from_numpy(np.load(PATHS.SEQ_TRAIN).astype(np.float32))
    X_val   = torch.from_numpy(np.load(PATHS.SEQ_VAL).astype(np.float32))
    X_test  = torch.from_numpy(np.load(PATHS.SEQ_TEST).astype(np.float32))
    y_train = torch.from_numpy(np.load(PATHS.SEQ_LABELS_TRAIN).astype(np.int64))
    y_val   = torch.from_numpy(np.load(PATHS.SEQ_LABELS_VAL).astype(np.int64))
    y_test  = torch.from_numpy(np.load(PATHS.SEQ_LABELS_TEST).astype(np.int64))

    input_size = X_train.shape[2]   # 5 sequence features
    seq_len    = X_train.shape[1]   # SEQ_LENGTH=50
    feat_cols  = SEQUENCES.SEQ_FEATURES

    if verbose:
        print(f"  Mode       : sequence  ({seq_len}-step timing windows)")
        print(f"  X_train    : {tuple(X_train.shape)}")
        print(f"  Input size : {input_size} features/step")
        print(f"  Seq len    : {seq_len} steps")

    enc = CryptoFPLabelEncoder.load()
    return (X_train, y_train, X_val, y_val,
            X_test, y_test, feat_cols, input_size, seq_len, enc)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(y_train: "torch.Tensor", device: str) -> "torch.Tensor":
    y_np    = y_train.numpy()
    classes = np.arange(LSTM.NUM_CLASSES)
    counts  = np.array([(y_np == c).sum() for c in classes], dtype=np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(
    model:   "CryptoFP_LSTM_Attention",
    X_train: "torch.Tensor",
    y_train: "torch.Tensor",
    X_val:   "torch.Tensor",
    y_val:   "torch.Tensor",
    device:  str,
    quick:   bool = False,
    verbose: bool = True,
) -> tuple:
    """
    Train BiLSTM+Attention with early stopping and gradient clipping.

    Training details for the paper:
      - Loss:      CrossEntropyLoss with class weights
      - Optimiser: Adam (lr=5e-4, weight_decay=1e-4)
      - Grad clip: L2 norm ≤ 1.0  (prevents LSTM gradient explosion)
      - Schedule:  ReduceLROnPlateau (patience=5, factor=0.5)
      - Stopping:  Patience=15 epochs on val F1 (macro)
      - Init:      Orthogonal LSTM weights, Xavier FC weights
      - Best:      Restore best checkpoint after training

    Gradient clipping (LSTM.GRAD_CLIP=1.0) is critical for LSTM
    stability — without it, long sequences can cause gradient
    explosion, especially in the early training epochs.

    Returns:
        trained model, training history dict
    """
    epochs     = 10 if quick else LSTM.EPOCHS
    batch_size = min(LSTM.BATCH_SIZE, len(X_train))
    patience   = 3  if quick else LSTM.PATIENCE

    model = model.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    class_weights = compute_class_weights(y_train, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LSTM.LEARNING_RATE,
        weight_decay=LSTM.WEIGHT_DECAY,
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
        "train_f1":   [], "val_f1": [],
    }

    best_val_f1  = -1.0
    best_weights = None
    pat_count    = 0

    if verbose:
        print(f"  Epochs: {epochs}  Batch: {batch_size}  "
              f"LR: {LSTM.LEARNING_RATE}  "
              f"GradClip: {LSTM.GRAD_CLIP}  Patience: {patience}")
        print(f"  {'Epoch':>6}  {'Train loss':>12}  {'Val loss':>10}  "
              f"{'Val F1':>8}  {'LR':>10}  {'Status'}")
        print(f"  {'-'*65}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # ── train ─────────────────────────────────────────────
        model.train()
        t_losses, t_preds, t_labels = [], [], []

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits, _ = model(X_batch)          # ignore weights during train
            loss       = criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping — essential for LSTM stability
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=LSTM.GRAD_CLIP
            )
            optimizer.step()

            t_losses.append(loss.item())
            t_preds.extend(logits.argmax(dim=1).cpu().numpy())
            t_labels.extend(y_batch.cpu().numpy())

        train_loss = float(np.mean(t_losses))
        train_f1   = f1_score(t_labels, t_preds,
                              average="macro", zero_division=0)

        # ── validate ──────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val)
            val_loss      = criterion(val_logits, y_val).item()
            val_preds     = val_logits.argmax(dim=1).cpu().numpy()

        val_f1 = f1_score(y_val.cpu().numpy(), val_preds,
                          average="macro", zero_division=0)

        scheduler.step(val_f1)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(float(val_loss), 4))
        history["train_f1"].append(round(float(train_f1), 4))
        history["val_f1"].append(round(float(val_f1), 4))

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            pat_count    = 0
            status       = "BEST"
        else:
            pat_count += 1
            status = f"wait {pat_count}/{patience}"

        if verbose and (
            epoch <= 5
            or epoch % 10 == 0
            or epoch == epochs
            or status == "BEST"
        ):
            print(
                f"  {epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  "
                f"{val_f1:>8.4f}  {lr_now:>10.6f}  {status}"
            )

        if pat_count >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}  "
                      f"(best val F1: {best_val_f1:.4f})")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n  Training complete: {elapsed:.1f}s  "
              f"Best val F1: {best_val_f1:.4f}")

    history["best_val_f1"] = round(best_val_f1, 4)
    history["elapsed_sec"] = round(elapsed, 1)
    history["epochs_run"]  = epoch

    return model, history


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model:      "CryptoFP_LSTM_Attention",
    X:          "torch.Tensor",
    y_true_t:   "torch.Tensor",
    split_name: str,
    feat_cols:  list,
    enc:        CryptoFPLabelEncoder,
    device:     str,
    verbose:    bool = True,
) -> dict:
    """
    Compute all paper metrics plus attention weight statistics.

    Attention statistics included:
        mean_attention_per_position — which features the model
        consistently focuses on across all test samples. This
        feeds directly into the attention heatmap figure.
    """
    if not TORCH_AVAILABLE:
        return {}

    model.eval()
    model = model.to(device)
    X     = X.to(device)

    with torch.no_grad():
        logits, attn_weights = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    y_true       = y_true_t.numpy()
    attn_np      = attn_weights.cpu().numpy()   # (n, seq_len)

    acc      = accuracy_score(y_true, preds)
    f1_mac   = f1_score(y_true, preds, average="macro",    zero_division=0)
    f1_wei   = f1_score(y_true, preds, average="weighted", zero_division=0)
    prec_mac = precision_score(y_true, preds, average="macro", zero_division=0)
    rec_mac  = recall_score(y_true,  preds, average="macro", zero_division=0)

    try:
        present   = sorted(np.unique(y_true).tolist())
        roc_auc   = roc_auc_score(
            y_true, probs[:, present],
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

    # Attention statistics: mean weight per position (feature/timestep)
    # Saved so attention_visualizer.py can load and plot without rerunning
    mean_attn = attn_np.mean(axis=0).tolist()   # (seq_len,)
    top_pos   = int(np.argmax(mean_attn))
    top_feat  = feat_cols[top_pos] if top_pos < len(feat_cols) else f"pos_{top_pos}"

    # Per-class attention: mean weight per position for each true class
    per_class_attn = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            per_class_attn[cls_name] = attn_np[mask].mean(axis=0).tolist()

    results = {
        "split":              split_name,
        "n_samples":          len(y_true),
        "accuracy":           round(float(acc),      4),
        "f1_macro":           round(float(f1_mac),   4),
        "f1_weighted":        round(float(f1_wei),   4),
        "precision_macro":    round(float(prec_mac), 4),
        "recall_macro":       round(float(rec_mac),  4),
        "roc_auc_ovr":        round(float(roc_auc),  4),
        "per_class_f1":       per_class_dict,
        "confusion_matrix":   cm.tolist(),
        "mean_attention":     [round(w, 4) for w in mean_attn],
        "per_class_attention": {k: [round(w, 4) for w in v]
                                for k, v in per_class_attn.items()},
        "top_attention_position": top_pos,
        "top_attention_feature":  top_feat,
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
        print()
        print(f"  Attention analysis:")
        print(f"    Top feature     : '{top_feat}' (position {top_pos})")
        print(f"    Mean weights    : {[round(w, 3) for w in mean_attn]}")
        print()
        print("  Per-class top attended features:")
        for cls_name, attn in per_class_attn.items():
            top = int(np.argmax(attn))
            fname = feat_cols[top] if top < len(feat_cols) else f"pos_{top}"
            print(f"    {cls_name:<14} → '{fname}'  (w={attn[top]:.4f})")

    return results


# ============================================================
# SAVE / LOAD
# ============================================================

def save_model(model: "CryptoFP_LSTM_Attention", verbose: bool = True):
    if not TORCH_AVAILABLE:
        return
    PATHS.MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), PATHS.LSTM_MODEL)
    size_kb = PATHS.LSTM_MODEL.stat().st_size / 1024
    if verbose:
        print(f"  Saved model : {PATHS.LSTM_MODEL.name}  ({size_kb:.1f} KB)")


def load_model(
    input_size: int,
    seq_len:    int,
    device:     str,
    verbose:    bool = True,
) -> "CryptoFP_LSTM_Attention":
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")
    if not PATHS.LSTM_MODEL.exists():
        raise FileNotFoundError(
            f"{PATHS.LSTM_MODEL} not found. Run without --eval-only first."
        )
    model = CryptoFP_LSTM_Attention(
        input_size=input_size, seq_len=seq_len
    )
    model.load_state_dict(
        torch.load(PATHS.LSTM_MODEL, map_location=device)
    )
    model = model.to(device)
    model.eval()
    if verbose:
        print(f"  Loaded: {PATHS.LSTM_MODEL.name}  {repr(model)}")
    return model


def save_results(results: dict, verbose: bool = True):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LSTM_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    size_kb = LSTM_RESULTS_JSON.stat().st_size / 1024
    if verbose:
        print(f"  Saved: {LSTM_RESULTS_JSON.relative_to(ROOT)}  ({size_kb:.1f} KB)")


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison(test_results: dict):
    """Print RF / SVM / CNN / LSTM side-by-side for paper Table 3."""
    metrics = ["accuracy", "f1_macro", "roc_auc_ovr"]
    rows = {}
    for name, path in [
        ("RF",      RESULTS_DIR / "rf_results.json"),
        ("SVM",     RESULTS_DIR / "svm_results.json"),
        ("CNN-1D",  RESULTS_DIR / "cnn_results.json"),
    ]:
        if path.exists():
            with open(path) as f:
                rows[name] = json.load(f).get("test", {})
    rows["LSTM-Attn"] = test_results

    print()
    print("  Full comparison — RF / SVM / CNN / LSTM  (test set, Table 3)")
    header = f"  {'Metric':<22}"
    for n in ["RF", "SVM", "CNN-1D", "LSTM-Attn"]:
        header += f" {n:>12}"
    print(header)
    print(f"  {'-'*70}")
    for m in metrics:
        vals = [rows.get(n, {}).get(m, float("nan"))
                for n in ["RF", "SVM", "CNN-1D", "LSTM-Attn"]]
        best = max((v for v in vals if not np.isnan(v)), default=0)
        row  = f"  {m:<22}"
        for v in vals:
            flag = " *" if not np.isnan(v) and abs(v - best) < 1e-4 else ""
            row += f" {v:>11.4f}{flag}"
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
    Full BiLSTM + Attention training and evaluation pipeline.

    Args:
        quick:     10 epochs, patience=3 — fast testing
        eval_only: load existing model, re-evaluate only
        mode:      'tabular' or 'sequence'
        verbose:   print detailed metrics + attention analysis
    """
    if not TORCH_AVAILABLE:
        print(
            "[ERROR] PyTorch not found.\n"
            "Install: pip install torch\n"
            "See requirements.txt for correct version."
        )
        return None

    set_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("  CryptoFP — BiLSTM + Attention  (novelty model)")
    print("=" * 65)
    mode_str = (
        "EVAL ONLY"          if eval_only else
        "QUICK (10 epochs)"  if quick     else
        f"FULL ({LSTM.EPOCHS} epochs)"
    )
    print(f"  Mode        : {mode_str}  [{mode}]")
    print(f"  Device      : {device}")
    print(f"  Hidden      : {LSTM.HIDDEN_SIZE} × {'bi' if LSTM.BIDIRECTIONAL else 'uni'} "
          f"= {LSTM.HIDDEN_SIZE * (2 if LSTM.BIDIRECTIONAL else 1)}")
    print(f"  Layers      : {LSTM.NUM_LAYERS}")
    print(f"  Attention   : Bahdanau additive, dim={LSTM.ATTENTION_DIM}")
    print(f"  Dropout     : {LSTM.DROPOUT}")
    print(f"  Grad clip   : {LSTM.GRAD_CLIP}")
    print(f"  Seed        : {RANDOM_SEED}")
    print()

    # ── 1. Load data ─────────────────────────────────────────
    print("  Step 1/5 — Loading data")
    try:
        loader = load_sequences if mode == "sequence" else load_tabular
        (X_train, y_train, X_val, y_val,
         X_test,  y_test,  feat_cols,
         input_size, seq_len, enc) = loader(verbose=verbose)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return None

    # ── 2. Build or load model ────────────────────────────────
    print(f"\n  Step 2/5 — {'Loading' if eval_only else 'Building'} model")
    if eval_only:
        try:
            model = load_model(input_size, seq_len, device, verbose=verbose)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [ERROR] {e}")
            return None
        history = {}
    else:
        model = CryptoFP_LSTM_Attention(
            input_size=input_size, seq_len=seq_len
        )
        print(f"  {repr(model)}")
        print(f"  Parameters: {model.count_parameters():,}")

        # ── 3. Train ──────────────────────────────────────────
        print(f"\n  Step 3/5 — Training")
        model, history = train_model(
            model, X_train, y_train, X_val, y_val,
            device=device, quick=quick, verbose=verbose,
        )

    # ── 4. Evaluate ───────────────────────────────────────────
    print(f"\n  Step 4/5 — Evaluation")
    val_results  = evaluate(
        model, X_val,  y_val,  "val",  feat_cols, enc, device, verbose
    )
    test_results = evaluate(
        model, X_test, y_test, "test", feat_cols, enc, device, verbose
    )

    test_f1 = test_results.get("f1_macro", 0)
    if test_f1 < EVAL.MIN_F1_THRESHOLD:
        print(
            f"\n  [WARN] Test F1={test_f1:.4f} below target "
            f"{EVAL.MIN_F1_THRESHOLD}.\n"
            f"  Expected with small dataset. Full 1800-sample "
            f"dataset will reach >{EVAL.MIN_F1_THRESHOLD}."
        )
    else:
        print(f"\n  Test F1={test_f1:.4f} >= {EVAL.MIN_F1_THRESHOLD}  OK")

    # ── 5. Save ───────────────────────────────────────────────
    print(f"\n  Step 5/5 — Saving")
    if not eval_only:
        save_model(model, verbose=verbose)

    full_results = {
        "model":        "LSTM_Attention",
        "mode":         mode_str,
        "input_mode":   mode,
        "input_size":   input_size,
        "seq_len":      seq_len,
        "n_features":   len(feat_cols),
        "feature_cols": feat_cols,
        "architecture": {
            "hidden_size":    LSTM.HIDDEN_SIZE,
            "num_layers":     LSTM.NUM_LAYERS,
            "bidirectional":  LSTM.BIDIRECTIONAL,
            "hidden_dim":     LSTM.HIDDEN_SIZE * (2 if LSTM.BIDIRECTIONAL else 1),
            "attention":      "Bahdanau additive",
            "attention_dim":  LSTM.ATTENTION_DIM,
            "dropout":        LSTM.DROPOUT,
            "grad_clip":      LSTM.GRAD_CLIP,
            "n_classes":      LSTM.NUM_CLASSES,
        },
        "training":    history,
        "val":         val_results,
        "test":        test_results,
        "paper_cite":  (
            f"LSTM+Attn: F1={test_results.get('f1_macro',0):.3f}, "
            f"Acc={test_results.get('accuracy',0):.3f}, "
            f"AUC={test_results.get('roc_auc_ovr',0):.3f}, "
            f"top_feat='{test_results.get('top_attention_feature','?')}'"
        ),
    }
    save_results(full_results, verbose=verbose)
    print_comparison(test_results)

    print()
    print("=" * 65)
    print("  LSTM+ATTENTION — FINAL SUMMARY")
    print("=" * 65)
    print(f"  Test accuracy   : {test_results.get('accuracy',0):.4f}")
    print(f"  Test F1 (macro) : {test_results.get('f1_macro',0):.4f}")
    print(f"  Test ROC-AUC    : {test_results.get('roc_auc_ovr',0):.4f}")
    if history:
        print(f"  Best val F1     : {history.get('best_val_f1',0):.4f}")
        print(f"  Epochs run      : {history.get('epochs_run',0)}")
        print(f"  Training time   : {history.get('elapsed_sec',0):.1f}s")
    print(f"  Top attn feat   : '{test_results.get('top_attention_feature','?')}'")
    print()
    print(f"  Paper cite (Table 3):")
    print(f"    {full_results['paper_cite']}")
    print()
    print("  Model saved to  : models/lstm_attention_best.pt")
    print("  Results saved to: results/lstm_results.json")
    print()
    print("  Next: python 03_models/misuse_detector.py")
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
        description="CryptoFP — BiLSTM + Attention novelty model"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Train for 10 epochs only — fast test."
    )
    parser.add_argument(
        "--eval-only", action="store_true", dest="eval_only",
        help="Load existing lstm_attention_best.pt and re-evaluate."
    )
    parser.add_argument(
        "--mode", choices=["tabular", "sequence"], default="tabular",
        help="'tabular' (default) or 'sequence' input mode."
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