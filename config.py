import os
from pathlib import Path


# ============================================================
# 1. PROJECT ROOT
# ============================================================

ROOT = Path(__file__).parent.resolve()


# ============================================================
# 2. DIRECTORY PATHS
# ============================================================

class PATHS:
    DATA_COLLECTION   = ROOT / "01_data_collection"
    FEATURE_ENG       = ROOT / "02_feature_engineering"
    MODELS_DIR        = ROOT / "03_models"
    EXPLAINABILITY    = ROOT / "04_explainability"
    RESULTS           = ROOT / "05_results"

    RAW               = ROOT / "data" / "raw"
    RAW_RSA_1024      = RAW / "rsa_1024.csv"
    RAW_RSA_2048      = RAW / "rsa_2048.csv"
    RAW_RSA_4096      = RAW / "rsa_4096.csv"
    RAW_KYBER_512     = RAW / "kyber_512.csv"
    RAW_KYBER_768     = RAW / "kyber_768.csv"
    RAW_KYBER_1024    = RAW / "kyber_1024.csv"

    PROCESSED         = ROOT / "data" / "processed"
    MASTER_DATASET    = PROCESSED / "master_dataset.csv"
    TRAIN_CSV         = PROCESSED / "train.csv"
    VAL_CSV           = PROCESSED / "val.csv"
    TEST_CSV          = PROCESSED / "test.csv"

    SEQUENCES         = ROOT / "data" / "sequences"
    SEQ_TRAIN         = SEQUENCES / "sequences_train.npy"
    SEQ_VAL           = SEQUENCES / "sequences_val.npy"
    SEQ_TEST          = SEQUENCES / "sequences_test.npy"
    SEQ_LABELS_TRAIN  = SEQUENCES / "labels_train.npy"
    SEQ_LABELS_VAL    = SEQUENCES / "labels_val.npy"
    SEQ_LABELS_TEST   = SEQUENCES / "labels_test.npy"

    MODELS            = ROOT / "models"
    RF_MODEL          = MODELS / "rf_model.pkl"
    SVM_MODEL         = MODELS / "svm_model.pkl"
    CNN_MODEL         = MODELS / "cnn1d_best.pt"
    LSTM_MODEL        = MODELS / "lstm_attention_best.pt"
    MISUSE_MODEL      = MODELS / "misuse_detector.pkl"
    SCALER            = MODELS / "scaler.pkl"
    LABEL_ENCODER     = MODELS / "label_encoder.pkl"

    PAPER             = ROOT / "paper"
    FIGURES           = PAPER / "figures"
    TABLES            = PAPER / "tables"

    MLFLOW_DIR        = ROOT / "mlruns"


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

import random
import numpy as np

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ============================================================
# 4. DATA COLLECTION SETTINGS
# ============================================================

class COLLECTION:
    SAMPLES_PER_CLASS   = 1800
    REPEAT_PER_SAMPLE   = 10
    RSA_KEY_SIZES       = [1024, 2048, 4096]
    KYBER_VARIANTS      = ["Kyber512", "Kyber768", "Kyber1024"]
    OPERATIONS          = ["keygen", "encrypt", "decrypt"]
    LOAD_LEVELS         = ["idle", "low", "medium", "high"]
    RSA_PLAINTEXT_BYTES = 64
    KYBER_MSG_BYTES     = 32
    INTER_SAMPLE_SLEEP  = 0.05


# ============================================================
# 5. MISUSE LABEL RULES
# ============================================================

class MISUSE:
    RSA_WEAK_KEY_SIZES        = [1024]
    KYBER_WEAK_VARIANTS       = []
    PQC_REQUIRED_CONTEXT_FLAG = True
    CORRECT_USE               = 0
    MISUSE                    = 1


# ============================================================
# 6. FEATURE ENGINEERING SETTINGS
# ============================================================

class FEATURES:
    RAW_FEATURES = [
        "keygen_time_ms",
        "enc_time_ms",
        "dec_time_ms",
        "memory_peak_kb",
        "cpu_percent",
    ]

    DERIVED_FEATURES = [
        "enc_dec_ratio",
        "timing_variance",
        "memory_delta_kb",
        "keygen_enc_ratio",
        "total_time_ms",
    ]

    ALL_FEATURES        = RAW_FEATURES + DERIVED_FEATURES
    TARGET_ALGO_KEY     = "label_algo_key"
    TARGET_MISUSE       = "misuse_flag"

    TRAIN_RATIO         = 0.70
    VAL_RATIO           = 0.15
    TEST_RATIO          = 0.15

    SMOTE_ENABLED       = True
    SMOTE_K_NEIGHBORS   = 5


# ============================================================
# 7. SEQUENCE SETTINGS (CNN / LSTM input)
# ============================================================

class SEQUENCES:
    SEQ_LENGTH   = 50
    SEQ_FEATURES = [
        "keygen_time_ms",
        "enc_time_ms",
        "dec_time_ms",
        "memory_peak_kb",
        "timing_variance",
    ]
    STRIDE       = 10


# ============================================================
# 8. MACHINE LEARNING — BASELINE MODELS
# ============================================================

class RF:
    PARAM_GRID = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features":      ["sqrt", "log2"],
    }
    CV_FOLDS = 5
    SCORING  = "f1_macro"
    N_JOBS   = -1


class SVM:
    PARAM_GRID = {
        "C":      [0.1, 1, 10, 100],
        "gamma":  ["scale", "auto"],
        "kernel": ["rbf"],
    }
    CV_FOLDS = 5
    SCORING  = "f1_macro"
    MAX_ITER = 5000


# ============================================================
# 9. MACHINE LEARNING — DEEP LEARNING MODELS
# ============================================================

class CNN:
    IN_CHANNELS  = len(SEQUENCES.SEQ_FEATURES)
    SEQ_LEN      = SEQUENCES.SEQ_LENGTH
    NUM_FILTERS  = [64, 128, 256]
    KERNEL_SIZE  = 3
    DROPOUT      = 0.3
    NUM_CLASSES  = 6

    EPOCHS        = 50
    BATCH_SIZE    = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-4
    PATIENCE      = 10


class LSTM:
    INPUT_SIZE    = len(SEQUENCES.SEQ_FEATURES)
    SEQ_LEN       = SEQUENCES.SEQ_LENGTH
    HIDDEN_SIZE   = 128
    NUM_LAYERS    = 2
    DROPOUT       = 0.3
    BIDIRECTIONAL = True

    ATTENTION_HEADS = 4
    ATTENTION_DIM   = 64
    NUM_CLASSES     = 6

    EPOCHS        = 80
    BATCH_SIZE    = 64
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY  = 1e-4
    PATIENCE      = 15
    GRAD_CLIP     = 1.0


# ============================================================
# 10. MISUSE DETECTOR
# ============================================================

class MISUSE_MODEL:
    N_ESTIMATORS = 200
    MAX_DEPTH    = 15
    CLASS_WEIGHT = "balanced"
    THRESHOLD    = 0.40


# ============================================================
# 11. EVALUATION METRICS
# ============================================================

class EVAL:
    AVERAGE          = "macro"
    ROC_MULTICLASS   = "ovr"
    CV_FOLDS         = 5
    MIN_F1_THRESHOLD = 0.90


# ============================================================
# 12. EXPLAINABILITY SETTINGS
# ============================================================

class EXPLAINABILITY:
    SHAP_BACKGROUND_SAMPLES = 100
    SHAP_EXPLAIN_SAMPLES    = 500
    SHAP_PLOT_STYLE         = "beeswarm"
    NOISE_LEVELS            = [0.01, 0.05, 0.10, 0.20, 0.50]


# ============================================================
# 13. FIGURE / OUTPUT SETTINGS
# ============================================================

class FIGURES:
    DPI            = 300
    FORMAT         = "png"
    FIGSIZE_SINGLE = (6, 5)
    FIGSIZE_DOUBLE = (12, 5)
    COLORMAP       = "Blues"
    FONT_SIZE      = 11


# ============================================================
# 14. MLFLOW EXPERIMENT TRACKING
# ============================================================

class MLFLOW:
    EXPERIMENT_NAME = "CryptoFP"
    TRACKING_URI    = str(PATHS.MLFLOW_DIR)
    TAGS = {
        "project":      "CryptoFP",
        "paper_target": "IEEE Access",
        "task":         "crypto_fingerprinting",
    }


# ============================================================
# 15. LABEL MAPS
# ============================================================

CLASS_NAMES = [
    "RSA_1024",
    "RSA_2048",
    "RSA_4096",
    "Kyber_512",
    "Kyber_768",
    "Kyber_1024",
]

IDX_TO_CLASS  = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX  = {name: i for i, name in enumerate(CLASS_NAMES)}
MISUSE_CLASSES = ["RSA_1024"]


# ============================================================
# SELF-TEST
# ============================================================

def _create_directories():
    dirs = [
        PATHS.RAW, PATHS.PROCESSED, PATHS.SEQUENCES,
        PATHS.MODELS, PATHS.FIGURES, PATHS.TABLES, PATHS.MLFLOW_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _verify_config():
    assert FEATURES.TRAIN_RATIO + FEATURES.VAL_RATIO + FEATURES.TEST_RATIO == 1.0, \
        "ERROR: Train/val/test ratios must sum to 1.0"
    assert len(CLASS_NAMES) == CNN.NUM_CLASSES == LSTM.NUM_CLASSES, \
        "ERROR: NUM_CLASSES mismatch between CLASS_NAMES, CNN, and LSTM configs"
    assert RANDOM_SEED is not None, \
        "ERROR: RANDOM_SEED must be set for reproducibility"

    total_samples = COLLECTION.SAMPLES_PER_CLASS * len(CLASS_NAMES)

    print("=" * 55)
    print("  CryptoFP config loaded. All paths ready.")
    print("=" * 55)
    print(f"  Project root     : {ROOT}")
    print(f"  Random seed      : {RANDOM_SEED}")
    print(f"  Classes          : {CLASS_NAMES}")
    print(f"  Samples/class    : {COLLECTION.SAMPLES_PER_CLASS}")
    print(f"  Total dataset    : ~{total_samples:,} rows")
    print(f"  Sequence length  : {SEQUENCES.SEQ_LENGTH} steps")
    print(f"  Features (total) : {len(FEATURES.ALL_FEATURES)}")
    print(f"  Split            : {int(FEATURES.TRAIN_RATIO*100)} / "
          f"{int(FEATURES.VAL_RATIO*100)} / "
          f"{int(FEATURES.TEST_RATIO*100)} (train/val/test)")
    print(f"  CNN epochs       : {CNN.EPOCHS}")
    print(f"  LSTM epochs      : {LSTM.EPOCHS}")
    print(f"  Min F1 target    : {EVAL.MIN_F1_THRESHOLD}")
    print(f"  Output format    : {FIGURES.FORMAT.upper()} @ {FIGURES.DPI} DPI")
    print("=" * 55)
    print("  Directories created (or already exist):")
    for attr in ["RAW", "PROCESSED", "SEQUENCES", "MODELS", "FIGURES", "TABLES"]:
        p = getattr(PATHS, attr)
        print(f"    {attr:<12} -> {p.relative_to(ROOT)}")
    print("=" * 55)


if __name__ == "__main__":
    _create_directories()
    _verify_config()
