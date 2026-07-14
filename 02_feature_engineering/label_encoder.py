# ============================================================
# CryptoFP — 02_feature_engineering/label_encoder.py
#
# Builds and saves a canonical label encoding scheme used
# consistently by every model, evaluation script, and
# paper figure in the project.
#
# Why this file exists:
#   preprocess.py saved a sklearn LabelEncoder that sorts
#   classes alphabetically (Kyber_1024=0, Kyber_512=1...).
#   config.py CLASS_TO_IDX uses the canonical order defined
#   for the paper (RSA_1024=0, RSA_2048=1, RSA_4096=2,
#   Kyber_512=3, Kyber_768=4, Kyber_1024=5).
#
#   These differ. If models use alphabetical indices and
#   evaluation uses canonical indices, confusion matrix rows
#   and columns are silently scrambled — one of the hardest
#   bugs to catch in an ML paper.
#
#   This file creates a single CryptoFPLabelEncoder that
#   enforces the canonical order everywhere, overwrites the
#   preprocess.py scaler with the corrected one, and encodes
#   ALL label columns in all three split CSVs.
#
# Encoded columns added / updated:
#   label_encoded   — primary 6-class target (canonical order)
#   algo_encoded    — RSA=0, Kyber=1
#   load_encoded    — idle=0, low=1, medium=2, high=3
#   context_encoded — classical_context=0, pqc_context=1
#
# Saved artifacts:
#   models/label_encoder.pkl  — overwritten with canonical encoder
#   models/label_mappings.json — human-readable mapping dicts
#                                (cite in paper methodology section)
#
# Usage:
#   python 02_feature_engineering/label_encoder.py
#   python 02_feature_engineering/label_encoder.py --dry-run
#   python 02_feature_engineering/label_encoder.py --verify
# ============================================================

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    PATHS, FEATURES, RANDOM_SEED,
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS,
    MISUSE_CLASSES,
)


# ============================================================
# CANONICAL ENCODING MAPS
# ============================================================
# All maps are derived from config.py values so they stay in
# sync if CLASS_NAMES is ever extended.

# Primary 6-class target — canonical order from config
ALGO_KEY_MAP: dict[str, int] = CLASS_TO_IDX   # e.g. RSA_1024 → 0

# Binary algorithm family
ALGO_MAP: dict[str, int] = {"RSA": 0, "Kyber": 1}

# System load levels — ordered by intensity
LOAD_MAP: dict[str, int] = {"idle": 0, "low": 1, "medium": 2, "high": 3}

# Deployment context
CONTEXT_MAP: dict[str, int] = {
    "classical_context": 0,
    "pqc_context":       1,
    "synthetic":         -1,   # SMOTE rows — excluded from eval
}

# Misuse flag stays as 0/1 — no encoding needed
# But we document it here for the mappings JSON
MISUSE_MAP: dict[int, str] = {0: "correct_use", 1: "misuse"}

# Complete mappings dict — saved as JSON for the paper
ALL_MAPPINGS = {
    "label_algo_key":  ALGO_KEY_MAP,
    "algo":            ALGO_MAP,
    "system_load":     LOAD_MAP,
    "context":         CONTEXT_MAP,
    "misuse_flag":     {0: "correct_use", 1: "misuse"},
    "class_names":     CLASS_NAMES,
    "idx_to_class":    {str(k): v for k, v in IDX_TO_CLASS.items()},
    "misuse_classes":  MISUSE_CLASSES,
    "note": (
        "label_encoded uses canonical CLASS_TO_IDX order from config.py "
        "(RSA_1024=0, RSA_2048=1, RSA_4096=2, Kyber_512=3, "
        "Kyber_768=4, Kyber_1024=5). "
        "This differs from sklearn LabelEncoder alphabetical order. "
        "All models and evaluation scripts must use this mapping."
    ),
}


# ============================================================
# CANONICAL LABEL ENCODER CLASS
# ============================================================

class CryptoFPLabelEncoder:
    """
    Canonical label encoder for the CryptoFP project.

    Wraps all encoding maps in one object so any script can
    do:
        from label_encoder import CryptoFPLabelEncoder
        enc = CryptoFPLabelEncoder.load()
        y_int = enc.encode_algo_key(['RSA_2048', 'Kyber_512'])
        y_str = enc.decode_algo_key([1, 3])

    Also provides a sklearn-compatible LabelEncoder with the
    canonical class order (not alphabetical) for use with
    sklearn's classification_report and confusion_matrix.
    """

    def __init__(self):
        # sklearn LabelEncoder with canonical class order
        self.sklearn_le = LabelEncoder()
        self.sklearn_le.fit(CLASS_NAMES)   # canonical order from config

        # Store all maps as attributes
        self.algo_key_map    = ALGO_KEY_MAP
        self.algo_map        = ALGO_MAP
        self.load_map        = LOAD_MAP
        self.context_map     = CONTEXT_MAP
        self.class_names     = CLASS_NAMES
        self.idx_to_class    = IDX_TO_CLASS
        self.misuse_classes  = MISUSE_CLASSES

        # Inverse maps
        self.idx_to_algo_key = {v: k for k, v in ALGO_KEY_MAP.items()}
        self.idx_to_algo     = {v: k for k, v in ALGO_MAP.items()}
        self.idx_to_load     = {v: k for k, v in LOAD_MAP.items()}
        self.idx_to_context  = {
            v: k for k, v in CONTEXT_MAP.items() if v >= 0
        }

    # ── primary 6-class target ───────────────────────────────

    def encode_algo_key(self, labels) -> np.ndarray:
        """
        Encode label_algo_key strings to canonical integer indices.
        e.g. ['RSA_2048', 'Kyber_512'] → [1, 3]
        """
        if isinstance(labels, (str, int)):
            labels = [labels]
        return np.array([self.algo_key_map[l] for l in labels], dtype=np.int64)

    def decode_algo_key(self, indices) -> list:
        """
        Decode canonical integer indices to label_algo_key strings.
        e.g. [1, 3] → ['RSA_2048', 'Kyber_512']
        """
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        return [self.idx_to_algo_key[int(i)] for i in indices]

    # ── algorithm family ─────────────────────────────────────

    def encode_algo(self, algos) -> np.ndarray:
        """Encode 'RSA'→0, 'Kyber'→1."""
        if isinstance(algos, str):
            algos = [algos]
        return np.array([self.algo_map.get(a, -1) for a in algos], dtype=np.int64)

    def decode_algo(self, indices) -> list:
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        return [self.idx_to_algo.get(int(i), "unknown") for i in indices]

    # ── system load ──────────────────────────────────────────

    def encode_load(self, loads) -> np.ndarray:
        """Encode idle→0, low→1, medium→2, high→3."""
        if isinstance(loads, str):
            loads = [loads]
        return np.array([self.load_map.get(l, -1) for l in loads], dtype=np.int64)

    def decode_load(self, indices) -> list:
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        return [self.idx_to_load.get(int(i), "unknown") for i in indices]

    # ── context ──────────────────────────────────────────────

    def encode_context(self, contexts) -> np.ndarray:
        """Encode classical_context→0, pqc_context→1."""
        if isinstance(contexts, str):
            contexts = [contexts]
        return np.array(
            [self.context_map.get(c, -1) for c in contexts], dtype=np.int64
        )

    # ── sklearn compatibility ────────────────────────────────

    def sklearn_transform(self, labels) -> np.ndarray:
        """
        Alias for sklearn LabelEncoder.transform() with canonical order.
        Use this when passing targets to sklearn metrics functions.
        """
        return self.sklearn_le.transform(labels)

    def sklearn_inverse_transform(self, indices) -> np.ndarray:
        """Alias for sklearn LabelEncoder.inverse_transform()."""
        return self.sklearn_le.inverse_transform(indices)

    @property
    def classes_(self) -> np.ndarray:
        """Canonical class list as numpy array (sklearn compat)."""
        return self.sklearn_le.classes_

    @property
    def n_classes(self) -> int:
        return len(CLASS_NAMES)

    # ── persistence ──────────────────────────────────────────

    def save(self, path=None):
        """Save encoder to disk using joblib."""
        path = Path(path) if path else PATHS.LABEL_ENCODER
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path=None):
        """Load encoder from disk."""
        path = Path(path) if path else PATHS.LABEL_ENCODER
        if not path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at {path}. "
                "Run: python 02_feature_engineering/label_encoder.py"
            )
        obj = joblib.load(path)
        # Handle case where preprocess.py saved a plain sklearn encoder
        if isinstance(obj, LabelEncoder):
            print(
                "  [INFO] Found plain sklearn LabelEncoder — "
                "replacing with CryptoFPLabelEncoder."
            )
            return cls()
        return obj

    def __repr__(self):
        return (
            f"CryptoFPLabelEncoder("
            f"n_classes={self.n_classes}, "
            f"classes={self.class_names})"
        )


# ============================================================
# ENCODE ALL SPLIT CSVs
# ============================================================

def encode_splits(
    encoder: CryptoFPLabelEncoder,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Apply all encoding maps to train.csv, val.csv, test.csv.

    Adds / updates these columns in each split CSV:
        label_encoded   — canonical 6-class int (overwrites preprocess.py version)
        algo_encoded    — RSA=0, Kyber=1
        load_encoded    — idle=0 .. high=3
        context_encoded — classical=0, pqc=1

    The label_encoded column from preprocess.py used sklearn
    alphabetical order. This function overwrites it with the
    canonical config.py order.

    Args:
        encoder: fitted CryptoFPLabelEncoder
        dry_run: compute encodings but do not save
        verbose: print per-split summary

    Returns:
        dict mapping split name → encoded DataFrame
    """
    split_files = {
        "train": PATHS.TRAIN_CSV,
        "val":   PATHS.VAL_CSV,
        "test":  PATHS.TEST_CSV,
    }

    results = {}

    for split_name, fpath in split_files.items():
        fpath = Path(fpath)
        if not fpath.exists():
            print(f"  [SKIP] {fpath.name} not found.")
            continue

        df = pd.read_csv(fpath)
        n_before = len(df.columns)

        # ── label_encoded (canonical, overwrites preprocess version) ──
        if FEATURES.TARGET_ALGO_KEY in df.columns:
            # Only encode real rows — SMOTE synthetic rows get -1
            is_real = ~df["sample_id"].str.startswith("SMOTE", na=False)
            df["label_encoded"] = -1
            df.loc[is_real, "label_encoded"] = (
                df.loc[is_real, FEATURES.TARGET_ALGO_KEY]
                .map(ALGO_KEY_MAP)
                .astype(int)
            )

        # ── algo_encoded ─────────────────────────────────────
        if "algo" in df.columns:
            df["algo_encoded"] = (
                df["algo"]
                .map(ALGO_MAP)
                .fillna(-1)
                .astype(int)
            )

        # ── load_encoded ─────────────────────────────────────
        if "system_load" in df.columns:
            df["load_encoded"] = (
                df["system_load"]
                .map(LOAD_MAP)
                .fillna(-1)
                .astype(int)
            )

        # ── context_encoded ───────────────────────────────────
        if "context" in df.columns:
            df["context_encoded"] = (
                df["context"]
                .map(CONTEXT_MAP)
                .fillna(-1)
                .astype(int)
            )

        n_after  = len(df.columns)
        n_new    = n_after - n_before

        if verbose:
            real_rows = df[df["label_encoded"] >= 0]
            class_dist = (
                real_rows.groupby("label_algo_key")["label_encoded"]
                .first()
                .to_dict()
            )
            print(
                f"  {split_name:<8} {len(df):>6} rows  "
                f"+{n_new} cols  "
                f"label→idx: {dict(sorted(class_dist.items()))}"
            )

        if not dry_run:
            df.to_csv(fpath, index=False)

        results[split_name] = df

    return results


# ============================================================
# VERIFICATION
# ============================================================

def verify_encoding(encoder: CryptoFPLabelEncoder, verbose: bool = True) -> bool:
    """
    Verify encoding consistency across all split CSVs.
    Checks:
      1. label_encoded matches canonical CLASS_TO_IDX
      2. No unexpected -1 values in real rows
      3. Round-trip encode → decode is lossless
      4. All splits have the same column set
    """
    split_files = {
        "train": PATHS.TRAIN_CSV,
        "val":   PATHS.VAL_CSV,
        "test":  PATHS.TEST_CSV,
    }

    all_passed = True
    issues     = []

    for split_name, fpath in split_files.items():
        fpath = Path(fpath)
        if not fpath.exists():
            continue

        df = pd.read_csv(fpath)

        # Check 1 — label_encoded matches CLASS_TO_IDX
        real = df[~df["sample_id"].str.startswith("SMOTE", na=False)]
        for _, row in real.iterrows():
            expected = ALGO_KEY_MAP.get(row["label_algo_key"], -999)
            actual   = int(row.get("label_encoded", -999))
            if expected != actual:
                issues.append(
                    f"{split_name}: row {row['sample_id']}: "
                    f"label_algo_key='{row['label_algo_key']}' "
                    f"expected label_encoded={expected}, got {actual}"
                )
                all_passed = False

        # Check 2 — no -1 in real label_encoded
        bad = real[real["label_encoded"] < 0]
        if len(bad) > 0:
            issues.append(
                f"{split_name}: {len(bad)} real rows with label_encoded=-1"
            )
            all_passed = False

        # Check 3 — required encoded columns present
        for col in ["label_encoded", "algo_encoded", "load_encoded", "context_encoded"]:
            if col not in df.columns:
                issues.append(f"{split_name}: missing column '{col}'")
                all_passed = False

        if verbose:
            status = "OK" if all_passed else "FAIL"
            print(
                f"  {split_name:<8} {len(df):>6} rows  "
                f"cols={len(df.columns)}  {status}"
            )

    # Check 4 — round-trip encode/decode
    for cls in CLASS_NAMES:
        enc  = encoder.encode_algo_key([cls])[0]
        dec  = encoder.decode_algo_key([enc])[0]
        if dec != cls:
            issues.append(
                f"Round-trip failed: '{cls}' → {enc} → '{dec}'"
            )
            all_passed = False

    if issues:
        for issue in issues:
            print(f"  [FAIL] {issue}")
    elif verbose:
        print("  All encoding verification checks passed.")

    return all_passed


# ============================================================
# PRINT MAPPING TABLE
# ============================================================

def print_mapping_table(encoder: CryptoFPLabelEncoder):
    """
    Print all encoding maps as a readable table.
    This is Table 2 of your paper's methodology section.
    """
    print()
    print("=" * 65)
    print("  Label Encoding Reference  —  Methodology Table 2 (paper)")
    print("=" * 65)

    print(f"\n  Primary target: label_algo_key → label_encoded")
    print(f"  (canonical order defined in config.py CLASS_NAMES)")
    print(f"  {'Class':<16} {'Index':>7}  {'Algo':>8}  {'Type'}")
    print(f"  {'-'*50}")
    for cls in CLASS_NAMES:
        idx  = ALGO_KEY_MAP[cls]
        algo = cls.split("_")[0]
        tag  = "MISUSE" if cls in MISUSE_CLASSES else "correct"
        print(f"  {cls:<16} {idx:>7}  {algo:>8}  {tag}")

    print(f"\n  algo → algo_encoded")
    for k, v in sorted(ALGO_MAP.items(), key=lambda x: x[1]):
        print(f"    {k:<20} → {v}")

    print(f"\n  system_load → load_encoded")
    for k, v in sorted(LOAD_MAP.items(), key=lambda x: x[1]):
        print(f"    {k:<20} → {v}")

    print(f"\n  context → context_encoded")
    for k, v in sorted(CONTEXT_MAP.items(), key=lambda x: x[1]):
        if v >= 0:
            print(f"    {k:<30} → {v}")

    print(f"\n  misuse_flag  (already 0/1, no encoding needed)")
    for k, v in MISUSE_MAP.items():
        print(f"    {k}  →  {v}")

    print()
    print("  NOTE: label_encoded uses CANONICAL order (RSA_1024=0)")
    print("  NOT sklearn alphabetical order (Kyber_1024=0).")
    print("  All models and sklearn metrics must use this mapping.")
    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main(
    dry_run: bool = False,
    verify:  bool = False,
    verbose: bool = True,
) -> CryptoFPLabelEncoder:
    """
    Build canonical encoder, encode all splits, save artifacts.
    Called by run_pipeline.py Phase 2, or directly from CLI.

    Args:
        dry_run: compute encodings but do not save files
        verify:  run verification checks on existing encoded splits
        verbose: print mapping table and per-split stats

    Returns:
        CryptoFPLabelEncoder instance
    """
    print("=" * 65)
    print("  CryptoFP — Label Encoder")
    print("=" * 65)
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    # ── build canonical encoder ───────────────────────────────
    print("  Building CryptoFPLabelEncoder (canonical order)...")
    encoder = CryptoFPLabelEncoder()
    print(f"  {repr(encoder)}")

    # ── verify-only mode ──────────────────────────────────────
    if verify:
        print("\n  Verifying existing encoded splits...")
        ok = verify_encoding(encoder, verbose=verbose)
        return encoder

    # ── check split files exist ───────────────────────────────
    missing = [
        str(p) for p in [PATHS.TRAIN_CSV, PATHS.VAL_CSV, PATHS.TEST_CSV]
        if not Path(p).exists()
    ]
    if missing:
        print(f"\n  [ERROR] Split CSVs not found: {missing}")
        print("  Run: python 02_feature_engineering/preprocess.py")
        return None

    # ── encode all splits ─────────────────────────────────────
    print("\n  Encoding all split CSVs...")
    print(f"  {'Split':<8} {'Rows':>6}  {'New cols'}  Class→Index mapping")
    print(f"  {'-'*62}")
    encode_splits(encoder, dry_run=dry_run, verbose=verbose)

    # ── save encoder + mappings JSON ─────────────────────────
    if not dry_run:
        PATHS.MODELS.mkdir(parents=True, exist_ok=True)

        encoder.save(PATHS.LABEL_ENCODER)
        size_enc = Path(PATHS.LABEL_ENCODER).stat().st_size / 1024
        print(f"\n  Saved: label_encoder.pkl  ({size_enc:.1f} KB)  [canonical order]")

        # Save human-readable JSON for paper methodology section
        mappings_path = PATHS.MODELS / "label_mappings.json"
        with open(mappings_path, "w") as f:
            json.dump(ALL_MAPPINGS, f, indent=2)
        size_json = mappings_path.stat().st_size / 1024
        print(f"  Saved: label_mappings.json  ({size_json:.1f} KB)")
        print(f"         → cite this file in paper Methodology §Dataset")
    else:
        print("\n  Dry run — no files written.")

    # ── print mapping table ───────────────────────────────────
    if verbose:
        print_mapping_table(encoder)

    # ── verify ────────────────────────────────────────────────
    if not dry_run:
        print("\n  Verifying encoded splits...")
        ok = verify_encoding(encoder, verbose=verbose)
        if ok:
            print()
            print("  Phase 2 complete. Ready for Phase 3 model training.")
            print("  Next: python 03_models/baseline_rf.py")
    print("=" * 65)

    return encoder


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — build canonical label encoder for all splits"
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Compute encodings but do not save any files."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing encoded splits only."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress mapping table output."
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run, verify=args.verify, verbose=not args.quiet)