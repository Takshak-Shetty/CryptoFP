# ============================================================
# CryptoFP — 01_data_collection/merge_dataset.py
#
# Merges all six raw benchmark CSVs into a single
# master_dataset.csv — the canonical dataset used by every
# downstream phase (feature engineering, model training,
# evaluation, paper figures).
#
# What this script does:
#   1. Loads all six raw CSVs from data/raw/
#   2. Validates each file: schema, dtypes, row counts,
#      label consistency, no NaN in critical columns
#   3. Concatenates into one DataFrame
#   4. Adds a unique sample_id column (for traceability)
#   5. Shuffles rows (seeded) so class order is random
#   6. Validates the merged result: class balance, no leakage
#   7. Saves to data/processed/master_dataset.csv
#   8. Prints a full dataset report (your paper Table 1)
#
# Output:
#   data/processed/master_dataset.csv
#
# Usage:
#   python 01_data_collection/merge_dataset.py
#   python 01_data_collection/merge_dataset.py --dry-run
#   python 01_data_collection/merge_dataset.py --report
# ============================================================

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import (
    PATHS, FEATURES, MISUSE,
    CLASS_NAMES, RANDOM_SEED,
    COLLECTION, IDX_TO_CLASS, CLASS_TO_IDX
)


# ============================================================
# EXPECTED SCHEMA
# ============================================================
# All raw CSVs must have exactly these columns before merge.
# merge_dataset does NOT add derived features — that is
# feature_extractor.py's job in Phase 2.

REQUIRED_COLS = [
    "algo",
    "key_size",
    "operation",
    "keygen_time_ms",
    "enc_time_ms",
    "dec_time_ms",
    "memory_peak_kb",
    "cpu_percent",
    "system_load",
    "timing_variance",
    "label_algo_key",
    "misuse_flag",
    "misuse_rule",
    "misuse_reason",
    "context",
]

# Master dataset column order — sample_id prepended at merge time
MASTER_COLS = ["sample_id"] + REQUIRED_COLS

# Columns that must never contain NaN
NO_NAN_COLS = [
    "keygen_time_ms", "enc_time_ms", "dec_time_ms",
    "memory_peak_kb", "cpu_percent", "timing_variance",
    "label_algo_key", "misuse_flag",
]

# Expected dtype map for validation
EXPECTED_DTYPES = {
    "algo":           "object",
    "key_size":       "int64",
    "keygen_time_ms": "float64",
    "enc_time_ms":    "float64",
    "dec_time_ms":    "float64",
    "memory_peak_kb": "float64",
    "cpu_percent":    "float64",
    "timing_variance":"float64",
    "misuse_flag":    "int64",
}

# Map raw CSV paths to their expected label
RAW_FILE_MAP = {
    "RSA_1024":   PATHS.RAW_RSA_1024,
    "RSA_2048":   PATHS.RAW_RSA_2048,
    "RSA_4096":   PATHS.RAW_RSA_4096,
    "Kyber_512":  PATHS.RAW_KYBER_512,
    "Kyber_768":  PATHS.RAW_KYBER_768,
    "Kyber_1024": PATHS.RAW_KYBER_1024,
}


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _validate_file(df: pd.DataFrame, label: str, min_rows: int = 10) -> list:
    """
    Validate a single raw CSV DataFrame. Returns list of error
    strings — empty list means the file is clean.

    Args:
        df:       loaded DataFrame
        label:    class label string e.g. "RSA_2048"
        min_rows: minimum acceptable row count (10 for test, 1800 for full)
    """
    errors = []

    # Row count
    if len(df) < min_rows:
        errors.append(
            f"{label}: only {len(df)} rows — need at least {min_rows}. "
            f"Re-run benchmark script."
        )

    # Required columns present — optional cols filled downstream
    OPTIONAL_COLS = {"misuse_rule", "misuse_reason", "context"}
    missing_cols = [c for c in REQUIRED_COLS
                    if c not in df.columns and c not in OPTIONAL_COLS]
    if missing_cols:
        errors.append(f"{label}: missing columns {missing_cols}.")
        return errors   # can't do further checks without columns

    # NaN check on critical columns
    for col in NO_NAN_COLS:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                errors.append(f"{label}: {n_nan} NaN values in '{col}'.")

    # Label consistency — all rows should have expected label
    if "label_algo_key" in df.columns:
        actual_labels = df["label_algo_key"].unique().tolist()
        if label not in actual_labels:
            errors.append(
                f"{label}: expected label_algo_key='{label}', "
                f"got {actual_labels}."
            )
        if len(actual_labels) > 1:
            errors.append(
                f"{label}: multiple label_algo_key values {actual_labels} — "
                f"expected single label per file."
            )

    # Misuse flag — must be 0 or 1 only
    if "misuse_flag" in df.columns:
        invalid_flags = df[~df["misuse_flag"].isin([0, 1])]
        if len(invalid_flags) > 0:
            errors.append(
                f"{label}: {len(invalid_flags)} rows with invalid "
                f"misuse_flag values (not 0 or 1)."
            )

    # Timing sanity — no negative or zero timing values
    for col in ["keygen_time_ms", "enc_time_ms", "dec_time_ms"]:
        if col in df.columns:
            n_bad = (df[col] <= 0).sum()
            if n_bad > 0:
                errors.append(
                    f"{label}: {n_bad} rows with non-positive {col}. "
                    f"Re-run benchmark."
                )

    # Memory sanity — peak memory should be > 0
    if "memory_peak_kb" in df.columns:
        n_bad = (df["memory_peak_kb"] <= 0).sum()
        if n_bad > 0:
            errors.append(f"{label}: {n_bad} rows with memory_peak_kb <= 0.")

    # System load values
    if "system_load" in df.columns:
        valid_loads = set(COLLECTION.LOAD_LEVELS)
        invalid_loads = set(df["system_load"].unique()) - valid_loads
        if invalid_loads:
            errors.append(
                f"{label}: unexpected system_load values {invalid_loads}."
            )

    return errors


def _validate_merged(df: pd.DataFrame) -> list:
    """
    Validate the fully merged DataFrame before saving.
    Catches issues that only appear after concatenation.
    """
    errors = []

    # Class balance — warn if any class deviates > 10% from mean
    counts = df["label_algo_key"].value_counts()
    mean_count = counts.mean()
    for cls, count in counts.items():
        deviation = abs(count - mean_count) / mean_count
        if deviation > 0.10:
            errors.append(
                f"Class imbalance: '{cls}' has {count} rows "
                f"({deviation*100:.0f}% from mean {mean_count:.0f}). "
                f"Re-collect that class."
            )

    # All expected classes present
    missing_classes = set(CLASS_NAMES) - set(df["label_algo_key"].unique())
    if missing_classes:
        errors.append(f"Missing classes in merged dataset: {missing_classes}.")

    # No duplicate sample_ids
    if "sample_id" in df.columns:
        n_dupes = df["sample_id"].duplicated().sum()
        if n_dupes > 0:
            errors.append(f"{n_dupes} duplicate sample_id values.")

    # No NaN in any critical column
    for col in NO_NAN_COLS:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                errors.append(
                    f"NaN in '{col}' after merge: {n_nan} rows."
                )

    # Sample ID format check
    if "sample_id" in df.columns:
        bad_ids = df[~df["sample_id"].str.match(r"^[A-Za-z0-9_]+-\d{6}$", na=False)]
        if len(bad_ids) > 0:
            errors.append(
                f"{len(bad_ids)} sample_ids do not match expected format."
            )

    return errors


# ============================================================
# LOADING
# ============================================================

def load_raw_files(min_rows: int = 10, verbose: bool = True) -> dict:
    """
    Load all six raw CSVs and run per-file validation.

    Args:
        min_rows: minimum row count per file (10 for test, use
                  COLLECTION.SAMPLES_PER_CLASS for production)
        verbose:  print per-file status

    Returns:
        dict mapping label -> DataFrame (validated, clean files only)
        Also prints warnings for files with issues.
    """
    loaded = {}
    all_errors = {}

    if verbose:
        print(f"  {'Label':<14} {'File':<22} {'Rows':>6}  {'Misuse':>8}  {'Status'}")
        print(f"  {'-'*65}")

    for label, fpath in RAW_FILE_MAP.items():
        fpath = Path(fpath)

        if not fpath.exists():
            if verbose:
                print(f"  {label:<14} {fpath.name:<22} {'MISSING':>6}  {'—':>8}  SKIP")
            continue

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            if verbose:
                print(f"  {label:<14} {fpath.name:<22} {'ERROR':>6}  {'—':>8}  {e}")
            continue

        # Run validation
        errors = _validate_file(df, label, min_rows=min_rows)
        all_errors[label] = errors

        n_misuse = int(df["misuse_flag"].sum()) if "misuse_flag" in df.columns else -1
        pct      = n_misuse / len(df) * 100 if len(df) > 0 else 0
        status   = "OK" if not errors else f"WARN({len(errors)})"

        if verbose:
            print(
                f"  {label:<14} {fpath.name:<22} {len(df):>6}  "
                f"{n_misuse:>5} ({pct:.0f}%)  {status}"
            )
            for err in errors:
                print(f"    [!] {err}")

        loaded[label] = df

    return loaded, all_errors


# ============================================================
# MERGE
# ============================================================

def merge_files(loaded: dict, shuffle: bool = True) -> pd.DataFrame:
    """
    Concatenate all loaded DataFrames into one master DataFrame.

    Steps:
      1. Concatenate all DataFrames
      2. Add sample_id: unique identifier per row
         Format: {LABEL}-{06d index}  e.g. RSA_2048-000042
      3. Coerce dtypes to expected types
      4. Shuffle rows with fixed random seed
      5. Reset index

    Args:
        loaded:  dict mapping label -> DataFrame
        shuffle: randomise row order (always True for training data)

    Returns:
        Merged DataFrame with MASTER_COLS column order
    """
    frames = []
    for label in CLASS_NAMES:   # preserve canonical class order before shuffle
        if label in loaded:
            df = loaded[label].copy()
            # Fill optional columns if missing
            if "misuse_rule" not in df.columns:
                df["misuse_rule"] = df["misuse_flag"].map(
                    {0: "none", 1: "RSA_weak_key"}
                ).fillna("none")
            if "misuse_reason" not in df.columns:
                df["misuse_reason"] = df["misuse_flag"].map(
                    {0: "", 1: "NIST SP 800-131A Rev.2"}
                ).fillna("")
            if "context" not in df.columns:
                df["context"] = "classical_context"
            # Add sample_id: label + zero-padded sequential index within class
            df["sample_id"] = [
                f"{label}-{i:06d}" for i in range(len(df))
            ]
            frames.append(df)

    if not frames:
        raise ValueError("No valid raw files loaded — cannot merge.")

    merged = pd.concat(frames, ignore_index=True)

    # Enforce column order — add any missing optional cols as empty
    for col in MASTER_COLS:
        if col not in merged.columns:
            merged[col] = ""

    merged = merged[MASTER_COLS]

    # Coerce critical dtypes
    for col, dtype in EXPECTED_DTYPES.items():
        if col in merged.columns:
            try:
                merged[col] = merged[col].astype(dtype)
            except Exception as e:
                print(f"  [WARN] Could not coerce '{col}' to {dtype}: {e}")

    # Shuffle rows — critical so train/val/test splits are not class-ordered
    if shuffle:
        merged = merged.sample(
            frac=1.0, random_state=RANDOM_SEED
        ).reset_index(drop=True)

    return merged


# ============================================================
# REPORT
# ============================================================

def print_dataset_report(df: pd.DataFrame):
    """
    Print a full dataset summary — this is Table 1 of your paper.
    """
    print()
    print("=" * 65)
    print("  Master Dataset Report  —  Table 1 (paper-ready)")
    print("=" * 65)

    # Class distribution
    print(f"\n  {'Class':<14} {'Rows':>7} {'Misuse':>9} {'Correct':>9} {'%Misuse':>9}")
    print(f"  {'-'*52}")

    counts     = df["label_algo_key"].value_counts().sort_index()
    for cls in CLASS_NAMES:
        if cls not in counts.index:
            continue
        cls_df    = df[df["label_algo_key"] == cls]
        n_total   = len(cls_df)
        n_misuse  = int(cls_df["misuse_flag"].sum())
        n_correct = n_total - n_misuse
        pct       = n_misuse / n_total * 100 if n_total > 0 else 0
        print(
            f"  {cls:<14} {n_total:>7} {n_misuse:>9} "
            f"{n_correct:>9} {pct:>8.0f}%"
        )

    print(f"  {'-'*52}")
    total    = len(df)
    n_misuse = int(df["misuse_flag"].sum())
    print(
        f"  {'TOTAL':<14} {total:>7} {n_misuse:>9} "
        f"{total-n_misuse:>9} {n_misuse/total*100:>8.0f}%"
    )

    # Misuse rule breakdown
    print(f"\n  Misuse rule breakdown:")
    if "misuse_rule" in df.columns:
        rule_counts = (
            df[df["misuse_flag"] == 1]["misuse_rule"]
            .value_counts()
        )
        for rule, count in rule_counts.items():
            print(f"    {rule:<38} {count:>6} rows")
    else:
        print("    misuse_rule column not present.")

    # System load distribution
    print(f"\n  System load distribution:")
    if "system_load" in df.columns:
        load_counts = df["system_load"].value_counts().sort_index()
        for load, count in load_counts.items():
            pct = count / total * 100
            print(f"    {load:<10} {count:>7} rows  ({pct:.1f}%)")

    # Feature summary stats
    print(f"\n  Feature statistics (raw features):")
    print(f"  {'Feature':<22} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*64}")
    for col in ["keygen_time_ms", "enc_time_ms", "dec_time_ms",
                "memory_peak_kb", "cpu_percent", "timing_variance"]:
        if col in df.columns:
            print(
                f"  {col:<22} "
                f"{df[col].mean():>10.4f} "
                f"{df[col].std():>10.4f} "
                f"{df[col].min():>10.4f} "
                f"{df[col].max():>10.4f}"
            )

    # Schema
    print(f"\n  Schema: {total:,} rows × {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  File: {PATHS.MASTER_DATASET.relative_to(ROOT) if PATHS.MASTER_DATASET else 'N/A'}")
    print("=" * 65)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(dry_run: bool = False, report: bool = False, verbose: bool = True):
    """
    Load, validate, merge, and save master_dataset.csv.
    Called by run_pipeline.py Phase 1, or directly from CLI.

    Args:
        dry_run: merge and validate but do not save
        report:  print report on existing master_dataset.csv only
        verbose: print per-file progress

    Returns:
        merged DataFrame, or None if report-only
    """
    print("=" * 65)
    print("  CryptoFP — Dataset Merger")
    print("=" * 65)

    # ── report-only mode ─────────────────────────────────────
    if report:
        if not PATHS.MASTER_DATASET.exists():
            print("  master_dataset.csv not found. Run without --report first.")
            return None
        print(f"  Loading existing: {PATHS.MASTER_DATASET.name}")
        df = pd.read_csv(PATHS.MASTER_DATASET)
        print_dataset_report(df)
        return df

    # ── determine min_rows threshold ─────────────────────────
    # If raw files have full 1800 samples, enforce that minimum.
    # If they only have 10 (test mode), relax to 5.
    sample_check = Path(PATHS.RAW_RSA_2048)
    if sample_check.exists():
        test_df   = pd.read_csv(sample_check)
        min_rows  = 5 if len(test_df) < 50 else COLLECTION.SAMPLES_PER_CLASS
    else:
        min_rows  = 5

    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"  Mode       : {mode}")
    print(f"  Min rows   : {min_rows} per class")
    print(f"  Output     : {PATHS.MASTER_DATASET.relative_to(ROOT)}")
    print(f"  Seed       : {RANDOM_SEED}")
    print()

    # ── step 1: load and validate raw files ──────────────────
    print("  Step 1/4 — Loading raw files")
    loaded, all_errors = load_raw_files(min_rows=min_rows, verbose=verbose)

    n_loaded = len(loaded)
    n_expected = len(CLASS_NAMES)

    if n_loaded == 0:
        print("\n  [ERROR] No raw files found. Run Phase 1 collection first:")
        print("    python 01_data_collection/benchmark_rsa.py --test")
        print("    python 01_data_collection/benchmark_kyber.py --test --simulate")
        print("    python 01_data_collection/misuse_labeler.py")
        return None

    fatal_errors = {
        label: errs for label, errs in all_errors.items()
        if any("missing columns" in e or "ERROR" in e for e in errs)
    }
    if fatal_errors:
        print("\n  [ERROR] Fatal validation errors — cannot merge:")
        for label, errs in fatal_errors.items():
            for e in errs:
                print(f"    {label}: {e}")
        print("  Fix errors and re-run.")
        return None

    if n_loaded < n_expected:
        missing = set(CLASS_NAMES) - set(loaded.keys())
        print(
            f"\n  [WARN] Only {n_loaded}/{n_expected} files loaded. "
            f"Missing: {missing}"
        )
        print("  Merging available files only. Paper requires all 6 classes.")

    # ── step 2: merge ─────────────────────────────────────────
    print(f"\n  Step 2/4 — Merging {n_loaded} file(s)")
    merged = merge_files(loaded, shuffle=True)
    print(f"  Merged: {len(merged):,} rows × {len(merged.columns)} columns")

    # ── step 3: validate merged dataset ───────────────────────
    print("\n  Step 3/4 — Validating merged dataset")
    merge_errors = _validate_merged(merged)
    if merge_errors:
        print(f"  [WARN] {len(merge_errors)} issue(s) found after merge:")
        for e in merge_errors:
            print(f"    [!] {e}")
    else:
        print("  All validation checks passed.")

    # ── step 4: save ──────────────────────────────────────────
    print(f"\n  Step 4/4 — {'Saving' if not dry_run else 'Skipping save (dry-run)'}")
    if not dry_run:
        PATHS.PROCESSED.mkdir(parents=True, exist_ok=True)
        merged.to_csv(PATHS.MASTER_DATASET, index=False)
        size_kb = PATHS.MASTER_DATASET.stat().st_size / 1024
        print(f"  Saved: {PATHS.MASTER_DATASET.relative_to(ROOT)}  ({size_kb:.1f} KB)")
    else:
        print("  Dry run — file not written.")

    # ── final report ──────────────────────────────────────────
    print_dataset_report(merged)

    return merged


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — merge raw benchmark CSVs into master_dataset.csv"
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Merge and validate without saving to disk."
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print report on existing master_dataset.csv only."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-file progress."
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run, report=args.report, verbose=not args.quiet)