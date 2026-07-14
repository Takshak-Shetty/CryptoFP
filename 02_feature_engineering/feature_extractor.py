# ============================================================
# CryptoFP — 02_feature_engineering/feature_extractor.py
#
# Computes derived features from raw benchmark measurements
# and adds them to master_dataset.csv in-place.
#
# Raw features (already in master_dataset.csv):
#   keygen_time_ms, enc_time_ms, dec_time_ms,
#   memory_peak_kb, cpu_percent, timing_variance
#
# Derived features added by this script:
#   enc_dec_ratio     — enc_time / dec_time
#                       RSA: dec >> enc (private key op is slow)
#                       Kyber: enc ≈ dec (KEM is symmetric)
#                       → strongest discriminating feature per SHAP
#
#   memory_delta_kb   — memory_peak_kb - per-algo minimum baseline
#                       captures memory growth relative to idle state
#                       rather than absolute RSS which varies by OS
#
#   keygen_enc_ratio  — keygen_time / enc_time
#                       RSA: keygen >> enc (prime generation is costly)
#                       Kyber: keygen ≈ enc (lattice ops are uniform)
#                       → second strongest discriminating feature
#
#   total_time_ms     — keygen + enc + dec combined
#                       overall operation cost — useful for PQC
#                       readiness scoring in pqc_readiness_score.py
#
#   log_keygen_ms     — log1p(keygen_time_ms)
#                       RSA keygen spans 3 orders of magnitude
#                       (7ms to 1148ms) — log transform normalises
#                       this for linear models (SVM)
#
#   log_dec_ms        — log1p(dec_time_ms)
#                       same reason — RSA dec varies widely with
#                       key size; Kyber dec is near-constant
#
# Note on timing_variance:
#   Already computed in benchmark_rsa.py / benchmark_kyber.py
#   (variance of REPEAT_PER_SAMPLE encrypt runs). This script
#   does NOT recompute it — it is kept as-is from collection.
#
# Output:
#   Overwrites data/processed/master_dataset.csv with 6 new
#   columns. All existing columns are preserved unchanged.
#
# Usage:
#   python 02_feature_engineering/feature_extractor.py
#   python 02_feature_engineering/feature_extractor.py --dry-run
#   python 02_feature_engineering/feature_extractor.py --plot
# ============================================================

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import PATHS, FEATURES, CLASS_NAMES, RANDOM_SEED


# ============================================================
# DERIVED FEATURE DEFINITIONS
# ============================================================
# Each entry is a dict describing one derived feature.
# formula(df) receives the full DataFrame and returns a Series.
# This structure makes it trivial to add new features — just
# append a new dict and the rest of the pipeline picks it up.

DERIVED_FEATURE_DEFS = [

    {
        "name": "enc_dec_ratio",
        "formula": lambda df: df["enc_time_ms"] / df["dec_time_ms"],
        "description": (
            "Encryption-to-decryption time ratio. "
            "RSA: dec >> enc → ratio < 1.0. "
            "Kyber: enc ≈ dec → ratio near 1.0. "
            "Strong discriminating feature."
        ),
        "dtype": "float64",
        "clip": (1e-6, 1e6),    # guard against extreme values
    },

    {
        "name": "memory_delta_kb",
        "formula": lambda df: (
            df["memory_peak_kb"]
            - df.groupby("algo")["memory_peak_kb"].transform("min")
        ),
        "description": (
            "Peak memory minus per-algo minimum baseline. "
            "Captures memory growth relative to idle rather "
            "than absolute RSS, which varies by OS and Python version."
        ),
        "dtype": "float64",
        "clip": (0.0, None),    # delta cannot be negative
    },

    {
        "name": "keygen_enc_ratio",
        "formula": lambda df: df["keygen_time_ms"] / df["enc_time_ms"],
        "description": (
            "Key generation to encryption time ratio. "
            "RSA: keygen >> enc (prime search is expensive) → ratio >> 1. "
            "Kyber: keygen ≈ enc (uniform lattice ops) → ratio near 1. "
            "Second strongest discriminating feature."
        ),
        "dtype": "float64",
        "clip": (1e-6, 1e6),
    },

    {
        "name": "total_time_ms",
        "formula": lambda df: (
            df["keygen_time_ms"] + df["enc_time_ms"] + df["dec_time_ms"]
        ),
        "description": (
            "Total operation time: keygen + enc + dec. "
            "Used in PQC readiness scoring as the overall cost metric."
        ),
        "dtype": "float64",
        "clip": (0.0, None),
    },

    {
        "name": "log_keygen_ms",
        "formula": lambda df: np.log1p(df["keygen_time_ms"]),
        "description": (
            "log1p(keygen_time_ms). RSA keygen spans 3 orders of "
            "magnitude (7ms–1148ms) due to prime search. Log transform "
            "normalises this skewed distribution for SVM and other "
            "linear models. Kyber keygen is near-constant so log "
            "compression has minimal effect there."
        ),
        "dtype": "float64",
        "clip": (0.0, None),
    },

    {
        "name": "log_dec_ms",
        "formula": lambda df: np.log1p(df["dec_time_ms"]),
        "description": (
            "log1p(dec_time_ms). RSA decryption varies significantly "
            "with key size (0.3ms for 1024-bit, 5.6ms for 4096-bit). "
            "Kyber decapsulation is near-constant across variants."
        ),
        "dtype": "float64",
        "clip": (0.0, None),
    },

]

# Names of all derived features (used for validation)
DERIVED_NAMES = [d["name"] for d in DERIVED_FEATURE_DEFS]

# Complete ALL_FEATURES list after extraction
# (config.py FEATURES.ALL_FEATURES + log features)
FINAL_FEATURE_COLS = FEATURES.ALL_FEATURES + ["log_keygen_ms", "log_dec_ms"]


# ============================================================
# CORE EXTRACTION FUNCTION
# ============================================================

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features and add them as new columns.
    Does not modify any existing columns.

    Each formula is applied with safe division / clipping to
    prevent NaN or Inf values from propagating into model training.

    Args:
        df: master_dataset DataFrame with raw features present

    Returns:
        Copy of df with derived feature columns appended.
    """
    df = df.copy()

    for feat_def in DERIVED_FEATURE_DEFS:
        name    = feat_def["name"]
        clip    = feat_def.get("clip", (None, None))
        dtype   = feat_def.get("dtype", "float64")

        try:
            series = feat_def["formula"](df)

            # Guard: replace inf values before clipping
            series = series.replace([np.inf, -np.inf], np.nan)

            # Fill NaN with column median (robust to outliers)
            # NaN can arise from 0/0 in ratio features
            if series.isna().any():
                median_val = series.median()
                n_nan = series.isna().sum()
                series = series.fillna(median_val)
                print(
                    f"  [WARN] '{name}': {n_nan} NaN values filled "
                    f"with median ({median_val:.6f})"
                )

            # Clip to valid range
            lo, hi = clip
            if lo is not None or hi is not None:
                series = series.clip(lower=lo, upper=hi)

            # Cast to target dtype
            series = series.astype(dtype)

            df[name] = series

        except Exception as e:
            print(f"  [ERROR] Feature '{name}' computation failed: {e}")
            # Fill with zeros so downstream steps don't break
            df[name] = 0.0

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_features(df: pd.DataFrame, verbose: bool = True) -> list:
    """
    Validate derived features after extraction.
    Returns list of warning strings — empty means clean.
    """
    warnings = []

    for name in DERIVED_NAMES:
        if name not in df.columns:
            warnings.append(f"Feature '{name}' missing from DataFrame.")
            continue

        col = df[name]

        # No NaN
        n_nan = col.isna().sum()
        if n_nan > 0:
            warnings.append(f"'{name}': {n_nan} NaN values remain.")

        # No Inf
        n_inf = np.isinf(col).sum()
        if n_inf > 0:
            warnings.append(f"'{name}': {n_inf} Inf values.")

        # Constant column (zero variance) — useless for ML
        if col.std() == 0:
            warnings.append(
                f"'{name}': zero variance — constant value {col.iloc[0]:.4f}. "
                f"Check formula or data."
            )

        # Reasonable range checks
        if name == "enc_dec_ratio":
            out_of_range = ((col < 0) | (col > 1000)).sum()
            if out_of_range > 0:
                warnings.append(
                    f"'{name}': {out_of_range} values outside [0, 1000]."
                )

        if name in ("log_keygen_ms", "log_dec_ms"):
            negative = (col < 0).sum()
            if negative > 0:
                warnings.append(f"'{name}': {negative} negative values (log should be ≥ 0).")

    # Cross-feature sanity: RSA enc_dec_ratio should be < Kyber ratio
    # (RSA: enc < dec → ratio < 1; Kyber: enc ≈ dec → ratio near 1)
    if "enc_dec_ratio" in df.columns:
        rsa_ratio   = df[df["algo"] == "RSA"]["enc_dec_ratio"].mean()
        kyber_ratio = df[df["algo"] == "Kyber"]["enc_dec_ratio"].mean()
        if rsa_ratio is not np.nan and kyber_ratio is not np.nan:
            if rsa_ratio >= kyber_ratio and len(df) >= 100:
                warnings.append(
                    f"Unexpected: RSA enc_dec_ratio ({rsa_ratio:.3f}) >= "
                    f"Kyber enc_dec_ratio ({kyber_ratio:.3f}). "
                    f"Check benchmark data quality."
                )

    return warnings


# ============================================================
# FEATURE REPORT
# ============================================================

def print_feature_report(df: pd.DataFrame):
    """
    Print per-class feature statistics.
    Shows how well each derived feature discriminates classes —
    large between-class differences = strong ML signal.
    This analysis belongs in your paper's methodology section.
    """
    print()
    print("=" * 70)
    print("  Derived Feature Report — discrimination power by class")
    print("=" * 70)

    all_feat_cols = FEATURES.ALL_FEATURES + ["log_keygen_ms", "log_dec_ms"]
    feat_cols     = [c for c in all_feat_cols if c in df.columns]

    for feat in feat_cols:
        print(f"\n  {feat}")
        print(f"  {'Class':<14} {'Mean':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*58}")
        for cls in CLASS_NAMES:
            cls_df = df[df["label_algo_key"] == cls]
            if len(cls_df) == 0:
                continue
            col = cls_df[feat]
            print(
                f"  {cls:<14} "
                f"{col.mean():>12.4f} "
                f"{col.std():>10.4f} "
                f"{col.min():>10.4f} "
                f"{col.max():>10.4f}"
            )

        # Between-class range as a discrimination score
        class_means = df.groupby("label_algo_key")[feat].mean()
        score = (class_means.max() - class_means.min()) / (class_means.std() + 1e-9)
        signal = "HIGH" if score > 5 else "MEDIUM" if score > 1 else "LOW"
        print(f"  discrimination score: {score:.2f}  [{signal}]")

    print()
    print("  Feature matrix summary:")
    print(f"    Rows     : {len(df):,}")
    print(f"    Features : {len(feat_cols)}  ({len(FEATURES.RAW_FEATURES)} raw + "
          f"{len(DERIVED_NAMES)} derived)")
    print(f"    Classes  : {df['label_algo_key'].nunique()}")
    print(f"    Columns  : {len(df.columns)}")
    print("=" * 70)


# ============================================================
# OPTIONAL PLOT
# ============================================================

def plot_feature_distributions(df: pd.DataFrame):
    """
    Generate distribution plots for each derived feature,
    coloured by class. Saved to paper/figures/ for the paper.

    Requires matplotlib + seaborn (already in requirements.txt).
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  [SKIP] matplotlib/seaborn not available — skipping plots.")
        return

    PATHS.FIGURES.mkdir(parents=True, exist_ok=True)

    plot_features = ["enc_dec_ratio", "keygen_enc_ratio",
                     "log_keygen_ms", "log_dec_ms",
                     "memory_delta_kb", "total_time_ms"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    palette = {
        "RSA_1024":   "#E24B4A",
        "RSA_2048":   "#D85A30",
        "RSA_4096":   "#BA7517",
        "Kyber_512":  "#1D9E75",
        "Kyber_768":  "#378ADD",
        "Kyber_1024": "#534AB7",
    }

    for i, feat in enumerate(plot_features):
        if feat not in df.columns:
            continue
        ax = axes[i]
        for cls in CLASS_NAMES:
            cls_df = df[df["label_algo_key"] == cls]
            if len(cls_df) == 0:
                continue
            vals = cls_df[feat].dropna()
            ax.hist(
                vals, bins=20, alpha=0.5,
                label=cls, color=palette.get(cls, "#888"),
                density=True,
            )
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

    handles = [
        plt.Rectangle((0,0),1,1, color=palette.get(c,"#888"), alpha=0.6)
        for c in CLASS_NAMES
    ]
    fig.legend(handles, CLASS_NAMES, loc="lower center",
               ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Derived Feature Distributions by Class", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = PATHS.FIGURES / "feature_distributions.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(ROOT)}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(dry_run: bool = False, plot: bool = False, verbose: bool = True):
    """
    Load master_dataset.csv, compute derived features, save back.
    Called by run_pipeline.py Phase 2, or directly from CLI.

    Args:
        dry_run: compute features but do not save
        plot:    generate distribution plots
        verbose: print progress

    Returns:
        DataFrame with all features, or None on error.
    """
    print("=" * 65)
    print("  CryptoFP — Feature Extractor")
    print("=" * 65)

    # ── load ─────────────────────────────────────────────────
    if not PATHS.MASTER_DATASET.exists():
        print(
            "  [ERROR] master_dataset.csv not found.\n"
            "  Run Phase 1 first:\n"
            "    python 01_data_collection/merge_dataset.py"
        )
        return None

    print(f"  Loading: {PATHS.MASTER_DATASET.relative_to(ROOT)}")
    df = pd.read_csv(PATHS.MASTER_DATASET)
    print(f"  Rows: {len(df):,}  |  Columns before: {len(df.columns)}")

    # Check required raw features are present
    missing_raw = [c for c in FEATURES.RAW_FEATURES if c not in df.columns]
    if missing_raw:
        print(f"  [ERROR] Missing raw feature columns: {missing_raw}")
        print("  Re-run merge_dataset.py to regenerate master_dataset.csv")
        return None

    # Skip features already present (idempotent re-run)
    already_present = [n for n in DERIVED_NAMES if n in df.columns]
    if already_present and not dry_run:
        print(
            f"  [INFO] {len(already_present)} derived feature(s) already present: "
            f"{already_present}"
        )
        print("  Re-computing all derived features (overwrite mode).")

    # ── extract ──────────────────────────────────────────────
    print(f"\n  Computing {len(DERIVED_FEATURE_DEFS)} derived features:")
    for d in DERIVED_FEATURE_DEFS:
        print(f"    + {d['name']:<22}  {d['description'][:55]}...")

    df_enriched = extract_features(df)
    n_new_cols  = len(df_enriched.columns) - len(df.columns)
    print(f"\n  Columns after: {len(df_enriched.columns)}  (+{n_new_cols} derived)")

    # ── validate ─────────────────────────────────────────────
    print("\n  Validating derived features...")
    warnings = validate_features(df_enriched, verbose=verbose)
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")
    else:
        print("  All feature validation checks passed.")

    # ── report ───────────────────────────────────────────────
    if verbose:
        print_feature_report(df_enriched)

    # ── plot ─────────────────────────────────────────────────
    if plot:
        print("\n  Generating distribution plots...")
        plot_feature_distributions(df_enriched)

    # ── save ─────────────────────────────────────────────────
    if not dry_run:
        df_enriched.to_csv(PATHS.MASTER_DATASET, index=False)
        size_kb = PATHS.MASTER_DATASET.stat().st_size / 1024
        print(
            f"\n  Saved: {PATHS.MASTER_DATASET.relative_to(ROOT)}  "
            f"({size_kb:.1f} KB)"
        )
        print(f"  Final schema: {len(df_enriched):,} rows × "
              f"{len(df_enriched.columns)} columns")
        print(f"  Feature cols: {FINAL_FEATURE_COLS}")
    else:
        print("\n  Dry run — file not written.")
        print("  Derived feature preview:")
        preview_cols = ["label_algo_key"] + DERIVED_NAMES
        print(
            df_enriched[[c for c in preview_cols if c in df_enriched.columns]]
            .head(8)
            .round(4)
            .to_string()
        )

    return df_enriched


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — compute derived features for master_dataset.csv"
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Compute features but do not save to disk."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate and save feature distribution plots."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-class feature report."
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run, plot=args.plot, verbose=not args.quiet)