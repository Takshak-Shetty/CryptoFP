# ============================================================
# CryptoFP — 01_data_collection/misuse_labeler.py
#
# Applies cryptographic misuse labels to raw benchmark CSVs.
# This is the most important file for your paper's novelty —
# it defines WHAT counts as misuse, WHY, and adds context
# labels that the misuse detection model will learn from.
#
# Three misuse rules are implemented (all driven by config.py):
#
#   Rule 1 — Weak key size (RSA)
#     RSA-1024 is deprecated by NIST since 2013. Any use of
#     RSA-1024 is flagged as misuse regardless of context.
#     Source: NIST SP 800-131A Rev.2
#
#   Rule 2 — Wrong algorithm in PQC-required context
#     When a system policy mandates post-quantum cryptography
#     (e.g. post-NIST FIPS 203 migration), using any RSA
#     variant is flagged as misuse. This models the real-world
#     migration compliance scenario.
#     Source: NIST FIPS 203 (ML-KEM, 2024)
#
#   Rule 3 — Weak Kyber variant (future-proofing)
#     Currently empty — no Kyber variants are considered weak.
#     Config-driven so you can add variants without code changes.
#
# Output:
#   - Updates misuse_flag column in each raw CSV in-place
#   - Adds misuse_reason column explaining WHY each row is flagged
#   - Adds context column ("classical_context"/"pqc_context")
#   - Prints a full audit report with row counts per rule
#
# Usage:
#   python 01_data_collection/misuse_labeler.py
#   python 01_data_collection/misuse_labeler.py --dry-run
#   python 01_data_collection/misuse_labeler.py --report-only
# ============================================================

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from config import PATHS, MISUSE, RANDOM_SEED, CLASS_NAMES


# ============================================================
# MISUSE RULE DEFINITIONS
# ============================================================
# Each rule is a dict with:
#   name      — short identifier used in misuse_reason column
#   condition — function(row) -> bool: True if row is misuse
#   reason    — human-readable explanation stored per flagged row
#   reference — NIST / standard citation for the paper

MISUSE_RULES = [

    {
        "name": "weak_rsa_key",
        "condition": lambda row: (
            row["algo"] == "RSA"
            and int(row["key_size"]) in MISUSE.RSA_WEAK_KEY_SIZES
        ),
        "reason": (
            "RSA key size below NIST minimum (2048-bit). "
            "RSA-1024 deprecated per NIST SP 800-131A Rev.2 (2019)."
        ),
        "reference": "NIST SP 800-131A Rev.2",
        "severity": "HIGH",
    },

    {
        "name": "classical_in_pqc_context",
        "condition": lambda row: (
            MISUSE.PQC_REQUIRED_CONTEXT_FLAG
            and row["algo"] == "RSA"
            and int(row["key_size"]) not in MISUSE.RSA_WEAK_KEY_SIZES
            # Only flag RSA-2048 and RSA-4096 here — RSA-1024 already
            # caught by weak_rsa_key rule above to avoid double-counting
        ),
        "reason": (
            "Classical RSA used in a post-quantum migration context. "
            "Post-NIST FIPS 203 (ML-KEM, 2024) mandates PQC for "
            "new deployments. RSA is not quantum-safe."
        ),
        "reference": "NIST FIPS 203 (ML-KEM, 2024)",
        "severity": "MEDIUM",
    },

    {
        "name": "weak_kyber_variant",
        "condition": lambda row: (
            row["algo"] == "Kyber"
            and row.get("label_algo_key", "") in [
                f"Kyber_{int(v.replace('Kyber',''))}"
                for v in MISUSE.KYBER_WEAK_VARIANTS
            ]
        ),
        "reason": (
            "Kyber variant below recommended security level."
        ),
        "reference": "NIST FIPS 203",
        "severity": "MEDIUM",
    },

]


# ============================================================
# CORE LABELING FUNCTION
# ============================================================

def apply_misuse_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all misuse rules to a DataFrame and return a labelled copy.

    Adds / updates these columns:
        misuse_flag   (int)  — 0 = correct use, 1 = misuse
        misuse_reason (str)  — explains why flagged, empty if correct
        misuse_rule   (str)  — rule name that triggered the flag
        context       (str)  — "classical_context" or "pqc_context"

    Rules are applied in order. First matching rule wins — a row
    flagged by weak_rsa_key is NOT also flagged by
    classical_in_pqc_context (no double-counting).

    Args:
        df: raw benchmark DataFrame

    Returns:
        Copy of df with misuse columns populated
    """
    df = df.copy()

    # Initialise / reset misuse columns
    df["misuse_flag"]   = MISUSE.CORRECT_USE
    df["misuse_reason"] = ""
    df["misuse_rule"]   = ""

    # Apply rules row by row
    # Using iterrows is fine here — this runs once at labeling time,
    # not inside a hot benchmark loop
    for idx, row in df.iterrows():
        for rule in MISUSE_RULES:
            try:
                if rule["condition"](row):
                    df.at[idx, "misuse_flag"]   = MISUSE.MISUSE
                    df.at[idx, "misuse_reason"] = rule["reason"]
                    df.at[idx, "misuse_rule"]   = rule["name"]
                    break   # first matching rule wins
            except Exception as e:
                # Rule evaluation error — skip this rule, log warning
                print(f"  [WARN] Rule '{rule['name']}' error on row {idx}: {e}")
                continue

    # Add context column — classifies deployment scenario
    # "pqc_context"      = PQC migration is mandated (post-FIPS 203)
    # "classical_context" = classical crypto still acceptable
    df["context"] = df["algo"].apply(
        lambda a: "pqc_context" if a == "Kyber" else "classical_context"
    )

    return df


# ============================================================
# PER-FILE PROCESSING
# ============================================================

def label_file(
    csv_path: Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Load a raw benchmark CSV, apply misuse labels, and save in-place.

    Args:
        csv_path: path to raw CSV file
        dry_run:  if True, compute labels but do not write to disk
        verbose:  print per-file summary

    Returns:
        dict with labeling statistics for this file
    """
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} — file not found.")
        return {}

    df_original = pd.read_csv(csv_path)
    df_labelled = apply_misuse_labels(df_original)

    # Compute stats for the audit report
    total     = len(df_labelled)
    n_misuse  = int(df_labelled["misuse_flag"].sum())
    n_correct = total - n_misuse
    pct_misuse = n_misuse / total * 100 if total > 0 else 0

    rule_counts = (
        df_labelled[df_labelled["misuse_flag"] == MISUSE.MISUSE]["misuse_rule"]
        .value_counts()
        .to_dict()
    )

    stats = {
        "file":       csv_path.name,
        "total":      total,
        "correct":    n_correct,
        "misuse":     n_misuse,
        "pct_misuse": round(pct_misuse, 1),
        "rules_hit":  rule_counts,
        "dry_run":    dry_run,
    }

    if verbose:
        status = "[DRY RUN]" if dry_run else "[WRITTEN]"
        print(
            f"  {status} {csv_path.name:<22} "
            f"total={total:<6} "
            f"misuse={n_misuse:<6} ({pct_misuse:.0f}%)  "
            f"rules={rule_counts if rule_counts else 'none'}"
        )

    if not dry_run:
        df_labelled.to_csv(csv_path, index=False)

    return stats


# ============================================================
# AUDIT REPORT
# ============================================================

def print_audit_report(all_stats: list):
    """
    Print a full audit report across all labelled files.
    This report should be included as Table 1 in your paper
    (dataset composition and misuse breakdown).
    """
    if not all_stats:
        print("  No files processed.")
        return

    total_rows    = sum(s.get("total", 0)   for s in all_stats)
    total_misuse  = sum(s.get("misuse", 0)  for s in all_stats)
    total_correct = sum(s.get("correct", 0) for s in all_stats)

    print()
    print("=" * 65)
    print("  Misuse Label Audit Report")
    print("=" * 65)
    print(f"  {'File':<24} {'Total':>7} {'Correct':>9} {'Misuse':>8} {'%':>6}")
    print(f"  {'-'*60}")

    for s in all_stats:
        if not s:
            continue
        print(
            f"  {s['file']:<24} "
            f"{s['total']:>7} "
            f"{s['correct']:>9} "
            f"{s['misuse']:>8} "
            f"{s['pct_misuse']:>5.0f}%"
        )

    print(f"  {'-'*60}")
    pct = total_misuse / total_rows * 100 if total_rows > 0 else 0
    print(
        f"  {'TOTAL':<24} "
        f"{total_rows:>7} "
        f"{total_correct:>9} "
        f"{total_misuse:>8} "
        f"{pct:>5.0f}%"
    )
    print("=" * 65)

    # Rules breakdown
    all_rule_counts: dict = {}
    for s in all_stats:
        for rule, count in s.get("rules_hit", {}).items():
            all_rule_counts[rule] = all_rule_counts.get(rule, 0) + count

    print()
    print("  Misuse rule breakdown:")
    for rule_def in MISUSE_RULES:
        name  = rule_def["name"]
        count = all_rule_counts.get(name, 0)
        ref   = rule_def["reference"]
        sev   = rule_def["severity"]
        print(f"    {name:<35} {count:>6} rows  [{sev}]  ref: {ref}")

    print()
    print("  Misuse columns added to each CSV:")
    print("    misuse_flag    — 0=correct, 1=misuse")
    print("    misuse_reason  — human-readable explanation per flagged row")
    print("    misuse_rule    — rule name that triggered the flag")
    print("    context        — 'classical_context' or 'pqc_context'")
    print()

    # Paper-ready summary line
    if total_rows > 0:
        print("  Paper dataset summary (Table 1 candidate):")
        print(f"    Total samples    : {total_rows:,}")
        print(f"    Correct use      : {total_correct:,}  ({total_correct/total_rows*100:.1f}%)")
        print(f"    Misuse cases     : {total_misuse:,}  ({total_misuse/total_rows*100:.1f}%)")
        print(f"    Misuse rules     : {len([r for r in MISUSE_RULES if r['name'] in all_rule_counts])}")
        print()
        if total_misuse < total_rows * 0.05:
            print("  [NOTE] Misuse class is <5% of dataset.")
            print("         SMOTE will be applied in preprocess.py to balance this.")
            print("         This is expected — real cryptographic misuse is rare.")
    print("=" * 65)


# ============================================================
# VERIFY LABELS
# ============================================================

def verify_labels(verbose: bool = True) -> bool:
    """
    Sanity-check labelled CSVs against expected misuse rules.
    Called after labeling to confirm correctness before merge.

    Returns:
        True if all checks pass, False if any fail.
    """
    checks_passed = 0
    checks_failed = 0

    expected = {
        # (algo, key_size) -> expected misuse_flag
        ("RSA",   1024): MISUSE.MISUSE,    # weak key — always misuse
        ("RSA",   2048): MISUSE.MISUSE,    # classical in PQC context
        ("RSA",   4096): MISUSE.MISUSE,    # classical in PQC context
        ("Kyber",  512): MISUSE.CORRECT_USE,
        ("Kyber",  768): MISUSE.CORRECT_USE,
        ("Kyber", 1024): MISUSE.CORRECT_USE,
    }

    raw_files = {
        ("RSA",   1024): PATHS.RAW_RSA_1024,
        ("RSA",   2048): PATHS.RAW_RSA_2048,
        ("RSA",   4096): PATHS.RAW_RSA_4096,
        ("Kyber",  512): PATHS.RAW_KYBER_512,
        ("Kyber",  768): PATHS.RAW_KYBER_768,
        ("Kyber", 1024): PATHS.RAW_KYBER_1024,
    }

    if verbose:
        print()
        print("  Verifying labels...")
        print(f"  {'File':<22} {'Expected flag':<16} {'Actual flag':<14} {'OK?'}")
        print(f"  {'-'*60}")

    for (algo, key_size), expected_flag in expected.items():
        fpath = raw_files.get((algo, key_size))
        if fpath is None or not Path(fpath).exists():
            if verbose:
                print(f"  {str(fpath.name) if fpath else 'N/A':<22} SKIP (file missing)")
            continue

        df   = pd.read_csv(fpath)
        flag = df["misuse_flag"].mode()[0] if len(df) > 0 else -1
        ok   = int(flag) == expected_flag
        sym  = "OK" if ok else "FAIL"

        if ok:
            checks_passed += 1
        else:
            checks_failed += 1

        if verbose:
            fname = Path(fpath).name
            print(
                f"  {fname:<22} "
                f"{'MISUSE' if expected_flag else 'CORRECT':<16} "
                f"{'MISUSE' if flag else 'CORRECT':<14} "
                f"{sym}"
            )

    if verbose:
        print(f"  {'-'*60}")
        print(f"  Checks passed: {checks_passed}  |  Failed: {checks_failed}")
        if checks_failed == 0:
            print("  All label checks passed.")
        else:
            print("  LABEL VERIFICATION FAILED — check misuse rules in config.py")

    return checks_failed == 0


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(dry_run: bool = False, report_only: bool = False, verbose: bool = True):
    """
    Apply misuse labels to all raw CSVs.
    Called by run_pipeline.py Phase 1, or directly from CLI.

    Args:
        dry_run:     compute and print labels without saving
        report_only: print current label stats without re-labeling
        verbose:     print per-file progress

    Returns:
        list of stats dicts, one per file
    """
    raw_files = [
        PATHS.RAW_RSA_1024,
        PATHS.RAW_RSA_2048,
        PATHS.RAW_RSA_4096,
        PATHS.RAW_KYBER_512,
        PATHS.RAW_KYBER_768,
        PATHS.RAW_KYBER_1024,
    ]

    print("=" * 65)
    print("  CryptoFP — Misuse Labeler")
    print("=" * 65)

    if dry_run:
        print("  Mode: DRY RUN — labels computed but NOT saved.")
    elif report_only:
        print("  Mode: REPORT ONLY — reading existing labels.")
    else:
        print("  Mode: LIVE — labels will be written to CSVs.")

    print()
    print("  Misuse rules active:")
    for rule in MISUSE_RULES:
        print(f"    [{rule['severity']}] {rule['name']} — {rule['reference']}")
    print()

    all_stats = []

    if report_only:
        # Just read existing flags — do not re-apply rules
        for fpath in raw_files:
            fpath = Path(fpath)
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            if "misuse_flag" not in df.columns:
                print(f"  [SKIP] {fpath.name} — no misuse_flag column yet.")
                continue
            n_misuse = int(df["misuse_flag"].sum())
            pct      = n_misuse / len(df) * 100
            rule_counts = (
                df[df["misuse_flag"] == 1]["misuse_rule"].value_counts().to_dict()
                if "misuse_rule" in df.columns else {}
            )
            all_stats.append({
                "file":       fpath.name,
                "total":      len(df),
                "correct":    len(df) - n_misuse,
                "misuse":     n_misuse,
                "pct_misuse": round(pct, 1),
                "rules_hit":  rule_counts,
                "dry_run":    False,
            })
        print_audit_report(all_stats)
        return all_stats

    # Apply labels to all files
    print(f"  {'Status':<12} {'File':<24} {'Total':>7} {'Misuse':>8}  Rules hit")
    print(f"  {'-'*65}")

    for fpath in raw_files:
        stats = label_file(Path(fpath), dry_run=dry_run, verbose=verbose)
        if stats:
            all_stats.append(stats)

    print_audit_report(all_stats)

    if not dry_run:
        verify_labels(verbose=verbose)

    return all_stats


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — apply cryptographic misuse labels to raw CSVs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Compute and print labels without saving to disk."
    )
    parser.add_argument(
        "--report-only", action="store_true", dest="report_only",
        help="Print current label statistics without re-labeling."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run label verification checks only."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-file progress output."
    )
    args = parser.parse_args()

    if args.verify:
        ok = verify_labels(verbose=True)
        sys.exit(0 if ok else 1)

    main(
        dry_run=args.dry_run,
        report_only=args.report_only,
        verbose=not args.quiet,
    )