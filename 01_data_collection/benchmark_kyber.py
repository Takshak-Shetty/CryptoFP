"""
CryptoFP — 01_data_collection/benchmark_kyber.py

Collects Kyber benchmark samples for Kyber512, Kyber768, Kyber1024
(standardised as ML-KEM in NIST FIPS 203).

For each sample measures:
  - Key generation time (ms)      — keypair generation
  - Encryption time (ms)          — KEM encapsulation
  - Decryption time (ms)          — KEM decapsulation
  - Peak memory usage (KB)
  - CPU utilization (%)
  - Timing variance across REPEAT_PER_SAMPLE runs

Output: one CSV per Kyber variant in data/raw/
  data/raw/kyber_512.csv
  data/raw/kyber_768.csv
  data/raw/kyber_1024.csv

Requirements:
  liboqs C library  — https://github.com/open-quantum-safe/liboqs
  liboqs-python     — pip install liboqs-python

Usage:
  python 01_data_collection/benchmark_kyber.py
  python 01_data_collection/benchmark_kyber.py --variant Kyber512
  python 01_data_collection/benchmark_kyber.py --samples 500

Quick test (10 samples, no liboqs needed — uses simulation):
  python 01_data_collection/benchmark_kyber.py --test
  python 01_data_collection/benchmark_kyber.py --test --simulate
"""

import os
import sys
import time
import argparse
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import psutil

from config import (
    PATHS, COLLECTION, MISUSE,
    RANDOM_SEED, CLASS_NAMES
)


# ============================================================
# LIBOQS IMPORT — with graceful fallback
# ============================================================

OQS_AVAILABLE = False
try:
    import oqs
    OQS_AVAILABLE = True
    print(f"  [liboqs] version {oqs.oqs_version()} loaded.")
except ImportError:
    print(
        "  [WARNING] liboqs-python not found.\n"
        "  Real Kyber benchmarking requires the liboqs C library.\n"
        "  Install guide: https://github.com/open-quantum-safe/liboqs-python\n"
        "  Running with --simulate flag will generate realistic synthetic data\n"
        "  for development and testing. Do NOT use simulated data in your paper."
    )


# ============================================================
# KYBER VARIANT CONFIG
# ============================================================

KYBER_OQS_NAMES = {
    "Kyber512":  ["Kyber512",  "ML-KEM-512"],
    "Kyber768":  ["Kyber768",  "ML-KEM-768"],
    "Kyber1024": ["Kyber1024", "ML-KEM-1024"],
}

KYBER_KEY_BITS = {
    "Kyber512":  512,
    "Kyber768":  768,
    "Kyber1024": 1024,
}

# Realistic timing ranges (ms) for simulation mode — derived from
# published Kyber benchmarks on x86-64 hardware.
# Source: NIST PQC Round 3 submission benchmarks (cycles @ 3GHz)
SIMULATION_PARAMS = {
    "Kyber512": {
        "keygen": (0.02, 0.008),
        "enc":    (0.025, 0.009),
        "dec":    (0.027, 0.010),
        "mem_kb": (2100, 50),
    },
    "Kyber768": {
        "keygen": (0.035, 0.012),
        "enc":    (0.040, 0.013),
        "dec":    (0.042, 0.014),
        "mem_kb": (2400, 60),
    },
    "Kyber1024": {
        "keygen": (0.050, 0.016),
        "enc":    (0.056, 0.018),
        "dec":    (0.058, 0.019),
        "mem_kb": (2800, 70),
    },
}

COLUMNS = [
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
]


# ============================================================
# LIBOQS KEM RESOLVER
# ============================================================

def _resolve_kem_name(variant: str) -> str:
    """
    Find the correct liboqs KEM identifier for a given variant name.
    Tries all aliases in KYBER_OQS_NAMES until one is enabled.

    Raises:
        RuntimeError if no alias is found in the enabled KEM list.
    """
    if not OQS_AVAILABLE:
        raise RuntimeError("liboqs not available — use --simulate flag.")

    enabled = oqs.get_enabled_KEMs()
    for candidate in KYBER_OQS_NAMES.get(variant, [variant]):
        if candidate in enabled:
            return candidate

    raise RuntimeError(
        f"No KEM found for '{variant}' in liboqs enabled list.\n"
        f"Available KEMs: {[k for k in enabled if 'Kyber' in k or 'KEM' in k]}\n"
        f"Check your liboqs installation or try a newer version."
    )


# ============================================================
# TIMING + MEMORY HELPERS
# ============================================================

def _time_ms(func, *args, **kwargs):
    """Call func and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return result, elapsed


def _memory_peak_kb() -> float:
    """Current RSS of this process in KB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024.0


def _cpu_percent() -> float:
    """CPU utilization snapshot over 0.1s interval."""
    return psutil.Process(os.getpid()).cpu_percent(interval=0.1)


def _timing_variance_enc(kem_name: str, public_key: bytes, n: int = None) -> float:
    """
    Run Kyber encapsulation n times and return timing variance (ms).
    Uses a fresh KEM object each call to avoid state contamination.
    """
    if n is None:
        n = COLLECTION.REPEAT_PER_SAMPLE
    times = []
    for _ in range(n):
        with oqs.KeyEncapsulation(kem_name) as kem:
            t0 = time.perf_counter()
            kem.encap_secret(public_key)
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.variance(times) if len(times) > 1 else 0.0


# ============================================================
# REAL COLLECTION — liboqs
# ============================================================

def collect_one_sample_real(variant: str, kem_name: str, system_load: str) -> dict:
    """
    Collect one Kyber benchmark sample using liboqs.

    Kyber is a Key Encapsulation Mechanism (KEM), not a cipher:
      - keygen:  generate public/secret key pair
      - enc:     encapsulate → produces (ciphertext, shared_secret)
      - dec:     decapsulate ciphertext → recovers shared_secret

    The shared secrets from enc and dec must match — this is
    verified as a sanity check on every sample.
    """
    # 1. Key generation
    with oqs.KeyEncapsulation(kem_name) as kem_keygen:
        t0 = time.perf_counter()
        public_key = kem_keygen.generate_keypair()
        keygen_ms = (time.perf_counter() - t0) * 1000.0
        secret_key = kem_keygen.export_secret_key()

    # 2. Encapsulation (enc)
    with oqs.KeyEncapsulation(kem_name) as kem_enc:
        t0 = time.perf_counter()
        ciphertext, shared_secret_enc = kem_enc.encap_secret(public_key)
        enc_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Decapsulation (dec)
    with oqs.KeyEncapsulation(kem_name, secret_key=secret_key) as kem_dec:
        t0 = time.perf_counter()
        shared_secret_dec = kem_dec.decap_secret(ciphertext)
        dec_ms = (time.perf_counter() - t0) * 1000.0

    assert shared_secret_enc == shared_secret_dec, (
        f"Kyber {variant} shared secret mismatch — "
        "KEM correctness failure. Check liboqs installation."
    )

    mem_kb  = _memory_peak_kb()
    cpu_pct = _cpu_percent()
    t_var   = _timing_variance_enc(kem_name, public_key)

    key_bits       = KYBER_KEY_BITS[variant]
    label_algo_key = f"Kyber_{key_bits}"
    misuse_flag    = (
        MISUSE.MISUSE
        if variant in MISUSE.KYBER_WEAK_VARIANTS
        else MISUSE.CORRECT_USE
    )

    return {
        "algo":            "Kyber",
        "key_size":        key_bits,
        "operation":       "keygen+enc+dec",
        "keygen_time_ms":  round(keygen_ms, 6),
        "enc_time_ms":     round(enc_ms, 6),
        "dec_time_ms":     round(dec_ms, 6),
        "memory_peak_kb":  round(mem_kb, 2),
        "cpu_percent":     round(cpu_pct, 2),
        "system_load":     system_load,
        "timing_variance": round(t_var, 8),
        "label_algo_key":  label_algo_key,
        "misuse_flag":     misuse_flag,
    }


# ============================================================
# SIMULATION MODE — for development without liboqs
# ============================================================

def collect_one_sample_simulated(variant: str, system_load: str, rng) -> dict:
    """
    Generate a realistic synthetic Kyber sample without liboqs.

    Uses Gaussian noise around published benchmark medians from
    the NIST PQC Round 3 submission. Timing distributions are
    modelled per variant so RSA vs Kyber patterns remain
    distinguishable.

    IMPORTANT: Simulated data must NOT be used in the paper.
    Replace with real liboqs data before submission.
    """
    params = SIMULATION_PARAMS[variant]

    load_factor = {
        "idle":   1.00,
        "low":    1.05,
        "medium": 1.12,
        "high":   1.22,
    }.get(system_load, 1.0)

    def _sample(mean, std):
        val = rng.normal(mean * load_factor, std * load_factor)
        return max(val, mean * 0.3)

    keygen_ms = _sample(*params["keygen"])
    enc_ms    = _sample(*params["enc"])
    dec_ms    = _sample(*params["dec"])
    mem_kb    = _sample(*params["mem_kb"])
    cpu_pct   = rng.uniform(5, 35) * load_factor

    enc_times = [_sample(*params["enc"]) for _ in range(COLLECTION.REPEAT_PER_SAMPLE)]
    t_var = statistics.variance(enc_times) if len(enc_times) > 1 else 0.0

    key_bits       = KYBER_KEY_BITS[variant]
    label_algo_key = f"Kyber_{key_bits}"
    misuse_flag    = (
        MISUSE.MISUSE
        if variant in MISUSE.KYBER_WEAK_VARIANTS
        else MISUSE.CORRECT_USE
    )

    return {
        "algo":            "Kyber",
        "key_size":        key_bits,
        "operation":       "keygen+enc+dec",
        "keygen_time_ms":  round(keygen_ms, 6),
        "enc_time_ms":     round(enc_ms, 6),
        "dec_time_ms":     round(dec_ms, 6),
        "memory_peak_kb":  round(mem_kb, 2),
        "cpu_percent":     round(cpu_pct, 2),
        "system_load":     system_load,
        "timing_variance": round(t_var, 8),
        "label_algo_key":  label_algo_key,
        "misuse_flag":     misuse_flag,
    }


# ============================================================
# MAIN COLLECTION LOOP
# ============================================================

def collect_for_variant(
    variant: str,
    n_samples: int,
    output_path: Path,
    simulate: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Collect n_samples benchmark rows for one Kyber variant.

    Args:
        variant:     "Kyber512", "Kyber768", or "Kyber1024"
        n_samples:   rows to collect
        output_path: CSV save path
        simulate:    use synthetic data (no liboqs required)
        verbose:     print progress

    Returns:
        DataFrame of collected samples
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    kem_name = None
    if not simulate:
        if not OQS_AVAILABLE:
            print(
                f"\n  [WARNING] liboqs not available for {variant}.\n"
                f"  Falling back to simulation mode.\n"
            )
            simulate = True
        else:
            try:
                kem_name = _resolve_kem_name(variant)
                print(f"  [liboqs] Using KEM: {kem_name}")
            except RuntimeError as e:
                print(f"  [WARNING] {e}\n  Falling back to simulation mode.")
                simulate = True

    mode_label = "SIMULATED" if simulate else "REAL liboqs"

    load_cycle = COLLECTION.LOAD_LEVELS * (
        n_samples // len(COLLECTION.LOAD_LEVELS) + 1
    )
    random.shuffle(load_cycle)
    load_cycle = load_cycle[:n_samples]

    rows   = []
    errors = 0

    key_bits = KYBER_KEY_BITS[variant]
    label    = f"Kyber-{key_bits}"

    print(f"\n  Collecting {n_samples} samples for {label} [{mode_label}]...")
    print(f"  Output: {output_path.relative_to(ROOT)}")
    print(f"  {'Sample':<10} {'Keygen(ms)':<14} {'Enc(ms)':<12} "
          f"{'Dec(ms)':<12} {'Mem(KB)':<12} {'Load'}")
    print(f"  {'-'*70}")

    t_start = time.time()

    for i in range(n_samples):
        try:
            if simulate:
                row = collect_one_sample_simulated(variant, load_cycle[i], rng)
            else:
                row = collect_one_sample_real(variant, kem_name, load_cycle[i])

            rows.append(row)

            if COLLECTION.INTER_SAMPLE_SLEEP > 0:
                time.sleep(COLLECTION.INTER_SAMPLE_SLEEP)

            if verbose and (i % 100 == 0 or i == n_samples - 1):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta  = (n_samples - i - 1) / rate if rate > 0 else 0
                print(
                    f"  {i+1:<10} "
                    f"{row['keygen_time_ms']:<14.4f}"
                    f"{row['enc_time_ms']:<12.4f}"
                    f"{row['dec_time_ms']:<12.4f}"
                    f"{row['memory_peak_kb']:<12.1f}"
                    f"{row['system_load']:<10}"
                    f"[{rate:.1f}/s  ETA {eta:.0f}s]"
                )

        except AssertionError as e:
            errors += 1
            print(f"  [WARNING] Sample {i+1} KEM correctness check failed: {e}")
            if errors > n_samples * 0.05:
                print("  [ERROR] Error rate > 5%. Stopping. Check liboqs install.")
                break

        except Exception as e:
            errors += 1
            print(f"  [WARNING] Sample {i+1} error: {e}")

    df = pd.DataFrame(rows, columns=COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    total_time = time.time() - t_start
    print(f"\n  Done. {len(df)} samples saved in {total_time:.1f}s")
    print(f"  Mode: {mode_label}  |  Errors: {errors}")
    _print_summary_stats(df, label, simulate)

    return df


def _print_summary_stats(df: pd.DataFrame, label: str, simulated: bool):
    """Print collection summary statistics."""
    sim_note = "  *** SIMULATED DATA — replace with liboqs before paper ***" if simulated else ""
    print(f"\n  --- {label} collection summary {sim_note} ---")
    for col in ["keygen_time_ms", "enc_time_ms", "dec_time_ms", "memory_peak_kb"]:
        if col in df.columns and len(df) > 0:
            print(
                f"  {col:<20}  "
                f"mean={df[col].mean():.4f}  "
                f"std={df[col].std():.4f}  "
                f"min={df[col].min():.4f}  "
                f"max={df[col].max():.4f}"
            )

    rsa_2048_path = PATHS.RAW / "rsa_2048.csv"
    if rsa_2048_path.exists() and len(df) > 0:
        rsa_df = pd.read_csv(rsa_2048_path)
        print(f"\n  --- Quick comparison: {label} vs RSA-2048 ---")
        for col in ["keygen_time_ms", "enc_time_ms", "dec_time_ms"]:
            kyber_mean = df[col].mean()
            rsa_mean   = rsa_df[col].mean()
            ratio      = rsa_mean / kyber_mean if kyber_mean > 0 else float("inf")
            print(f"  {col:<20}  Kyber={kyber_mean:.4f}ms  RSA={rsa_mean:.4f}ms  "
                  f"(RSA is {ratio:.1f}x {'slower' if ratio > 1 else 'faster'})")

    misuse_count = df["misuse_flag"].sum() if len(df) > 0 else 0
    print(f"  misuse_flag=1:  {misuse_count} / {len(df)} rows")
    print(f"  label_algo_key: {df['label_algo_key'].unique().tolist()}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(variants=None, n_samples=None, simulate=False, verbose=True):
    """
    Collect Kyber benchmark data for all (or specified) variants.

    Args:
        variants:  list of variant names, defaults to config
        n_samples: samples per variant, defaults to config
        simulate:  use synthetic data (development/testing)
        verbose:   print progress

    Returns:
        dict mapping variant → DataFrame
    """
    if variants is None:
        variants = COLLECTION.KYBER_VARIANTS
    if n_samples is None:
        n_samples = COLLECTION.SAMPLES_PER_CLASS

    print("=" * 55)
    print("  CryptoFP — Kyber Benchmark Collection")
    print("=" * 55)
    print(f"  Variants    : {variants}")
    print(f"  Samples each: {n_samples}")
    print(f"  Message size: {COLLECTION.KYBER_MSG_BYTES} bytes")
    print(f"  Repeat/var  : {COLLECTION.REPEAT_PER_SAMPLE} runs")
    print(f"  liboqs      : {'available' if OQS_AVAILABLE else 'NOT available'}")
    print(f"  Mode        : {'REAL' if OQS_AVAILABLE and not simulate else 'SIMULATED'}")
    print(f"  Output dir  : {PATHS.RAW.relative_to(ROOT)}")
    print(f"  Random seed : {RANDOM_SEED}")

    if simulate and not OQS_AVAILABLE:
        print("\n  [NOTE] --simulate flag active: generating synthetic data.")
        print("  Install liboqs before final paper data collection.\n")

    results = {}

    for variant in variants:
        key_bits    = KYBER_KEY_BITS[variant]
        output_path = PATHS.RAW / f"kyber_{key_bits}.csv"

        if output_path.exists():
            existing = pd.read_csv(output_path)
            if len(existing) >= n_samples:
                print(
                    f"\n  [SKIP] kyber_{key_bits}.csv already has "
                    f"{len(existing)} rows — skipping."
                )
                results[variant] = existing
                continue
            else:
                print(
                    f"\n  [INFO] kyber_{key_bits}.csv exists but only has "
                    f"{len(existing)} rows — re-collecting."
                )

        df = collect_for_variant(
            variant=variant,
            n_samples=n_samples,
            output_path=output_path,
            simulate=simulate or not OQS_AVAILABLE,
            verbose=verbose,
        )
        results[variant] = df

    print("\n" + "=" * 55)
    print("  Kyber collection complete.")
    print("  Files written:")
    for variant in variants:
        key_bits = KYBER_KEY_BITS[variant]
        p = PATHS.RAW / f"kyber_{key_bits}.csv"
        if p.exists():
            df = pd.read_csv(p)
            size_kb = p.stat().st_size / 1024
            print(f"    kyber_{key_bits}.csv  —  {len(df)} rows  ({size_kb:.1f} KB)")
    print("=" * 55)

    return results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — Kyber (ML-KEM) benchmark data collection"
    )
    parser.add_argument(
        "--variant", choices=["Kyber512", "Kyber768", "Kyber1024"],
        help="Collect for a single Kyber variant only."
    )
    parser.add_argument(
        "--samples", type=int,
        help=f"Samples per variant (default: {COLLECTION.SAMPLES_PER_CLASS})."
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help=(
            "Generate synthetic data without liboqs. "
            "Use for development and pipeline testing only. "
            "NOT suitable for paper results."
        )
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test: 10 samples per variant."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-sample progress output."
    )
    args = parser.parse_args()

    variants  = [args.variant] if args.variant else COLLECTION.KYBER_VARIANTS
    n_samples = 10 if args.test else (args.samples or COLLECTION.SAMPLES_PER_CLASS)
    simulate  = args.simulate or not OQS_AVAILABLE

    if args.test:
        print("\n  [TEST MODE] 10 samples per variant.")
        print("  Remove --test for full collection.\n")

    if simulate and not args.simulate:
        print("\n  [AUTO-SIMULATE] liboqs not found — using simulation mode.")
        print("  Install liboqs for real Kyber data.\n")

    main(variants=variants, n_samples=n_samples, simulate=simulate, verbose=not args.quiet)
