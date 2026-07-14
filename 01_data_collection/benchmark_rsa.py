# ============================================================
# CryptoFP — 01_data_collection/benchmark_rsa.py
#
# Collects RSA benchmark samples for key sizes 1024, 2048, 4096.
# For each sample measures:
#   - Key generation time (ms)
#   - Encryption time (ms)
#   - Decryption time (ms)
#   - Peak memory usage (KB)
#   - CPU utilization (%)
#   - Timing variance across REPEAT_PER_SAMPLE runs
#
# Output: one CSV per key size in data/raw/
#   data/raw/rsa_1024.csv
#   data/raw/rsa_2048.csv
#   data/raw/rsa_4096.csv
#
# Usage:
#   python 01_data_collection/benchmark_rsa.py
#   python 01_data_collection/benchmark_rsa.py --key-size 2048
#   python 01_data_collection/benchmark_rsa.py --samples 500 --key-size 1024
#
# Test (quick, 10 samples per key size):
#   python 01_data_collection/benchmark_rsa.py --test
# ============================================================

import os
import sys
import time
import argparse
import random
import statistics
from pathlib import Path

# ── resolve project root ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import psutil
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

from config import (
    PATHS, COLLECTION, FEATURES,
    MISUSE, RANDOM_SEED, CLASS_NAMES
)


# ============================================================
# CONSTANTS
# ============================================================

# RSA OAEP with SHA-256:
#   max plaintext = key_bytes - 2 * hash_len - 2
#                 = (key_size // 8) - 66
#
#   RSA-1024: 128 - 66 = 62 bytes max  ← config default 64 exceeds this
#   RSA-2048: 256 - 66 = 190 bytes max
#   RSA-4096: 512 - 66 = 446 bytes max
#
# We compute a safe plaintext per key size at runtime so the
# same script works for all three without manual adjustment.
OAEP_OVERHEAD = 66   # 2 * SHA-256 digest (32) + 2 bytes

def _safe_plaintext(key_size_bits: int) -> bytes:
    """Return the largest safe OAEP plaintext for this key size."""
    max_bytes = (key_size_bits // 8) - OAEP_OVERHEAD
    # Use the configured size if it fits, otherwise use the safe max
    target = min(COLLECTION.RSA_PLAINTEXT_BYTES, max_bytes)
    return b"A" * target

# Padding scheme used consistently across all RSA operations
OAEP_PADDING = padding.OAEP(
    mgf=padding.MGF1(algorithm=hashes.SHA256()),
    algorithm=hashes.SHA256(),
    label=None,
)

# CSV column order — must match master_dataset.csv schema in config.py
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
# TIMING + MEMORY HELPERS
# ============================================================

def _time_ms(func, *args, **kwargs):
    """
    Call func(*args, **kwargs) and return (result, elapsed_ms).
    Uses time.perf_counter for sub-millisecond resolution.
    """
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return result, elapsed


def _memory_peak_kb():
    """
    Return current RSS (Resident Set Size) of this process in KB.
    Called immediately after the operation to capture peak usage.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024.0


def _cpu_percent():
    """
    Snapshot CPU utilization of this process over a short interval.
    interval=0.1s gives a stable reading without slowing collection.
    """
    process = psutil.Process(os.getpid())
    return process.cpu_percent(interval=0.1)


def _timing_variance(func, *args, n=None, **kwargs):
    """
    Run func n times and return the variance of elapsed times (ms).
    This is the 'timing_variance' derived feature — higher variance
    indicates the operation is more sensitive to system conditions.
    """
    if n is None:
        n = COLLECTION.REPEAT_PER_SAMPLE
    times = []
    for _ in range(n):
        _, t = _time_ms(func, *args, **kwargs)
        times.append(t)
    return statistics.variance(times) if len(times) > 1 else 0.0


# ============================================================
# SINGLE SAMPLE COLLECTOR
# ============================================================

def collect_one_sample(key_size: int, system_load: str) -> dict:
    """
    Collect one complete benchmark sample for a given RSA key size.

    Steps:
      1. Generate RSA key pair — measure keygen_time_ms
      2. Encrypt plaintext with public key — measure enc_time_ms
      3. Decrypt ciphertext with private key — measure dec_time_ms
      4. Capture memory and CPU after decrypt (peak of the operation)
      5. Compute timing_variance over REPEAT_PER_SAMPLE encrypt runs
      6. Assign misuse_flag based on MISUSE config rules

    Args:
        key_size:    RSA key size in bits (1024, 2048, or 4096)
        system_load: current load label ("idle","low","medium","high")

    Returns:
        dict with all COLUMNS fields populated
    """
    # Compute safe plaintext for this key size at runtime.
    # RSA-1024 max = 62 bytes with OAEP SHA-256, so we can't use
    # the global 64-byte default — _safe_plaintext() handles this.
    plaintext = _safe_plaintext(key_size)

    # ── 1. Key generation ────────────────────────────────────
    (private_key, keygen_ms) = _time_ms(
        rsa.generate_private_key,
        public_exponent=65537,
        key_size=key_size,
        backend=default_backend(),
    )
    public_key = private_key.public_key()

    # ── 2. Encryption ────────────────────────────────────────
    (ciphertext, enc_ms) = _time_ms(
        public_key.encrypt,
        plaintext,
        OAEP_PADDING,
    )

    # ── 3. Decryption ────────────────────────────────────────
    (decrypted, dec_ms) = _time_ms(
        private_key.decrypt,
        ciphertext,
        OAEP_PADDING,
    )

    # Sanity check — decrypted plaintext must match original
    assert decrypted == plaintext, (
        f"RSA-{key_size} decrypt mismatch — "
        "possible library or padding configuration error."
    )

    # ── 4. Memory and CPU ────────────────────────────────────
    mem_kb  = _memory_peak_kb()
    cpu_pct = _cpu_percent()

    # ── 5. Timing variance (encrypt repeated n times) ────────
    t_var = _timing_variance(
        public_key.encrypt,
        plaintext,
        OAEP_PADDING,
    )

    # ── 6. Labels ────────────────────────────────────────────
    label_algo_key = f"RSA_{key_size}"
    misuse_flag = (
        MISUSE.MISUSE
        if key_size in MISUSE.RSA_WEAK_KEY_SIZES
        else MISUSE.CORRECT_USE
    )

    return {
        "algo":            "RSA",
        "key_size":        key_size,
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

def collect_for_key_size(
    key_size: int,
    n_samples: int,
    output_path: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Collect n_samples benchmark rows for one RSA key size and
    save to output_path as CSV.

    Load levels are distributed evenly across samples so the
    dataset has equal representation of idle/low/medium/high.
    The actual CPU stress is applied externally by
    system_load_controller.py — here we only record the label.

    Args:
        key_size:    RSA key bits
        n_samples:   number of rows to collect
        output_path: where to save the CSV
        verbose:     print progress every 100 samples

    Returns:
        DataFrame of collected samples
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Distribute load levels evenly across samples
    load_cycle = COLLECTION.LOAD_LEVELS * (
        n_samples // len(COLLECTION.LOAD_LEVELS) + 1
    )
    random.shuffle(load_cycle)
    load_cycle = load_cycle[:n_samples]

    rows = []
    errors = 0

    label = f"RSA-{key_size}"
    print(f"\n  Collecting {n_samples} samples for {label}...")
    print(f"  Output: {output_path.relative_to(ROOT)}")
    print(f"  {'Sample':<10} {'Keygen(ms)':<14} {'Enc(ms)':<12} "
          f"{'Dec(ms)':<12} {'Mem(KB)':<12} {'Load'}")
    print(f"  {'-'*70}")

    t_start = time.time()

    for i in range(n_samples):
        try:
            row = collect_one_sample(key_size, load_cycle[i])
            rows.append(row)

            # Inter-sample sleep to avoid thermal throttling
            if COLLECTION.INTER_SAMPLE_SLEEP > 0:
                time.sleep(COLLECTION.INTER_SAMPLE_SLEEP)

            # Progress print every 100 samples (and first + last)
            if verbose and (i % 100 == 0 or i == n_samples - 1):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_samples - i - 1) / rate if rate > 0 else 0
                print(
                    f"  {i+1:<10} "
                    f"{row['keygen_time_ms']:<14.3f}"
                    f"{row['enc_time_ms']:<12.3f}"
                    f"{row['dec_time_ms']:<12.3f}"
                    f"{row['memory_peak_kb']:<12.1f}"
                    f"{row['system_load']:<10}"
                    f"[{rate:.1f}/s  ETA {eta:.0f}s]"
                )

        except AssertionError as e:
            errors += 1
            print(f"  [WARNING] Sample {i+1} skipped: {e}")
            if errors > n_samples * 0.05:
                print(f"  [ERROR] Error rate exceeds 5% — stopping collection.")
                print(f"          Check your cryptography library installation.")
                break

        except Exception as e:
            errors += 1
            print(f"  [WARNING] Sample {i+1} unexpected error: {e}")

    df = pd.DataFrame(rows, columns=COLUMNS)

    # Save CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    total_time = time.time() - t_start
    print(f"\n  Done. {len(df)} samples saved in {total_time:.1f}s")
    print(f"  Errors/skipped: {errors}")
    _print_summary_stats(df, label)

    return df


def _print_summary_stats(df: pd.DataFrame, label: str):
    """Print a quick stats summary after collection completes."""
    print(f"\n  --- {label} collection summary ---")
    for col in ["keygen_time_ms", "enc_time_ms", "dec_time_ms", "memory_peak_kb"]:
        if col in df.columns:
            print(
                f"  {col:<20}  "
                f"mean={df[col].mean():.3f}  "
                f"std={df[col].std():.3f}  "
                f"min={df[col].min():.3f}  "
                f"max={df[col].max():.3f}"
            )
    misuse_count = df["misuse_flag"].sum()
    print(f"  misuse_flag=1:  {misuse_count} / {len(df)} rows")
    print(f"  label_algo_key: {df['label_algo_key'].unique().tolist()}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(key_sizes=None, n_samples=None, verbose=True):
    """
    Collect RSA benchmark data for all (or specified) key sizes.
    Called by run_pipeline.py Phase 1, or directly from CLI.

    Args:
        key_sizes: list of int key sizes, defaults to config value
        n_samples: samples per key size, defaults to config value
        verbose:   print progress

    Returns:
        dict mapping key_size → DataFrame
    """
    if key_sizes is None:
        key_sizes = COLLECTION.RSA_KEY_SIZES
    if n_samples is None:
        n_samples = COLLECTION.SAMPLES_PER_CLASS

    print("=" * 55)
    print("  CryptoFP — RSA Benchmark Collection")
    print("=" * 55)
    print(f"  Key sizes   : {key_sizes}")
    print(f"  Samples each: {n_samples}")
    print(f"  Plaintext   : {COLLECTION.RSA_PLAINTEXT_BYTES} bytes")
    print(f"  Repeat/var  : {COLLECTION.REPEAT_PER_SAMPLE} runs")
    print(f"  Output dir  : {PATHS.RAW.relative_to(ROOT)}")
    print(f"  Random seed : {RANDOM_SEED}")

    results = {}

    for key_size in key_sizes:
        output_path = PATHS.RAW / f"rsa_{key_size}.csv"

        # Skip if already collected and has enough rows
        if output_path.exists():
            existing = pd.read_csv(output_path)
            if len(existing) >= n_samples:
                print(
                    f"\n  [SKIP] rsa_{key_size}.csv already has "
                    f"{len(existing)} rows — skipping."
                )
                results[key_size] = existing
                continue
            else:
                print(
                    f"\n  [INFO] rsa_{key_size}.csv exists but only has "
                    f"{len(existing)} rows — re-collecting."
                )

        df = collect_for_key_size(
            key_size=key_size,
            n_samples=n_samples,
            output_path=output_path,
            verbose=verbose,
        )
        results[key_size] = df

    print("\n" + "=" * 55)
    print("  RSA collection complete.")
    print("  Files written:")
    for ks in key_sizes:
        p = PATHS.RAW / f"rsa_{ks}.csv"
        if p.exists():
            rows = len(pd.read_csv(p))
            size_kb = p.stat().st_size / 1024
            print(f"    rsa_{ks}.csv  —  {rows} rows  ({size_kb:.1f} KB)")
    print("=" * 55)

    return results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — RSA benchmark data collection"
    )
    parser.add_argument(
        "--key-size", type=int, choices=[1024, 2048, 4096],
        help="Collect for a single key size only."
    )
    parser.add_argument(
        "--samples", type=int,
        help=f"Samples per key size (default: {COLLECTION.SAMPLES_PER_CLASS})."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test run: 10 samples per key size."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-sample progress output."
    )
    args = parser.parse_args()

    key_sizes = [args.key_size] if args.key_size else COLLECTION.RSA_KEY_SIZES
    n_samples = 10 if args.test else (args.samples or COLLECTION.SAMPLES_PER_CLASS)

    if args.test:
        print("\n  [TEST MODE] Collecting 10 samples per key size only.")
        print("  This takes ~30 seconds. For full collection remove --test.\n")

    main(key_sizes=key_sizes, n_samples=n_samples, verbose=not args.quiet)