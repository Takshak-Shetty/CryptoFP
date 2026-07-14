# ============================================================
# CryptoFP — 01_data_collection/system_load_controller.py
#
# Injects controlled CPU background load during benchmark
# collection so the dataset contains samples from four real
# system conditions: idle, low (~25%), medium (~50%), high (~75%).
#
# This is critical for a publishable dataset. A dataset collected
# only at idle is not representative of real-world deployment
# conditions. IEEE reviewers will ask about this.
#
# Usage:
#   python 01_data_collection/system_load_controller.py         # self-test
#   python 01_data_collection/system_load_controller.py --quick # fast test
#   python 01_data_collection/system_load_controller.py --verify-env
# ============================================================

import os
import sys
import time
import argparse
import multiprocessing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psutil
from config import COLLECTION, RANDOM_SEED

# ============================================================
# LOAD LEVEL DEFINITIONS
# ============================================================

LOAD_TARGETS = {
    "idle":   0.00,
    "low":    0.25,
    "medium": 0.50,
    "high":   0.75,
}

CPU_COUNT = psutil.cpu_count(logical=True) or 1

def _worker_count(level):
    target = LOAD_TARGETS.get(level, 0.0)
    if target == 0.0:
        return 0
    return max(1, round(target * CPU_COUNT))

WORKER_COUNTS = {level: _worker_count(level) for level in LOAD_TARGETS}


# ============================================================
# BACKGROUND WORKER
# ============================================================

def _cpu_burn_worker(stop_event, worker_id):
    """
    Worker process: runs CPU-bound arithmetic until stop_event set.
    Runs at reduced OS priority so the benchmark process is favoured.
    """
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass

    x = 1
    while not stop_event.is_set():
        for i in range(1, 500):
            x = (x * i + i) % 999983
        time.sleep(0.0001)


# ============================================================
# CONTROLLER CLASS
# ============================================================

class SystemLoadController:
    """
    Context manager that maintains a target CPU load level by
    running N background worker processes during a benchmark window.

    Usage:
        with SystemLoadController("medium") as ctrl:
            row = collect_one_sample(key_size, "medium")

        # OR:
        ctrl = SystemLoadController("high", verbose=True)
        ctrl.start()
        row = collect_one_sample(key_size, "high")
        ctrl.stop()

    Args:
        level:   "idle" | "low" | "medium" | "high"
        verbose: print start/stop messages
    """

    def __init__(self, level="idle", verbose=False):
        if level not in LOAD_TARGETS:
            raise ValueError(
                f"Unknown load level '{level}'. "
                f"Choose from: {list(LOAD_TARGETS.keys())}"
            )
        self.level     = level
        self.verbose   = verbose
        self.n_workers = WORKER_COUNTS[level]
        self._workers  = []
        self._stop_evt = None
        self._running  = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def start(self):
        """Launch background workers. No-op for idle level."""
        if self._running:
            return

        if self.n_workers == 0:
            self._running = True
            if self.verbose:
                print("  [LoadCtrl] idle — no workers launched.")
            return

        self._stop_evt = multiprocessing.Event()
        self._workers  = []

        for i in range(self.n_workers):
            p = multiprocessing.Process(
                target=_cpu_burn_worker,
                args=(self._stop_evt, i),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Warm-up: let workers reach steady state before measurement
        time.sleep(0.3)
        self._running = True

        if self.verbose:
            target_pct = int(LOAD_TARGETS[self.level] * 100)
            print(
                f"  [LoadCtrl] Started {self.n_workers} worker(s) "
                f"for '{self.level}' load (target ~{target_pct}% CPU)"
            )

    def stop(self):
        """Signal workers to stop and wait for clean exit."""
        if not self._running:
            return

        if self._stop_evt is not None:
            self._stop_evt.set()

        for p in self._workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)

        self._workers  = []
        self._stop_evt = None
        self._running  = False

        if self.verbose:
            print(f"  [LoadCtrl] Stopped workers for '{self.level}'.")

    def measure_achieved_load(self, duration=2.0):
        """
        Measure actual system CPU% while workers are running.
        Call after start() to calibrate on your hardware.

        Returns:
            Mean CPU utilization (0.0-100.0) over the window.
        """
        if not self._running:
            raise RuntimeError("Call start() before measuring.")

        psutil.cpu_percent(interval=None)   # prime counter
        time.sleep(0.2)

        samples = []
        step    = 0.2
        for _ in range(max(1, int(duration / step))):
            samples.append(psutil.cpu_percent(interval=step))

        return sum(samples) / len(samples) if samples else 0.0

    @property
    def is_running(self):
        return self._running

    def __repr__(self):
        return (
            f"SystemLoadController(level='{self.level}', "
            f"n_workers={self.n_workers}, running={self._running})"
        )


# ============================================================
# FACTORY FUNCTION
# ============================================================

def get_controller(level, verbose=False):
    """
    Convenience factory. Preferred import for benchmark scripts.

    Example:
        from system_load_controller import get_controller

        with get_controller("medium"):
            row = collect_one_sample(key_size, "medium")
    """
    return SystemLoadController(level=level, verbose=verbose)


# ============================================================
# ENVIRONMENT VERIFICATION
# ============================================================

def verify_environment():
    """
    Detect whether this environment supports meaningful CPU
    load injection. Some sandboxed/throttled environments
    report near-zero cpu_percent() regardless of actual load.

    Returns:
        dict with environment diagnostics and recommendation.
    """
    report = {
        "cpu_count_logical":  CPU_COUNT,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_freq_mhz":       getattr(psutil.cpu_freq(), "current", "unknown"),
        "worker_counts":      WORKER_COUNTS,
        "load_measurement_works": False,
        "achieved_single_worker_pct": 0.0,
        "recommendation": "",
    }

    stop = multiprocessing.Event()
    p    = multiprocessing.Process(
        target=_cpu_burn_worker, args=(stop, 0), daemon=True
    )
    p.start()
    time.sleep(0.4)

    psutil.cpu_percent(interval=None)
    measured = psutil.cpu_percent(interval=1.0)

    stop.set()
    p.join(timeout=2)
    if p.is_alive():
        p.terminate()

    report["achieved_single_worker_pct"] = round(measured, 1)
    report["load_measurement_works"]     = measured > 5.0

    if measured > 5.0:
        report["recommendation"] = (
            "Environment supports CPU load measurement. "
            "Load variation in your dataset will be meaningful."
        )
    else:
        report["recommendation"] = (
            "CPU measurement appears throttled (sandboxed/container env). "
            "Workers still run and samples are labeled correctly, "
            "but cpu_percent values may cluster near 0 for all levels. "
            "Collect final paper data on a bare-metal machine or "
            "an unthrottled VM for meaningful load variation."
        )

    return report


# ============================================================
# SELF-TEST
# ============================================================

def self_test(quick=False):
    """
    Test all four load levels and print a calibration table.
    Run this on your collection machine before gathering the
    full dataset to verify load levels are achievable.

    Args:
        quick: shorter measurement windows (faster, less accurate)
    """
    measure_secs = 1.5 if quick else 3.0

    print("=" * 60)
    print("  CryptoFP — SystemLoadController self-test")
    print("=" * 60)
    print(f"  Logical CPUs    : {CPU_COUNT}")
    print(f"  Physical CPUs   : {psutil.cpu_count(logical=False)}")
    print(f"  CPU freq (MHz)  : {getattr(psutil.cpu_freq(), 'current', 'N/A')}")
    print(f"  Measure window  : {measure_secs}s per level")
    print(f"  Worker counts   : {WORKER_COUNTS}")
    print()

    print("  Running environment check...")
    env = verify_environment()
    print(f"  Load measurement works : {env['load_measurement_works']}")
    print(f"  Single-worker CPU%     : {env['achieved_single_worker_pct']}%")
    print(f"  Note: {env['recommendation']}")
    print()

    print(f"  {'Level':<10} {'Workers':<10} {'Target %':<12} "
          f"{'Achieved %':<14} {'Status'}")
    print(f"  {'-'*58}")

    results = {}

    for level in ["idle", "low", "medium", "high"]:
        target_pct = LOAD_TARGETS[level] * 100
        n_workers  = WORKER_COUNTS[level]

        ctrl = SystemLoadController(level=level, verbose=False)
        ctrl.start()

        try:
            achieved = ctrl.measure_achieved_load(duration=measure_secs)
        finally:
            ctrl.stop()

        # Status: OK if idle (always passes) or achieved >= 50% of target
        on_target = (
            level == "idle"
            or achieved >= target_pct * 0.5
        )
        status = "OK" if on_target else "LOW"

        results[level] = {
            "target":   target_pct,
            "achieved": achieved,
            "workers":  n_workers,
            "ok":       on_target,
        }

        print(
            f"  {level:<10} {n_workers:<10} {target_pct:<12.0f} "
            f"{achieved:<14.1f} {status}"
        )

        time.sleep(0.5)   # let system settle between levels

    print()
    all_ok = all(r["ok"] for r in results.values())

    if all_ok:
        print("  All load levels functional.")
        print("  Safe to run full data collection with load variation.")
    else:
        low_levels = [l for l, r in results.items() if not r["ok"]]
        print(f"  Note: {low_levels} levels below 50% of target.")
        print("  This is normal in sandboxed environments.")
        print("  Samples are labeled correctly — load variation in")
        print("  cpu_percent column may be limited on this machine.")

    print()
    print("  Integration pattern for benchmark scripts:")
    print()
    print("    from system_load_controller import get_controller")
    print()
    print("    for i in range(n_samples):")
    print("        level = load_cycle[i]")
    print("        with get_controller(level):")
    print("            row = collect_one_sample(key_size, level)")
    print("            rows.append(row)")
    print()
    print("  Tip: for faster collection, batch by load level:")
    print()
    print("    for level in ['idle', 'low', 'medium', 'high']:")
    print("        batch = [s for s in load_cycle if s == level]")
    print("        with get_controller(level):")
    print("            for lbl in batch:")
    print("                rows.append(collect_one_sample(key_size, lbl))")
    print()
    print("=" * 60)

    return results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — CPU load controller self-test"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Shorter measurement windows (faster, less accurate)."
    )
    parser.add_argument(
        "--verify-env", action="store_true", dest="verify_env",
        help="Check if this environment supports CPU load measurement."
    )
    parser.add_argument(
        "--level", choices=["idle", "low", "medium", "high"],
        help="Test a single load level only."
    )
    args = parser.parse_args()

    if args.verify_env:
        print("\n  Verifying environment...\n")
        report = verify_environment()
        for k, v in report.items():
            print(f"  {k:<40}: {v}")
        sys.exit(0)

    if args.level:
        print(f"\n  Testing single level: '{args.level}'")
        ctrl = SystemLoadController(args.level, verbose=True)
        ctrl.start()
        achieved = ctrl.measure_achieved_load(duration=2.0)
        ctrl.stop()
        target = LOAD_TARGETS[args.level] * 100
        print(f"  Target: {target:.0f}%  |  Achieved: {achieved:.1f}%")
        sys.exit(0)

    self_test(quick=args.quick)