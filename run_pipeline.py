"""
CryptoFP — Master pipeline script.
Runs all phases end-to-end in order.

Usage:
    python run_pipeline.py              # full pipeline
    python run_pipeline.py --from 2    # start from phase 2
    python run_pipeline.py --only 3    # run only phase 3
    python run_pipeline.py --quick     # fast test run (small data)
"""
import sys, argparse, time, importlib
sys.path.insert(0, ".")

from config import RANDOM_SEED, set_seed
set_seed(RANDOM_SEED)

_quick = False


def _step(label: str, fn):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    t0 = time.perf_counter()
    fn()
    print(f"  Done in {time.perf_counter()-t0:.1f}s")


def _mod(path):
    return importlib.import_module(path)


def phase1():
    from config import COLLECTION
    n = 10 if _quick else COLLECTION.SAMPLES_PER_CLASS

    _mod("01_data_collection.benchmark_rsa").main(n_samples=n, verbose=True)
    _mod("01_data_collection.benchmark_kyber").main(n_samples=n, simulate=True, verbose=True)
    _mod("01_data_collection.merge_dataset").main(verbose=True)
    _mod("01_data_collection.misuse_labeler").main(verbose=True)


def phase2():
    _mod("02_feature_engineering.feature_extractor").main(verbose=True)
    _mod("02_feature_engineering.label_encoder").main(verbose=True)
    _mod("02_feature_engineering.preprocess").main(verbose=True)


def phase3():
    _mod("03_models.baseline_rf").main(quick=_quick, verbose=True)
    _mod("03_models.baseline_svm").main(quick=_quick, verbose=True)
    _mod("03_models.misuse_detector").main(quick=_quick, verbose=True)
    _mod("03_models.cnn_1d").main(quick=True, verbose=True)
    _mod("03_models.lstm_attention").main(quick=True, verbose=True)


def phase4():
    _mod("04_explainability.shap_analysis").main()
    _mod("04_explainability.attention_visualizer").main()
    # adversarial_test.py is a standalone diagnostic — run manually if needed:
    # python3 04_explainability/adversarial_test.py


def phase5():
    _mod("05_results.plot_confusion_matrix").main(verbose=True)
    _mod("05_results.plot_roc_curves").main()
    _mod("05_results.comparison_table").main()

    # Call pqc_readiness_score directly to avoid sys.argv conflicts
    pqc = _mod("05_results.pqc_readiness_score")
    all_results = [pqc.compute_mrs(p) for p in pqc._BUILTIN_PROFILES.values()]
    for r in all_results:
        pqc._print_result(r)
    pqc._print_all_summary(all_results)
    pqc.plot_mrs(all_results, out_path="paper/figures/pqc_readiness_scores.png")
    pqc._save_json(all_results, "results")


PHASES = {1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5}
LABELS = {
    1: "Phase 1 — Data Collection",
    2: "Phase 2 — Feature Engineering",
    3: "Phase 3 — Model Training",
    4: "Phase 4 — Explainability",
    5: "Phase 5 — Results & Figures",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_phase", type=int, default=1)
    parser.add_argument("--only", dest="only_phase", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Fast test run with small data")
    args = parser.parse_args()

    _quick = args.quick

    to_run = ([args.only_phase] if args.only_phase
              else list(range(args.from_phase, 6)))

    for p in to_run:
        _step(LABELS[p], PHASES[p])

    print("\n✓ Pipeline complete.")
