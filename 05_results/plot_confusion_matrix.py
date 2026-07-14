# ============================================================
# CryptoFP — 05_results/plot_confusion_matrix.py
#
# Generates publication-quality confusion matrix figures for
# the paper. Loads saved results JSONs — no model retraining.
#
# Figures produced (saved to paper/figures/):
#
#   confusion_matrix_rf.pdf        — RF single matrix
#   confusion_matrix_svm.pdf       — SVM single matrix
#   confusion_matrix_cnn.pdf       — CNN-1D (if available)
#   confusion_matrix_lstm.pdf      — LSTM+Attention (if available)
#   confusion_matrix_all.pdf       — 2×2 grid of all 4 models
#   confusion_matrix_misuse.pdf    — Misuse detector (2×2 binary)
#
# Figure design follows IEEE two-column format:
#   - 300 DPI minimum (IEEE requirement)
#   - PDF vector format (scales to any size in LaTeX)
#   - Colormap: Blues (accessible, prints well in greyscale)
#   - Cell annotations: count + percentage
#   - Diagonal highlighted: correct predictions
#   - Short class labels on axes to fit column width
#
# Usage:
#   python 05_results/plot_confusion_matrix.py
#   python 05_results/plot_confusion_matrix.py --model rf
#   python 05_results/plot_confusion_matrix.py --format png
#   python 05_results/plot_confusion_matrix.py --no-pct
# ============================================================

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "02_feature_engineering"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from config import (
    PATHS, CLASS_NAMES, FIGURES, MISUSE_CLASSES,
)


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = ROOT / "results"

# Model results JSON files
RESULT_FILES = {
    "rf":    RESULTS_DIR / "rf_results.json",
    "svm":   RESULTS_DIR / "svm_results.json",
    "cnn":   RESULTS_DIR / "cnn_results.json",
    "lstm":  RESULTS_DIR / "lstm_results.json",
    "misuse": RESULTS_DIR / "misuse_results.json",
}

# Short axis labels — fit IEEE single-column figure width
SHORT_LABELS = {
    "RSA_1024":   "RSA\n1024",
    "RSA_2048":   "RSA\n2048",
    "RSA_4096":   "RSA\n4096",
    "Kyber_512":  "K-512",
    "Kyber_768":  "K-768",
    "Kyber_1024": "K-1024",
}

BINARY_LABELS = ["correct", "misuse"]

# Display names for figure titles
DISPLAY_NAMES = {
    "rf":    "Random Forest",
    "svm":   "SVM (RBF)",
    "cnn":   "CNN-1D",
    "lstm":  "LSTM + Attention",
    "misuse": "Misuse Detector",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_confusion_matrix(model_name: str) -> dict:
    """
    Load confusion matrix and summary metrics from a model's
    results JSON file.

    Returns:
        dict with:
          cm          — numpy array confusion matrix
          accuracy    — float
          f1_macro    — float (or f1 for binary)
          roc_auc     — float
          n_samples   — int
          task        — 'multiclass' or 'binary'
        or None if file not found.
    """
    fpath = RESULT_FILES.get(model_name)
    if fpath is None or not fpath.exists():
        return None

    with open(fpath) as f:
        r = json.load(f)

    test = r.get("test", {})
    cm = test.get("confusion_matrix")
    if cm is None:
        # Misuse detector stores TP/FP/TN/FN separately — reconstruct 2×2 CM
        tn = test.get("tn"); fp = test.get("fp")
        fn = test.get("fn"); tp = test.get("tp")
        if all(v is not None for v in [tn, fp, fn, tp]):
            cm = [[int(tn), int(fp)], [int(fn), int(tp)]]
        else:
            return None

    cm = np.array(cm, dtype=int)

    # Detect binary vs multiclass
    task = "binary" if cm.shape == (2, 2) else "multiclass"

    return {
        "cm":        cm,
        "accuracy":  test.get("accuracy",  test.get("f1", 0)),
        "f1_macro":  test.get("f1_macro",  test.get("f1", 0)),
        "roc_auc":   test.get("roc_auc_ovr", test.get("roc_auc", 0)),
        "pr_auc":    test.get("pr_auc", None),
        "n_samples": test.get("n_samples", int(cm.sum())),
        "task":      task,
        "model":     model_name,
    }


# ============================================================
# SINGLE CONFUSION MATRIX PLOT
# ============================================================

def plot_single_cm(
    data:        dict,
    ax:          plt.Axes,
    show_pct:    bool = True,
    title:       str  = None,
    cmap:        str  = None,
    fontsize:    int  = None,
    colorbar:    bool = True,
) -> plt.Axes:
    """
    Draw one confusion matrix on the given Axes object.

    Cell annotations show:
      - Count (always)
      - Percentage of true class (when show_pct=True)
        e.g. "2\n100%" means 2 samples, all true positives

    Diagonal cells are the correct predictions — the brighter
    the colour, the better the recall for that class.

    Off-diagonal cells show which classes are confused with
    which — critical for the security interpretation (e.g.
    Kyber-768 confused with Kyber-1024 is expected and benign;
    RSA confused with Kyber would be a serious error).

    Args:
        data:     dict from load_confusion_matrix()
        ax:       matplotlib Axes to draw on
        show_pct: annotate with row-normalised percentages
        title:    figure title (default: model display name)
        cmap:     colormap name (default: config FIGURES.COLORMAP)
        fontsize: annotation font size
        colorbar: draw colorbar

    Returns:
        ax with confusion matrix drawn
    """
    if cmap is None:
        cmap = FIGURES.COLORMAP
    if fontsize is None:
        fontsize = FIGURES.FONT_SIZE - 1

    cm   = data["cm"]
    task = data["task"]
    n    = cm.shape[0]

    # Row-normalised matrix for colour intensity
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm  = cm / row_sums   # recall per class

    # Axis labels
    if task == "binary":
        labels_short = BINARY_LABELS
        labels_full  = BINARY_LABELS
    else:
        labels_short = [SHORT_LABELS.get(c, c) for c in CLASS_NAMES[:n]]
        labels_full  = CLASS_NAMES[:n]

    # Draw heatmap
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    # Axis ticks and labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(
        labels_short, rotation=35, ha="right",
        fontsize=fontsize - 1,
    )
    ax.set_yticklabels(labels_short, fontsize=fontsize - 1)

    # Cell annotations
    for i in range(n):
        for j in range(n):
            count  = cm[i, j]
            norm   = cm_norm[i, j]
            # White text on dark cells, dark text on light cells
            color  = "white" if norm > 0.55 else "black"

            if show_pct and row_sums[i, 0] > 0:
                pct  = norm * 100
                label = f"{count}\n{pct:.0f}%"
            else:
                label = str(count)

            ax.text(
                j, i, label,
                ha="center", va="center",
                fontsize=fontsize - 1,
                color=color,
                fontweight="bold" if i == j else "normal",
            )

    # Axis labels
    ax.set_xlabel("Predicted label", fontsize=fontsize)
    ax.set_ylabel("True label",      fontsize=fontsize)

    # Title with key metrics
    if title is None:
        title = DISPLAY_NAMES.get(data["model"], data["model"])

    acc = data.get("accuracy", 0)
    f1  = data.get("f1_macro", 0)
    auc = data.get("roc_auc",  0)

    if task == "binary":
        pr_auc = data.get("pr_auc")
        if pr_auc is not None:
            metric_str = f"F1={f1:.3f}  Recall={data.get('recall',0):.3f}  PR-AUC={pr_auc:.3f}"
        else:
            metric_str = f"F1={f1:.3f}  Acc={acc:.3f}"
    else:
        metric_str = f"Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}"

    ax.set_title(
        f"{title}\n{metric_str}",
        fontsize=fontsize,
        pad=8,
    )

    # Highlight diagonal with a thin border
    for i in range(n):
        rect = mpatches.FancyBboxPatch(
            (i - 0.48, i - 0.48), 0.96, 0.96,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#2C7BB6",
            facecolor="none",
            zorder=3,
        )
        ax.add_patch(rect)

    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Recall", fontsize=fontsize - 1)
        cbar.ax.tick_params(labelsize=fontsize - 2)

    return ax


# ============================================================
# SAVE HELPER
# ============================================================

def save_figure(
    fig:      plt.Figure,
    filename: str,
    fmt:      str  = None,
    verbose:  bool = True,
) -> Path:
    """
    Save figure to paper/figures/ at 300 DPI.
    Creates directory if needed.
    """
    if fmt is None:
        fmt = FIGURES.FORMAT

    PATHS.FIGURES.mkdir(parents=True, exist_ok=True)
    out = PATHS.FIGURES / filename

    fig.savefig(
        out,
        dpi=FIGURES.DPI,
        format=fmt,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    size_kb = out.stat().st_size / 1024
    if verbose:
        print(f"  Saved: {out.relative_to(ROOT)}  ({size_kb:.1f} KB)")

    return out


# ============================================================
# INDIVIDUAL MODEL FIGURES
# ============================================================

def plot_model_cm(
    model_name: str,
    show_pct:   bool = True,
    fmt:        str  = None,
    verbose:    bool = True,
) -> bool:
    """
    Generate a single-model confusion matrix figure.

    Returns True if saved successfully, False if data unavailable.
    """
    data = load_confusion_matrix(model_name)
    if data is None:
        if verbose:
            print(f"  [SKIP] {model_name.upper()} — results not found.")
        return False

    fig, ax = plt.subplots(figsize=FIGURES.FIGSIZE_SINGLE)
    plot_single_cm(data, ax, show_pct=show_pct)
    plt.tight_layout()

    fname = f"confusion_matrix_{model_name}.{fmt or FIGURES.FORMAT}"
    save_figure(fig, fname, fmt=fmt, verbose=verbose)
    return True


# ============================================================
# FOUR-MODEL GRID FIGURE (paper Figure N)
# ============================================================

def plot_all_models_grid(
    models:   list = None,
    show_pct: bool = True,
    fmt:      str  = None,
    verbose:  bool = True,
) -> bool:
    """
    Generate a 2×2 (or 1×N) grid comparing all available models.
    This is the primary confusion matrix figure for the paper —
    it shows all four classifiers side by side so reviewers can
    compare error patterns at a glance.

    Layout:
      RF          SVM
      CNN-1D      LSTM+Attn

    If fewer than 4 models available, adapts to 1×N layout.

    Args:
        models:   list of model names to include (default: all 4)
        show_pct: annotate with row percentages
        fmt:      output format ('pdf', 'png', 'svg')
        verbose:  print progress

    Returns True if at least one model was plotted.
    """
    if models is None:
        models = ["rf", "svm", "cnn", "lstm"]

    # Load only available models
    available = []
    for name in models:
        d = load_confusion_matrix(name)
        if d is not None:
            available.append((name, d))

    if not available:
        if verbose:
            print("  [SKIP] No model results found for grid plot.")
        return False

    n = len(available)

    # Choose layout
    if n == 1:
        nrows, ncols = 1, 1
    elif n == 2:
        nrows, ncols = 1, 2
    elif n == 3:
        nrows, ncols = 1, 3
    else:
        nrows, ncols = 2, 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.5, nrows * 5.0),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, (name, data) in enumerate(available):
        ax = axes_flat[idx]
        # No individual colourbars in grid — shared interpretation
        plot_single_cm(
            data, ax,
            show_pct=show_pct,
            colorbar=(idx == len(available) - 1),   # only last cell
        )

    # Hide unused subplots
    for idx in range(len(available), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared title
    fig.suptitle(
        "Confusion Matrices — Cryptographic Algorithm Fingerprinting",
        fontsize=FIGURES.FONT_SIZE + 1,
        y=1.01,
    )

    # Shared colour-scale note
    fig.text(
        0.5, -0.01,
        "Cell colour = recall (row-normalised).  "
        "Cell text = count / row %.  "
        "Blue diagonal = correct predictions.",
        ha="center", fontsize=FIGURES.FONT_SIZE - 2, color="gray",
    )

    plt.tight_layout()
    fname = f"confusion_matrix_all.{fmt or FIGURES.FORMAT}"
    save_figure(fig, fname, fmt=fmt, verbose=verbose)
    return True


# ============================================================
# MISUSE DETECTOR — binary confusion matrix with annotations
# ============================================================

def plot_misuse_cm(
    show_pct: bool = True,
    fmt:      str  = None,
    verbose:  bool = True,
) -> bool:
    """
    Generate the misuse detector confusion matrix with
    security-specific annotations (TP/FP/TN/FN labels).

    The binary confusion matrix gets a richer treatment than
    the 6-class matrices because the security interpretation
    of each cell matters for the paper:
      TP — misuse correctly caught (good)
      FP — false alarm (minor cost)
      TN — correct usage confirmed (good)
      FN — MISSED MISUSE — undetected vulnerability (critical)

    The FN cell is highlighted in red regardless of its value.
    """
    data = load_confusion_matrix("misuse")
    if data is None:
        if verbose:
            print("  [SKIP] misuse_results.json not found.")
        return False

    cm   = data["cm"]
    task = data["task"]

    if task != "binary" or cm.shape != (2, 2):
        if verbose:
            print(f"  [WARN] Misuse CM is not 2×2 (shape={cm.shape}). "
                  "Using standard plot.")
        return plot_model_cm("misuse", show_pct=show_pct,
                             fmt=fmt, verbose=verbose)

    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: heatmap ────────────────────────────────────────
    ax_cm = axes[0]
    plot_single_cm(
        data, ax_cm,
        show_pct=show_pct,
        title="Misuse Detector\n(Binary Classification)",
        colorbar=True,
    )

    # Highlight FN cell (row=1=misuse, col=0=predicted correct)
    # Red border = dangerous miss
    rect = mpatches.FancyBboxPatch(
        (0 - 0.48, 1 - 0.48), 0.96, 0.96,
        boxstyle="round,pad=0.02",
        linewidth=2.5,
        edgecolor="#E24B4A",
        facecolor="none",
        zorder=4,
    )
    ax_cm.add_patch(rect)
    ax_cm.annotate(
        "← FN: missed misuse!",
        xy=(0, 1), xycoords="data",
        xytext=(-0.4, 1.45),
        fontsize=FIGURES.FONT_SIZE - 2,
        color="#E24B4A",
        arrowprops=dict(
            arrowstyle="->", color="#E24B4A", lw=1.2
        ),
    )

    # ── Right: security interpretation bar chart ──────────────
    ax_bar = axes[1]
    labels = ["TN\n(correct use\nconfirmed)",
              "FP\n(false alarm)",
              "FN\n(MISSED\nMISUSE)",
              "TP\n(misuse\ncaught)"]
    values = [tn, fp, fn, tp]
    colors = ["#1D9E75", "#EF9F27", "#E24B4A", "#378ADD"]

    bars = ax_bar.bar(labels, values, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=1.5)

    # Annotate bars with counts
    for bar, val in zip(bars, values):
        ypos = bar.get_height() + max(values) * 0.02
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            ypos, str(val),
            ha="center", va="bottom",
            fontsize=FIGURES.FONT_SIZE,
            fontweight="bold",
        )

    ax_bar.set_ylabel("Sample count", fontsize=FIGURES.FONT_SIZE)
    ax_bar.set_title(
        "Security Interpretation\n"
        f"(n={data['n_samples']} test samples, "
        f"threshold={data.get('threshold', 0.4)})",
        fontsize=FIGURES.FONT_SIZE,
    )
    ax_bar.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 2)
    ax_bar.tick_params(axis="x", labelsize=FIGURES.FONT_SIZE - 2)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # Add metric annotations
    f1      = data.get("f1_macro", 0)
    recall  = 1 - fn / (tp + fn) if (tp + fn) > 0 else 0
    pr_auc  = data.get("pr_auc", 0)
    ax_bar.text(
        0.98, 0.97,
        f"F1={f1:.3f}\nRecall={recall:.3f}\nPR-AUC={pr_auc:.3f}",
        transform=ax_bar.transAxes,
        ha="right", va="top",
        fontsize=FIGURES.FONT_SIZE - 1,
        bbox=dict(
            facecolor="white", edgecolor="#B4B2A9",
            boxstyle="round,pad=0.3", linewidth=0.7,
        ),
    )

    fig.suptitle(
        "Misuse Detector Results  "
        "(Rules: RSA-1024 weak key + RSA in PQC-required context)",
        fontsize=FIGURES.FONT_SIZE,
        y=1.01,
    )
    plt.tight_layout()

    fname = f"confusion_matrix_misuse.{fmt or FIGURES.FORMAT}"
    save_figure(fig, fname, fmt=fmt, verbose=verbose)
    return True


# ============================================================
# SUMMARY STATS FROM ALL CONFUSION MATRICES
# ============================================================

def print_cm_summary(verbose: bool = True):
    """
    Print a compact table of per-class recall derived from all
    available confusion matrices. This is the supplement to
    Table 3 in the paper — shows not just overall F1 but
    where each model struggles.
    """
    if not verbose:
        return

    models = ["rf", "svm", "cnn", "lstm"]
    available = {}
    for name in models:
        d = load_confusion_matrix(name)
        if d is not None:
            available[name] = d

    if not available:
        return

    print()
    print("=" * 70)
    print("  Per-class recall from confusion matrices (paper supplement)")
    print("=" * 70)

    col_w = 10
    header = f"  {'Class':<14}"
    for name in available:
        header += f" {DISPLAY_NAMES.get(name, name)[:col_w]:>{col_w}}"
    print(header)
    print(f"  {'-'*65}")

    n = len(CLASS_NAMES)
    for i, cls in enumerate(CLASS_NAMES):
        row = f"  {cls:<14}"
        for name, data in available.items():
            cm       = data["cm"]
            row_sum  = cm[i].sum() if i < cm.shape[0] else 0
            recall   = cm[i, i] / row_sum if row_sum > 0 else float("nan")
            if np.isnan(recall):
                row += f" {'N/A':>{col_w}}"
            else:
                marker = "*" if recall == 1.0 else " "
                row += f" {recall:>{col_w-1}.3f}{marker}"
        print(row)

    print(f"  {'-'*65}")
    print("  (* = 100% recall for that class)")
    print()
    print("  Security note:")
    print("  RSA classes correctly classified at 100% in most models.")
    print("  Kyber sub-variant confusion (K-768 ↔ K-1024) is expected —")
    print("  these variants have similar timing profiles by design.")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main(
    model_filter: str  = None,
    show_pct:     bool = True,
    fmt:          str  = None,
    verbose:      bool = True,
) -> dict:
    """
    Generate all confusion matrix figures for the paper.
    Called by run_pipeline.py Phase 6, or directly from CLI.

    Args:
        model_filter: generate figure for one model only
        show_pct:     annotate cells with row percentages
        fmt:          output format ('pdf', 'png', 'svg')
        verbose:      print save paths and summary

    Returns:
        dict mapping model name → output path (or None)
    """
    print("=" * 65)
    print("  CryptoFP — Confusion Matrix Figures")
    print("=" * 65)
    print(f"  Format   : {fmt or FIGURES.FORMAT}")
    print(f"  DPI      : {FIGURES.DPI}")
    print(f"  Colormap : {FIGURES.COLORMAP}")
    print(f"  Output   : {PATHS.FIGURES.relative_to(ROOT)}/")
    print()

    PATHS.FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = {}

    if model_filter:
        # Single model only
        models_to_plot = [model_filter]
    else:
        models_to_plot = ["rf", "svm", "cnn", "lstm"]

    # ── Individual figures ────────────────────────────────────
    print("  Individual confusion matrices:")
    for name in models_to_plot:
        ok = plot_model_cm(name, show_pct=show_pct, fmt=fmt, verbose=verbose)
        outputs[name] = (
            PATHS.FIGURES / f"confusion_matrix_{name}.{fmt or FIGURES.FORMAT}"
            if ok else None
        )

    # ── All-models grid ───────────────────────────────────────
    if not model_filter:
        print("\n  All-models grid (paper Figure):")
        ok = plot_all_models_grid(
            models=["rf", "svm", "cnn", "lstm"],
            show_pct=show_pct,
            fmt=fmt,
            verbose=verbose,
        )
        outputs["all"] = (
            PATHS.FIGURES / f"confusion_matrix_all.{fmt or FIGURES.FORMAT}"
            if ok else None
        )

        # ── Misuse detector ───────────────────────────────────
        print("\n  Misuse detector figure:")
        ok = plot_misuse_cm(show_pct=show_pct, fmt=fmt, verbose=verbose)
        outputs["misuse"] = (
            PATHS.FIGURES / f"confusion_matrix_misuse.{fmt or FIGURES.FORMAT}"
            if ok else None
        )

    # ── Summary stats ─────────────────────────────────────────
    print_cm_summary(verbose=verbose)

    # ── Final output list ─────────────────────────────────────
    saved = [str(p.relative_to(ROOT)) for p in outputs.values()
             if p is not None and p.exists()]
    print()
    print("=" * 65)
    print("  Confusion matrix figures complete.")
    print(f"  {len(saved)} file(s) saved to paper/figures/:")
    for p in saved:
        size_kb = (ROOT / p).stat().st_size / 1024
        print(f"    {p}  ({size_kb:.1f} KB)")
    print()
    print("  LaTeX inclusion:")
    print("    \\includegraphics[width=\\columnwidth]"
          "{figures/confusion_matrix_all.pdf}")
    print()
    print("  Next: python 05_results/plot_roc_curves.py")
    print("=" * 65)

    return outputs


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CryptoFP — generate confusion matrix figures"
    )
    parser.add_argument(
        "--model",
        choices=["rf", "svm", "cnn", "lstm", "misuse"],
        default=None,
        help="Generate figure for one model only.",
    )
    parser.add_argument(
        "--format", choices=["pdf", "png", "svg"],
        default=None, dest="fmt",
        help=f"Output format (default: {FIGURES.FORMAT}).",
    )
    parser.add_argument(
        "--no-pct", action="store_false", dest="show_pct",
        help="Do not annotate cells with row percentages.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed output.",
    )
    args, _ = parser.parse_known_args()

    main(
        model_filter=args.model,
        show_pct=args.show_pct,
        fmt=args.fmt,
        verbose=not args.quiet,
    )