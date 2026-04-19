"""
Comprehensive analysis of attribution experiment results.

Produces ICML-grade academic figures and summary tables from
the full-dataset runs across multiple models.

Usage:
    python -m scripts.run_analysis                           # auto-detect available runs
    python -m scripts.run_analysis --output ./figures        # specify output dir
    python -m scripts.run_analysis --models tinyllama phi3   # specific models only
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# ICML-style plot configuration
# ---------------------------------------------------------------------------
ICML_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.figsize": (5.5, 3.5),
}
plt.rcParams.update(ICML_STYLE)

# Consistent colour palette across all figures
SECTION_COLORS = {
    "background": "#4C72B0",
    "objective":  "#DD8452",
    "method":     "#55A868",
    "result":     "#C44E52",
    "other":      "#8172B3",
}
SECTION_ORDER = ["background", "objective", "method", "result", "other"]
SECTION_LABELS = {
    "background": "Background",
    "objective":  "Objective",
    "method":     "Method",
    "result":     "Result",
    "other":      "Other",
}
MODEL_DISPLAY = {
    "tinyllama": "TinyLlama-1.1B",
    "phi3":      "Phi-3-mini-4k",
    "llama8b":   "Llama-3.1-8B",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary_csv(path):
    """Load a summary CSV into a list of dicts (avoids pandas dependency)."""
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for k, v in row.items():
                if v == "" or v is None or v == "None":
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except (ValueError, TypeError):
                        if v == "True":
                            row[k] = True
                        elif v == "False":
                            row[k] = False
            rows.append(row)
    return rows


def discover_runs(base_dir):
    """Find all completed model runs in the output directory."""
    runs = {}
    for entry in sorted(Path(base_dir).iterdir()):
        if entry.is_dir() and entry.name.startswith("full_"):
            csv_path = entry / "summary.csv"
            meta_path = entry / "run_metadata.json"
            if csv_path.exists():
                model_key = entry.name.replace("full_", "")
                meta = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                runs[model_key] = {
                    "dir": entry,
                    "csv": csv_path,
                    "meta": meta,
                    "data": load_summary_csv(csv_path),
                }
    return runs


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_section_scores(data, method, section):
    """Extract non-None scores for a given method and section."""
    key = f"{method}_{section}"
    return [row[key] for row in data if row.get(key) is not None]


def get_top_section_counts(data, method):
    """Count how often each section ranks #1."""
    key = f"{method}_top_section"
    counts = {}
    for row in data:
        s = row.get(key)
        if s:
            counts[s] = counts.get(s, 0) + 1
    return counts


def save_fig(fig, path, name):
    """Save figure in both PDF (for paper) and PNG (for quick viewing)."""
    fig.savefig(path / f"{name}.pdf")
    fig.savefig(path / f"{name}.png")
    plt.close(fig)
    print(f"    Saved {name}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 1: Mean attribution scores per section (single model)
# ---------------------------------------------------------------------------

def fig_mean_attribution_bars(data, model_key, out_dir):
    """Bar chart of mean attribution per section for both methods."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0), sharey=True)

    for ax, (method, method_label) in zip(axes, [("fa", "Feature Ablation"),
                                                   ("sh", "Shapley Value")]):
        means = []
        stds = []
        colors = []
        labels = []
        for sec in SECTION_ORDER:
            scores = get_section_scores(data, method, sec)
            if scores:
                means.append(np.mean(scores))
                stds.append(np.std(scores))
            else:
                means.append(0)
                stds.append(0)
            colors.append(SECTION_COLORS[sec])
            labels.append(SECTION_LABELS[sec])

        x = np.arange(len(SECTION_ORDER))
        bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                      linewidth=0.5, capsize=3, error_kw={"linewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(method_label)
        ax.set_ylabel("Mean attribution score" if ax == axes[0] else "")

    fig.suptitle(f"Sectional Attribution — {MODEL_DISPLAY.get(model_key, model_key)}",
                 y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, f"fig1_mean_attribution_{model_key}")


# ---------------------------------------------------------------------------
# Figure 2: Top-section frequency (single model, both methods side by side)
# ---------------------------------------------------------------------------

def fig_top_section_frequency(data, model_key, out_dir):
    """Grouped bar chart showing how often each section ranks #1."""
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    fa_counts = get_top_section_counts(data, "fa")
    sh_counts = get_top_section_counts(data, "sh")
    total = len(data)

    x = np.arange(len(SECTION_ORDER))
    width = 0.35

    fa_vals = [fa_counts.get(s, 0) / total * 100 for s in SECTION_ORDER]
    sh_vals = [sh_counts.get(s, 0) / total * 100 for s in SECTION_ORDER]

    ax.bar(x - width / 2, fa_vals, width, label="Feature Ablation",
           color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, sh_vals, width, label="Shapley Value",
           color="#DD8452", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([SECTION_LABELS[s] for s in SECTION_ORDER])
    ax.set_ylabel("Frequency as top section (%)")
    ax.set_title(f"Top-Ranked Section Distribution — "
                 f"{MODEL_DISPLAY.get(model_key, model_key)}")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, out_dir, f"fig2_top_section_freq_{model_key}")


# ---------------------------------------------------------------------------
# Figure 3: FA vs Shapley scatter (method agreement)
# ---------------------------------------------------------------------------

def fig_method_scatter(data, model_key, out_dir):
    """Scatter plot of FA vs Shapley scores per section, all abstracts."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    for sec in SECTION_ORDER:
        fa_scores = []
        sh_scores = []
        for row in data:
            fa_val = row.get(f"fa_{sec}")
            sh_val = row.get(f"sh_{sec}")
            if fa_val is not None and sh_val is not None:
                fa_scores.append(fa_val)
                sh_scores.append(sh_val)
        if fa_scores:
            ax.scatter(fa_scores, sh_scores, alpha=0.3, s=12,
                       color=SECTION_COLORS[sec], label=SECTION_LABELS[sec],
                       edgecolors="none")

    # Diagonal reference line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", linewidth=0.6, alpha=0.5, zorder=0)

    # Compute overall Spearman correlation
    all_fa, all_sh = [], []
    for sec in SECTION_ORDER:
        for row in data:
            fa_val = row.get(f"fa_{sec}")
            sh_val = row.get(f"sh_{sec}")
            if fa_val is not None and sh_val is not None:
                all_fa.append(fa_val)
                all_sh.append(sh_val)

    if len(all_fa) > 2:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(all_fa, all_sh)
        ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}\np < {max(pval, 1e-300):.1e}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

    ax.set_xlabel("Feature Ablation score")
    ax.set_ylabel("Shapley Value score")
    ax.set_title(f"Method Agreement — {MODEL_DISPLAY.get(model_key, model_key)}")
    ax.legend(frameon=False, loc="lower right", markerscale=2)
    fig.tight_layout()
    save_fig(fig, out_dir, f"fig3_method_scatter_{model_key}")


# ---------------------------------------------------------------------------
# Figure 4: Attribution distributions (violin / box plot)
# ---------------------------------------------------------------------------

def fig_attribution_violins(data, model_key, out_dir):
    """Violin plot of attribution score distributions per section."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.5), sharey=True)

    for ax, (method, method_label) in zip(axes, [("fa", "Feature Ablation"),
                                                   ("sh", "Shapley Value")]):
        all_scores = []
        positions = []
        colors_list = []
        for i, sec in enumerate(SECTION_ORDER):
            scores = get_section_scores(data, method, sec)
            if scores:
                all_scores.append(scores)
                positions.append(i)
                colors_list.append(SECTION_COLORS[sec])

        if all_scores:
            parts = ax.violinplot(all_scores, positions=positions,
                                  showmeans=True, showmedians=True,
                                  showextrema=False)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors_list[i])
                pc.set_alpha(0.7)
            parts["cmeans"].set_color("black")
            parts["cmedians"].set_color("gray")
            parts["cmedians"].set_linestyle("--")

        ax.set_xticks(range(len(SECTION_ORDER)))
        ax.set_xticklabels([SECTION_LABELS[s] for s in SECTION_ORDER],
                           rotation=35, ha="right")
        ax.set_title(method_label)
        ax.set_ylabel("Attribution score" if ax == axes[0] else "")

    fig.suptitle(f"Attribution Distributions — "
                 f"{MODEL_DISPLAY.get(model_key, model_key)}", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, f"fig4_violins_{model_key}")


# ---------------------------------------------------------------------------
# Figure 5: Section word count vs attribution (confound check)
# ---------------------------------------------------------------------------

def fig_length_vs_attribution(data, model_key, out_dir):
    """Scatter: section word count vs attribution score (checks length confound)."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0))

    for ax, (method, method_label) in zip(axes, [("fa", "Feature Ablation"),
                                                   ("sh", "Shapley Value")]):
        all_words, all_scores = [], []
        for sec in SECTION_ORDER:
            for row in data:
                w = row.get(f"words_{sec}")
                s = row.get(f"{method}_{sec}")
                if w is not None and s is not None and w > 0:
                    ax.scatter(w, s, alpha=0.2, s=8, color=SECTION_COLORS[sec],
                               edgecolors="none")
                    all_words.append(w)
                    all_scores.append(s)

        if len(all_words) > 2:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(all_words, all_scores)
            ax.text(0.05, 0.95, f"ρ = {rho:.3f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="gray", alpha=0.8))

        ax.set_xlabel("Section word count")
        ax.set_ylabel("Attribution score" if ax == axes[0] else "")
        ax.set_title(method_label)

    fig.suptitle(f"Length vs Attribution — {MODEL_DISPLAY.get(model_key, model_key)}",
                 y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, f"fig5_length_confound_{model_key}")


# ---------------------------------------------------------------------------
# Figure 6: Cross-model comparison (requires ≥2 models)
# ---------------------------------------------------------------------------

def fig_cross_model_comparison(all_runs, out_dir):
    """Grouped bar chart comparing mean Shapley values across models."""
    if len(all_runs) < 2:
        print("    Skipping cross-model comparison (need ≥2 models)")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    model_keys = list(all_runs.keys())
    n_models = len(model_keys)
    n_sections = len(SECTION_ORDER)
    x = np.arange(n_sections)
    width = 0.8 / n_models

    model_colors = ["#4C72B0", "#DD8452", "#55A868"]

    for j, model_key in enumerate(model_keys):
        data = all_runs[model_key]["data"]
        means = []
        stds = []
        for sec in SECTION_ORDER:
            scores = get_section_scores(data, "sh", sec)
            means.append(np.mean(scores) if scores else 0)
            stds.append(np.std(scores) if scores else 0)

        offset = (j - (n_models - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=MODEL_DISPLAY.get(model_key, model_key),
               color=model_colors[j % len(model_colors)],
               edgecolor="white", linewidth=0.5,
               capsize=2, error_kw={"linewidth": 0.6})

    ax.set_xticks(x)
    ax.set_xticklabels([SECTION_LABELS[s] for s in SECTION_ORDER])
    ax.set_ylabel("Mean Shapley value")
    ax.set_title("Cross-Model Comparison of Sectional Influence")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig6_cross_model_comparison")


# ---------------------------------------------------------------------------
# Figure 7: Cross-model top-section agreement heatmap
# ---------------------------------------------------------------------------

def fig_cross_model_top_section(all_runs, out_dir):
    """Heatmap: % of abstracts where each section ranks #1, per model."""
    if len(all_runs) < 2:
        print("    Skipping cross-model heatmap (need ≥2 models)")
        return

    model_keys = list(all_runs.keys())
    matrix = []
    for model_key in model_keys:
        data = all_runs[model_key]["data"]
        counts = get_top_section_counts(data, "sh")
        total = len(data)
        row = [counts.get(s, 0) / total * 100 for s in SECTION_ORDER]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(SECTION_ORDER)))
    ax.set_xticklabels([SECTION_LABELS[s] for s in SECTION_ORDER])
    ax.set_yticks(range(len(model_keys)))
    ax.set_yticklabels([MODEL_DISPLAY.get(k, k) for k in model_keys])

    # Annotate cells
    for i in range(len(model_keys)):
        for j in range(len(SECTION_ORDER)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_title("Top-Ranked Section Frequency by Model (Shapley, %)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="%")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig7_cross_model_heatmap")


# ---------------------------------------------------------------------------
# Table outputs (LaTeX for paper, plain text for terminal)
# ---------------------------------------------------------------------------

def print_summary_table(all_runs, out_dir):
    """Generate summary statistics table as LaTeX and plain text."""
    lines_txt = []
    lines_tex = []

    lines_txt.append("=" * 80)
    lines_txt.append("TABLE 1: Mean Attribution Scores by Section and Method")
    lines_txt.append("=" * 80)

    lines_tex.append(r"\begin{table}[t]")
    lines_tex.append(r"\centering")
    lines_tex.append(r"\caption{Mean attribution scores by section across models. "
                     r"Standard deviations in parentheses.}")
    lines_tex.append(r"\label{tab:mean_attribution}")
    lines_tex.append(r"\small")

    n_models = len(all_runs)
    col_spec = "l" + "cc" * n_models
    lines_tex.append(r"\begin{tabular}{" + col_spec + "}")
    lines_tex.append(r"\toprule")

    # Header
    header_parts = [r"\textbf{Section}"]
    for model_key in all_runs:
        display = MODEL_DISPLAY.get(model_key, model_key)
        header_parts.append(r"\multicolumn{2}{c}{\textbf{" + display + "}}")
    lines_tex.append(" & ".join(header_parts) + r" \\")

    subheader_parts = [""]
    for _ in all_runs:
        subheader_parts.extend([r"\textbf{FA}", r"\textbf{Shapley}"])
    lines_tex.append(" & ".join(subheader_parts) + r" \\")
    lines_tex.append(r"\midrule")

    for sec in SECTION_ORDER:
        txt_parts = [f"{SECTION_LABELS[sec]:<12}"]
        tex_parts = [SECTION_LABELS[sec]]

        for model_key, run in all_runs.items():
            data = run["data"]
            for method in ["fa", "sh"]:
                scores = get_section_scores(data, method, sec)
                if scores:
                    m = np.mean(scores)
                    s = np.std(scores)
                    txt_parts.append(f"{m:>8.1f} ± {s:<6.1f}")
                    tex_parts.append(f"{m:.1f} ({s:.1f})")
                else:
                    txt_parts.append(f"{'N/A':>17}")
                    tex_parts.append("--")

        lines_txt.append("  ".join(txt_parts))
        lines_tex.append(" & ".join(tex_parts) + r" \\")

    lines_tex.append(r"\bottomrule")
    lines_tex.append(r"\end{tabular}")
    lines_tex.append(r"\end{table}")

    # Agreement row
    lines_txt.append("-" * 80)
    agree_parts = ["Agreement   "]
    agree_tex = ["Top-1 agree (\\%)"]
    for model_key, run in all_runs.items():
        meta = run["meta"]
        pct = meta.get("top_section_agreement_pct", "?")
        agree_parts.append(f"{pct}%")
        agree_tex.extend([r"\multicolumn{2}{c}{" + f"{pct}" + r"\%}"])
    lines_txt.append("  ".join(agree_parts))

    lines_txt.append("=" * 80)

    # Print to terminal
    print("\n".join(lines_txt))

    # Save LaTeX
    tex_path = out_dir / "table1_mean_attribution.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines_tex))
    print(f"\n    LaTeX table saved to {tex_path}")

    # Table 2: Top-section frequency
    lines_tex2 = []
    lines_tex2.append(r"\begin{table}[t]")
    lines_tex2.append(r"\centering")
    lines_tex2.append(r"\caption{Percentage of abstracts where each section "
                      r"ranks as most influential (Shapley values).}")
    lines_tex2.append(r"\label{tab:top_section}")
    lines_tex2.append(r"\small")
    col_spec2 = "l" + "c" * len(all_runs)
    lines_tex2.append(r"\begin{tabular}{" + col_spec2 + "}")
    lines_tex2.append(r"\toprule")

    header2 = [r"\textbf{Section}"]
    for model_key in all_runs:
        header2.append(r"\textbf{" + MODEL_DISPLAY.get(model_key, model_key) + "}")
    lines_tex2.append(" & ".join(header2) + r" \\")
    lines_tex2.append(r"\midrule")

    for sec in SECTION_ORDER:
        row_parts = [SECTION_LABELS[sec]]
        for model_key, run in all_runs.items():
            counts = get_top_section_counts(run["data"], "sh")
            total = len(run["data"])
            pct = counts.get(sec, 0) / total * 100
            row_parts.append(f"{pct:.1f}\\%")
        lines_tex2.append(" & ".join(row_parts) + r" \\")

    lines_tex2.append(r"\bottomrule")
    lines_tex2.append(r"\end{tabular}")
    lines_tex2.append(r"\end{table}")

    tex2_path = out_dir / "table2_top_section.tex"
    with open(tex2_path, "w") as f:
        f.write("\n".join(lines_tex2))
    print(f"    LaTeX table saved to {tex2_path}")


# ---------------------------------------------------------------------------
# Table 3: Method comparison statistics
# ---------------------------------------------------------------------------

def print_method_comparison(all_runs, out_dir):
    """Compute and print method comparison statistics."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("TABLE 3: Method Comparison Statistics")
    lines.append("=" * 80)

    tex_lines = []
    tex_lines.append(r"\begin{table}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{Comparison of Feature Ablation and Shapley Value "
                     r"Sampling across models.}")
    tex_lines.append(r"\label{tab:method_comparison}")
    tex_lines.append(r"\small")
    tex_lines.append(r"\begin{tabular}{l" + "c" * len(all_runs) + "}")
    tex_lines.append(r"\toprule")

    header = [r"\textbf{Metric}"]
    for mk in all_runs:
        header.append(r"\textbf{" + MODEL_DISPLAY.get(mk, mk) + "}")
    tex_lines.append(" & ".join(header) + r" \\")
    tex_lines.append(r"\midrule")

    # Compute per-model stats
    for model_key, run in all_runs.items():
        data = run["data"]
        lines.append(f"\n--- {MODEL_DISPLAY.get(model_key, model_key)} ---")

        # Spearman correlation
        all_fa, all_sh = [], []
        for sec in SECTION_ORDER:
            for row in data:
                fa_val = row.get(f"fa_{sec}")
                sh_val = row.get(f"sh_{sec}")
                if fa_val is not None and sh_val is not None:
                    all_fa.append(fa_val)
                    all_sh.append(sh_val)

        if len(all_fa) > 2:
            try:
                from scipy.stats import spearmanr, kendalltau
                rho, p_rho = spearmanr(all_fa, all_sh)
                tau, p_tau = kendalltau(all_fa, all_sh)
                lines.append(f"  Spearman rho:        {rho:.4f} (p={p_rho:.2e})")
                lines.append(f"  Kendall tau:         {tau:.4f} (p={p_tau:.2e})")
            except ImportError:
                lines.append("  (scipy not available for correlation stats)")

        # Top-1 agreement
        agree = sum(1 for row in data
                    if row.get("fa_top_section") == row.get("sh_top_section"))
        agree_pct = agree / len(data) * 100
        lines.append(f"  Top-1 agreement:     {agree}/{len(data)} ({agree_pct:.1f}%)")

        # Timing
        fa_avg = np.mean([row["fa_time_s"] for row in data
                          if row.get("fa_time_s") is not None])
        sh_avg = np.mean([row["sh_time_s"] for row in data
                          if row.get("sh_time_s") is not None])
        lines.append(f"  Avg FA time:         {fa_avg:.3f}s")
        lines.append(f"  Avg Shapley time:    {sh_avg:.3f}s")
        lines.append(f"  Shapley/FA ratio:    {sh_avg/fa_avg:.1f}x")

    print("\n".join(lines))

    # Friedman test across sections (if scipy available)
    print("\n--- Statistical Tests ---")
    for model_key, run in all_runs.items():
        data = run["data"]
        print(f"\n  {MODEL_DISPLAY.get(model_key, model_key)}:")

        try:
            from scipy.stats import friedmanchisquare
            # For Friedman: need matched observations across sections
            # Use only abstracts that have all 5 sections
            matched = []
            for row in data:
                vals = [row.get(f"sh_{sec}") for sec in SECTION_ORDER]
                if all(v is not None for v in vals):
                    matched.append(vals)

            if len(matched) >= 10:
                cols = list(zip(*matched))
                stat, pval = friedmanchisquare(*cols)
                print(f"    Friedman test (Shapley, n={len(matched)}): "
                      f"χ²={stat:.2f}, p={pval:.2e}")
            else:
                print(f"    Friedman test: insufficient abstracts with all 5 sections "
                      f"(n={len(matched)})")
        except ImportError:
            print("    (scipy not available)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for figures (default: $PROJECT_SCRATCH/outputs/figures)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific model keys to analyse (default: all available)")
    args = parser.parse_args()

    # Determine base output dir
    scratch = os.environ.get("PROJECT_SCRATCH", ".")
    base_dir = Path(scratch) / "outputs"

    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = base_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Attribution Analysis ===")
    print(f"  Base dir: {base_dir}")
    print(f"  Output:   {out_dir}\n")

    # Discover runs
    all_runs = discover_runs(base_dir)
    if args.models:
        all_runs = {k: v for k, v in all_runs.items() if k in args.models}

    if not all_runs:
        print("ERROR: No completed runs found. Check $PROJECT_SCRATCH/outputs/")
        sys.exit(1)

    print(f"  Found {len(all_runs)} model run(s): {list(all_runs.keys())}")
    for mk, run in all_runs.items():
        print(f"    {mk}: {len(run['data'])} abstracts")
    print()

    # Per-model figures
    for model_key, run in all_runs.items():
        data = run["data"]
        print(f"\n--- Generating figures for {MODEL_DISPLAY.get(model_key, model_key)} ---")
        fig_mean_attribution_bars(data, model_key, out_dir)
        fig_top_section_frequency(data, model_key, out_dir)
        fig_method_scatter(data, model_key, out_dir)
        fig_attribution_violins(data, model_key, out_dir)
        fig_length_vs_attribution(data, model_key, out_dir)

    # Cross-model figures
    print(f"\n--- Cross-model analysis ---")
    fig_cross_model_comparison(all_runs, out_dir)
    fig_cross_model_top_section(all_runs, out_dir)

    # Tables
    print(f"\n--- Summary Tables ---")
    print_summary_table(all_runs, out_dir)
    print_method_comparison(all_runs, out_dir)

    print(f"\n=== Analysis complete. All outputs in {out_dir} ===")


if __name__ == "__main__":
    main()
