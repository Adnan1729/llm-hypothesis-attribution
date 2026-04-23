"""
Quick standalone script to compute appendix statistics from CSABSTRUCT.

Outputs:
  1. Sentence-level label distribution (Table 6)
  2. Section co-occurrence percentages (Table 7)
  3. Abstract length histogram (app_abstract_length_dist.png)

Usage:
    python -m scripts.compute_appendix_stats
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.load_csabstruct import load_csabstruct

# ICML-ish style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SECTION_ORDER = ["background", "objective", "method", "result", "other"]
SECTION_LABELS = {
    "background": "Background",
    "objective": "Objective",
    "method": "Method",
    "result": "Result",
    "other": "Other",
}


def main():
    # Load all splits
    all_abstracts = []
    for split in ["train", "validation", "test"]:
        abstracts = load_csabstruct(split)
        all_abstracts.extend(abstracts)

    # Filter same as experiments
    filtered = [a for a in all_abstracts if len(a.sections) >= 2]
    print(f"Total abstracts: {len(all_abstracts)}")
    print(f"After filtering (>=2 sections): {len(filtered)}")
    print()

    # ----------------------------------------------------------------
    # 1. Sentence-level label distribution
    # ----------------------------------------------------------------
    print("=" * 60)
    print("TABLE 6: Sentence-level label distribution")
    print("=" * 60)

    label_counts = {s: 0 for s in SECTION_ORDER}
    total_sentences = 0
    for a in filtered:
        for label in a.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
            total_sentences += 1

    print(f"{'Label':<15} {'Sentences':>10} {'Proportion':>12}")
    print("-" * 40)
    for sec in SECTION_ORDER:
        count = label_counts.get(sec, 0)
        prop = count / total_sentences if total_sentences > 0 else 0
        print(f"{SECTION_LABELS[sec]:<15} {count:>10} {prop:>12.3f}")
    print(f"{'Total':<15} {total_sentences:>10} {'1.000':>12}")
    print()

    # LaTeX-ready
    print("LaTeX rows:")
    for sec in SECTION_ORDER:
        count = label_counts.get(sec, 0)
        prop = count / total_sentences
        print(f"\\textsc{{{SECTION_LABELS[sec]}}} & {count:,} & {prop:.2f} \\\\")
    print()

    # ----------------------------------------------------------------
    # 2. Section co-occurrence percentages
    # ----------------------------------------------------------------
    print("=" * 60)
    print("TABLE 7: Section co-occurrence (%)")
    print("=" * 60)

    n = len(filtered)

    # Presence counts (diagonal)
    presence = {}
    for sec in SECTION_ORDER:
        presence[sec] = sum(1 for a in filtered if sec in a.sections)

    print(f"\nSection presence (diagonal):")
    for sec in SECTION_ORDER:
        pct = presence[sec] / n * 100
        print(f"  {SECTION_LABELS[sec]:<15} {presence[sec]:>6} / {n} = {pct:.1f}%")

    # Co-occurrence (off-diagonal)
    print(f"\nCo-occurrence matrix (%):")
    header = f"{'':>15}" + "".join(f"{SECTION_LABELS[s]:>10}" for s in SECTION_ORDER)
    print(header)
    print("-" * (15 + 10 * len(SECTION_ORDER)))

    cooccur = {}
    for i, sec_i in enumerate(SECTION_ORDER):
        row = f"{SECTION_LABELS[sec_i]:>15}"
        for j, sec_j in enumerate(SECTION_ORDER):
            count = sum(1 for a in filtered
                        if sec_i in a.sections and sec_j in a.sections)
            pct = count / n * 100
            cooccur[(sec_i, sec_j)] = pct
            if j < i:
                row += f"{pct:>9.1f}%"
            elif j == i:
                row += f"{pct:>9.1f}%"  # diagonal = presence
            else:
                row += f"{'--':>10}"
        print(row)

    # LaTeX-ready lower triangle
    print(f"\nLaTeX rows (lower triangle + diagonal):")
    for i, sec_i in enumerate(SECTION_ORDER):
        parts = [f"\\textbf{{{SECTION_LABELS[sec_i]}}}"]
        for j, sec_j in enumerate(SECTION_ORDER):
            if j <= i:
                parts.append(f"{cooccur[(sec_i, sec_j)]:.1f}")
            else:
                parts.append("--")
        print(" & ".join(parts) + " \\\\")
    print()

    # ----------------------------------------------------------------
    # 3. Abstract length histogram
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Generating abstract length histogram...")
    print("=" * 60)

    lengths = [len(a.sentences) for a in filtered]
    mean_len = sum(lengths) / len(lengths)
    std_len = (sum((x - mean_len) ** 2 for x in lengths) / len(lengths)) ** 0.5
    min_len = min(lengths)
    max_len = max(lengths)

    print(f"  Mean: {mean_len:.1f}")
    print(f"  Std:  {std_len:.1f}")
    print(f"  Min:  {min_len}")
    print(f"  Max:  {max_len}")

    # Determine output path
    scratch = os.environ.get("PROJECT_SCRATCH", ".")
    out_dir = os.path.join(scratch, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "app_abstract_length_dist.png")

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.hist(lengths, bins=range(min_len, max_len + 2),
            color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(mean_len, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_len:.1f}")
    ax.set_xlabel("Sentences per abstract")
    ax.set_ylabel("Number of abstracts")
    ax.set_title("Abstract Length Distribution (CSABSTRUCT)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also save to local outputs if not on HPC
    local_path = os.path.join("results", "figures", "app_abstract_length_dist.png")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    fig2, ax2 = plt.subplots(figsize=(5.0, 3.0))
    ax2.hist(lengths, bins=range(min_len, max_len + 2),
             color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax2.axvline(mean_len, color="#C44E52", linestyle="--", linewidth=1.2,
                label=f"Mean = {mean_len:.1f}")
    ax2.set_xlabel("Sentences per abstract")
    ax2.set_ylabel("Number of abstracts")
    ax2.set_title("Abstract Length Distribution (CSABSTRUCT)")
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(local_path, dpi=300)
    plt.close(fig2)
    print(f"  Saved: {local_path}")

    # ----------------------------------------------------------------
    # 4. Section count distribution (bonus — for Table 1)
    # ----------------------------------------------------------------
    print()
    print("=" * 60)
    print("BONUS: Abstracts by number of distinct sections")
    print("=" * 60)
    section_count_dist = {}
    for a in filtered:
        n_sec = len(a.sections)
        section_count_dist[n_sec] = section_count_dist.get(n_sec, 0) + 1
    for k in sorted(section_count_dist):
        print(f"  {k} sections: {section_count_dist[k]}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()