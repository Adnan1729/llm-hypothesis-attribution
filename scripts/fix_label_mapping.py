"""Fix the label mapping error in summary CSVs.

The original LABEL_NAMES had indices 1-4 wrong:
  Wrong:   ["background", "objective", "method", "result", "other"]
  Correct: ["background", "method", "objective", "other", "result"]

This script reads each summary CSV, swaps the column names, and overwrites.
The attribution SCORES are correct — only the labels need swapping.
"""
import csv
import os
import shutil
from pathlib import Path


# Mapping from wrong label to correct label
SWAP = {
    "objective": "method",
    "method": "objective",
    "result": "other",
    "other": "result",
}


def swap_label(label):
    return SWAP.get(label, label)


def fix_csv(csv_path):
    """Read a summary CSV, swap labels in all relevant columns, overwrite."""
    # Read all rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        print(f"  SKIP (empty): {csv_path}")
        return

    # Backup
    backup = str(csv_path) + ".bak"
    shutil.copy2(csv_path, backup)

    # Columns that contain label names as VALUES (need value swapping)
    value_swap_cols = ["fa_top_section", "sh_top_section", "sections_present"]

    # Columns that contain label names in their COLUMN NAME (need column renaming)
    # e.g., fa_objective -> fa_method, sh_result -> sh_other, words_method -> words_objective
    prefixes = ["fa_", "sh_", "words_"]

    new_rows = []
    for row in rows:
        new_row = {}
        for col, val in row.items():
            # Step 1: rename the column if it contains a wrong label
            new_col = col
            for prefix in prefixes:
                for wrong, correct in SWAP.items():
                    if col == f"{prefix}{wrong}":
                        new_col = f"{prefix}{correct}"
                        break

            # Step 2: swap the value if it's a label-value column
            new_val = val
            if col in value_swap_cols:
                if col == "sections_present":
                    # Comma-separated list of labels
                    parts = val.split(",")
                    parts = [swap_label(p.strip()) for p in parts]
                    new_val = ",".join(sorted(parts))
                else:
                    new_val = swap_label(val)

            # Also fix top_section_agree (recompute after swapping)
            new_row[new_col] = new_val

        # Recompute agreement after swapping
        if "fa_top_section" in new_row and "sh_top_section" in new_row:
            new_row["top_section_agree"] = str(
                new_row["fa_top_section"] == new_row["sh_top_section"]
            )

        new_rows.append(new_row)

    # Build new fieldnames with swapped column names
    new_fieldnames = []
    for col in fieldnames:
        new_col = col
        for prefix in prefixes:
            for wrong, correct in SWAP.items():
                if col == f"{prefix}{wrong}":
                    new_col = f"{prefix}{correct}"
        new_fieldnames.append(new_col)

    # Write
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"  FIXED: {csv_path} (backup: {backup})")


def main():
    print("=== Fixing label mapping in summary CSVs ===\n")

    # Find all summary CSVs
    scratch = os.environ.get("PROJECT_SCRATCH", ".")
    locations = [
        Path(scratch) / "outputs",
        Path("results"),
    ]

    csv_files = []
    for base in locations:
        if base.exists():
            for csv_path in base.rglob("summary.csv"):
                csv_files.append(csv_path)

    if not csv_files:
        print("No summary.csv files found!")
        return

    print(f"Found {len(csv_files)} CSV files to fix:\n")
    for csv_path in csv_files:
        fix_csv(csv_path)

    print(f"\n=== Done. Now re-run the analysis script to regenerate figures. ===")
    print(f"    python -m scripts.run_analysis")


if __name__ == "__main__":
    main()