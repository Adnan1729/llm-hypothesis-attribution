"""Full dataset run: both attribution methods on all CSABSTRUCT abstracts."""
import time
import json
import csv
import torch
from pathlib import Path
from src.data.load_csabstruct import load_csabstruct, LABEL_NAMES
from src.models.load_model import load_model
from src.attribution.feature_ablation import feature_ablation
from src.attribution.shapley import shapley_value_sampling
from src.generation.prompts import HYPOTHESIS_PROMPT_V1
from src.utils.io import get_output_dir, save_json

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 80
SEED = 42
SHAPLEY_SAMPLES = 100
MIN_SECTIONS = 2  # include all abstracts with at least 2 sections

GENERATION_PARAMS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "seed": SEED,
}


def generate_hypothesis(model, tokenizer, context, prompt_template):
    prompt = prompt_template.format(context=context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(SEED)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    print(f"=== Full dataset run: FA + Shapley ===\n")

    # Load all splits
    print("[1] Loading all CSABSTRUCT splits...")
    all_abstracts = []
    for split in ["train", "validation", "test"]:
        abstracts = load_csabstruct(split)
        print(f"    {split}: {len(abstracts)} abstracts")
        all_abstracts.extend(abstracts)

    # Filter
    selected = [a for a in all_abstracts if len(a.sections) >= MIN_SECTIONS]
    print(f"\n    Total: {len(all_abstracts)} abstracts")
    print(f"    After filtering (>={MIN_SECTIONS} sections): {len(selected)}")

    # Section distribution summary
    section_counts = {}
    for a in selected:
        n = len(a.sections)
        section_counts[n] = section_counts.get(n, 0) + 1
    print(f"    Section count distribution: {dict(sorted(section_counts.items()))}\n")

    # Load model
    print(f"[2] Loading model {MODEL_NAME}...")
    t0 = time.time()
    model, tokenizer = load_model(MODEL_NAME)
    model_load_time = time.time() - t0
    print(f"    Model loaded on {model.device} in {model_load_time:.1f}s\n")

    # Output directories
    run_name = "full_tinyllama"
    out_dir_fa = get_output_dir(f"{run_name}/feature_ablation")
    out_dir_sh = get_output_dir(f"{run_name}/shapley")
    out_dir_summary = get_output_dir(run_name)
    print(f"[3] Output dirs:")
    print(f"    FA:      {out_dir_fa}")
    print(f"    Shapley: {out_dir_sh}")
    print(f"    Summary: {out_dir_summary}\n")

    # Process all abstracts
    print(f"[4] Processing {len(selected)} abstracts...\n")
    summary_rows = []
    fa_times = []
    sh_times = []
    agreements = 0
    errors = []
    t_run_start = time.time()

    for i, abstract in enumerate(selected, 1):
        try:
            # Generate hypothesis
            t_gen = time.time()
            hypothesis = generate_hypothesis(
                model, tokenizer, abstract.full_text, HYPOTHESIS_PROMPT_V1
            )
            gen_time = round(time.time() - t_gen, 3)

            # Feature Ablation
            fa_result = feature_ablation(
                model, tokenizer, abstract, hypothesis, HYPOTHESIS_PROMPT_V1,
                model_name=MODEL_NAME,
                generation_params=GENERATION_PARAMS,
            )
            fa_result.generation_time_s = gen_time
            fa_result.total_time_s = round(gen_time + fa_result.attribution_time_s, 3)
            save_json(fa_result, out_dir_fa / f"{abstract.abstract_id}.json")
            fa_times.append(fa_result.attribution_time_s)

            # Shapley
            sh_result = shapley_value_sampling(
                model, tokenizer, abstract, hypothesis, HYPOTHESIS_PROMPT_V1,
                num_samples=SHAPLEY_SAMPLES,
                model_name=MODEL_NAME,
                generation_params=GENERATION_PARAMS,
                seed=SEED,
            )
            sh_result.generation_time_s = gen_time
            sh_result.total_time_s = round(gen_time + sh_result.attribution_time_s, 3)
            save_json(sh_result, out_dir_sh / f"{abstract.abstract_id}.json")
            sh_times.append(sh_result.attribution_time_s)

            # Compare top sections
            fa_top = max(fa_result.attributions, key=fa_result.attributions.get)
            sh_top = max(sh_result.shapley_values, key=sh_result.shapley_values.get)
            agree = fa_top == sh_top
            if agree:
                agreements += 1

            # Build summary row
            row = {
                "abstract_id": abstract.abstract_id,
                "num_sentences": len(abstract.sentences),
                "num_sections": len(abstract.sections),
                "sections_present": ",".join(sorted(abstract.sections.keys())),
                "hypothesis_length": len(hypothesis.split()),
                "fa_baseline_lp": fa_result.baseline_log_prob,
                "fa_top_section": fa_top,
                "sh_top_section": sh_top,
                "top_section_agree": agree,
                "fa_time_s": fa_result.attribution_time_s,
                "sh_time_s": sh_result.attribution_time_s,
                "sh_forward_passes": sh_result.num_forward_passes,
            }
            # Add per-section scores for both methods
            for label in LABEL_NAMES:
                row[f"fa_{label}"] = fa_result.attributions.get(label, None)
                row[f"sh_{label}"] = sh_result.shapley_values.get(label, None)
                # Section word count
                info = fa_result.sections_info.get(label, {})
                row[f"words_{label}"] = info.get("num_words", 0)

            summary_rows.append(row)

            # Progress logging every 50 abstracts
            if i % 50 == 0 or i == len(selected):
                elapsed = time.time() - t_run_start
                rate = i / elapsed
                remaining = (len(selected) - i) / rate if rate > 0 else 0
                agree_pct = (agreements / i) * 100
                print(f"  [{i}/{len(selected)}] "
                      f"elapsed={elapsed:.0f}s, "
                      f"rate={rate:.1f} abs/s, "
                      f"ETA={remaining:.0f}s, "
                      f"agreement={agree_pct:.0f}%")

        except Exception as e:
            errors.append({"abstract_id": abstract.abstract_id, "error": str(e)})
            print(f"  [{i}/{len(selected)}] ERROR on {abstract.abstract_id}: {e}")
            continue

    # Write summary CSV
    csv_path = out_dir_summary / "summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    # Write error log
    if errors:
        save_json(errors, out_dir_summary / "errors.json")

    # Write run metadata
    total_time = time.time() - t_run_start
    metadata = {
        "model_name": MODEL_NAME,
        "generation_params": GENERATION_PARAMS,
        "shapley_samples": SHAPLEY_SAMPLES,
        "min_sections": MIN_SECTIONS,
        "total_abstracts": len(all_abstracts),
        "processed_abstracts": len(summary_rows),
        "skipped_abstracts": len(all_abstracts) - len(selected),
        "errors": len(errors),
        "top_section_agreement_pct": round((agreements / len(summary_rows)) * 100, 1)
            if summary_rows else 0,
        "total_time_s": round(total_time, 1),
        "model_load_time_s": round(model_load_time, 1),
        "avg_fa_time_s": round(sum(fa_times) / len(fa_times), 3) if fa_times else 0,
        "avg_sh_time_s": round(sum(sh_times) / len(sh_times), 3) if sh_times else 0,
        "prompt_template": HYPOTHESIS_PROMPT_V1,
    }
    save_json(metadata, out_dir_summary / "run_metadata.json")

    # Final report
    print(f"\n{'=' * 60}")
    print(f"FINAL REPORT")
    print(f"{'=' * 60}")
    print(f"Abstracts processed: {len(summary_rows)}")
    print(f"Errors:              {len(errors)}")
    print(f"Total time:          {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Avg FA time:         {metadata['avg_fa_time_s']:.3f}s per abstract")
    print(f"Avg Shapley time:    {metadata['avg_sh_time_s']:.3f}s per abstract")
    print(f"Top-section agree:   {metadata['top_section_agreement_pct']}%")

    # Per-section summary (FA)
    print(f"\nFA: times each section ranked #1:")
    fa_top_counts = {}
    for row in summary_rows:
        s = row["fa_top_section"]
        fa_top_counts[s] = fa_top_counts.get(s, 0) + 1
    for s in sorted(fa_top_counts, key=fa_top_counts.get, reverse=True):
        pct = fa_top_counts[s] / len(summary_rows) * 100
        print(f"    {s:<15} {fa_top_counts[s]:>5} ({pct:.1f}%)")

    # Per-section summary (Shapley)
    print(f"\nShapley: times each section ranked #1:")
    sh_top_counts = {}
    for row in summary_rows:
        s = row["sh_top_section"]
        sh_top_counts[s] = sh_top_counts.get(s, 0) + 1
    for s in sorted(sh_top_counts, key=sh_top_counts.get, reverse=True):
        pct = sh_top_counts[s] / len(summary_rows) * 100
        print(f"    {s:<15} {sh_top_counts[s]:>5} ({pct:.1f}%)")

    if errors:
        print(f"\nErrors logged to: {out_dir_summary / 'errors.json'}")

    print(f"\nOutputs:")
    print(f"  JSON (FA):      {out_dir_fa}")
    print(f"  JSON (Shapley): {out_dir_sh}")
    print(f"  Summary CSV:    {csv_path}")
    print(f"  Metadata:       {out_dir_summary / 'run_metadata.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
