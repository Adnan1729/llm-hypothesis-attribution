"""Step 4b: 10-abstract benchmark with BOTH Feature Ablation and Shapley."""
import time
import torch
from src.data.load_csabstruct import load_csabstruct
from src.models.load_model import load_model
from src.attribution.feature_ablation import feature_ablation
from src.attribution.shapley import shapley_value_sampling
from src.generation.prompts import HYPOTHESIS_PROMPT_V1
from src.utils.io import get_output_dir, save_json

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 80
SEED = 42
N_ABSTRACTS = 10
MIN_SECTIONS = 4
SHAPLEY_SAMPLES = 100

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
    print(f"=== Step 4b: {N_ABSTRACTS}-abstract benchmark (FA + Shapley) ===\n")

    print("[1] Loading CSABSTRUCT test split and filtering...")
    all_abstracts = load_csabstruct("test")
    rich_abstracts = [a for a in all_abstracts if len(a.sections) >= MIN_SECTIONS]
    print(f"    {len(rich_abstracts)}/{len(all_abstracts)} abstracts have "
          f">={MIN_SECTIONS} sections")
    selected = rich_abstracts[:N_ABSTRACTS]
    print(f"    Selected: {[a.abstract_id for a in selected]}\n")

    print(f"[2] Loading model {MODEL_NAME}...")
    t0 = time.time()
    model, tokenizer = load_model(MODEL_NAME)
    print(f"    Model loaded on {model.device} in {time.time() - t0:.1f}s\n")

    out_dir_fa = get_output_dir("benchmark_both/feature_ablation")
    out_dir_sh = get_output_dir("benchmark_both/shapley")
    print(f"[3] Output dirs:")
    print(f"    FA:      {out_dir_fa}")
    print(f"    Shapley: {out_dir_sh}\n")

    print(f"[4] Processing {len(selected)} abstracts...\n")
    fa_times = []
    sh_times = []

    for i, abstract in enumerate(selected, 1):
        # Generate hypothesis once, use for both methods
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

        # Shapley Value Sampling
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

        # Print comparison
        fa_ranked = sorted(fa_result.attributions.items(),
                           key=lambda x: x[1], reverse=True)
        sh_ranked = sorted(sh_result.shapley_values.items(),
                           key=lambda x: x[1], reverse=True)

        fa_str = ", ".join(f"{s}:{v:.1f}" for s, v in fa_ranked)
        sh_str = ", ".join(f"{s}:{v:.1f}" for s, v in sh_ranked)

        # Check if they agree on the top section
        fa_top = fa_ranked[0][0]
        sh_top = sh_ranked[0][0]
        agree = "AGREE" if fa_top == sh_top else "DISAGREE"

        print(f"  [{i}/{len(selected)}] {abstract.abstract_id} "
              f"({len(abstract.sections)} sections)")
        print(f"    FA     ({fa_result.attribution_time_s:.1f}s, "
              f"{fa_result.num_forward_passes} fwd): {fa_str}")
        print(f"    Shapley({sh_result.attribution_time_s:.1f}s, "
              f"{sh_result.num_forward_passes} fwd): {sh_str}")
        print(f"    Top section: {agree} (FA={fa_top}, Sh={sh_top})")
        print()

    # Summary
    print(f"[5] Timing summary:")
    print(f"    Feature Ablation:  avg {sum(fa_times)/len(fa_times):.1f}s per abstract")
    print(f"    Shapley (n={SHAPLEY_SAMPLES}): avg {sum(sh_times)/len(sh_times):.1f}s per abstract")

    fa_proj = (sum(fa_times) / len(fa_times) * 2189) / 3600
    sh_proj = (sum(sh_times) / len(sh_times) * 2189) / 3600
    print(f"\n[6] Full-dataset projection (2189 abstracts, single GPU):")
    print(f"    Feature Ablation: ~{fa_proj:.1f} hours")
    print(f"    Shapley:          ~{sh_proj:.1f} hours")

    # Agreement summary
    print(f"\n[7] Method agreement on top section: shown per abstract above")

    print(f"\n=== Done. Results in {out_dir_fa.parent} ===")


if __name__ == "__main__":
    main()
