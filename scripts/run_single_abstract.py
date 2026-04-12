"""Step 3: end-to-end pipeline on a single abstract, with Feature Ablation."""
import torch
from src.data.load_csabstruct import load_csabstruct
from src.models.load_model import load_model
from src.attribution.value_function import hypothesis_log_prob
from src.attribution.feature_ablation import feature_ablation
from src.generation.prompts import HYPOTHESIS_PROMPT_V1

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 80
SEED = 42


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


def pick_abstract_with_enough_sections(abstracts, min_sections=3):
    """Find an abstract that has at least `min_sections` different labels,
    so ablation is actually interesting."""
    for a in abstracts:
        if len(a.sections) >= min_sections:
            return a
    return abstracts[0]  # fallback


def main():
    print("=== Step 3: single-abstract pipeline with Feature Ablation ===\n")

    print("[1] Loading CSABSTRUCT test split...")
    abstracts = load_csabstruct("test")
    abstract = pick_abstract_with_enough_sections(abstracts, min_sections=3)
    print(f"    Using abstract {abstract.abstract_id}")
    print(f"    Sentences: {len(abstract.sentences)}")
    print(f"    Sections: {[(k, len(v)) for k, v in abstract.sections.items()]}\n")

    print(f"[2] Loading model {MODEL_NAME}...")
    model, tokenizer = load_model(MODEL_NAME)
    print(f"    Device: {model.device}\n")

    print("[3] Generating hypothesis from full abstract...")
    hypothesis = generate_hypothesis(
        model, tokenizer, abstract.full_text, HYPOTHESIS_PROMPT_V1
    )
    print(f"    Hypothesis: {hypothesis}\n")

    print("[4] Running Feature Ablation across all sections...")
    result = feature_ablation(
        model, tokenizer, abstract, hypothesis, HYPOTHESIS_PROMPT_V1
    )

    print(f"\n    Baseline log P(hyp | full abstract): {result.baseline_log_prob:.4f}\n")
    print(f"    {'Section':<15} {'Ablated LP':>12} {'Drop':>10}")
    print(f"    {'-' * 40}")
    # Sort by attribution magnitude, largest drop first
    for section in sorted(result.attributions,
                          key=lambda k: result.attributions[k], reverse=True):
        lp = result.ablated_log_probs[section]
        drop = result.attributions[section]
        print(f"    {section:<15} {lp:>12.4f} {drop:>10.4f}")

    print(f"\n    Interpretation: a larger drop means removing that section")
    print(f"    hurt the hypothesis log-probability more, so it contributed more.\n")

    print("=== Done. Check that the ranking makes intuitive sense. ===")


if __name__ == "__main__":
    main()