"""Step 4: 10-abstract benchmark."""
import time
import torch
from src.data.load_csabstruct import load_csabstruct
from src.models.load_model import load_model
from src.attribution.feature_ablation import feature_ablation
from src.generation.prompts import HYPOTHESIS_PROMPT_V1
from src.utils.io import get_output_dir, save_json

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 80
SEED = 42
N_ABSTRACTS = 10
MIN_SECTIONS = 4

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
    print(f"=== Step 4: {N_ABSTRACTS}-abstract benchmark ===\n")

    print("[1] Loading CSABSTRUCT test split and filtering...")
    all_abstracts = load_csabstruct("test")
    rich_abstracts = [a for a in all_abstracts if len(a.sections) >= MIN_SECTIONS]
    print(f"    {len(rich_abstracts)}/{len(all_abstracts)} abstracts have "
          f">={MIN_SECTIONS} sections")
    selected = rich_abstracts[:N_ABSTRACTS]
    if len(selected) < N_ABSTRACTS:
        print(f"    WARNING: only {len(selected)} qualifying abstracts found")
    print(f"    Selected: {[a.abstract_id for a in selected]}\n")

    print(f"[2] Loading model {MODEL_NAME}...")
    t0 = time.time()
    model, tokenizer = load_model(MODEL_NAME)
    print(f"    Model loaded on {model.device} in {time.time() - t0:.1f}s\n")

    out_dir = get_output_dir("benchmark_10")
    print(f"[3] Output dir: {out_dir}\n")

    print(f"[4] Processing {len(selected)} abstracts...\n")
    per_abstract_times = []
    all_results = []

    for i, abstract in enumerate(selected, 1):
        t_start = time.time()

        t_gen = time.time()
        hypothesis = generate_hypothesis(
            model, tokenizer, abstract.full_text, HYPOTHESIS_PROMPT_V1
        )
        gen_time = time.time() - t_gen

        result = feature_ablation(
            model, tokenizer, abstract, hypothesis, HYPOTHESIS_PROMPT_V1,
            model_name=MODEL_NAME,
            generation_params=GENERATION_PARAMS,
        )
        result.generation_time_s = round(gen_time, 3)
        result.total_time_s = round(time.time() - t_start, 3)

        save_json(result, out_dir / f"{abstract.abstract_id}.json")
        all_results.append(result)
        per_abstract_times.append(result.total_time_s)

        ranked = sorted(result.attributions.items(), key=lambda x: x[1], reverse=True)
        ranked_str = ", ".join(f"{s}:{v:.1f}" for s, v in ranked)
        print(f"  [{i}/{len(selected)}] {abstract.abstract_id} "
              f"({len(abstract.sections)} sections, {result.total_time_s:.1f}s) "
              f"-> {ranked_str}")

    total = sum(per_abstract_times)
    avg = total / len(per_abstract_times)
    print(f"\n[5] Throughput summary:")
    print(f"    Total compute time: {total:.1f}s")
    print(f"    Avg per abstract:   {avg:.1f}s")
    print(f"    Min / max per abs:  {min(per_abstract_times):.1f}s / "
          f"{max(per_abstract_times):.1f}s")

    full_size = 2189
    projected_hours = (avg * full_size) / 3600
    print(f"\n[6] Projection for full dataset ({full_size} abstracts):")
    print(f"    ~{projected_hours:.1f} hours single-GPU (Feature Ablation only)")
    print(f"    ~{projected_hours * 25:.1f} hours adding Shapley (100 samples)")

    print(f"\n=== Done. Results in {out_dir} ===")


if __name__ == "__main__":
    main()
