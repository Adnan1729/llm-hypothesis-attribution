"""Value function: log-probability of a target hypothesis given a context.

This is the shared building block for Feature Ablation and Shapley Value Sampling.
For attribution, we compute log P(hypothesis | context) and measure how it changes
when context is modified (full abstract vs. ablated sections).
"""
import torch


@torch.no_grad()
def hypothesis_log_prob(model, tokenizer, context: str, hypothesis: str,
                        prompt_template: str) -> float:
    """Compute log P(hypothesis | context) under the model.

    We tokenize [prompt(context) + hypothesis], run a forward pass, and sum
    log-probs over only the hypothesis tokens.

    Args:
        model: causal LM (in eval mode).
        tokenizer: matching tokenizer.
        context: the (possibly ablated) abstract text.
        hypothesis: the fixed target hypothesis we're scoring.
        prompt_template: format string with a '{context}' placeholder.

    Returns:
        Sum of log-probabilities over hypothesis tokens (a float).
    """
    prompt = prompt_template.format(context=context)
    full_text = prompt + hypothesis

    # Tokenize prompt alone to find where the hypothesis starts
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids
    full_ids = full_ids.to(model.device)

    prompt_len = prompt_ids.shape[1]
    hyp_len = full_ids.shape[1] - prompt_len
    if hyp_len <= 0:
        raise ValueError("Hypothesis produced no tokens after prompt.")

    outputs = model(full_ids)
    logits = outputs.logits  # (1, seq_len, vocab)

    # Shift: token at position t is predicted from logits at position t-1
    # We want log-probs of hypothesis tokens only.
    hyp_logits = logits[0, prompt_len - 1 : -1, :]   # (hyp_len, vocab)
    hyp_targets = full_ids[0, prompt_len:]           # (hyp_len,)

    log_probs = torch.log_softmax(hyp_logits, dim=-1)
    token_log_probs = log_probs.gather(1, hyp_targets.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum().item()