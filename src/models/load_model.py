"""Unified wrapper for loading HuggingFace causal LMs."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device: str = "auto", dtype: str = "float16"):
    """Load a causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        device: 'auto', 'cuda', 'cpu', or 'mps'.
        dtype: 'float16', 'bfloat16', or 'float32'.

    Returns:
        (model, tokenizer) tuple. Model is in eval mode.
    """
    torch_dtype = {"float16": torch.float16,
                   "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    return model, tokenizer