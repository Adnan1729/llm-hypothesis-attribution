"""Shapley Value Sampling for rhetorical sections of scientific abstracts.

Estimates each section's Shapley value via Monte Carlo sampling: randomly
permute the sections, add them one by one, and measure each section's
marginal contribution to the hypothesis log-probability.
"""
import time
import random
from dataclasses import dataclass
from typing import Dict, List
from src.data.load_csabstruct import Abstract
from src.attribution.value_function import hypothesis_log_prob
from src.attribution.feature_ablation import _build_sections_info


@dataclass
class ShapleyResult:
    """Complete record of Shapley value estimation for one abstract-hypothesis pair."""
    # Identity
    abstract_id: str
    hypothesis: str

    # Abstract metadata
    full_text: str
    sentences: List[str]
    labels: List[str]
    sections_info: Dict[str, dict]

    # Shapley values (mean marginal contribution per section)
    shapley_values: Dict[str, float]

    # Raw data: all marginal contributions per section across all permutations
    marginal_contributions: Dict[str, List[float]]

    # Experiment config
    model_name: str
    prompt_template: str
    generation_params: dict
    attribution_method: str = "shapley_value_sampling"
    num_samples: int = 0

    # Timing
    generation_time_s: float = 0.0
    attribution_time_s: float = 0.0
    total_time_s: float = 0.0

    # Cost
    num_forward_passes: int = 0


def _build_context_from_sections(abstract: Abstract, section_labels: List[str]) -> str:
    """Build abstract text containing only sentences from the given sections,
    preserving original sentence order."""
    kept = [s for s, lbl in zip(abstract.sentences, abstract.labels)
            if lbl in section_labels]
    return " ".join(kept)


def shapley_value_sampling(model, tokenizer, abstract: Abstract, hypothesis: str,
                           prompt_template: str, num_samples: int = 100,
                           model_name: str = "", generation_params: dict = None,
                           seed: int = 42) -> ShapleyResult:
    """Estimate Shapley values via Monte Carlo permutation sampling.

    For each sample:
      1. Generate a random permutation of the sections
      2. Walk through the permutation, adding one section at a time
      3. Record the marginal contribution of each section
         (log-prob with section minus log-prob without)

    The Shapley value for each section is the mean of its marginal
    contributions across all samples.
    """
    t_start = time.time()
    rng = random.Random(seed)

    section_labels = list(abstract.sections.keys())
    num_sections = len(section_labels)

    # Collect marginal contributions
    marginals: Dict[str, List[float]] = {s: [] for s in section_labels}
    forward_passes = 0

    # Cache for coalition log-probs to avoid redundant computation
    # Key: frozenset of section labels, Value: log-prob
    cache: Dict[frozenset, float] = {}

    def get_coalition_logprob(coalition_labels: frozenset) -> float:
        nonlocal forward_passes
        if coalition_labels not in cache:
            if len(coalition_labels) == 0:
                context = ""
            else:
                context = _build_context_from_sections(abstract, list(coalition_labels))
            cache[coalition_labels] = hypothesis_log_prob(
                model, tokenizer, context, hypothesis, prompt_template
            )
            forward_passes += 1
        return cache[coalition_labels]

    for sample_idx in range(num_samples):
        # Random permutation of sections
        perm = section_labels.copy()
        rng.shuffle(perm)

        # Walk through permutation, computing marginal contributions
        current_coalition = frozenset()
        for section in perm:
            # Log-prob without this section
            lp_without = get_coalition_logprob(current_coalition)

            # Log-prob with this section added
            new_coalition = current_coalition | {section}
            lp_with = get_coalition_logprob(new_coalition)

            # Marginal contribution
            marginals[section].append(lp_with - lp_without)

            current_coalition = new_coalition

    # Compute mean Shapley values
    shapley_values = {s: sum(m) / len(m) for s, m in marginals.items()}

    t_end = time.time()

    return ShapleyResult(
        abstract_id=abstract.abstract_id,
        hypothesis=hypothesis,
        full_text=abstract.full_text,
        sentences=abstract.sentences,
        labels=abstract.labels,
        sections_info=_build_sections_info(abstract),
        shapley_values=shapley_values,
        marginal_contributions=marginals,
        model_name=model_name,
        prompt_template=prompt_template,
        generation_params=generation_params or {},
        num_samples=num_samples,
        attribution_time_s=round(t_end - t_start, 3),
        num_forward_passes=forward_passes,
    )
