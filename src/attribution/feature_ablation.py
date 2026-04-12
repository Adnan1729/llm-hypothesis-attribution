"""Feature Ablation for rhetorical sections of scientific abstracts.

For each section, we remove its sentences from the abstract and measure how
much the hypothesis log-probability drops. A larger drop means that section
contributed more to the model's generation of the hypothesis.
"""
from dataclasses import dataclass
from typing import Dict, List
from src.data.load_csabstruct import Abstract
from src.attribution.value_function import hypothesis_log_prob


@dataclass
class AblationResult:
    """Results of feature ablation for one abstract-hypothesis pair."""
    abstract_id: str
    hypothesis: str
    baseline_log_prob: float              # log P(hyp | full abstract)
    ablated_log_probs: Dict[str, float]   # section name -> log P(hyp | abstract minus section)
    attributions: Dict[str, float]        # section name -> baseline - ablated (drop magnitude)


def _abstract_without_section(abstract: Abstract, section_to_remove: str) -> str:
    """Reconstruct abstract text with all sentences of one section removed.

    We iterate sentences in their original order so the remaining text stays
    coherent; we just skip sentences whose label matches the ablated section.
    """
    kept = [s for s, lbl in zip(abstract.sentences, abstract.labels)
            if lbl != section_to_remove]
    return " ".join(kept)


def feature_ablation(model, tokenizer, abstract: Abstract, hypothesis: str,
                     prompt_template: str) -> AblationResult:
    """Run feature ablation across all sections present in this abstract.

    Note: only sections actually present in the abstract are ablated. If an
    abstract has no 'method' sentences, there's no 'method' ablation to run.
    """
    # Baseline: full abstract
    baseline_lp = hypothesis_log_prob(
        model, tokenizer, abstract.full_text, hypothesis, prompt_template
    )

    ablated_lps = {}
    attributions = {}
    for section in abstract.sections.keys():
        ablated_text = _abstract_without_section(abstract, section)
        lp = hypothesis_log_prob(
            model, tokenizer, ablated_text, hypothesis, prompt_template
        )
        ablated_lps[section] = lp
        # Drop in log-prob when this section is removed = its attribution
        attributions[section] = baseline_lp - lp

    return AblationResult(
        abstract_id=abstract.abstract_id,
        hypothesis=hypothesis,
        baseline_log_prob=baseline_lp,
        ablated_log_probs=ablated_lps,
        attributions=attributions,
    )