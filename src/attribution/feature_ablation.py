"""Feature Ablation for rhetorical sections of scientific abstracts."""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.data.load_csabstruct import Abstract
from src.attribution.value_function import hypothesis_log_prob


@dataclass
class SectionInfo:
    """Metadata about one rhetorical section of an abstract."""
    label: str
    sentences: List[str]
    num_sentences: int
    num_words: int
    num_chars: int


@dataclass
class AblationResult:
    """Complete record of feature ablation for one abstract-hypothesis pair."""
    # Identity
    abstract_id: str
    hypothesis: str

    # Abstract metadata
    full_text: str
    sentences: List[str]
    labels: List[str]
    sections_info: Dict[str, dict]       # label -> SectionInfo as dict

    # Attribution scores
    baseline_log_prob: float
    ablated_log_probs: Dict[str, float]
    attributions: Dict[str, float]

    # Experiment config
    model_name: str
    prompt_template: str
    generation_params: dict
    attribution_method: str = "feature_ablation"

    # Timing
    generation_time_s: float = 0.0
    attribution_time_s: float = 0.0
    total_time_s: float = 0.0

    # Metadata
    num_forward_passes: int = 0


def _build_sections_info(abstract: Abstract) -> Dict[str, dict]:
    """Build detailed metadata for each section."""
    info = {}
    for label, sents in abstract.sections.items():
        text = " ".join(sents)
        info[label] = {
            "label": label,
            "sentences": sents,
            "num_sentences": len(sents),
            "num_words": len(text.split()),
            "num_chars": len(text),
        }
    return info


def _abstract_without_section(abstract: Abstract, section_to_remove: str) -> str:
    kept = [s for s, lbl in zip(abstract.sentences, abstract.labels)
            if lbl != section_to_remove]
    return " ".join(kept)


def feature_ablation(model, tokenizer, abstract: Abstract, hypothesis: str,
                     prompt_template: str, model_name: str = "",
                     generation_params: dict = None) -> AblationResult:
    """Run feature ablation across all sections present in this abstract."""
    t_start = time.time()

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
        attributions[section] = baseline_lp - lp

    t_end = time.time()
    num_passes = 1 + len(abstract.sections)  # baseline + one per section

    return AblationResult(
        abstract_id=abstract.abstract_id,
        hypothesis=hypothesis,
        full_text=abstract.full_text,
        sentences=abstract.sentences,
        labels=abstract.labels,
        sections_info=_build_sections_info(abstract),
        baseline_log_prob=baseline_lp,
        ablated_log_probs=ablated_lps,
        attributions=attributions,
        model_name=model_name,
        prompt_template=prompt_template,
        generation_params=generation_params or {},
        attribution_time_s=round(t_end - t_start, 3),
        num_forward_passes=num_passes,
    )
