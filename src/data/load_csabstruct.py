"""Load CSABSTRUCT from HuggingFace and expose it as simple Python objects."""
from dataclasses import dataclass
from typing import List
from datasets import load_dataset


# CSABSTRUCT label order (from the HuggingFace dataset's ClassLabel feature)
LABEL_NAMES = ["background", "objective", "method", "result", "other"]


@dataclass
class Abstract:
    abstract_id: str
    sentences: List[str]
    labels: List[str]          # now string names, e.g. "background"
    sections: dict             # {label_name: [sentences]}

    @property
    def full_text(self) -> str:
        return " ".join(self.sentences)


def _group_by_label(sentences, labels):
    sections = {}
    for sent, label in zip(sentences, labels):
        sections.setdefault(label, []).append(sent)
    return sections


def _map_label(label):
    """Handle both integer and string labels defensively."""
    if isinstance(label, int):
        return LABEL_NAMES[label]
    return str(label).lower()


def load_csabstruct(split: str = "test") -> List[Abstract]:
    ds = load_dataset("allenai/csabstruct", split=split)
    abstracts = []
    for i, row in enumerate(ds):
        sentences = row["sentences"]
        labels = [_map_label(l) for l in row["labels"]]
        abstracts.append(
            Abstract(
                abstract_id=f"{split}_{i}",
                sentences=sentences,
                labels=labels,
                sections=_group_by_label(sentences, labels),
            )
        )
    return abstracts


if __name__ == "__main__":
    abstracts = load_csabstruct("test")
    print(f"Loaded {len(abstracts)} abstracts from test split")
    a = abstracts[0]
    print(f"\nFirst abstract ({a.abstract_id}):")
    print(f"  Num sentences: {len(a.sentences)}")
    print(f"  Label distribution: {[(k, len(v)) for k, v in a.sections.items()]}")
    print(f"  First 2 sentences: {a.sentences[:2]}")