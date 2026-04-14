"""I/O utilities for saving attribution results."""
import json
import os
from dataclasses import asdict
from pathlib import Path


def get_output_dir(subdir: str = "") -> Path:
    """Return the base output directory, creating it if needed.
    Reads PROJECT_SCRATCH env var; falls back to local 'outputs/' for laptop runs."""
    base = os.environ.get("PROJECT_SCRATCH")
    if base:
        out = Path(base) / "outputs" / subdir
    else:
        out = Path("outputs") / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj, path: Path):
    """Save a dataclass or dict as JSON. Handles dataclasses automatically."""
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
