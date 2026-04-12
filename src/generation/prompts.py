"""Prompt templates. Keep these explicit and versioned."""

HYPOTHESIS_PROMPT_V1 = (
    "Read this scientific paper abstract and identify its main hypothesis. "
    "Be specific and concise.\n\n"
    "Abstract: {context}\n\n"
    "Main hypothesis:"
)