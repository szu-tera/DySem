"""Language-code normalization helpers for prompts and translation."""

from .constants import LANGUAGE_ALIASES


DEFAULT_LANGUAGE = "eng_Latn"


def normalize_language(language: str | None, fallback: str = DEFAULT_LANGUAGE) -> str:
    """Normalize short language aliases to NLLB-style language codes."""
    if language is None:
        return fallback
    raw = str(language).strip()
    if not raw:
        return fallback
    if "_" in raw and raw in set(LANGUAGE_ALIASES.values()):
        return raw
    return LANGUAGE_ALIASES.get(raw.lower(), fallback)


def infer_subset_language(subset_name: str, sentence_index: int) -> str:
    """Infer source language from multilingual STS subset names when possible."""
    subset = str(subset_name or "default")
    if subset == "default":
        return DEFAULT_LANGUAGE
    for separator in ("-", "_"):
        if separator in subset:
            pieces = [piece for piece in subset.split(separator) if piece]
            if len(pieces) >= 2 and sentence_index in {0, 1}:
                return normalize_language(pieces[sentence_index])
    return DEFAULT_LANGUAGE
