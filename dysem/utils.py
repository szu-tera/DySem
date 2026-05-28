"""Small utility functions shared across DyDim modules."""

import csv
import hashlib
import re
from pathlib import Path
from typing import Iterable


def project_root() -> Path:
    """Return the DyDim project root directory."""
    return Path(__file__).resolve().parents[1]


def parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse common shell-style boolean values."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def split_values(value: str | None, default: Iterable[str]) -> list[str]:
    """Split comma/whitespace separated environment values."""
    if value is None:
        return list(default)
    tokens = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    return tokens or list(default)


def parse_positive_int_list(value: str | None, default: Iterable[int]) -> list[int]:
    """Parse positive integers and simple closed ranges like 1-4."""
    raw_tokens = split_values(value, [str(item) for item in default])
    values: list[int] = []
    seen: set[int] = set()
    for token in raw_tokens:
        expanded: list[int]
        if "-" in token[1:]:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start <= 0 or end <= 0 or start > end:
                raise ValueError(f"Invalid positive integer range: {token!r}.")
            expanded = list(range(start, end + 1))
        else:
            expanded = [int(token)]
        for value_int in expanded:
            if value_int <= 0:
                raise ValueError(f"Expected a positive integer, got {value_int}.")
            if value_int not in seen:
                seen.add(value_int)
                values.append(value_int)
    if not values:
        raise ValueError("Expected at least one positive integer.")
    return values


def slugify(value: str) -> str:
    """Make a stable filesystem-safe identifier."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "unnamed"


def short_hash(values: Iterable[object], length: int = 10) -> str:
    """Hash ordered config values into a short cache-key token."""
    payload = "\n".join(str(value) for value in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def validate_csv_columns(path: Path, required_columns: set[str]) -> bool:
    """Check whether a CSV exists and contains required headers."""
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except Exception:
        return False
    return required_columns.issubset(set(header))


def pairwise_cosine(a, b):
    """Compute row-wise cosine similarity."""
    import numpy as np

    dot = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    scores = np.zeros((a.shape[0],), dtype=np.float32)
    valid = denom > 1e-9
    scores[valid] = dot[valid] / denom[valid]
    return scores


def safe_spearman_percent(gold_scores, predicted_scores) -> float:
    """Return Spearman correlation as a percentage, mapping NaN to zero."""
    import numpy as np
    from scipy.stats import spearmanr

    corr, _ = spearmanr(gold_scores, predicted_scores)
    if np.isnan(corr):
        return 0.0
    return float(corr * 100.0)
