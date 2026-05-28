"""Environment and model configuration parsing for DyDim.

Only DyDim-facing parameter names are accepted here. Legacy experiment knobs
from earlier repositories are intentionally not parsed.
"""

from dataclasses import dataclass
import os
from pathlib import Path

from .constants import (
    DEFAULT_CANDIDATE_LANGUAGES,
    DEFAULT_TASKS,
    VALID_PROMPT_SETTINGS,
    VALID_SEMANTIC_VECTORS,
)
from .languages import normalize_language
from .utils import parse_bool, parse_positive_int_list, project_root, split_values


@dataclass(frozen=True)
class ModelSpec:
    """A local causal-LM checkpoint and the model label used in CSV outputs."""

    path: Path
    tag: str


@dataclass(frozen=True)
class DyDimConfig:
    """Fully resolved runtime configuration shared by all DyDim modules."""

    root_dir: Path
    models: list[ModelSpec]
    prompt_settings: list[str]
    semantic_vectors: list[str]
    language_counts: list[int]
    dimension_sizes: list[int]
    tasks: list[str]
    candidate_languages: list[str]
    batch_size: int
    max_length: int
    translation_model: str
    translation_device: str
    translation_batch_size: int
    translation_max_length: int
    translation_cache_namespace: str
    translation_cache_dir: Path
    rank_cache_dir: Path
    results_dir: Path
    force_rerank: bool
    dry_run: bool


def _normalize_tasks(raw_tasks: list[str]) -> list[str]:
    """Deduplicate task names while preserving the original MTEB names."""
    normalized: list[str] = []
    seen: set[str] = set()
    for task in raw_tasks:
        task_name = task.strip()
        if task_name and task_name not in seen:
            seen.add(task_name)
            normalized.append(task_name)
    if not normalized:
        raise ValueError("At least one task must be configured.")
    return normalized


def _normalize_choice_list(name: str, values: list[str], valid: set[str]) -> list[str]:
    """Validate a whitespace/comma separated enum-style option list."""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        if item not in valid:
            raise ValueError(f"Unsupported {name}: {item!r}. Choices: {sorted(valid)}.")
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _parse_model_specs_env(value: str) -> list[ModelSpec]:
    """Parse MODEL_SPECS='path|tag;path2|tag2' for batch model runs."""
    specs: list[ModelSpec] = []
    for raw_item in value.replace("\n", ";").split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if "|" not in item:
            raise ValueError(
                "MODEL_SPECS entries must use 'local_model_path|model_tag' format. "
                f"Got {item!r}."
            )
        path_s, tag = item.split("|", 1)
        specs.append(ModelSpec(path=Path(path_s).expanduser(), tag=tag.strip()))
    if not specs:
        raise ValueError("MODEL_SPECS did not contain any model entries.")
    return specs


def _read_model_config(path: Path) -> list[ModelSpec]:
    """Read configs/models.yaml when present.

    The checked-in repository only contains models.example.yaml, so paper
    readers must provide their own local paths without editing source code.
    """
    if not path.exists():
        return []
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Reading YAML model configs requires PyYAML.") from exc

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    models = payload.get("models", [])
    specs: list[ModelSpec] = []
    for item in models:
        if not isinstance(item, dict):
            raise ValueError(f"Each model entry must be a mapping, got {item!r}.")
        model_path = item.get("path")
        model_tag = item.get("tag")
        if not model_path or not model_tag:
            raise ValueError("Each model entry must include 'path' and 'tag'.")
        specs.append(ModelSpec(path=Path(str(model_path)).expanduser(), tag=str(model_tag)))
    return specs


def _load_models(root: Path, dry_run: bool) -> list[ModelSpec]:
    """Resolve model specs from env vars first, then project config."""
    model_specs_env = os.environ.get("MODEL_SPECS")
    if model_specs_env and model_specs_env.strip():
        return _parse_model_specs_env(model_specs_env)

    model_path = os.environ.get("MODEL_PATH")
    model_tag = os.environ.get("MODEL_TAG")
    if model_path and model_tag:
        return [ModelSpec(path=Path(model_path).expanduser(), tag=model_tag)]

    config_path = Path(os.environ.get("MODEL_CONFIG", root / "configs" / "models.yaml")).expanduser()
    specs = _read_model_config(config_path)
    if specs:
        return specs

    if dry_run:
        return [ModelSpec(path=Path("<set MODEL_PATH or configs/models.yaml>"), tag="example/causal-lm")]

    raise ValueError(
        "No model configured. Set MODEL_SPECS, set MODEL_PATH and MODEL_TAG, "
        "or create configs/models.yaml from configs/models.example.yaml."
    )


def load_config_from_env() -> DyDimConfig:
    """Build a DyDimConfig from the public shell/env interface."""
    root = Path(os.environ.get("DYDIM_ROOT", project_root())).expanduser().resolve()
    dry_run = parse_bool(os.environ.get("DYDIM_DRY_RUN"), default=False)
    models = _load_models(root, dry_run=dry_run)

    prompt_settings = _normalize_choice_list(
        "PROMPT_SETTINGS",
        split_values(os.environ.get("PROMPT_SETTINGS"), ["english", "language-specific"]),
        VALID_PROMPT_SETTINGS,
    )
    semantic_vectors = _normalize_choice_list(
        "SEMANTIC_VECTORS",
        split_values(os.environ.get("SEMANTIC_VECTORS"), ["source", "mean"]),
        VALID_SEMANTIC_VECTORS,
    )
    language_counts = parse_positive_int_list(os.environ.get("LANGUAGE_COUNTS"), [12])
    dimension_sizes = parse_positive_int_list(
        os.environ.get("DIMENSION_SIZES"),
        [256, 512, 768, 1024, 1280, 2048],
    )
    tasks = _normalize_tasks(split_values(os.environ.get("TASKS"), DEFAULT_TASKS))
    candidate_languages = [
        normalize_language(language)
        for language in split_values(os.environ.get("CANDIDATE_LANGUAGES"), DEFAULT_CANDIDATE_LANGUAGES)
    ]
    candidate_languages = list(dict.fromkeys(candidate_languages))
    if not candidate_languages:
        raise ValueError("CANDIDATE_LANGUAGES must not be empty.")
    if max(language_counts) > len(candidate_languages):
        raise ValueError(
            f"Max LANGUAGE_COUNTS={max(language_counts)} exceeds candidate language count "
            f"{len(candidate_languages)}."
        )

    translation_model = os.environ.get("TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")
    translation_device = os.environ.get("TRANSLATION_DEVICE", "cpu")
    translation_batch_size = int(os.environ.get("TRANSLATION_BATCH_SIZE", "8"))
    if translation_batch_size <= 0:
        raise ValueError("TRANSLATION_BATCH_SIZE must be positive.")
    translation_namespace = os.environ.get("TRANSLATION_CACHE_NAMESPACE", "nllb-200-distilled-600M")

    return DyDimConfig(
        root_dir=root,
        models=models,
        prompt_settings=prompt_settings,
        semantic_vectors=semantic_vectors,
        language_counts=language_counts,
        dimension_sizes=dimension_sizes,
        tasks=tasks,
        candidate_languages=candidate_languages,
        batch_size=int(os.environ.get("BATCH_SIZE", "1")),
        max_length=int(os.environ.get("MAX_LENGTH", "512")),
        translation_model=translation_model,
        translation_device=translation_device,
        translation_batch_size=translation_batch_size,
        translation_max_length=int(os.environ.get("TRANSLATION_MAX_LENGTH", "256")),
        translation_cache_namespace=translation_namespace,
        translation_cache_dir=Path(
            os.environ.get("TRANSLATION_CACHE_DIR", root / "translation_cache")
        ).expanduser(),
        rank_cache_dir=Path(os.environ.get("RANK_CACHE_DIR", root / "rank_cache")).expanduser(),
        results_dir=Path(os.environ.get("RESULTS_DIR", root / "results")).expanduser(),
        force_rerank=parse_bool(os.environ.get("FORCE_RERANK"), default=False),
        dry_run=dry_run,
    )
