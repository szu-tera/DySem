"""Language ranking cache generation and reuse."""

from pathlib import Path

import numpy as np

from .config import DyDimConfig, ModelSpec
from .constants import REPRESENTATION_NAME
from .mteb_tasks import has_sts_columns, iter_sts_test_subsets, load_mteb_tasks, source_languages_for_subset, task_name
from .utils import pairwise_cosine, safe_spearman_percent, short_hash, slugify, validate_csv_columns


RANK_CACHE_COLUMNS = {
    "Rank",
    "Language",
    "Score",
    "Model",
    "Prompt_Setting",
    "Tasks",
    "Candidate_Languages",
}


def rank_cache_path(config: DyDimConfig, model_spec: ModelSpec, prompt_setting: str) -> Path:
    """Build the deterministic CSV path for a prompt/model ranking run."""
    tasks_hash = short_hash(config.tasks)
    languages_hash = short_hash(config.candidate_languages)
    translation_hash = short_hash(
        [
            config.translation_model,
            config.translation_cache_namespace,
            REPRESENTATION_NAME,
        ]
    )
    file_name = (
        f"dydim_rank__model-{slugify(model_spec.tag)}"
        f"__prompt-{slugify(prompt_setting)}"
        f"__tasks-{tasks_hash}"
        f"__langs-{languages_hash}"
        f"__trans-{translation_hash}.csv"
    )
    return config.rank_cache_dir / file_name


def _score_language(task, language: str, prompt_setting: str, config: DyDimConfig, encoder, translator) -> float:
    """Score one candidate language on one STS task for rank ordering."""
    subset_scores: list[float] = []
    translator.set_dataset(task_name(task))
    for subset_name, dataset in iter_sts_test_subsets(task):
        if not has_sts_columns(dataset):
            continue
        sentence1 = list(dataset["sentence1"])
        sentence2 = list(dataset["sentence2"])
        scores = list(dataset["score"])
        source_langs_s1 = source_languages_for_subset(subset_name, 0, len(sentence1))
        source_langs_s2 = source_languages_for_subset(subset_name, 1, len(sentence2))
        translated_s1 = translator.translate_texts(
            sentence1,
            source_languages=source_langs_s1,
            target_language=language,
        )
        translated_s2 = translator.translate_texts(
            sentence2,
            source_languages=source_langs_s2,
            target_language=language,
        )
        language_codes = [language] * len(translated_s1)
        vectors_s1 = encoder.encode(
            translated_s1,
            prompt_setting=prompt_setting,
            languages=language_codes,
            batch_size=config.batch_size,
            progress_desc=f"Ranking {task_name(task)}/{subset_name}/{language} sentence1",
        )
        vectors_s2 = encoder.encode(
            translated_s2,
            prompt_setting=prompt_setting,
            languages=language_codes,
            batch_size=config.batch_size,
            progress_desc=f"Ranking {task_name(task)}/{subset_name}/{language} sentence2",
        )
        subset_scores.append(safe_spearman_percent(scores, pairwise_cosine(vectors_s1, vectors_s2)))
    return float(np.mean(subset_scores)) if subset_scores else 0.0


def ensure_rank_cache(
    config: DyDimConfig,
    model_spec: ModelSpec,
    *,
    prompt_setting: str,
    encoder,
    translator,
) -> Path:
    """Return an existing valid rank cache or generate it once."""
    config.rank_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = rank_cache_path(config, model_spec, prompt_setting)
    if not config.force_rerank and validate_csv_columns(cache_path, RANK_CACHE_COLUMNS):
        print(f"Using existing rank cache: {cache_path}")
        return cache_path

    if config.force_rerank and cache_path.exists():
        print(f"FORCE_RERANK=1, regenerating rank cache: {cache_path}")
    else:
        print(f"Generating rank cache: {cache_path}")

    tasks = load_mteb_tasks(config.tasks)
    rows = []
    for language in config.candidate_languages:
        # Ranking uses dense attention-sum cosine before any dimension slicing.
        task_scores = [
            _score_language(task, language, prompt_setting, config, encoder, translator)
            for task in tasks
        ]
        rows.append(
            {
                "Language": language,
                "Score": float(np.mean(task_scores)) if task_scores else 0.0,
                "Model": model_spec.tag,
                "Prompt_Setting": prompt_setting,
                "Tasks": "|".join(config.tasks),
                "Candidate_Languages": "|".join(config.candidate_languages),
                "Representation": REPRESENTATION_NAME,
                "Translation_Model": config.translation_model,
                "Translation_Cache_Namespace": config.translation_cache_namespace,
            }
        )

    ranked = sorted(rows, key=lambda row: row["Score"], reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["Rank"] = idx

    ordered_columns = [
        "Rank",
        "Language",
        "Score",
        "Model",
        "Prompt_Setting",
        "Tasks",
        "Candidate_Languages",
        "Representation",
        "Translation_Model",
        "Translation_Cache_Namespace",
    ]
    import pandas as pd

    pd.DataFrame(ranked)[ordered_columns].to_csv(cache_path, index=False)
    print(f"Saved rank cache: {cache_path}")
    return cache_path


def selected_languages(rank_cache_csv: Path, language_count: int) -> list[str]:
    """Read top-N languages from a DyDim rank cache CSV."""
    import pandas as pd

    df = pd.read_csv(rank_cache_csv)
    df = df.sort_values("Rank", ascending=True)
    languages = df["Language"].astype(str).head(int(language_count)).tolist()
    if len(languages) < int(language_count):
        raise ValueError(
            f"Rank cache {rank_cache_csv} only has {len(languages)} languages, "
            f"but LANGUAGE_COUNTS requested {language_count}."
        )
    return languages
