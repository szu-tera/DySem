"""Final DyDim STS evaluation and CSV writing."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DyDimConfig, ModelSpec
from .constants import REPRESENTATION_NAME
from .language_ranking import selected_languages
from .mteb_tasks import has_sts_columns, iter_sts_test_subsets, load_mteb_tasks, source_languages_for_subset, task_name
from .utils import safe_spearman_percent, slugify


def _topk_indices(
    values: np.ndarray,
    dimension_size: int,
    positive_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Select top dimensions inside the translated-language support mask.

    This mirrors the original Semantic_neurons behavior: top-k positions are
    ranked by the active value vector, but candidate dimensions must be positive
    across all selected translated-language vectors.
    """
    if values.ndim != 2:
        raise ValueError("values must be a 2D array.")
    if positive_mask is not None and positive_mask.shape != values.shape:
        raise ValueError(
            f"positive_mask shape {positive_mask.shape} does not match values shape {values.shape}."
        )
    batch_size, vector_dim = values.shape
    k = min(int(dimension_size), vector_dim)
    out = np.full((batch_size, int(dimension_size)), -1, dtype=np.int32)
    if k <= 0:
        return out

    for row_idx, row in enumerate(values):
        mask_row = row > 0 if positive_mask is None else positive_mask[row_idx]
        candidates = np.flatnonzero(mask_row)
        if candidates.size == 0:
            continue
        scores = row[candidates]
        if k >= candidates.size:
            order = np.argsort(scores)[::-1]
        else:
            partial = np.argpartition(scores, -k)[-k:]
            order = partial[np.argsort(scores[partial])[::-1]]
        selected = candidates[order][:k]
        out[row_idx, : selected.size] = selected.astype(np.int32, copy=False)
    return out


def _translated_positive_mask(translated_stack: np.ndarray, language_count: int) -> np.ndarray:
    """Return dimensions positive in every selected translated language."""
    selected_stack = translated_stack[:, : int(language_count), :]
    if selected_stack.ndim != 3 or selected_stack.shape[1] <= 0:
        raise ValueError("translated_stack must contain at least one selected language.")
    return np.all(selected_stack > 0, axis=1)


def _union_topk_cosine(vectors_a: np.ndarray, vectors_b: np.ndarray, topk_a: np.ndarray, topk_b: np.ndarray) -> np.ndarray:
    """Cosine similarity on the union of each sentence pair's top dimensions."""
    scores = np.zeros((vectors_a.shape[0],), dtype=np.float32)
    for row_idx, (indices_a, indices_b) in enumerate(zip(topk_a, topk_b)):
        valid_a = indices_a[indices_a >= 0]
        valid_b = indices_b[indices_b >= 0]
        support = np.union1d(valid_a, valid_b)
        if support.size == 0:
            continue
        vec_a = vectors_a[row_idx, support]
        vec_b = vectors_b[row_idx, support]
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom > 1e-9:
            scores[row_idx] = float(np.dot(vec_a, vec_b) / denom)
    return scores


def _result_path(
    config: DyDimConfig,
    model_spec: ModelSpec,
    *,
    prompt_setting: str,
    semantic_vector: str,
    language_count: int,
    dimension_size: int,
) -> Path:
    """Build the final CSV path for one public DyDim grid point."""
    return config.results_dir / (
        f"dydim_{slugify(model_spec.tag)}"
        f"__prompt-{slugify(prompt_setting)}"
        f"__semantic-{slugify(semantic_vector)}"
        f"__lang{int(language_count)}"
        f"__dim{int(dimension_size)}.csv"
    )


def _encode_translated_stack(
    texts: list[str],
    source_languages: list[str],
    selected_langs: list[str],
    *,
    prompt_setting: str,
    config: DyDimConfig,
    encoder,
    translator,
    progress_prefix: str,
) -> np.ndarray:
    """Encode each selected translation language and stack as [N, L, D]."""
    per_language_vectors = []
    for language in selected_langs:
        translated = translator.translate_texts(
            texts,
            source_languages=source_languages,
            target_language=language,
        )
        vectors = encoder.encode(
            translated,
            prompt_setting=prompt_setting,
            languages=[language] * len(translated),
            batch_size=config.batch_size,
            progress_desc=f"{progress_prefix}/{language}",
        )
        per_language_vectors.append(vectors)
    return np.stack(per_language_vectors, axis=1)


def _encode_source_vectors(
    texts: list[str],
    source_languages: list[str],
    *,
    prompt_setting: str,
    config: DyDimConfig,
    encoder,
    progress_desc: str,
) -> np.ndarray:
    """Encode original sentences for the `source` semantic vector setting."""
    prompt_languages = source_languages if prompt_setting == "language-specific" else ["eng_Latn"] * len(texts)
    return encoder.encode(
        texts,
        prompt_setting=prompt_setting,
        languages=prompt_languages,
        batch_size=config.batch_size,
        progress_desc=progress_desc,
    )


def evaluate_grid(
    config: DyDimConfig,
    model_spec: ModelSpec,
    *,
    prompt_setting: str,
    rank_cache_csv: Path,
    encoder,
    translator,
):
    """Evaluate all semantic-vector, language-count, and dimension-size combos."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_mteb_tasks(config.tasks)
    max_language_count = max(config.language_counts)
    selected_full = selected_languages(rank_cache_csv, max_language_count)

    grid_rows: dict[tuple[str, int, int], list[dict]] = {
        (semantic_vector, language_count, dimension_size): []
        for semantic_vector in config.semantic_vectors
        for language_count in config.language_counts
        for dimension_size in config.dimension_sizes
    }

    for task in tasks:
        name = task_name(task)
        translator.set_dataset(name)
        for subset_name, dataset in iter_sts_test_subsets(task):
            if not has_sts_columns(dataset):
                continue
            sentence1 = list(dataset["sentence1"])
            sentence2 = list(dataset["sentence2"])
            gold_scores = list(dataset["score"])
            source_langs_s1 = source_languages_for_subset(subset_name, 0, len(sentence1))
            source_langs_s2 = source_languages_for_subset(subset_name, 1, len(sentence2))

            source_s1 = source_s2 = None
            if "source" in config.semantic_vectors:
                # Source vectors are encoded once per subset and reused for all
                # language-count and dimension-size settings.
                source_s1 = _encode_source_vectors(
                    sentence1,
                    source_langs_s1,
                    prompt_setting=prompt_setting,
                    config=config,
                    encoder=encoder,
                    progress_desc=f"Eval {name}/{subset_name}/source sentence1",
                )
                source_s2 = _encode_source_vectors(
                    sentence2,
                    source_langs_s2,
                    prompt_setting=prompt_setting,
                    config=config,
                    encoder=encoder,
                    progress_desc=f"Eval {name}/{subset_name}/source sentence2",
                )

            # Both `source` and `mean` use the selected-language stack to define
            # the top-k support. For `source`, only the value vector is the
            # original sentence; the support still comes from translations.
            translated_stack_s1 = _encode_translated_stack(
                sentence1,
                source_langs_s1,
                selected_full,
                prompt_setting=prompt_setting,
                config=config,
                encoder=encoder,
                translator=translator,
                progress_prefix=f"Eval {name}/{subset_name}/support sentence1",
            )
            translated_stack_s2 = _encode_translated_stack(
                sentence2,
                source_langs_s2,
                selected_full,
                prompt_setting=prompt_setting,
                config=config,
                encoder=encoder,
                translator=translator,
                progress_prefix=f"Eval {name}/{subset_name}/support sentence2",
            )

            for language_count in config.language_counts:
                selected_langs = selected_full[:language_count]
                positive_mask_s1 = _translated_positive_mask(translated_stack_s1, language_count)
                positive_mask_s2 = _translated_positive_mask(translated_stack_s2, language_count)
                for semantic_vector in config.semantic_vectors:
                    if semantic_vector == "source":
                        vectors_s1 = source_s1
                        vectors_s2 = source_s2
                    elif semantic_vector == "mean":
                        vectors_s1 = translated_stack_s1[:, :language_count, :].mean(axis=1)
                        vectors_s2 = translated_stack_s2[:, :language_count, :].mean(axis=1)
                    else:
                        raise ValueError(f"Unsupported semantic vector: {semantic_vector!r}.")

                    for dimension_size in config.dimension_sizes:
                        topk_s1 = _topk_indices(vectors_s1, dimension_size, positive_mask_s1)
                        topk_s2 = _topk_indices(vectors_s2, dimension_size, positive_mask_s2)
                        predicted = _union_topk_cosine(vectors_s1, vectors_s2, topk_s1, topk_s2)
                        score = safe_spearman_percent(gold_scores, predicted)
                        grid_rows[(semantic_vector, language_count, dimension_size)].append(
                            {
                                "Task": name,
                                "Subset": subset_name,
                                "Score": score,
                                "Prompt_Setting": prompt_setting,
                                "Semantic_Vector": semantic_vector,
                                "Language_Count": language_count,
                                "Dimension_Size": dimension_size,
                                "Selected_Languages": "|".join(selected_langs),
                                "Model": model_spec.tag,
                                "Representation": REPRESENTATION_NAME,
                                "Rank_Cache_CSV": str(rank_cache_csv),
                                "Translation_Cache_Dir": str(config.translation_cache_dir),
                            }
                        )

    for (semantic_vector, language_count, dimension_size), rows in grid_rows.items():
        # The project writes only final result CSVs plus rank cache CSVs.
        selected_langs = selected_full[:language_count]
        average = float(np.mean([row["Score"] for row in rows])) if rows else 0.0
        rows.append(
            {
                "Task": "AVERAGE",
                "Subset": "AVERAGE",
                "Score": average,
                "Prompt_Setting": prompt_setting,
                "Semantic_Vector": semantic_vector,
                "Language_Count": language_count,
                "Dimension_Size": dimension_size,
                "Selected_Languages": "|".join(selected_langs),
                "Model": model_spec.tag,
                "Representation": REPRESENTATION_NAME,
                "Rank_Cache_CSV": str(rank_cache_csv),
                "Translation_Cache_Dir": str(config.translation_cache_dir),
            }
        )
        result_path = _result_path(
            config,
            model_spec,
            prompt_setting=prompt_setting,
            semantic_vector=semantic_vector,
            language_count=language_count,
            dimension_size=dimension_size,
        )
        pd.DataFrame(rows).to_csv(result_path, index=False)
        print(f"Saved result CSV: {result_path}")
