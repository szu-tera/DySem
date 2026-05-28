"""Thin wrappers around MTEB STS task loading and subset handling."""

from .languages import infer_subset_language


def load_mteb_tasks(task_names: list[str]):
    """Load MTEB tasks lazily so dry-run does not require the package."""
    from mteb import get_tasks

    return get_tasks(tasks=task_names)


def task_name(task) -> str:
    """Return the stable MTEB task name used in caches and CSV files."""
    metadata = getattr(task, "metadata", None)
    return str(getattr(metadata, "name", task.__class__.__name__))


def iter_sts_test_subsets(task):
    """Yield test subsets that follow the STS sentence1/sentence2/score shape."""
    task.load_data()
    dataset = task.dataset
    if "test" in dataset and hasattr(dataset["test"], "column_names"):
        yield "default", dataset["test"]
        return
    for subset_name, subset_payload in dataset.items():
        if hasattr(subset_payload, "keys") and "test" in subset_payload:
            test_dataset = subset_payload["test"]
            if hasattr(test_dataset, "column_names"):
                yield str(subset_name), test_dataset


def has_sts_columns(dataset) -> bool:
    """Check that a dataset subset has the columns DyDim evaluates."""
    columns = set(getattr(dataset, "column_names", []))
    return {"sentence1", "sentence2", "score"}.issubset(columns)


def source_languages_for_subset(subset_name: str, sentence_index: int, count: int) -> list[str]:
    """Infer repeated source-language labels for one side of an STS pair."""
    return [infer_subset_language(subset_name, sentence_index)] * count
