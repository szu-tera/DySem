"""Command-line orchestration for a full DyDim evaluation run."""

from .config import DyDimConfig, load_config_from_env


def _print_config(config: DyDimConfig):
    """Print the resolved run configuration before touching model weights."""
    print("DyDim configuration")
    print(f"  Root: {config.root_dir}")
    print(f"  Prompt settings: {' '.join(config.prompt_settings)}")
    print(f"  Semantic vectors: {' '.join(config.semantic_vectors)}")
    print(f"  Language counts: {' '.join(str(v) for v in config.language_counts)}")
    print(f"  Dimension sizes: {' '.join(str(v) for v in config.dimension_sizes)}")
    print(f"  Tasks: {' '.join(config.tasks)}")
    print(f"  Candidate languages: {' '.join(config.candidate_languages)}")
    print(f"  Translation model: {config.translation_model}")
    print(f"  Translation device: {config.translation_device}")
    print(f"  Translation batch size: {config.translation_batch_size}")
    print(f"  Translation cache: {config.translation_cache_dir}")
    print(f"  Rank cache: {config.rank_cache_dir}")
    print(f"  Results: {config.results_dir}")
    print(f"  Force rerank: {config.force_rerank}")
    for model in config.models:
        print(f"  Model: {model.tag} | {model.path}")


def _dry_run(config: DyDimConfig):
    """Show planned cache/result files without importing heavy ML dependencies."""
    from .language_ranking import rank_cache_path
    from .utils import slugify

    _print_config(config)
    print("\nDry-run planned files")
    for model in config.models:
        for prompt_setting in config.prompt_settings:
            print(f"  Rank cache: {rank_cache_path(config, model, prompt_setting)}")
            for semantic_vector in config.semantic_vectors:
                for language_count in config.language_counts:
                    for dimension_size in config.dimension_sizes:
                        result_path = config.results_dir / (
                            f"dydim_{slugify(model.tag)}"
                            f"__prompt-{slugify(prompt_setting)}"
                            f"__semantic-{slugify(semantic_vector)}"
                            f"__lang{int(language_count)}"
                            f"__dim{int(dimension_size)}.csv"
                        )
                        print(
                            f"  Result: {result_path}"
                        )


def _validate_model_paths(config: DyDimConfig):
    """Fail early when a configured local model path is missing."""
    for model in config.models:
        if not model.path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model.path}")


def main():
    config = load_config_from_env()
    if config.dry_run:
        _dry_run(config)
        return

    # Keep heavyweight imports out of dry-run so configuration checks work in a
    # lightweight environment.
    from .attention_vectors import AttentionSumEncoder
    from .evaluator import evaluate_grid
    from .language_ranking import ensure_rank_cache
    from .model_loader import load_causal_lm
    from .translation import TranslationManager

    _print_config(config)
    _validate_model_paths(config)
    config.translation_cache_dir.mkdir(parents=True, exist_ok=True)
    config.rank_cache_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    for model_spec in config.models:
        print("\n" + "=" * 80)
        print(f"Loading model: {model_spec.tag}")
        print("=" * 80)
        model, tokenizer = load_causal_lm(model_spec.path, model_spec.tag)
        encoder = AttentionSumEncoder(model, tokenizer, max_length=config.max_length)
        translator = TranslationManager(
            config.translation_cache_dir,
            config.translation_cache_namespace,
            config.translation_model,
            device=config.translation_device,
            batch_size=config.translation_batch_size,
            max_length=config.translation_max_length,
        )
        try:
            for prompt_setting in config.prompt_settings:
                print("\n" + "-" * 80)
                print(f"Prompt setting: {prompt_setting}")
                print("-" * 80)
                rank_csv = ensure_rank_cache(
                    config,
                    model_spec,
                    prompt_setting=prompt_setting,
                    encoder=encoder,
                    translator=translator,
                )
                # Main evaluation reuses the prompt-specific language ranking
                # and writes only final per-grid CSV files.
                evaluate_grid(
                    config,
                    model_spec,
                    prompt_setting=prompt_setting,
                    rank_cache_csv=rank_csv,
                    encoder=encoder,
                    translator=translator,
                )
        finally:
            translator.close()
