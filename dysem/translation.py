"""Project-local translation cache and online translation support."""

import json
from pathlib import Path

import torch

from .languages import normalize_language
from .utils import slugify


class TranslationCache:
    """A JSON cache scoped to DyDim, translation namespace, and dataset."""

    def __init__(self, root_dir: Path, namespace: str, dataset_name: str):
        self.dataset_name = str(dataset_name)
        self.cache_dir = Path(root_dir) / slugify(namespace) / slugify(self.dataset_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "translation_cache.json"
        self.data = self._load()
        self.dirty = False

    def _load(self) -> dict:
        """Load cached translations if this DyDim cache file exists."""
        if not self.cache_path.exists():
            return {}
        with self.cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _source_key(source_language: str | None) -> str:
        return normalize_language(source_language, fallback="unknown")

    def get(self, text: str, source_language: str | None, target_language: str) -> str | None:
        """Return a cached translation for a source/target language pair."""
        source_key = self._source_key(source_language)
        target_key = normalize_language(target_language)
        return self.data.get(source_key, {}).get(target_key, {}).get(text)

    def put(self, text: str, source_language: str | None, target_language: str, translated_text: str):
        """Stage a translation update; save() persists it to disk."""
        source_key = self._source_key(source_language)
        target_key = normalize_language(target_language)
        self.data.setdefault(source_key, {}).setdefault(target_key, {})[text] = translated_text
        self.dirty = True

    def save(self):
        if not self.dirty:
            return
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
        self.dirty = False


class OnlineTranslator:
    """Lazy wrapper around a seq2seq translation model."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = None
        self.model = None

    def _ensure_loaded(self):
        """Load the translation model only after the first cache miss."""
        if self.model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"Loading translation model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_kwargs = {}
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_kwargs).to(self.device).eval()

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language: str | None,
        target_language: str,
        max_length: int,
    ) -> list[str]:
        """Translate a same-source-language batch into one target language."""
        if not texts:
            return []
        self._ensure_loaded()
        resolved_target = normalize_language(target_language)
        resolved_source = normalize_language(source_language, fallback="")
        if resolved_source and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = resolved_source

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(resolved_target)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=1,
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


class TranslationManager:
    """Coordinates dataset-scoped caches and lazy online translation."""

    def __init__(
        self,
        cache_root: Path,
        namespace: str,
        translation_model: str,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 256,
    ):
        self.cache_root = Path(cache_root)
        self.namespace = str(namespace)
        self.translation_model = str(translation_model)
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        if self.batch_size <= 0:
            raise ValueError("Translation batch_size must be positive.")
        self.current_cache: TranslationCache | None = None
        self.online_translator: OnlineTranslator | None = None

    def set_dataset(self, dataset_name: str):
        """Switch the active cache to the current MTEB task."""
        if self.current_cache is not None:
            self.current_cache.save()
        self.current_cache = TranslationCache(self.cache_root, self.namespace, dataset_name)

    def close(self):
        if self.current_cache is not None:
            self.current_cache.save()

    def _translator(self) -> OnlineTranslator:
        if self.online_translator is None:
            self.online_translator = OnlineTranslator(self.translation_model, device=self.device)
        return self.online_translator

    def translate_texts(
        self,
        texts: list[str],
        *,
        source_languages: list[str],
        target_language: str,
    ) -> list[str]:
        """Translate with cache-first semantics and project-local persistence."""
        if self.current_cache is None:
            raise RuntimeError("Translation dataset context is not set.")
        if len(texts) != len(source_languages):
            raise ValueError("texts and source_languages must have the same length.")

        target = normalize_language(target_language)
        results: list[str | None] = [None] * len(texts)
        missing_by_source: dict[str, list[tuple[int, str]]] = {}

        for idx, (text, source_language) in enumerate(zip(texts, source_languages)):
            source = normalize_language(source_language)
            if source == target:
                # Identity translations are not stored; the original text is
                # already the exact target-language input.
                results[idx] = text
                continue
            cached = self.current_cache.get(text, source, target)
            if cached is not None:
                results[idx] = cached
                continue
            missing_by_source.setdefault(source, []).append((idx, text))

        for source, missing_items in missing_by_source.items():
            for start in range(0, len(missing_items), self.batch_size):
                chunk = missing_items[start : start + self.batch_size]
                translated = self._translator().translate_batch(
                    [text for _, text in chunk],
                    source_language=source,
                    target_language=target,
                    max_length=self.max_length,
                )
                for (idx, text), translated_text in zip(chunk, translated):
                    self.current_cache.put(text, source, target, translated_text)
                    results[idx] = translated_text
                self.current_cache.save()

        self.current_cache.save()
        return [str(item) for item in results]
