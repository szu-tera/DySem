"""Attention-sum vector extraction for DyDim.

The encoder hooks every decoder self-attention block, pools the last-token
attention output from each layer, then sums those layer outputs into one dense
semantic vector per input sentence.
"""

import numpy as np
import torch
from tqdm import tqdm

from .prompts import apply_prompts


ATTENTION_ATTR_ALIASES = ("self_attn", "attention", "attn")


def _resolve_attention_module(layer_module):
    """Resolve common attention attribute names across model families."""
    for attr in ATTENTION_ATTR_ALIASES:
        if hasattr(layer_module, attr):
            return getattr(layer_module, attr)
    raise AttributeError(f"Could not find an attention module on {type(layer_module).__name__}.")


def _decoder_layers(model):
    """Find decoder layers for common Hugging Face causal-LM layouts."""
    candidates = (
        getattr(model, "model", None),
        model,
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
    )
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    raise AttributeError(f"Could not locate decoder layers for {type(model).__name__}.")


def _input_device(model):
    """Infer the device where tokenized inputs should be placed."""
    embeddings = getattr(model, "get_input_embeddings", lambda: None)()
    if embeddings is not None and hasattr(embeddings, "weight"):
        device = embeddings.weight.device
        if str(device) != "meta":
            return device
    for param in model.parameters():
        if str(param.device) != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return last non-padding token indices for right-padded batches."""
    mask = attention_mask.to(dtype=torch.long)
    seq_len = mask.shape[1]
    return seq_len - 1 - mask.flip(dims=[1]).argmax(dim=1)


class _HookStore:
    """Small helper for registering and cleaning forward hooks."""

    def __init__(self):
        self.activations = {}
        self.handles = []

    def register(self, module, name: str):
        def hook_fn(_, __, output):
            tensor = output[0] if isinstance(output, tuple) else output
            self.activations[name] = tensor.detach()

        self.handles.append(module.register_forward_hook(hook_fn))

    def clear_batch(self):
        self.activations = {}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = {}


class AttentionSumEncoder:
    """Encode prompted text into DyDim attention-sum vectors."""

    def __init__(self, model, tokenizer, *, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.layers = list(_decoder_layers(model))
        self.input_device = _input_device(model)

    @property
    def dimension(self) -> int | None:
        config = getattr(self.model, "config", None)
        if config is None:
            return None
        return getattr(config, "hidden_size", None)

    def encode(
        self,
        texts: list[str],
        *,
        prompt_setting: str,
        languages: list[str],
        batch_size: int,
        progress_desc: str,
    ) -> np.ndarray:
        """Encode texts using the selected prompt setting and language codes."""
        if len(texts) != len(languages):
            raise ValueError("texts and languages must have the same length.")
        if not texts:
            dim = self.dimension or 0
            return np.zeros((0, dim), dtype=np.float32)

        hook_store = _HookStore()
        for layer_idx, layer in enumerate(self.layers):
            hook_store.register(_resolve_attention_module(layer), f"layer_{layer_idx}")

        vectors: list[np.ndarray] = []
        try:
            iterator = tqdm(range(0, len(texts), batch_size), desc=progress_desc, leave=False)
            for start in iterator:
                batch_texts = texts[start : start + batch_size]
                batch_languages = languages[start : start + batch_size]
                prompted = apply_prompts(batch_texts, prompt_setting=prompt_setting, languages=batch_languages)
                encoded = self.tokenizer(
                    prompted,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
                hook_store.clear_batch()

                # Attention outputs are captured by hooks; the model return
                # value is only needed to trigger the forward pass.
                with torch.inference_mode():
                    forward_kwargs = {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "return_dict": True,
                    }
                    try:
                        self.model(use_cache=False, **forward_kwargs)
                    except TypeError:
                        self.model(**forward_kwargs)

                seq_lens = _last_token_indices(encoded["attention_mask"])
                per_layer = []
                for layer_idx in range(len(self.layers)):
                    output = hook_store.activations[f"layer_{layer_idx}"]
                    if self.tokenizer.padding_side == "left":
                        pooled = output[:, -1, :]
                    else:
                        batch_indices = torch.arange(output.shape[0], device=output.device)
                        pooled = output[batch_indices, seq_lens.to(output.device), :]
                    per_layer.append(pooled.to(torch.float32))
                attention_sum = torch.stack(per_layer, dim=0).sum(dim=0)
                vectors.append(attention_sum.detach().cpu().numpy().astype(np.float32, copy=False))
        finally:
            hook_store.close()

        return np.vstack(vectors).astype(np.float32, copy=False)
