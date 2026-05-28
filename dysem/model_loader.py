"""Causal language model loading utilities.

DyDim evaluates attention-sum vectors from decoder LMs only; embedding models
and embedding-specific prompts are rejected at this boundary.
"""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _is_embedding_model(model_path: Path, model_tag: str) -> bool:
    """Conservative guard against accidentally running embedding checkpoints."""
    markers = (model_path.name.lower(), str(model_tag).lower())
    return any("embedding" in marker for marker in markers)


def _read_declared_dtype(model_path: Path):
    """Use the model config dtype when available."""
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    return getattr(config, "torch_dtype", None)


def resolve_model_dtype(model_path: Path, device: str):
    """Choose a dtype that matches both checkpoint metadata and local device."""
    declared = _read_declared_dtype(model_path)
    if isinstance(declared, torch.dtype):
        dtype = declared
    elif isinstance(declared, str):
        dtype = getattr(torch, declared, None)
    else:
        dtype = None

    if isinstance(dtype, torch.dtype):
        if dtype == torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
            return torch.float16
        return dtype
    return torch.float16 if device == "cuda" else torch.float32


def load_causal_lm(model_path: Path, model_tag: str):
    """Load a local causal LM/tokenizer pair for attention extraction."""
    if _is_embedding_model(model_path, model_tag):
        raise ValueError(
            "DyDim only supports causal language models. "
            f"Embedding model-like identifier rejected: {model_tag!r}."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_model_dtype(model_path, device)
    visible_cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_map = "auto" if visible_cuda_devices > 1 else None

    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if device == "cuda":
        model_kwargs["low_cpu_mem_usage"] = True
        if device_map is not None:
            model_kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs).eval()
    except TypeError as exc:
        if "dtype" in str(exc) and "unexpected" in str(exc):
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs).eval()
        else:
            raise

    if device == "cuda" and device_map is None:
        model = model.to(device)
    if hasattr(model, "config"):
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer
