"""Model loading, caching, and forward-pass logic.

Everything that touches `transformers` or `torch` lives in this module. FastAPI
routes call into these functions and do nothing else.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .models import ModelInfo, get_model
from .schemas import AnalyzeResponse, CompareResponse, ModelMagnitudes

logger = logging.getLogger(__name__)

# Inference is CPU-only and explicit. On an M2 you *could* use "mps", but MPS
# has historically returned subtly wrong attention values for some ops, and this
# tool exists to show real numbers — so CPU it is.
DEVICE = torch.device("cpu")

# Attention payloads scale as layers x heads x seq^2. GPT-2 small at 32 tokens
# is 12 x 12 x 1024 = ~147k floats, which measures ~1.8 MB of JSON; 64 tokens
# would be 4x that. Keep the ceiling low enough that the browser stays responsive.
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "32"))

# Attention weights are probabilities; 4 decimals is well past what a heatmap
# can render, and it roughly halves the response size.
_ROUND_DP = 4


class ModelLoadError(RuntimeError):
    """Raised when a checkpoint exists in the catalog but can't be loaded."""


class EmptyPromptError(ValueError):
    """Raised when a prompt contains no tokens the model can consume."""


@dataclass(frozen=True)
class LoadedModel:
    """A tokenizer + model pair, resident in memory."""

    info: ModelInfo
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def num_heads(self) -> int:
        return int(self.model.config.num_attention_heads)


# Cache keyed by our internal model id, so repeat requests never re-read from disk.
_CACHE: Dict[str, LoadedModel] = {}
# Guards the cache. Loading is slow (first call also downloads weights), so the
# lock is held across the load to stop two concurrent requests loading the same
# 500 MB checkpoint twice.
_CACHE_LOCK = threading.Lock()


def load_model(model_id: str) -> LoadedModel:
    """Return a cached model, loading it on first use.

    Raises `UnknownModelError` for ids outside the catalog and `ModelLoadError`
    if the checkpoint can't be fetched or instantiated.
    """
    cached = _CACHE.get(model_id)
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        # Re-check: another thread may have loaded it while we waited.
        cached = _CACHE.get(model_id)
        if cached is not None:
            return cached

        info = get_model(model_id)
        logger.info("Loading %s (%s) — first call may download weights.", info.id, info.hf_model_id)
        try:
            tokenizer = AutoTokenizer.from_pretrained(info.hf_model_id)
            model = AutoModelForCausalLM.from_pretrained(
                info.hf_model_id,
                output_attentions=True,
                output_hidden_states=True,
            )
        except Exception as exc:  # network failure, bad repo, corrupt cache…
            raise ModelLoadError(f"Could not load '{info.hf_model_id}': {exc}") from exc

        # GPT-2 ships no pad token. We never batch here, but the tokenizer warns
        # loudly (and `padding=True` would hard-fail) without one.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.to(DEVICE)
        model.eval()

        loaded = LoadedModel(info=info, tokenizer=tokenizer, model=model)
        _CACHE[model_id] = loaded
        logger.info("Loaded %s: %d layers, %d heads.", info.id, loaded.num_layers, loaded.num_heads)
        return loaded


def cached_model_ids() -> List[str]:
    return sorted(_CACHE)


def _resolve_token_budget(requested: Optional[int]) -> int:
    if requested is None:
        return MAX_PROMPT_TOKENS
    return max(1, min(int(requested), MAX_PROMPT_TOKENS))


def _display_token(raw: str) -> str:
    """Turn a GPT-2 BPE token into something readable on a chart axis.

    GPT-2 encodes a leading space as 'Ġ' and a newline as 'Ċ'. Rendering those
    raw makes the heatmap axes unreadable, so map them to visible stand-ins.
    """
    return raw.replace("Ġ", "␣").replace("Ċ", "⏎")


def _forward(loaded: LoadedModel, prompt: str, max_tokens: Optional[int]) -> Tuple[List[str], Any, bool]:
    """Run one CPU forward pass and return (display tokens, model output, truncated)."""
    budget = _resolve_token_budget(max_tokens)

    # Tokenize once unbounded to detect truncation honestly, then cut.
    full_ids = loaded.tokenizer.encode(prompt)
    if not full_ids:
        raise EmptyPromptError("Prompt tokenized to zero tokens.")

    truncated = len(full_ids) > budget
    input_ids = torch.tensor([full_ids[:budget]], dtype=torch.long, device=DEVICE)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )

    raw_tokens = loaded.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return [_display_token(t) for t in raw_tokens], outputs, truncated


def _attentions_to_nested(outputs: Any) -> List[List[List[List[float]]]]:
    """(layer, 1, head, seq, seq) tensors -> plain nested lists, layer -> head -> seq x seq."""
    nested: List[List[List[List[float]]]] = []
    for layer_attn in outputs.attentions:
        # Drop the batch dim, then round in torch (far faster than a Python loop).
        rounded = torch.round(layer_attn[0] * (10**_ROUND_DP)) / (10**_ROUND_DP)
        nested.append(rounded.tolist())
    return nested


def _hidden_state_magnitudes(outputs: Any) -> List[float]:
    """Mean absolute activation for each hidden state.

    `hidden_states` has `num_layers + 1` entries: index 0 is the embedding
    output, index i>0 is the output of transformer block i.

    Note the last entry is taken *after* GPT-2's final layer norm (`ln_f`), so
    it drops sharply rather than continuing the ramp. That's the architecture,
    not a bug — the UI says so too.
    """
    return [round(float(h[0].abs().mean().item()), 6) for h in outputs.hidden_states]


def analyze(model_id: str, prompt: str, max_tokens: Optional[int] = None) -> AnalyzeResponse:
    """Run one model on one prompt and extract tokens, attention, activations."""
    loaded = load_model(model_id)
    tokens, outputs, truncated = _forward(loaded, prompt, max_tokens)

    return AnalyzeResponse(
        model_id=loaded.info.id,
        tokens=tokens,
        attentions=_attentions_to_nested(outputs),
        hidden_state_magnitudes=_hidden_state_magnitudes(outputs),
        num_layers=loaded.num_layers,
        num_heads=loaded.num_heads,
        truncated=truncated,
    )


def compare(
    base_model_id: str,
    finetuned_model_id: str,
    prompt: str,
    max_tokens: Optional[int] = None,
) -> CompareResponse:
    """Run two models on the same prompt and diff their per-layer activations."""
    base = load_model(base_model_id)
    finetuned = load_model(finetuned_model_id)

    base_tokens, base_out, _ = _forward(base, prompt, max_tokens)
    ft_tokens, ft_out, _ = _forward(finetuned, prompt, max_tokens)

    base_mags = _hidden_state_magnitudes(base_out)
    ft_mags = _hidden_state_magnitudes(ft_out)

    # Models of different depth (e.g. gpt2 vs gpt2-medium) still line up at the
    # embedding and early blocks, so diff the shared prefix rather than refusing.
    shared = min(len(base_mags), len(ft_mags))
    delta = [round(ft_mags[i] - base_mags[i], 6) for i in range(shared)]

    notes: List[str] = []
    if len(base_mags) != len(ft_mags):
        notes.append(
            f"Different depths ({len(base_mags) - 1} vs {len(ft_mags) - 1} blocks); "
            f"delta covers the first {shared} hidden states only."
        )
    if base_tokens != ft_tokens:
        notes.append("The two tokenizers disagree on this prompt; token labels come from the base model.")

    return CompareResponse(
        prompt=prompt,
        tokens=base_tokens,
        base=ModelMagnitudes(
            model_id=base.info.id,
            display_name=base.info.display_name,
            hidden_state_magnitudes=base_mags,
        ),
        finetuned=ModelMagnitudes(
            model_id=finetuned.info.id,
            display_name=finetuned.info.display_name,
            hidden_state_magnitudes=ft_mags,
        ),
        delta=delta,
        layers_compared=shared,
        note=" ".join(notes) if notes else None,
    )
