"""Hardcoded model catalog.

Every `hf_model_id` below was verified to exist on the Hugging Face Hub and to
report `model_type: gpt2` / `GPT2LMHeadModel`, so they all share the GPT-2
architecture and the GPT-2 BPE tokenizer. That matters: per-layer comparisons
are only meaningful between models with the same architecture family.
"""

from typing import Dict, List

from .schemas import ModelInfo

MODELS: List[ModelInfo] = [
    ModelInfo(
        id="gpt2",
        display_name="GPT-2 (base)",
        hf_model_id="gpt2",
        role="base",
        description="124M reference checkpoint. The reliability anchor — everything is compared against this.",
    ),
    ModelInfo(
        id="gpt2-imdb",
        display_name="GPT-2 IMDB (fine-tuned)",
        hf_model_id="lvwerra/gpt2-imdb",
        role="finetuned",
        description="GPT-2 small fine-tuned on IMDB movie reviews. Same 12 layers as base.",
    ),
    ModelInfo(
        id="dialogpt-small",
        display_name="DialoGPT-small (fine-tuned)",
        hf_model_id="microsoft/DialoGPT-small",
        role="finetuned",
        description="GPT-2 small fine-tuned on Reddit dialogue. Same 12 layers as base.",
    ),
    ModelInfo(
        id="gpt2-medium",
        display_name="GPT-2 Medium",
        hf_model_id="gpt2-medium",
        role="base",
        description="355M, 24 layers. Deeper than base — comparisons cover the shared prefix only.",
    ),
]

_BY_ID: Dict[str, ModelInfo] = {m.id: m for m in MODELS}


class UnknownModelError(ValueError):
    """Raised when a request names a model id that isn't in the catalog."""


def list_models() -> List[ModelInfo]:
    return list(MODELS)


def get_model(model_id: str) -> ModelInfo:
    try:
        return _BY_ID[model_id]
    except KeyError:
        known = ", ".join(sorted(_BY_ID))
        raise UnknownModelError(f"Unknown model_id '{model_id}'. Known ids: {known}") from None
