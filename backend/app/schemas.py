"""Pydantic request/response schemas.

These are the single source of truth for the HTTP contract. The frontend
mirrors them by hand in `frontend/src/api/types.ts` — if you change a field
here, change it there too.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    """Base for every schema in this app.

    Pydantic v2 reserves the `model_` prefix for its own methods and warns on
    fields like `model_id`. `model_id` is the natural name here, so the
    protected namespace is cleared once, in one place.
    """

    model_config = ConfigDict(protected_namespaces=())


class ModelInfo(Schema):
    """One entry in the hardcoded model catalog (`GET /models`)."""

    id: str = Field(..., description="Stable internal id used by the API.")
    display_name: str = Field(..., description="Human label for the selector.")
    hf_model_id: str = Field(..., description="Hugging Face Hub repo id.")
    role: str = Field(
        ...,
        description="'base' or 'finetuned' — lets the UI preselect sensible compare defaults.",
    )
    description: str = Field("", description="One-line note about the checkpoint.")


class AnalyzeRequest(Schema):
    model_id: str
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Truncate the prompt to this many tokens. Attention payloads grow as "
            "layers x heads x seq^2, so the server clamps this to its own ceiling."
        ),
    )


class AnalyzeResponse(Schema):
    model_id: str
    tokens: List[str] = Field(..., description="Display-ready token strings, in order.")
    attentions: List[List[List[List[float]]]] = Field(
        ...,
        description="layer -> head -> seq x seq attention weights (rows are queries).",
    )
    hidden_state_magnitudes: List[float] = Field(
        ...,
        description=(
            "Mean absolute activation per layer. Index 0 is the embedding output, "
            "index i>0 is the output of transformer block i. The final entry is "
            "measured after GPT-2's final layer norm, so it dips."
        ),
    )
    num_layers: int = Field(..., description="Number of transformer blocks.")
    num_heads: int = Field(..., description="Attention heads per block.")
    truncated: bool = Field(..., description="True if the prompt was cut to fit the token ceiling.")


class CompareRequest(Schema):
    base_model_id: str
    finetuned_model_id: str
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: Optional[int] = Field(None, ge=1)


class ModelMagnitudes(Schema):
    """Per-layer activation profile for one model in a comparison."""

    model_id: str
    display_name: str
    hidden_state_magnitudes: List[float]


class CompareResponse(Schema):
    prompt: str
    tokens: List[str] = Field(..., description="Tokenization from the base model.")
    base: ModelMagnitudes
    finetuned: ModelMagnitudes
    delta: List[float] = Field(
        ..., description="finetuned[i] - base[i], over the layers the two models share."
    )
    layers_compared: int
    note: Optional[str] = Field(
        None, description="Set when the two models don't line up exactly (e.g. different depths)."
    )


class ErrorResponse(Schema):
    detail: str
