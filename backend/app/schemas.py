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
    prompt_notice: Optional[str] = Field(
        None,
        description="Set when the server had to adjust the prompt, e.g. trimming whitespace.",
    )


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
    prompt_notice: Optional[str] = Field(
        None,
        description="Set when the server had to adjust the prompt, e.g. trimming whitespace.",
    )


class ErrorResponse(Schema):
    detail: str


# --------------------------------------------------------------------------
# Logit lens — "what is the model thinking at each layer?"
# --------------------------------------------------------------------------


class TokenPrediction(Schema):
    """One candidate next-token and the probability the model assigns it."""

    token: str = Field(..., description="Display-ready token string.")
    token_id: int
    prob: float = Field(..., description="Softmax probability, 0-1.")


class LayerLens(Schema):
    """The model's best guess read out of one layer's residual stream."""

    layer: int = Field(..., description="Hidden-state index. 0 is the embedding output.")
    label: str = Field(..., description="Short display label: 'embed', 'L1', 'L2'…")
    top: List[TokenPrediction] = Field(..., description="Top-k candidates at this layer.")
    entropy: float = Field(
        ...,
        description=(
            "Shannon entropy of the full distribution, in bits. High = the model is "
            "undecided across many tokens; low = it has committed."
        ),
    )
    target_prob: float = Field(
        ...,
        description=(
            "Probability this layer assigns to the model's FINAL answer. Rising values "
            "trace the answer forming; this is the line worth animating."
        ),
    )
    changed: bool = Field(..., description="True if the top-1 token differs from the previous layer.")


class TokenTrajectory(Schema):
    """One candidate's probability at every layer.

    Built from the union of all layers' top-k, so a token that leads mid-network
    and then fades still has a complete, gap-free line to draw.
    """

    token: str
    token_id: int
    probs: List[float] = Field(..., description="Probability at each hidden state, in layer order.")
    peak_layer: int
    peak_prob: float
    final_prob: float


class LensRequest(Schema):
    model_id: str
    prompt: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=10)
    position: Optional[int] = Field(
        None,
        description="Token position to read out. Defaults to the last token (the one being predicted from).",
    )
    max_tokens: Optional[int] = Field(None, ge=1)


class LensResponse(Schema):
    model_id: str
    display_name: str
    tokens: List[str]
    position: int = Field(..., description="Token index the readout was taken at.")
    layers: List[LayerLens]
    trajectories: List[TokenTrajectory] = Field(
        ...,
        description="Gap-free probability lines for every token that led or placed at any layer.",
    )
    final_prediction: TokenPrediction = Field(..., description="What the model actually predicts.")
    narration: List[str] = Field(
        ..., description="Plain-English findings derived from the layer trace."
    )
    truncated: bool
    prompt_notice: Optional[str] = Field(
        None,
        description="Set when the server had to adjust the prompt, e.g. trimming whitespace.",
    )
