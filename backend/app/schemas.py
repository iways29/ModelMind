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
    num_layers: int = Field(..., description="Number of transformer blocks.")
    num_heads: int = Field(..., description="Attention heads per block.")
    truncated: bool = Field(..., description="True if the prompt was cut to fit the token ceiling.")
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


# --------------------------------------------------------------------------
# Attribution — which parts of the network built this answer
#
# The residual stream is a *sum*: every block writes into it and nothing is
# overwritten. The final read-out is linear. So the answer's logit decomposes
# exactly into one number per component — no ablation, no re-running, one pass.
# --------------------------------------------------------------------------


class LensTrace(Schema):
    """The layer-by-layer readout of a single forward pass."""

    layers: List[LayerLens]
    trajectories: List[TokenTrajectory]
    final_prediction: TokenPrediction


class Contribution(Schema):
    """One component's signed push toward the answer, in logits."""

    kind: str = Field(..., description="'embed', 'attn', 'mlp', or 'head'.")
    layer: Optional[int] = Field(None, description="0-based block index; None for 'embed'.")
    head: Optional[int] = Field(None, description="Set only when kind is 'head'.")
    label: str = Field(..., description="'L8 attn', 'L8 mlp', 'L8 H3', 'embed'.")
    logits: float = Field(
        ...,
        description=(
            "Signed contribution to (answer logit - contrast logit). Positive pushed "
            "toward the answer, negative pushed away."
        ),
    )
    share: float = Field(
        ...,
        description="logits as a fraction of the total margin. Can exceed 1 when others push back.",
    )


class AttributionRequest(Schema):
    model_id: str
    prompt: str = Field(..., min_length=1, max_length=2000)
    contrast_token_id: Optional[int] = Field(
        None,
        description=(
            "Token to measure the answer against. Defaults to the runner-up, which "
            "asks 'what made it pick this word over the next-best one?'"
        ),
    )
    position: Optional[int] = None
    max_tokens: Optional[int] = Field(None, ge=1)


class AttributionResponse(Schema):
    model_id: str
    display_name: str
    tokens: List[str]
    position: int
    answer: TokenPrediction = Field(..., description="The token the model actually predicts.")
    contrast: TokenPrediction = Field(..., description="The token the answer is measured against.")
    margin: float = Field(
        ...,
        description="answer logit - contrast logit. Every contribution below sums into this.",
    )
    blocks: List[Contribution] = Field(
        ...,
        description=(
            "Embeddings plus each block's attention and MLP. These sum to `margin` "
            "minus `unattributed`, exactly — they are the whole network."
        ),
    )
    heads: List[Contribution] = Field(
        ...,
        description="Every attention head, ranked by absolute contribution. A drill-down into the attn rows above.",
    )
    unattributed: float = Field(
        ...,
        description=(
            "The bias terms, which belong to no component. Small; reported so the "
            "arithmetic is checkable rather than hidden."
        ),
    )
    narration: List[str]
    truncated: bool
    prompt_notice: Optional[str] = None


# --------------------------------------------------------------------------
# Patching — run one checkpoint with the other's weights in one place
#
# Ablation asks "what if this part were silent?", which is a state the model was
# never trained for. Patching asks the question a fine-tuner actually has: if I
# reverted this one layer to the base weights, would the behaviour come back?
# Only possible because the two models share an architecture and a tokenizer.
# --------------------------------------------------------------------------


class LayerPatch(Schema):
    """The result of reverting one part of the recipient to the donor's weights."""

    layer: int = Field(
        ...,
        description="0-based block index; -1 for the embeddings, block count for the read-out.",
    )
    kind: str = Field(..., description="'embed', 'block', or 'readout'.")
    label: str
    answer: TokenPrediction = Field(..., description="What the recipient says with this patch in place.")
    donor_answer_prob: float = Field(
        ..., description="Probability the patched run assigns to the DONOR's answer."
    )
    recovery: Optional[float] = Field(
        None,
        description=(
            "How far this single patch moved the recipient toward the donor, as a "
            "fraction: 0 = no movement, 1 = fully the donor's answer. None when the "
            "two models already agreed and the question is undefined."
        ),
    )
    flipped: bool = Field(..., description="True if the patched answer matches the donor's.")


class PatchFocus(Schema):
    """Real generated text for one chosen layer, so the effect is legible as behavior."""

    layer: int
    label: str
    recipient_text: str
    donor_text: str
    patched_text: str


class PatchRequest(Schema):
    recipient_model_id: str = Field(..., description="The model being modified — usually your fine-tune.")
    donor_model_id: str = Field(..., description="Where the transplanted component comes from — usually the base.")
    prompt: str = Field(..., min_length=1, max_length=2000)
    layer: Optional[int] = Field(
        None,
        ge=-1,
        description=(
            "Omit to sweep everything. Name one to also generate real text for that "
            "swap, which is the version you can actually read. -1 is the embeddings and "
            "the block count is the read-out."
        ),
    )
    position: Optional[int] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    max_new_tokens: int = Field(24, ge=1, le=80)


class PatchResponse(Schema):
    recipient_model_id: str
    recipient_name: str
    donor_model_id: str
    donor_name: str
    tokens: List[str]
    position: int
    recipient_answer: TokenPrediction
    donor_answer: TokenPrediction
    agreed: bool = Field(
        ..., description="True if both checkpoints already predict the same token here."
    )
    layers: List[LayerPatch] = Field(..., description="One entry per block, in layer order.")
    best_layer: Optional[int] = Field(
        None, description="The part whose swap moved the recipient furthest toward the donor."
    )
    focus: Optional[PatchFocus] = None
    narration: List[str]
    truncated: bool
    prompt_notice: Optional[str] = None


# --------------------------------------------------------------------------
# Behavior — many prompts, real generated text, diffed between two checkpoints
# --------------------------------------------------------------------------


class BehaviorRequest(Schema):
    model_id: str
    compare_model_id: Optional[str] = Field(
        None,
        description="Second checkpoint to diff against. Omit to just read one model's output.",
    )
    prompts: List[str] = Field(
        ...,
        min_length=1,
        max_length=25,
        description="One prompt per row. Capped because generation is CPU-bound.",
    )
    max_new_tokens: int = Field(20, ge=4, le=48)


class Continuation(Schema):
    """What one model actually wrote after the prompt."""

    model_id: str
    display_name: str
    text: str = Field(..., description="Generated continuation only, without the prompt.")
    repeats: bool = Field(
        ...,
        description=(
            "True if the continuation falls into a repetition loop. Greedy decoding does "
            "this readily, and it is the most common way a small model's output goes bad."
        ),
    )


class Divergence(Schema):
    """The first token where the two models part ways."""

    index: int = Field(..., description="Position within the generated continuation, 0-based.")
    shared_prefix: str = Field(
        ..., description="Prompt plus the continuation both models agreed on, up to this point."
    )
    prefix_token_count: int = Field(
        ...,
        description=(
            "Tokens in `shared_prefix`. The lens reads out the last one, so this is the "
            "position to open the microscope at."
        ),
    )
    token: TokenPrediction = Field(..., description="What the first model chose here.")
    compare_token: TokenPrediction = Field(..., description="What the second model chose instead.")


class PromptBehavior(Schema):
    """One prompt, run through both models."""

    prompt: str
    primary: Continuation
    compare: Optional[Continuation] = None
    identical: bool = Field(
        ..., description="True when both models produced exactly the same continuation."
    )
    divergence: Optional[Divergence] = None
    surprise_bits: Optional[float] = Field(
        None,
        description=(
            "Average bits the FIRST model assigns to the SECOND model's continuation. High "
            "means the fine-tune went somewhere the base would not have — the single best "
            "scalar for 'how far did this drift'."
        ),
    )
    flags: List[str] = Field(
        ..., description="Short machine-checkable labels: 'identical', 'repeats', 'drifted'."
    )


class BehaviorResponse(Schema):
    model_id: str
    display_name: str
    compare_model_id: Optional[str] = None
    compare_display_name: Optional[str] = None
    rows: List[PromptBehavior]
    max_new_tokens: int
    narration: List[str]
