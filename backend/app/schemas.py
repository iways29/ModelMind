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


# --------------------------------------------------------------------------
# Ablation — "what does the model predict if this component never fired?"
# --------------------------------------------------------------------------


class Ablation(Schema):
    """One component to switch off for the duration of a forward pass.

    `layer` is the honest 0-based block index. Display labels shift by one so a
    block is named after the residual-stream row it writes — block 4 is "L5",
    the same row the lens shows it landing in.
    """

    layer: int = Field(..., ge=0, description="Transformer block index, 0-based.")
    head: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Attention head within that block. Omit to ablate the whole block, which "
            "passes the residual stream through untouched as if the layer weren't there."
        ),
    )


class LensTrace(Schema):
    """The layer-by-layer readout of a single forward pass.

    Shared shape so a baseline run and an ablated run can be drawn by the same
    component with no branching.
    """

    layers: List[LayerLens]
    trajectories: List[TokenTrajectory]
    final_prediction: TokenPrediction


class TokenShift(Schema):
    """How one candidate's final-layer probability moved under ablation."""

    token: str
    token_id: int
    baseline_prob: float
    ablated_prob: float
    delta: float = Field(..., description="ablated_prob - baseline_prob. Negative means suppressed.")


class AblationEffect(Schema):
    """Scalar summary of what switching a component off did to the answer."""

    answer_changed: bool = Field(
        ..., description="True if the ablated run's top-1 differs from the baseline's."
    )
    baseline_answer: TokenPrediction
    ablated_answer: TokenPrediction = Field(
        ..., description="The ablated run's own top-1, which may be a different token."
    )
    baseline_answer_prob_after: float = Field(
        ...,
        description="Probability the ablated run still assigns to the BASELINE answer token.",
    )
    prob_delta: float = Field(
        ...,
        description=(
            "baseline_answer_prob_after - baseline_answer.prob. The signed change in "
            "support for the answer the intact model gave."
        ),
    )
    kl_bits: float = Field(
        ...,
        description=(
            "KL(baseline || ablated) over the full vocabulary, in bits. The scalar "
            "effect size — 0 means the component made no difference to this prompt."
        ),
    )
    top_shifts: List[TokenShift] = Field(
        ..., description="Candidates whose probability moved most, largest magnitude first."
    )


class AblateRequest(Schema):
    model_id: str
    prompt: str = Field(..., min_length=1, max_length=2000)
    ablations: List[Ablation] = Field(
        ...,
        min_length=1,
        max_length=16,
        description="Components to switch off. All are applied to the same forward pass.",
    )
    top_k: int = Field(5, ge=1, le=10)
    position: Optional[int] = None
    max_tokens: Optional[int] = Field(None, ge=1)


class AblateResponse(Schema):
    model_id: str
    display_name: str
    tokens: List[str]
    position: int
    ablations: List[Ablation] = Field(..., description="Echoed back, resolved and validated.")
    ablation_label: str = Field(..., description="Human-readable summary, e.g. 'L7 H3'.")
    baseline: LensTrace
    ablated: LensTrace
    effect: AblationEffect
    narration: List[str]
    truncated: bool
    prompt_notice: Optional[str] = None


# --------------------------------------------------------------------------
# Attribution sweep — ablate every component of one kind, then rank them
# --------------------------------------------------------------------------


class ComponentEffect(Schema):
    """One component's measured contribution, from its own ablation run."""

    layer: int = Field(..., description="0-based block index; `label` carries the display name.")
    head: Optional[int] = Field(None, description="None when the whole block was ablated.")
    label: str = Field(..., description="'L8' for block 7, 'L8 H3' for head 3 inside it.")
    baseline_answer_prob_after: float
    prob_delta: float
    kl_bits: float
    top_token: str = Field(..., description="What the ablated run predicts instead.")
    top_token_id: int
    answer_changed: bool


class AttributionRequest(Schema):
    model_id: str
    prompt: str = Field(..., min_length=1, max_length=2000)
    scope: str = Field(
        "heads",
        description=(
            "'layers' ablates each block in turn (one run per block). 'heads' ablates "
            "each head inside `layer` (one run per head)."
        ),
        pattern="^(layers|heads)$",
    )
    layer: Optional[int] = Field(
        None,
        ge=0,
        description="Required when scope is 'heads'; ignored when scope is 'layers'.",
    )
    position: Optional[int] = None
    max_tokens: Optional[int] = Field(None, ge=1)


class AttributionResponse(Schema):
    model_id: str
    display_name: str
    tokens: List[str]
    position: int
    scope: str
    layer: Optional[int] = None
    baseline_answer: TokenPrediction
    components: List[ComponentEffect] = Field(
        ..., description="Ranked by kl_bits, strongest effect first."
    )
    runs: int = Field(..., description="Forward passes performed, excluding the baseline.")
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
