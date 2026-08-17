"""Model loading, caching, and forward-pass logic.

Everything that touches `transformers` or `torch` lives in this module. FastAPI
routes call into these functions and do nothing else.
"""

import logging
import math
import os
import threading
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .insights import narrate_attribution, narrate_behavior, narrate_lens, narrate_patch
from .models import ModelInfo, get_model
from .schemas import (
    AnalyzeResponse,
    AttributionResponse,
    BehaviorResponse,
    Continuation,
    Contribution,
    Divergence,
    LayerLens,
    LayerPatch,
    LensResponse,
    LensTrace,
    PatchFocus,
    PatchResponse,
    PromptBehavior,
    TokenPrediction,
    TokenTrajectory,
)

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

# Longest token string we'll render before eliding.
_MAX_TOKEN_CHARS = 16


class ModelLoadError(RuntimeError):
    """Raised when a checkpoint exists in the catalog but can't be loaded."""


class EmptyPromptError(ValueError):
    """Raised when a prompt contains no tokens the model can consume."""


class LensUnsupportedError(RuntimeError):
    """Raised when a model doesn't expose the final norm + output head the lens needs."""


class ComponentUnsupportedError(RuntimeError):
    """Raised when a model's block/attention layout isn't one we know how to switch off."""


class PatchIncompatibleError(ValueError):
    """Raised when two checkpoints are too different to transplant between."""


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


def _visible_char(char: str) -> str:
    """Make one character legible, including the ones that render as nothing."""
    if char == " ":
        return "␣"
    if char == "\n":
        return "⏎"
    if char == "\t":
        return "⇥"
    # Control, format, and exotic-separator codepoints (non-breaking space,
    # zero-width joiners, byte-order marks) draw as nothing or as a blank box.
    # Show the codepoint instead so a token is never silently invisible.
    if unicodedata.category(char)[0] in {"C", "Z"}:
        return f"<{ord(char):02X}>"
    return char


def _display_token(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    """Turn a token id into something readable on a chart axis or in prose.

    Must decode rather than read `convert_ids_to_tokens`. GPT-2 uses byte-level
    BPE and stores tokens in an internal byte->unicode mapping, so the raw token
    string for a non-breaking space is the mojibake 'Âł' and for 'ét' it is
    'Ã©t'. `decode` reverses that mapping; anything else shows users garbage.
    """
    text = tokenizer.decode([int(token_id)])
    if not text:
        return "∅"
    shown = "".join(_visible_char(c) for c in text)
    # GPT-2's vocabulary contains genuine oddities like a single 64-underscore
    # token (id 27193) scraped from web forms. Left whole they wreck every
    # layout they land in.
    if len(shown) > _MAX_TOKEN_CHARS:
        return shown[:_MAX_TOKEN_CHARS] + "…"
    return shown


def _clean_prompt(prompt: str) -> Tuple[str, Optional[str]]:
    """Trim surrounding whitespace, and say so when it mattered.

    A trailing space is the single most destructive thing you can do to a
    next-token prediction. GPT-2 puts the space *inside* the following token
    (' Paris', not 'Paris'), so a dangling space becomes its own token and the
    readout position lands on it — you end up asking "what follows a lone
    space?", which is meaningless, and the model answers with byte fragments.
    It looks exactly like a broken model.
    """
    cleaned = prompt.strip()
    if cleaned == prompt:
        return cleaned, None
    if not cleaned:
        return cleaned, None
    return cleaned, (
        "Trimmed whitespace from your prompt. A trailing space becomes its own "
        "token, and the prediction would have been read from that space rather "
        "than from your last real word."
    )


@dataclass(frozen=True)
class _Prepared:
    """A tokenized prompt, ready to be run through the model any number of times."""

    input_ids: torch.Tensor
    tokens: List[str]
    truncated: bool
    notice: Optional[str]


def _prepare(loaded: LoadedModel, prompt: str, max_tokens: Optional[int]) -> _Prepared:
    """Tokenize once. Patch sweeps re-run the model once per block on this."""
    budget = _resolve_token_budget(max_tokens)

    prompt, notice = _clean_prompt(prompt)

    # Tokenize once unbounded to detect truncation honestly, then cut.
    full_ids = loaded.tokenizer.encode(prompt)
    if not full_ids:
        raise EmptyPromptError("Prompt tokenized to zero tokens.")

    truncated = len(full_ids) > budget
    ids = full_ids[:budget]
    return _Prepared(
        input_ids=torch.tensor([ids], dtype=torch.long, device=DEVICE),
        tokens=[_display_token(loaded.tokenizer, t) for t in ids],
        truncated=truncated,
        notice=notice,
    )


def _run(loaded: LoadedModel, input_ids: torch.Tensor, internals: bool = True) -> Any:
    """One CPU forward pass.

    `internals=False` skips attention and hidden-state collection, which is most
    of the cost when all a caller wants is the final logits — the sweep case.
    """
    with torch.no_grad():
        return loaded.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_attentions=internals,
            output_hidden_states=internals,
            use_cache=False,
        )


def _forward(
    loaded: LoadedModel, prompt: str, max_tokens: Optional[int]
) -> Tuple[List[str], Any, bool, Optional[str]]:
    """Run one CPU forward pass and return (display tokens, output, truncated, notice)."""
    prepared = _prepare(loaded, prompt, max_tokens)
    outputs = _run(loaded, prepared.input_ids)
    return prepared.tokens, outputs, prepared.truncated, prepared.notice


# --------------------------------------------------------------------------
# Locating the pieces of a block
# --------------------------------------------------------------------------


def _blocks(model: PreTrainedModel) -> Any:
    """The indexable list of transformer blocks."""
    base = getattr(model, "transformer", None) or getattr(model, "model", None)
    blocks = getattr(base, "h", None) if base is not None else None
    if blocks is None and base is not None:
        blocks = getattr(base, "layers", None)
    if blocks is None:
        raise ComponentUnsupportedError(
            f"{model.__class__.__name__} does not expose an indexable list of transformer "
            "blocks, so its components can't be switched off."
        )
    return blocks


def _attentions_to_nested(outputs: Any) -> List[List[List[List[float]]]]:
    """(layer, 1, head, seq, seq) tensors -> plain nested lists, layer -> head -> seq x seq."""
    nested: List[List[List[List[float]]]] = []
    for layer_attn in outputs.attentions:
        # Drop the batch dim, then round in torch (far faster than a Python loop).
        rounded = torch.round(layer_attn[0] * (10**_ROUND_DP)) / (10**_ROUND_DP)
        nested.append(rounded.tolist())
    return nested


def analyze(model_id: str, prompt: str, max_tokens: Optional[int] = None) -> AnalyzeResponse:
    """Run one model on one prompt and extract tokens, attention, activations."""
    loaded = load_model(model_id)
    tokens, outputs, truncated, notice = _forward(loaded, prompt, max_tokens)

    return AnalyzeResponse(
        model_id=loaded.info.id,
        tokens=tokens,
        attentions=_attentions_to_nested(outputs),
        num_layers=loaded.num_layers,
        num_heads=loaded.num_heads,
        truncated=truncated,
        prompt_notice=notice,
    )


# --------------------------------------------------------------------------
# Logit lens
# --------------------------------------------------------------------------


def _output_projection(model: PreTrainedModel) -> Tuple[Any, Any]:
    """Find the final layer norm and the unembedding head.

    The logit lens works by taking a *mid-network* residual stream and pushing
    it through the same two operations the model uses at the very end. Attribute
    names differ across architectures, so probe for the common spellings rather
    than hardcoding GPT-2's — this is the seam that lets arbitrary models work
    later.
    """
    head = getattr(model, "lm_head", None)
    base = getattr(model, "transformer", None) or getattr(model, "model", None)

    norm = None
    if base is not None:
        for attr in ("ln_f", "norm", "final_layer_norm", "final_norm"):
            norm = getattr(base, attr, None)
            if norm is not None:
                break

    if head is None or norm is None:
        raise LensUnsupportedError(
            f"{model.__class__.__name__} does not expose a final norm + lm_head pair, "
            "so its residual stream can't be projected to vocabulary space."
        )
    return norm, head


def _entropy_bits(probs: torch.Tensor) -> float:
    """Shannon entropy in bits. Low means the model has committed to an answer."""
    safe = probs.clamp_min(1e-12)
    return float(-(safe * safe.log2()).sum().item())


def _resolve_position(tokens: Sequence[str], position: Optional[int]) -> int:
    return len(tokens) - 1 if position is None else max(0, min(int(position), len(tokens) - 1))


def _lens_trace(
    loaded: LoadedModel, outputs: Any, pos: int, top_k: int
) -> Tuple[LensTrace, torch.Tensor]:
    """Decode every layer's residual stream into a next-token distribution.

    Returns the trace plus the final-layer probability vector, which callers
    comparing two runs need at full vocabulary width.
    """
    norm, head = _output_projection(loaded.model)

    hidden = outputs.hidden_states
    last_index = len(hidden) - 1

    def distribution(layer_index: int) -> torch.Tensor:
        resid = hidden[layer_index][0, pos]
        # GPT-2's last hidden state has already been through ln_f. Norming it a
        # second time distorts the distribution, so pass it straight through.
        normed = resid if layer_index == last_index else norm(resid)
        with torch.no_grad():
            return torch.softmax(head(normed).float(), dim=-1)

    # Compute every layer's distribution up front. 13 x 50k floats is ~2.6 MB —
    # cheap, and it lets us build gap-free trajectories in one pass instead of
    # re-running the projection per token of interest.
    per_layer = [distribution(i) for i in range(len(hidden))]

    # The final layer's argmax is this run's actual answer; every earlier layer
    # is scored against it so we can watch that answer's probability grow.
    final_probs = per_layer[last_index]
    final_id = int(torch.argmax(final_probs).item())

    layers: List[LayerLens] = []
    previous_top: Optional[int] = None
    for i, probs in enumerate(per_layer):
        top_probs, top_ids = torch.topk(probs, top_k)

        top = [
            TokenPrediction(
                token=_display_token(loaded.tokenizer, int(tid)),
                token_id=int(tid),
                prob=round(float(p), 6),
            )
            for p, tid in zip(top_probs, top_ids)
        ]
        current_top = int(top_ids[0].item())

        layers.append(
            LayerLens(
                layer=i,
                label="embed" if i == 0 else f"L{i}",
                top=top,
                entropy=round(_entropy_bits(probs), 4),
                target_prob=round(float(probs[final_id].item()), 6),
                changed=previous_top is not None and current_top != previous_top,
            )
        )
        previous_top = current_top

    # Union of everything that placed anywhere, so a token like "Paris" that
    # leads at L10 and drops out of the top-k by L12 still has a complete line.
    candidate_ids = sorted({p.token_id for layer in layers for p in layer.top})
    index = torch.tensor(candidate_ids, dtype=torch.long)
    stacked = torch.stack([probs[index] for probs in per_layer])  # (layers, candidates)

    trajectories: List[TokenTrajectory] = []
    for column, token_id in enumerate(candidate_ids):
        series = stacked[:, column]
        peak = int(torch.argmax(series).item())
        trajectories.append(
            TokenTrajectory(
                token=_display_token(loaded.tokenizer, token_id),
                token_id=token_id,
                probs=[round(float(v), 6) for v in series],
                peak_layer=peak,
                peak_prob=round(float(series[peak].item()), 6),
                final_prob=round(float(series[-1].item()), 6),
            )
        )
    # Strongest first, so the frontend can draw the top N and drop the tail.
    trajectories.sort(key=lambda t: t.peak_prob, reverse=True)

    trace = LensTrace(
        layers=layers,
        trajectories=trajectories,
        final_prediction=TokenPrediction(
            token=_display_token(loaded.tokenizer, final_id),
            token_id=final_id,
            prob=round(float(final_probs[final_id].item()), 6),
        ),
    )
    return trace, final_probs


def logit_lens(
    model_id: str,
    prompt: str,
    top_k: int = 5,
    position: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> LensResponse:
    """Decode what the model 'believes' the next token is, at every layer.

    At each hidden state we run the residual stream through the final layer norm
    and the unembedding matrix — the same path the last layer takes — and read
    off a probability distribution over the vocabulary. Watching the top-1
    change from layer to layer is the model forming its answer.
    """
    loaded = load_model(model_id)
    prepared = _prepare(loaded, prompt, max_tokens)
    pos = _resolve_position(prepared.tokens, position)
    outputs = _run(loaded, prepared.input_ids)
    trace, _ = _lens_trace(loaded, outputs, pos, top_k)

    return LensResponse(
        model_id=loaded.info.id,
        display_name=loaded.info.display_name,
        tokens=prepared.tokens,
        position=pos,
        layers=trace.layers,
        trajectories=trace.trajectories,
        final_prediction=trace.final_prediction,
        narration=narrate_lens(
            trace.layers, trace.trajectories, trace.final_prediction, prepared.tokens[pos]
        ),
        truncated=prepared.truncated,
        prompt_notice=prepared.notice,
    )


# --------------------------------------------------------------------------
# Naming blocks
# --------------------------------------------------------------------------


def _block_label(layer: int) -> str:
    """Name a block the way the lens names its output.

    The lens labels hidden states, where "L5" is the residual stream *after*
    block 4. Numbering blocks 0-based in the UI too would mean patching "L4" and
    watching the trace change at "L5", which reads as an off-by-one bug. So the
    wire keeps the honest 0-based block index and the label is shifted to match
    the row the block writes.
    """
    return f"L{layer + 1}"


def _final_probs(outputs: Any, pos: int) -> torch.Tensor:
    """Softmax over the model's own output logits at one position."""
    return torch.softmax(outputs.logits[0, pos].float(), dim=-1)


# --------------------------------------------------------------------------
# Attribution — splitting the answer into one number per component
#
# The residual stream is a running sum: every block *adds* to it and nothing is
# overwritten. The read-out at the end is a layer norm followed by a linear map.
# So if the layer norm's scale is held fixed, the answer's logit is a plain
# linear function of that sum, and it splits exactly into one term per
# component — embeddings, and each block's attention and MLP.
#
# That is the whole trick, and it is why this needs one forward pass rather than
# one per component. Switching a part off and re-running measures something
# related but different (it also captures how the parts downstream react); this
# measures the direct push, exactly, and the terms are checkable because they
# must add back up to the number they came from.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Decomposition:
    """Every component's write into the residual stream, at one token position."""

    embed: torch.Tensor  # (d_model,)
    attn: List[torch.Tensor]  # per block, (d_model,)
    mlp: List[torch.Tensor]  # per block, (d_model,)
    heads: List[Optional[torch.Tensor]]  # per block, (n_heads, d_model)
    final_resid: torch.Tensor  # (d_model,) — the sum, before the final norm
    logits: torch.Tensor  # (vocab,)


def _block_parts(block: Any, layer: int) -> Tuple[Any, Any]:
    """The attention and MLP sub-modules, whatever this architecture calls them."""
    attn = getattr(block, "attn", None) or getattr(block, "self_attn", None)
    mlp = getattr(block, "mlp", None) or getattr(block, "feed_forward", None)
    if attn is None or mlp is None:
        raise ComponentUnsupportedError(
            f"Block {layer} does not expose separate attention and MLP sub-modules, "
            "so its contribution can't be split."
        )
    return attn, mlp


def _head_contributions(projection: Any, merged: torch.Tensor, n_heads: int) -> Optional[torch.Tensor]:
    """Split one block's attention output into per-head writes.

    After the output projection the heads are summed and can never be separated
    again, so the split has to happen on its *input*: each head owns a disjoint
    slice of that vector, and therefore a disjoint band of rows in the projection
    matrix. Multiplying one head's slice by its own band gives exactly what that
    head contributed.

    Returns None when the projection isn't a shape we can decompose, so head
    attribution degrades to "unavailable" rather than to a wrong number.
    """
    weight = getattr(projection, "weight", None)
    if weight is None:
        return None

    d_model = merged.shape[-1]
    head_dim = d_model // n_heads

    # GPT-2's Conv1D stores (in, out) and computes x @ W; nn.Linear stores
    # (out, in) and computes x @ W.T. Tell them apart by which axis is the input.
    if weight.shape[0] == d_model:
        rows = lambda lo, hi: weight[lo:hi, :]  # noqa: E731 — Conv1D
    elif weight.shape[1] == d_model:
        rows = lambda lo, hi: weight[:, lo:hi].T  # noqa: E731 — nn.Linear
    else:
        return None

    return torch.stack(
        [
            merged[h * head_dim : (h + 1) * head_dim].float()
            @ rows(h * head_dim, (h + 1) * head_dim).float()
            for h in range(n_heads)
        ]
    )


def _decompose(loaded: LoadedModel, input_ids: torch.Tensor, pos: int) -> _Decomposition:
    """Capture what every component wrote into the residual stream, in one pass."""
    model = loaded.model
    blocks = _blocks(model)
    norm, _ = _output_projection(model)
    n_heads = loaded.num_heads

    attn_out: Dict[int, torch.Tensor] = {}
    mlp_out: Dict[int, torch.Tensor] = {}
    head_out: Dict[int, Optional[torch.Tensor]] = {}
    resid: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []

    def written(output: Any) -> torch.Tensor:
        """Attention returns a tuple; the MLP returns a bare tensor."""
        tensor = output[0] if isinstance(output, tuple) else output
        return tensor[0, pos].detach()

    try:
        for index, block in enumerate(blocks):
            attn, mlp = _block_parts(block, index)
            handles.append(
                attn.register_forward_hook(
                    lambda _m, _a, out, i=index: attn_out.__setitem__(i, written(out))
                )
            )
            handles.append(
                mlp.register_forward_hook(
                    lambda _m, _a, out, i=index: mlp_out.__setitem__(i, written(out))
                )
            )
            projection = getattr(attn, "c_proj", None) or getattr(attn, "o_proj", None)
            if projection is not None:
                handles.append(
                    projection.register_forward_pre_hook(
                        lambda _m, args, i=index, proj=projection: head_out.__setitem__(
                            i, _head_contributions(proj, args[0][0, pos].detach(), n_heads)
                        )
                    )
                )

        # The final norm's *input* is the completed sum. It isn't in
        # `hidden_states` — that list's last entry is already normed.
        handles.append(
            norm.register_forward_pre_hook(lambda _m, args: resid.__setitem__("x", args[0][0, pos].detach()))
        )

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_hidden_states=True,
                use_cache=False,
            )
    finally:
        # The model is a process-wide singleton; a hook that outlived its request
        # would silently corrupt every later run.
        for handle in handles:
            handle.remove()

    layer_count = len(blocks)
    return _Decomposition(
        embed=outputs.hidden_states[0][0, pos].detach(),
        attn=[attn_out[i] for i in range(layer_count)],
        mlp=[mlp_out[i] for i in range(layer_count)],
        heads=[head_out.get(i) for i in range(layer_count)],
        final_resid=resid["x"],
        logits=outputs.logits[0, pos].detach(),
    )


def attribution(
    model_id: str,
    prompt: str,
    contrast_token_id: Optional[int] = None,
    position: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> AttributionResponse:
    """Split the answer's margin over the runner-up into one number per component.

    Measured as a *difference* between two tokens rather than one token's raw
    logit, because a raw logit means nothing on its own — adding a constant to
    every logit leaves the model's answer unchanged. The difference is what the
    model actually decided, so it's what's worth attributing.
    """
    loaded = load_model(model_id)
    norm, head = _output_projection(loaded.model)

    prepared = _prepare(loaded, prompt, max_tokens)
    pos = _resolve_position(prepared.tokens, position)
    decomposed = _decompose(loaded, prepared.input_ids, pos)

    logits = decomposed.logits.float()
    probs = torch.softmax(logits, dim=-1)
    ranked = torch.topk(logits, 2).indices
    answer_id = int(ranked[0].item())
    contrast_id = int(ranked[1].item()) if contrast_token_id is None else int(contrast_token_id)

    def prediction(token_id: int) -> TokenPrediction:
        return TokenPrediction(
            token=_display_token(loaded.tokenizer, token_id),
            token_id=token_id,
            prob=round(float(probs[token_id].item()), 6),
        )

    # The direction in activation space that separates these two tokens, folded
    # through the final norm's fixed scale so component writes map straight onto
    # logit difference.
    resid = decomposed.final_resid.float()
    scale = torch.sqrt(resid.var(unbiased=False) + float(getattr(norm, "eps", 1e-5)))
    weights = head.weight.float()
    direction = (weights[answer_id] - weights[contrast_id]) * norm.weight.float() / scale

    def push(vector: torch.Tensor) -> float:
        centred = vector.float()
        return float(((centred - centred.mean()) @ direction).item())

    margin = float((logits[answer_id] - logits[contrast_id]).item())

    blocks: List[Contribution] = [
        _contribution("embed", None, None, "embed", push(decomposed.embed), margin)
    ]
    for index in range(loaded.num_layers):
        label = _block_label(index)
        blocks.append(
            _contribution("attn", index, None, f"{label} attn", push(decomposed.attn[index]), margin)
        )
        blocks.append(
            _contribution("mlp", index, None, f"{label} mlp", push(decomposed.mlp[index]), margin)
        )

    heads: List[Contribution] = []
    for index, per_head in enumerate(decomposed.heads):
        if per_head is None:
            continue
        label = _block_label(index)
        heads.extend(
            _contribution("head", index, h, f"{label} H{h}", push(per_head[h]), margin)
            for h in range(per_head.shape[0])
        )
    heads.sort(key=lambda c: abs(c.logits), reverse=True)

    # The norm's shift term belongs to no component. Reporting it keeps the sum
    # honest and lets anyone check the arithmetic.
    bias = getattr(norm, "bias", None)
    unattributed = (
        float(((weights[answer_id] - weights[contrast_id]) @ bias.float()).item())
        if bias is not None
        else 0.0
    )

    return AttributionResponse(
        model_id=loaded.info.id,
        display_name=loaded.info.display_name,
        tokens=prepared.tokens,
        position=pos,
        answer=prediction(answer_id),
        contrast=prediction(contrast_id),
        margin=round(margin, 4),
        blocks=blocks,
        heads=heads,
        unattributed=round(unattributed, 4),
        narration=narrate_attribution(blocks, heads, prediction(answer_id), prediction(contrast_id), margin),
        truncated=prepared.truncated,
        prompt_notice=prepared.notice,
    )


def _contribution(
    kind: str, layer: Optional[int], head: Optional[int], label: str, value: float, margin: float
) -> Contribution:
    return Contribution(
        kind=kind,
        layer=layer,
        head=head,
        label=label,
        logits=round(value, 4),
        share=round(value / margin, 4) if abs(margin) > 1e-9 else 0.0,
    )


# --------------------------------------------------------------------------
# Patching — running one checkpoint with another's component spliced in
#
# The question a fine-tuner actually has is "which part of my model changed?",
# and neither of the obvious approaches answers it. Comparing weights tells you
# where the numbers moved, not where the *behaviour* moved. Switching a part off
# tells you what breaks without it, which is not the same as what the fine-tune
# did to it.
#
# Splicing does answer it. Take one block's output from the base model, drop it
# into the fine-tune in place of its own, and let the rest of the fine-tune run
# normally. If the fine-tuned behaviour disappears, that block was carrying it.
# --------------------------------------------------------------------------


def _require_swappable(donor: LoadedModel, recipient: LoadedModel, prompt_ids: Sequence[int]) -> None:
    """Refuse pairings where one model's weights would mean nothing in the other.

    Same depth and width is the mechanical requirement. Identical tokenization is
    the semantic one: if the two models cut the prompt into different pieces,
    they aren't reading the same sentence and no comparison between them holds.
    """
    if donor.num_layers != recipient.num_layers:
        raise PatchIncompatibleError(
            f"{donor.info.display_name} has {donor.num_layers} blocks and "
            f"{recipient.info.display_name} has {recipient.num_layers}. Swapping needs matching "
            "depth — pick two checkpoints of the same size."
        )
    donor_width = int(donor.model.config.hidden_size)
    recipient_width = int(recipient.model.config.hidden_size)
    if donor_width != recipient_width:
        raise PatchIncompatibleError(
            f"Residual streams are different widths ({donor_width} vs {recipient_width}), "
            "so one model's blocks don't fit in the other."
        )
    if donor.tokenizer.encode(donor.tokenizer.decode(list(prompt_ids))) != list(prompt_ids):
        raise PatchIncompatibleError(
            f"{donor.info.display_name} tokenizes this prompt differently from "
            f"{recipient.info.display_name}, so their positions don't line up."
        )


def _readout(model: PreTrainedModel) -> Tuple[Any, str]:
    """The module holding the final norm, and the attribute name it lives under."""
    base = getattr(model, "transformer", None) or getattr(model, "model", None)
    if base is not None:
        for attr in ("ln_f", "norm", "final_layer_norm", "final_norm"):
            if getattr(base, attr, None) is not None:
                return base, attr
    raise ComponentUnsupportedError(
        f"{model.__class__.__name__} does not expose a final norm, so its read-out "
        "can't be swapped."
    )


# The two ends of the network, which are weights too and change under
# fine-tuning like any block does. Sweeping only the blocks would be a quiet
# lie: a fine-tune that moved its output embedding shows nothing anywhere in a
# block sweep, and "the change isn't in the blocks" reads identically to "there
# is no change to find". Numbered outside the block range so one integer can
# address any part of the model.
EMBED_TARGET = -1


def _readout_target(loaded: LoadedModel) -> int:
    return loaded.num_layers


@contextmanager
def _reverted(recipient: LoadedModel, donor: LoadedModel, target: int) -> Iterator[None]:
    """Run the recipient with the donor's weights in exactly one place.

    This is a *weight* swap, not an activation transplant, and the difference is
    the whole point. Transplanting activations replaces the residual stream at
    that depth, which carries everything the earlier blocks did too — so it
    measures the accumulated difference up to that point, not the part. Swapping
    weights leaves the stream alone and changes only what this part does to it,
    which is the question a fine-tuner actually has: if I reverted this layer,
    would the behaviour come back?

    Swapping every target at once reproduces the donor exactly, which is the
    property that makes the sweep trustworthy — nothing is left unaccounted for.
    """
    restore: List[Any] = []
    try:
        if target == EMBED_TARGET:
            r_base = recipient.model.transformer
            d_base = donor.model.transformer
            # Only the input side moves here. `lm_head` holds its own reference
            # to the recipient's original embedding tensor, so the output side
            # stays put and remains the read-out's business.
            for attr in ("wte", "wpe"):
                restore.append((r_base, attr, getattr(r_base, attr)))
                setattr(r_base, attr, getattr(d_base, attr))
        elif target == _readout_target(recipient):
            r_base, attr = _readout(recipient.model)
            d_base, _ = _readout(donor.model)
            restore.append((r_base, attr, getattr(r_base, attr)))
            setattr(r_base, attr, getattr(d_base, attr))
            restore.append((recipient.model, "lm_head", recipient.model.lm_head))
            recipient.model.lm_head = donor.model.lm_head
        else:
            blocks = _blocks(recipient.model)
            restore.append((blocks, target, blocks[target]))
            blocks[target] = _blocks(donor.model)[target]
        yield
    finally:
        # The models are process-wide singletons. A swap that outlived its
        # request would quietly turn one checkpoint into a chimera of the two.
        for holder, key, original in restore:
            if isinstance(key, int):
                holder[key] = original
            else:
                setattr(holder, key, original)


def _target_kind(target: int, readout: int) -> str:
    if target == EMBED_TARGET:
        return "embed"
    return "readout" if target == readout else "block"


def _target_label(target: int, readout: int) -> str:
    if target == EMBED_TARGET:
        return "embeddings"
    return "read-out" if target == readout else _block_label(target)


def _generate_text(loaded: LoadedModel, ids: Sequence[int], max_new_tokens: int) -> str:
    """Greedy continuation for one prompt, decoded to a plain string."""
    continuation = _generate_batch(loaded, [list(ids)], max_new_tokens)[0]
    return loaded.tokenizer.decode(continuation).strip()


def patch(
    recipient_model_id: str,
    donor_model_id: str,
    prompt: str,
    layer: Optional[int] = None,
    position: Optional[int] = None,
    max_tokens: Optional[int] = None,
    max_new_tokens: int = 24,
) -> PatchResponse:
    """Revert each of the recipient's blocks to the donor's weights in turn, and measure.

    The sweep is one forward pass per block, so the whole thing costs about as
    much as a dozen predictions. Naming a layer additionally generates real text
    for that swap, which is the version worth reading — a changed token is
    evidence, but a changed sentence is the thing you actually care about.

    The read-out is swept alongside the blocks. Leaving it out would be a quiet
    lie: a fine-tune that moved its output embedding would show nothing anywhere
    in the block sweep, and the honest reading of that — "the change isn't in the
    blocks" — is indistinguishable from "there's no change to find".
    """
    recipient = load_model(recipient_model_id)
    donor = load_model(donor_model_id)

    prepared = _prepare(recipient, prompt, max_tokens)
    prompt_ids = prepared.input_ids[0].tolist()
    _require_swappable(donor, recipient, prompt_ids)

    pos = _resolve_position(prepared.tokens, position)
    if layer is not None and not EMBED_TARGET <= layer <= recipient.num_layers:
        raise PatchIncompatibleError(
            f"{recipient.info.display_name} has {recipient.num_layers} blocks (0-"
            f"{recipient.num_layers - 1}), the embeddings at {EMBED_TARGET}, and the read-out "
            f"at {recipient.num_layers}; target {layer} doesn't exist."
        )

    recipient_probs = _final_probs(_run(recipient, prepared.input_ids, internals=False), pos)
    donor_probs = _final_probs(_run(donor, prepared.input_ids, internals=False), pos)

    def prediction(probs: torch.Tensor, token_id: Optional[int] = None) -> TokenPrediction:
        tid = int(torch.argmax(probs).item()) if token_id is None else token_id
        return TokenPrediction(
            token=_display_token(recipient.tokenizer, tid),
            token_id=tid,
            prob=round(float(probs[tid].item()), 6),
        )

    recipient_answer = prediction(recipient_probs)
    donor_answer = prediction(donor_probs)
    agreed = recipient_answer.token_id == donor_answer.token_id

    # How far a swap moved the recipient toward the donor, as a fraction of the
    # distance between them. Scaling by that gap rather than reporting a raw
    # probability keeps the number comparable across prompts where the two models
    # started close together and prompts where they started far apart.
    start = float(recipient_probs[donor_answer.token_id].item())
    target = float(donor_probs[donor_answer.token_id].item())
    span = target - start

    # In the order the model runs: what it reads the words as, then each block,
    # then how it turns the result back into a word.
    readout = _readout_target(recipient)
    targets = [EMBED_TARGET, *range(recipient.num_layers), readout]

    patches: List[LayerPatch] = []
    for target in targets:
        with _reverted(recipient, donor, target):
            probs = _final_probs(_run(recipient, prepared.input_ids, internals=False), pos)
        moved = float(probs[donor_answer.token_id].item())
        answer = prediction(probs)
        patches.append(
            LayerPatch(
                layer=target,
                kind=_target_kind(target, readout),
                label=_target_label(target, readout),
                answer=answer,
                donor_answer_prob=round(moved, 6),
                # Undefined when the two models already agree: there is no gap to
                # close, and dividing by it would manufacture a huge number from
                # rounding noise.
                recovery=None if agreed or abs(span) < 1e-6 else round((moved - start) / span, 4),
                flipped=answer.token_id == donor_answer.token_id,
            )
        )

    scored = [p for p in patches if p.recovery is not None]
    best = max(scored, key=lambda p: p.recovery or 0.0) if scored else None

    focus: Optional[PatchFocus] = None
    if layer is not None:
        with _reverted(recipient, donor, layer):
            patched_text = _generate_text(recipient, prompt_ids, max_new_tokens)
        focus = PatchFocus(
            layer=layer,
            label=_target_label(layer, readout),
            recipient_text=_generate_text(recipient, prompt_ids, max_new_tokens),
            donor_text=_generate_text(donor, prompt_ids, max_new_tokens),
            patched_text=patched_text,
        )

    return PatchResponse(
        recipient_model_id=recipient.info.id,
        recipient_name=recipient.info.display_name,
        donor_model_id=donor.info.id,
        donor_name=donor.info.display_name,
        tokens=prepared.tokens,
        position=pos,
        recipient_answer=recipient_answer,
        donor_answer=donor_answer,
        agreed=agreed,
        layers=patches,
        best_layer=best.layer if best else None,
        focus=focus,
        narration=narrate_patch(
            recipient.info.display_name,
            donor.info.display_name,
            recipient_answer,
            donor_answer,
            agreed,
            patches,
            focus,
        ),
        truncated=prepared.truncated,
        prompt_notice=prepared.notice,
    )


# --------------------------------------------------------------------------
# Behavior — what the model actually writes, across many prompts
#
# Everything above this line reads a single forward pass. This section is the
# only place the model generates text, and it is the entry point the rest of the
# tool was missing: you cannot ask "where did my model go wrong" until you have
# seen it go wrong somewhere.
# --------------------------------------------------------------------------

# Greedy decoding, always. Sampling would mean two runs of the *same* model
# disagree, and then a diff between two checkpoints measures nothing.
_GENERATION_IS_GREEDY = True

# A 4-token window repeated this many times is a degeneration loop rather than
# ordinary English repetition.
_REPEAT_WINDOW = 4
_REPEAT_THRESHOLD = 3

# "Drifted" is relative to the rest of the run, not an absolute bit count. An
# absolute threshold is unusable here: surprise depends on prompt length, domain,
# and how far apart the two checkpoints are, so any constant is either never hit
# or always hit. A row is flagged when it is this many times the run's median.
_DRIFT_RATIO = 1.25


def _encode_prompt(loaded: LoadedModel, prompt: str) -> List[int]:
    """Clean, encode, and truncate one prompt to the server's token ceiling."""
    cleaned, _ = _clean_prompt(prompt)
    ids = loaded.tokenizer.encode(cleaned)[:MAX_PROMPT_TOKENS]
    if not ids:
        raise EmptyPromptError(f"Prompt {prompt!r} tokenized to zero tokens.")
    return ids


def _generate_batch(
    loaded: LoadedModel, id_lists: Sequence[List[int]], max_new_tokens: int
) -> List[List[int]]:
    """Greedy-generate a continuation for each prompt, in one batched pass.

    Padding is on the LEFT and built by hand rather than via the tokenizer.
    Decoder-only models continue from the last position, so right-padding would
    have them continue from pad tokens instead of from the prompt — and building
    the batch here avoids mutating `padding_side` on a cached, shared tokenizer.
    """
    pad_id = loaded.tokenizer.eos_token_id
    width = max(len(ids) for ids in id_lists)

    input_ids = torch.full((len(id_lists), width), pad_id, dtype=torch.long, device=DEVICE)
    attention_mask = torch.zeros((len(id_lists), width), dtype=torch.long, device=DEVICE)
    for row, ids in enumerate(id_lists):
        input_ids[row, width - len(ids) :] = torch.tensor(ids, dtype=torch.long, device=DEVICE)
        attention_mask[row, width - len(ids) :] = 1

    with torch.no_grad():
        generated = loaded.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=not _GENERATION_IS_GREEDY,
            pad_token_id=pad_id,
            # The checkpoints are loaded with output_attentions/hidden_states on
            # for the lens. Some generation configs pick those up and switch to
            # returning a ModelOutput instead of a plain tensor — iterating that
            # yields field *names*. Pin the return type rather than sniff it.
            return_dict_in_generate=False,
        )
    sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences

    continuations: List[List[int]] = []
    for row in sequences:
        ids = row[width:].tolist()
        # Everything at and after the first EOS is padding, not output.
        if pad_id in ids:
            ids = ids[: ids.index(pad_id)]
        continuations.append(ids)
    return continuations


def _repeats(token_ids: Sequence[int]) -> bool:
    """True if the continuation has fallen into a repetition loop."""
    if len(token_ids) < _REPEAT_WINDOW * _REPEAT_THRESHOLD:
        return False
    seen: Dict[Tuple[int, ...], int] = {}
    for start in range(len(token_ids) - _REPEAT_WINDOW + 1):
        window = tuple(token_ids[start : start + _REPEAT_WINDOW])
        seen[window] = seen.get(window, 0) + 1
        if seen[window] >= _REPEAT_THRESHOLD:
            return True
    return False


def _divergence_index(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    """First position where two continuations differ, or None if one prefixes the other."""
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return None


def _next_token_at(loaded: LoadedModel, ids: Sequence[int]) -> TokenPrediction:
    """What this model predicts after `ids`, and how sure it is."""
    tensor = torch.tensor([list(ids)], dtype=torch.long, device=DEVICE)
    probs = _final_probs(_run(loaded, tensor, internals=False), len(ids) - 1)
    top = int(torch.argmax(probs).item())
    return TokenPrediction(
        token=_display_token(loaded.tokenizer, top),
        token_id=top,
        prob=round(float(probs[top].item()), 6),
    )


def _surprise_bits(
    loaded: LoadedModel, prompt_ids: Sequence[int], continuation_ids: Sequence[int]
) -> Optional[float]:
    """Average bits this model assigns to someone else's continuation.

    Teacher-forced: we feed the other model's text through and ask how improbable
    it was. High means the second model wrote something the first would never
    have produced, which is exactly "how far did the fine-tune drift" as a single
    number.
    """
    if not continuation_ids:
        return None

    ids = list(prompt_ids) + list(continuation_ids)
    tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    outputs = _run(loaded, tensor, internals=False)
    log_probs = torch.log_softmax(outputs.logits[0].float(), dim=-1)

    # Position i's logits predict token i+1, so the score for the first
    # continuation token is read from the last prompt position.
    start = len(prompt_ids)
    total = sum(
        float(log_probs[start + offset - 1, token_id].item())
        for offset, token_id in enumerate(continuation_ids)
    )
    return round(-total / len(continuation_ids) / math.log(2), 4)


def _continuation(loaded: LoadedModel, ids: Sequence[int]) -> Continuation:
    return Continuation(
        model_id=loaded.info.id,
        display_name=loaded.info.display_name,
        text=loaded.tokenizer.decode(list(ids)),
        repeats=_repeats(ids),
    )


def _flag_drift(rows: List[PromptBehavior]) -> None:
    """Mark the rows that moved furthest, judged against this run's own median.

    Needs every row's score before it can flag any of them, so it runs as a
    second pass rather than inline.
    """
    scores = sorted(r.surprise_bits for r in rows if r.surprise_bits is not None)
    if len(scores) < 3:
        return
    median = scores[len(scores) // 2]
    if median <= 0:
        return
    for row in rows:
        if row.surprise_bits is not None and row.surprise_bits >= median * _DRIFT_RATIO:
            row.flags.append("drifted")


def behavior(
    model_id: str,
    prompts: Sequence[str],
    compare_model_id: Optional[str] = None,
    max_new_tokens: int = 20,
) -> BehaviorResponse:
    """Run many prompts through one or two checkpoints and diff what they write."""
    primary = load_model(model_id)
    other = load_model(compare_model_id) if compare_model_id else None

    prompt_ids = [_encode_prompt(primary, prompt) for prompt in prompts]
    cleaned = [primary.tokenizer.decode(ids) for ids in prompt_ids]

    primary_out = _generate_batch(primary, prompt_ids, max_new_tokens)
    compare_out = _generate_batch(other, prompt_ids, max_new_tokens) if other else None

    rows: List[PromptBehavior] = []
    for index, prompt in enumerate(cleaned):
        mine = primary_out[index]
        row_flags: List[str] = []

        if compare_out is None:
            if _repeats(mine):
                row_flags.append("repeats")
            rows.append(
                PromptBehavior(
                    prompt=prompt,
                    primary=_continuation(primary, mine),
                    identical=False,
                    flags=row_flags,
                )
            )
            continue

        theirs = compare_out[index]
        identical = mine == theirs

        divergence: Optional[Divergence] = None
        split = _divergence_index(mine, theirs)
        if split is not None:
            shared = prompt_ids[index] + mine[:split]
            divergence = Divergence(
                index=split,
                shared_prefix=primary.tokenizer.decode(shared),
                prefix_token_count=len(shared),
                token=_next_token_at(primary, shared),
                compare_token=_next_token_at(other, shared),
            )

        surprise = _surprise_bits(primary, prompt_ids[index], theirs)

        if identical:
            row_flags.append("identical")
        if _repeats(mine) or _repeats(theirs):
            row_flags.append("repeats")

        rows.append(
            PromptBehavior(
                prompt=prompt,
                primary=_continuation(primary, mine),
                compare=_continuation(other, theirs),
                identical=identical,
                divergence=divergence,
                surprise_bits=surprise,
                flags=row_flags,
            )
        )

    _flag_drift(rows)

    return BehaviorResponse(
        model_id=primary.info.id,
        display_name=primary.info.display_name,
        compare_model_id=other.info.id if other else None,
        compare_display_name=other.info.display_name if other else None,
        rows=rows,
        max_new_tokens=max_new_tokens,
        narration=narrate_behavior(rows, primary.info.display_name, other.info.display_name if other else None),
    )
