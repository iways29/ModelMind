"""Model loading, caching, and forward-pass logic.

Everything that touches `transformers` or `torch` lives in this module. FastAPI
routes call into these functions and do nothing else.
"""

import logging
import os
import threading
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .insights import narrate_ablation, narrate_attribution, narrate_lens
from .models import ModelInfo, get_model
from .schemas import (
    Ablation,
    AblateResponse,
    AblationEffect,
    AnalyzeResponse,
    AttributionResponse,
    CompareResponse,
    ComponentEffect,
    LayerLens,
    LensResponse,
    LensTrace,
    ModelMagnitudes,
    TokenPrediction,
    TokenShift,
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


class AblationUnsupportedError(RuntimeError):
    """Raised when a model's block/attention layout isn't one we know how to switch off."""


class InvalidAblationError(ValueError):
    """Raised when an ablation names a layer or head this model doesn't have."""


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
    """Tokenize once. Ablation sweeps re-run the model dozens of times on this."""
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


def _run(
    loaded: LoadedModel,
    input_ids: torch.Tensor,
    ablations: Sequence[Ablation] = (),
    internals: bool = True,
) -> Any:
    """One CPU forward pass, optionally with components switched off.

    `internals=False` skips attention and hidden-state collection, which is most
    of the cost when all a caller wants is the final logits — the sweep case.
    """
    with torch.no_grad(), _ablated(loaded, ablations):
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
# Ablation hooks
#
# Two different surgeries, because "switch this off" means two different things
# at the two granularities:
#
#   whole block — replace the block's output with its own input, so the residual
#                 stream flows past untouched. Attention and MLP both contribute
#                 nothing; the model runs as if the layer were not there.
#   single head — zero that head's slice of the concatenated attention output
#                 *before* the output projection mixes the heads together. After
#                 c_proj the heads are summed and can no longer be separated.
# --------------------------------------------------------------------------


def _blocks(model: PreTrainedModel) -> Any:
    """The indexable list of transformer blocks."""
    base = getattr(model, "transformer", None) or getattr(model, "model", None)
    blocks = getattr(base, "h", None) if base is not None else None
    if blocks is None and base is not None:
        blocks = getattr(base, "layers", None)
    if blocks is None:
        raise AblationUnsupportedError(
            f"{model.__class__.__name__} does not expose an indexable list of transformer "
            "blocks, so its components can't be switched off."
        )
    return blocks


def _skip_block_hook(module: Any, args: Any, kwargs: Any, output: Any) -> Any:
    """Return the block's input in place of its output."""
    resid = args[0] if args else kwargs.get("hidden_states")
    if resid is None:
        return output
    # GPT2Block returns (hidden_states,) plus attention weights when they were
    # requested. Keep the tail so `output_attentions=True` still works.
    if isinstance(output, tuple):
        return (resid,) + tuple(output[1:])
    return resid


def _zero_head_hook(head: int, head_dim: int) -> Any:
    """Pre-hook for `attn.c_proj` that blanks one head's contribution."""

    def hook(module: Any, args: Any) -> Any:
        merged = args[0]
        # Clone: the incoming tensor is the attention output, and writing into it
        # in place would corrupt autograd bookkeeping and any shared storage.
        patched = merged.clone()
        patched[..., head * head_dim : (head + 1) * head_dim] = 0.0
        return (patched,) + tuple(args[1:])

    return hook


def _validate_ablations(loaded: LoadedModel, ablations: Iterable[Ablation]) -> List[Ablation]:
    resolved = list(ablations)
    for ablation in resolved:
        if ablation.layer >= loaded.num_layers:
            raise InvalidAblationError(
                f"{loaded.info.display_name} has {loaded.num_layers} blocks (0-"
                f"{loaded.num_layers - 1}); layer {ablation.layer} doesn't exist."
            )
        if ablation.head is not None and ablation.head >= loaded.num_heads:
            raise InvalidAblationError(
                f"{loaded.info.display_name} has {loaded.num_heads} heads per block (0-"
                f"{loaded.num_heads - 1}); head {ablation.head} doesn't exist."
            )
    return resolved


@contextmanager
def _ablated(loaded: LoadedModel, ablations: Sequence[Ablation]) -> Iterator[None]:
    """Install ablation hooks for the duration of the block, then always remove them.

    The model is a process-wide cached singleton, so a hook that outlives its
    request would silently corrupt every later run. Hence the unconditional
    teardown in `finally`.
    """
    if not ablations:
        yield
        return

    blocks = _blocks(loaded.model)
    head_dim = int(loaded.model.config.hidden_size) // loaded.num_heads
    handles: List[Any] = []
    try:
        for ablation in ablations:
            block = blocks[ablation.layer]
            if ablation.head is None:
                handles.append(block.register_forward_hook(_skip_block_hook, with_kwargs=True))
                continue

            attn = getattr(block, "attn", None) or getattr(block, "self_attn", None)
            projection = getattr(attn, "c_proj", None) or getattr(attn, "o_proj", None)
            if projection is None:
                raise AblationUnsupportedError(
                    f"Block {ablation.layer} has no attention output projection to hook, "
                    "so individual heads can't be isolated."
                )
            handles.append(
                projection.register_forward_pre_hook(_zero_head_hook(ablation.head, head_dim))
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


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
    tokens, outputs, truncated, notice = _forward(loaded, prompt, max_tokens)

    return AnalyzeResponse(
        model_id=loaded.info.id,
        tokens=tokens,
        attentions=_attentions_to_nested(outputs),
        hidden_state_magnitudes=_hidden_state_magnitudes(outputs),
        num_layers=loaded.num_layers,
        num_heads=loaded.num_heads,
        truncated=truncated,
        prompt_notice=notice,
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

    base_tokens, base_out, _, notice = _forward(base, prompt, max_tokens)
    ft_tokens, ft_out, _, _ = _forward(finetuned, prompt, max_tokens)

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
# Ablation
# --------------------------------------------------------------------------


def _kl_bits(reference: torch.Tensor, other: torch.Tensor) -> float:
    """KL(reference || other) in bits — how much the ablated run surprises the baseline.

    Asymmetric on purpose: it weights the tokens the *intact* model cared about,
    which is the question being asked ("what did removing this cost the answer?").
    """
    p = reference.clamp_min(1e-12)
    q = other.clamp_min(1e-12)
    return float((p * (p.log2() - q.log2())).sum().item())


def _top_shifts(
    loaded: LoadedModel, baseline: torch.Tensor, ablated: torch.Tensor, limit: int = 6
) -> List[TokenShift]:
    """The candidates whose probability moved most, in either direction."""
    delta = ablated - baseline
    moved = torch.topk(delta.abs(), limit).indices

    shifts = [
        TokenShift(
            token=_display_token(loaded.tokenizer, int(tid)),
            token_id=int(tid),
            baseline_prob=round(float(baseline[tid].item()), 6),
            ablated_prob=round(float(ablated[tid].item()), 6),
            delta=round(float(delta[tid].item()), 6),
        )
        for tid in moved
    ]
    shifts.sort(key=lambda s: abs(s.delta), reverse=True)
    return shifts


def _block_label(layer: int) -> str:
    """Name a block the way the lens names its output.

    The lens labels hidden states, where "L5" is the residual stream *after*
    block 4. Numbering blocks 0-based in the UI too would mean ablating "L4" and
    watching the trace change at "L5", which reads as an off-by-one bug. So the
    wire keeps the honest 0-based block index and the label is shifted to match
    the row the block writes.
    """
    return f"L{layer + 1}"


def _ablation_label(ablations: Sequence[Ablation]) -> str:
    parts = [
        _block_label(a.layer) if a.head is None else f"{_block_label(a.layer)} H{a.head}"
        for a in ablations
    ]
    return " + ".join(parts)


def _final_probs(outputs: Any, pos: int) -> torch.Tensor:
    """Softmax over the model's own output logits at one position."""
    return torch.softmax(outputs.logits[0, pos].float(), dim=-1)


def ablate(
    model_id: str,
    prompt: str,
    ablations: Sequence[Ablation],
    top_k: int = 5,
    position: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> AblateResponse:
    """Run the model twice — intact, then with components switched off — and diff.

    This is the counterfactual the rest of the tool can only hint at: the lens
    shows *when* an answer formed, and this shows *what the answer depended on*.
    """
    loaded = load_model(model_id)
    resolved = _validate_ablations(loaded, ablations)

    prepared = _prepare(loaded, prompt, max_tokens)
    pos = _resolve_position(prepared.tokens, position)

    baseline_trace, baseline_probs = _lens_trace(
        loaded, _run(loaded, prepared.input_ids), pos, top_k
    )
    ablated_trace, ablated_probs = _lens_trace(
        loaded, _run(loaded, prepared.input_ids, resolved), pos, top_k
    )

    baseline_answer = baseline_trace.final_prediction
    after = float(ablated_probs[baseline_answer.token_id].item())

    effect = AblationEffect(
        answer_changed=ablated_trace.final_prediction.token_id != baseline_answer.token_id,
        baseline_answer=baseline_answer,
        ablated_answer=ablated_trace.final_prediction,
        baseline_answer_prob_after=round(after, 6),
        prob_delta=round(after - baseline_answer.prob, 6),
        kl_bits=round(_kl_bits(baseline_probs, ablated_probs), 4),
        top_shifts=_top_shifts(loaded, baseline_probs, ablated_probs),
    )

    label = _ablation_label(resolved)
    return AblateResponse(
        model_id=loaded.info.id,
        display_name=loaded.info.display_name,
        tokens=prepared.tokens,
        position=pos,
        ablations=resolved,
        ablation_label=label,
        baseline=baseline_trace,
        ablated=ablated_trace,
        effect=effect,
        narration=narrate_ablation(label, effect, baseline_trace, ablated_trace),
        truncated=prepared.truncated,
        prompt_notice=prepared.notice,
    )


def attribution(
    model_id: str,
    prompt: str,
    scope: str = "heads",
    layer: Optional[int] = None,
    position: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> AttributionResponse:
    """Ablate every component of one kind in turn and rank them by effect.

    One ablation at a time answers "did this matter?"; only a sweep answers
    "which one mattered most?", and a 12-head block is far too many to try by
    hand. Each run needs only the final logits, so `internals=False` keeps the
    sweep to roughly one baseline forward pass per component.
    """
    loaded = load_model(model_id)

    if scope == "heads":
        if layer is None:
            raise InvalidAblationError("scope='heads' needs a layer to sweep.")
        targets = [Ablation(layer=layer, head=h) for h in range(loaded.num_heads)]
    else:
        targets = [Ablation(layer=index) for index in range(loaded.num_layers)]
    _validate_ablations(loaded, targets)

    prepared = _prepare(loaded, prompt, max_tokens)
    pos = _resolve_position(prepared.tokens, position)

    baseline_probs = _final_probs(_run(loaded, prepared.input_ids, internals=False), pos)
    baseline_id = int(torch.argmax(baseline_probs).item())
    baseline_answer = TokenPrediction(
        token=_display_token(loaded.tokenizer, baseline_id),
        token_id=baseline_id,
        prob=round(float(baseline_probs[baseline_id].item()), 6),
    )

    components: List[ComponentEffect] = []
    for target in targets:
        probs = _final_probs(
            _run(loaded, prepared.input_ids, [target], internals=False), pos
        )
        top_id = int(torch.argmax(probs).item())
        after = float(probs[baseline_id].item())
        components.append(
            ComponentEffect(
                layer=target.layer,
                head=target.head,
                label=_ablation_label([target]),
                baseline_answer_prob_after=round(after, 6),
                prob_delta=round(after - baseline_answer.prob, 6),
                kl_bits=round(_kl_bits(baseline_probs, probs), 4),
                top_token=_display_token(loaded.tokenizer, top_id),
                top_token_id=top_id,
                answer_changed=top_id != baseline_id,
            )
        )

    components.sort(key=lambda c: c.kl_bits, reverse=True)

    return AttributionResponse(
        model_id=loaded.info.id,
        display_name=loaded.info.display_name,
        tokens=prepared.tokens,
        position=pos,
        scope=scope,
        layer=layer if scope == "heads" else None,
        baseline_answer=baseline_answer,
        components=components,
        runs=len(targets),
        narration=narrate_attribution(
            scope,
            components,
            baseline_answer,
            _block_label(layer) if scope == "heads" and layer is not None else None,
        ),
        truncated=prepared.truncated,
        prompt_notice=prepared.notice,
    )
