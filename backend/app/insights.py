"""Turning a layer trace into sentences a non-expert can read.

Pure functions over already-computed numbers — no torch, no model access. This
is the module that answers "so what?", which is the whole point of the tool:
the charts show what happened, these sentences say what it means.

Deliberately conservative: every claim here is a direct restatement of a number
in the trace, never an inference about *why* the model did something.
"""

from typing import List, Optional, Tuple

from .schemas import (
    AblationEffect,
    ComponentEffect,
    LayerLens,
    LensTrace,
    PromptBehavior,
    TokenPrediction,
    TokenTrajectory,
)

# Below this probability, a layer's "guess" is barely better than noise and
# calling it the model's belief would overstate things.
_MEANINGFUL_PROB = 0.05

# Entropy (bits) under which we're willing to call the model "committed".
# Uniform over GPT-2's 50k vocab is ~15.6 bits, so 6 is genuinely decided.
_COMMITTED_BITS = 6.0

# Entropy is log-scaled, so 2 bits is a 4x reduction in tokens genuinely in
# play — enough to be worth a sentence. Smaller moves are noise.
_MEANINGFUL_DROP_BITS = 2.0

# A "faded" candidate must have genuinely led the field mid-network (not just
# flickered) and must end at a third or less of its peak.
_FADE_PEAK_PROB = 0.25
_FADE_RATIO = 0.34

# The lens is known to be unreliable in early layers — the residual stream there
# isn't yet aligned with the output embedding, so its "predictions" are closer to
# noise. Only trust a peak from the back two-thirds of the network.
_FADE_EARLIEST_FRACTION = 1 / 3


def narrate_lens(
    layers: List[LayerLens],
    trajectories: List[TokenTrajectory],
    final: TokenPrediction,
    read_token: str,
) -> List[str]:
    """Describe how the answer formed, as a short list of plain-English findings."""
    if not layers:
        return []

    findings: List[str] = []
    answer = _clean(final.token)
    blocks = [layer for layer in layers if layer.layer > 0]
    if not blocks:
        return findings

    findings.append(
        f'Reading the residual stream above "{_clean(read_token)}", the model settles on '
        f'"{answer}" with {_pct(final.prob)} confidence.'
    )

    # The embedding readout is a known artifact, not a prediction: GPT-2 ties its
    # input and output embeddings, so projecting the embedding back through the
    # unembedding just recovers the input token at ~100%. Saying so up front stops
    # people reading it as "the model already knew the answer".
    if layers[0].top[0].token == read_token:
        findings.append(
            f'Ignore the "embed" row — it just echoes "{_clean(read_token)}" back. GPT-2 ties its '
            f"input and output embeddings, so the lens can only recover the input token there. "
            f"The readout starts meaning something at L1."
        )

    lock = _lock_in_layer(layers, final.token_id)
    if lock is not None and lock.layer < layers[-1].layer:
        findings.append(
            f'"{answer}" first becomes the leading candidate at {lock.label} and stays there. '
            f"Everything before that is the model still making up its mind."
        )
    else:
        # Winning only at the final layer is the interesting case, not a fallback:
        # it means the answer was decided by the last block alone.
        runner_up = _clean(layers[-2].top[0].token) if len(layers) > 1 else None
        tail = f' — {layers[-2].label} was still backing "{runner_up}"' if runner_up else ""
        findings.append(
            f'"{answer}" only takes the lead at the very last layer{tail}. '
            f"The final block is doing the deciding here."
        )

    peak = max(blocks, key=lambda layer: layer.target_prob, default=None)
    if peak is not None and peak.layer != layers[-1].layer:
        drop = peak.target_prob - layers[-1].target_prob
        if drop > 0.05:
            findings.append(
                f"Confidence peaks at {peak.label} ({_pct(peak.target_prob)}) and then falls to "
                f"{_pct(layers[-1].target_prob)}. The model commits early, then hedges — "
                f"spreading probability across alternatives it considers almost as good."
            )

    # Baseline from the first real block, never from the embedding artifact above.
    # Requiring a real drop stops us calling a 6.3 -> 5.9 wobble a "collapse".
    committed = next(
        (
            layer
            for layer in blocks
            if layer.entropy < _COMMITTED_BITS
            and blocks[0].entropy - layer.entropy >= _MEANINGFUL_DROP_BITS
        ),
        None,
    )
    if committed is not None and committed.layer > blocks[0].layer:
        findings.append(
            f"{committed.label} is where the field narrows: {committed.entropy:.1f} bits of "
            f"entropy, down from {blocks[0].entropy:.1f} at {blocks[0].label}. Each bit removed "
            f"halves the number of tokens genuinely in play."
        )

    changes = [layer for layer in blocks if layer.changed]
    if changes:
        names = ", ".join(layer.label for layer in changes[:6])
        more = "" if len(changes) <= 6 else f" (+{len(changes) - 6} more)"
        findings.append(
            f"The leading guess changes {len(changes)} time"
            f"{'s' if len(changes) != 1 else ''} on the way up: {names}{more}."
        )

    for faded in _faded_candidates(trajectories, layers, final.token_id):
        findings.append(
            f'"{_clean(faded.token)}" is a mid-network answer the model walks back: it reaches '
            f"{_pct(faded.peak_prob)} at {layers[faded.peak_layer].label}, then finishes at "
            f"{_pct(faded.final_prob)}. Later blocks moved that mass elsewhere."
        )

    rivals = [c for c in layers[-1].top[1:] if c.prob >= _MEANINGFUL_PROB * 0.4]
    if rivals:
        listed = ", ".join(f'"{_clean(c.token)}" ({_pct(c.prob)})' for c in rivals[:3])
        findings.append(f"Still in contention at the end: {listed}.")

    return findings


def _faded_candidates(
    trajectories: List[TokenTrajectory],
    layers: List[LayerLens],
    final_token_id: int,
    limit: int = 2,
) -> List[TokenTrajectory]:
    """Tokens that peaked convincingly mid-network and then lost most of that mass.

    This is the logit lens's signature finding — a model that 'knows' something
    at layer 10 and dilutes it by layer 12 — so it earns its own sentence.
    """
    if len(layers) < 3:
        return []

    last_block = len(layers) - 1
    earliest = max(1, int(last_block * _FADE_EARLIEST_FRACTION))
    faded = [
        t
        for t in trajectories
        # The winning token isn't "walked back" even if its probability dips;
        # that story is already told as hedging.
        if t.token_id != final_token_id
        # Peak must sit in a real block, late enough to be trustworthy, and
        # before the end.
        and earliest <= t.peak_layer < last_block
        and t.peak_prob >= _FADE_PEAK_PROB
        and t.final_prob <= t.peak_prob * _FADE_RATIO
    ]
    faded.sort(key=lambda t: t.peak_prob - t.final_prob, reverse=True)
    return faded[:limit]


def _lock_in_layer(layers: List[LayerLens], final_token_id: int) -> Optional[LayerLens]:
    """Earliest layer after which the final answer never stops leading.

    Scans backwards so a token that briefly leads early, loses, and returns
    isn't miscredited with an early lock-in.
    """
    lock: Optional[LayerLens] = None
    for layer in reversed(layers):
        leader = layer.top[0]
        if leader.token_id == final_token_id and leader.prob >= _MEANINGFUL_PROB:
            lock = layer
        else:
            break
    return lock


# --------------------------------------------------------------------------
# Ablation
# --------------------------------------------------------------------------

# KL in bits between the intact and ablated output distributions. Ablating a
# single GPT-2 head usually lands well under 0.05, so anything below this is
# indistinguishable from the component doing nothing on this prompt.
_NEGLIGIBLE_KL = 0.02

# Above this the component is carrying real weight: ~0.5 bits of divergence
# means the answer distribution has visibly reshaped, not just jittered.
_STRONG_KL = 0.5

# Probability swing worth naming in prose.
_MEANINGFUL_SHIFT = 0.02


def narrate_ablation(
    label: str,
    effect: AblationEffect,
    baseline: LensTrace,
    ablated: LensTrace,
) -> List[str]:
    """Describe what switching a component off did, as plain-English findings."""
    findings: List[str] = []
    answer = _clean(effect.baseline_answer.token)

    if effect.answer_changed:
        findings.append(
            f'Switching off {label} changes the answer: "{answer}" becomes '
            f'"{_clean(effect.ablated_answer.token)}" ({_pct(effect.ablated_answer.prob)}). '
            f"This component is load-bearing for this prompt."
        )
    elif effect.kl_bits < _NEGLIGIBLE_KL:
        findings.append(
            f'{label} makes almost no difference here — {effect.kl_bits:.3f} bits of divergence, '
            f'and "{answer}" still wins at {_pct(effect.baseline_answer_prob_after)}. '
            f"On this prompt the model routes around it."
        )
    else:
        findings.append(
            f'{label} is not decisive — "{answer}" still wins without it — but the output '
            f"distribution does move: {effect.kl_bits:.2f} bits of divergence from the intact run."
        )

    if abs(effect.prob_delta) >= _MEANINGFUL_SHIFT:
        direction = "drops" if effect.prob_delta < 0 else "rises"
        findings.append(
            f'Confidence in "{answer}" {direction} from {_pct(effect.baseline_answer.prob)} to '
            f"{_pct(effect.baseline_answer_prob_after)} — a {abs(effect.prob_delta) * 100:.1f} "
            f"point swing attributable to {label} alone."
        )

    divergence = _first_divergence(baseline, ablated)
    if divergence is not None:
        base_layer, abl_layer = divergence
        findings.append(
            f"The two runs first disagree at {base_layer.label}: intact says "
            f'"{_clean(base_layer.top[0].token)}", ablated says '
            f'"{_clean(abl_layer.top[0].token)}". Everything above that layer is identical, '
            f"which is the expected shape — a component can only affect what comes after it."
        )

    gainer = next(
        (
            s
            for s in effect.top_shifts
            if s.delta > 0 and s.token_id != effect.baseline_answer.token_id
        ),
        None,
    )
    if gainer is not None and gainer.delta >= _MEANINGFUL_SHIFT:
        findings.append(
            f'The mass goes to "{_clean(gainer.token)}", which climbs from '
            f"{_pct(gainer.baseline_prob)} to {_pct(gainer.ablated_prob)}. That is the "
            f"prediction the model falls back on without {label}."
        )

    if effect.kl_bits >= _NEGLIGIBLE_KL:
        findings.append(
            "One caveat: this measures what the component contributes *given the rest of the "
            "network is intact*. A component can look unimportant because another one "
            "compensates, which single-component ablation cannot see."
        )

    return findings


def _first_divergence(
    baseline: LensTrace, ablated: LensTrace
) -> Optional[Tuple[LayerLens, LayerLens]]:
    """First layer where the two runs' leading candidates differ.

    Skips the embedding row, which is identical by construction — no ablation
    can reach behind the input embeddings.
    """
    for base_layer, abl_layer in zip(baseline.layers[1:], ablated.layers[1:]):
        if base_layer.top[0].token_id != abl_layer.top[0].token_id:
            return base_layer, abl_layer
    return None


def narrate_attribution(
    scope: str,
    components: List[ComponentEffect],
    baseline_answer: TokenPrediction,
    layer_label: Optional[str],
) -> List[str]:
    """Describe a ranked sweep: what carried the prediction, and what was inert."""
    if not components:
        return []

    findings: List[str] = []
    answer = _clean(baseline_answer.token)
    kind = "head" if scope == "heads" else "block"
    where = f" in {layer_label}" if layer_label else ""
    strongest = components[0]

    findings.append(
        f'Intact, the model answers "{answer}" at {_pct(baseline_answer.prob)}. Every {kind}'
        f"{where} was then switched off one at a time and the output distribution re-measured."
    )

    if strongest.kl_bits < _NEGLIGIBLE_KL:
        findings.append(
            f"No single {kind}{where} matters much on this prompt — the strongest, "
            f"{strongest.label}, only moves the distribution {strongest.kl_bits:.3f} bits. "
            f"The prediction is spread across the network rather than carried by one {kind}."
        )
    else:
        findings.append(
            f"{strongest.label} carries the most: removing it costs {strongest.kl_bits:.2f} bits "
            f'and takes "{answer}" to {_pct(strongest.baseline_answer_prob_after)}'
            + (
                f', with "{_clean(strongest.top_token)}" winning instead.'
                if strongest.answer_changed
                else "."
            )
        )

    flippers = [c for c in components if c.answer_changed]
    if flippers:
        listed = ", ".join(c.label for c in flippers[:5])
        more = "" if len(flippers) <= 5 else f" (+{len(flippers) - 5} more)"
        findings.append(
            f"{len(flippers)} {kind}{'s' if len(flippers) != 1 else ''} change the answer outright "
            f"when removed: {listed}{more}."
        )

    inert = [c for c in components if c.kl_bits < _NEGLIGIBLE_KL]
    if inert and len(inert) < len(components):
        findings.append(
            f"{len(inert)} of {len(components)} {kind}s are effectively inert here "
            f"(under {_NEGLIGIBLE_KL} bits). That is normal — most components specialise, "
            f"and only some of them are relevant to any given prompt."
        )

    if strongest.kl_bits >= _STRONG_KL:
        findings.append(
            f"Read this as importance *on this prompt*, not in general. Re-run with a "
            f"different prompt and the ranking will usually change — that is the point of "
            f"the measurement, not a flaw in it."
        )

    return findings


# --------------------------------------------------------------------------
# Behavior
# --------------------------------------------------------------------------


def narrate_behavior(
    rows: List[PromptBehavior],
    name: str,
    compare_name: Optional[str],
) -> List[str]:
    """Summarise a whole run of prompts: what changed, what broke, where to look."""
    if not rows:
        return []

    findings: List[str] = []
    total = len(rows)

    if compare_name is None:
        looping = [r for r in rows if "repeats" in r.flags]
        findings.append(
            f"Generated {total} continuation{'s' if total != 1 else ''} from {name}, "
            f"greedily — no sampling, so re-running gives byte-identical output."
        )
        if looping:
            findings.append(
                f"{len(looping)} of {total} fall into a repetition loop. Greedy decoding does "
                f"this readily on small models; it is the single most common way output goes "
                f"bad, and it is a decoding problem before it is a training problem."
            )
        findings.append(
            "Add a second checkpoint to diff against and this view starts answering "
            "*what changed*, not just *what it says*."
        )
        return findings

    identical = [r for r in rows if r.identical]
    changed = [r for r in rows if not r.identical]
    findings.append(
        f"{len(changed)} of {total} prompts produce different text under {compare_name} "
        f"than under {name}. The other {len(identical)} are byte-identical."
    )

    if not changed:
        findings.append(
            f"On this set the two checkpoints are indistinguishable. That is a real result: "
            f"either the fine-tune did not touch this kind of input, or these prompts are not "
            f"the ones that would show it. Try prompts closer to what {compare_name} was tuned on."
        )
        return findings

    early = [r for r in changed if r.divergence is not None and r.divergence.index == 0]
    if early:
        findings.append(
            f"{len(early)} diverge on the very first generated word — the checkpoints disagree "
            f"immediately rather than drifting apart. Those are the clearest cases to open in "
            f"the lens, because the disagreement is not yet tangled up in different context."
        )

    drifted = sorted(
        (r for r in changed if r.surprise_bits is not None),
        key=lambda r: r.surprise_bits or 0,
        reverse=True,
    )
    if drifted:
        worst = drifted[0]
        findings.append(
            f'Furthest drift: "{_shorten(worst.prompt)}" — {name} finds {compare_name}\'s '
            f"continuation {worst.surprise_bits:.1f} bits per token surprising. High values mean "
            f"the fine-tune wrote something the base model would rarely have produced."
        )

    looping = [r for r in rows if "repeats" in r.flags]
    if looping:
        findings.append(
            f"{len(looping)} of {total} contain a repetition loop in at least one model. That is "
            f"a decoding artifact of greedy search, not evidence either checkpoint is broken — "
            f"but it does make those rows hard to read as behaviour."
        )

    findings.append(
        "Click any row to load that prompt into the lens and ablation views, positioned at the "
        "token where the two models parted ways. That is the point where 'what changed' becomes "
        "'which part of the network changed it'."
    )
    return findings


def _shorten(text: str, limit: int = 42) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clean(token: str) -> str:
    """Strip the visible whitespace markers so tokens read naturally in prose.

    '␣amazing' is right on a chart axis where leading spaces matter, and wrong
    inside a sentence.
    """
    return token.replace("␣", "").replace("⏎", "\\n").strip() or token


def _pct(value: float) -> str:
    """Percentages readable at a glance; sub-1% values stay honest rather than rounding to 0%."""
    if value < 0.01:
        return f"{value * 100:.2f}%"
    return f"{value * 100:.0f}%"
