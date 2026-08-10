"""Turning a layer trace into sentences a non-expert can read.

Pure functions over already-computed numbers — no torch, no model access. This
is the module that answers "so what?", which is the whole point of the tool:
the charts show what happened, these sentences say what it means.

Deliberately conservative: every claim here is a direct restatement of a number
in the trace, never an inference about *why* the model did something.
"""

from typing import List, Optional

from .schemas import LayerLens, TokenPrediction, TokenTrajectory

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
