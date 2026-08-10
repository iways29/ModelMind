/**
 * "Watch it think" — the model's answer decoded at every layer.
 *
 * The column is the residual stream, drawn top-to-bottom: embeddings enter at
 * the top, each block writes into the stream, the answer falls out at the
 * bottom. That orientation is a reading-order choice; the structure it shows is
 * the real one.
 *
 * The node's luminosity is how much probability that layer puts on the final
 * answer. Everything else is read off two labelled bars rather than encoded
 * optically — an earlier version blurred undecided layers, which was legible as
 * an effect and illegible as a number.
 */

import { useEffect, useMemo, useState } from 'react'

import type { LayerLens, LensResponse } from '../api/types'
import { PALETTE } from './chartTheme'
import TrajectoryRibbon from './TrajectoryRibbon'
import { EmptyState, Panel } from './ui'

export function LogitLens({ result }: { result: LensResponse | null }) {
  const [revealed, setRevealed] = useState<number>(Number.POSITIVE_INFINITY)
  const [highlightId, setHighlightId] = useState<number | null>(null)

  // Replay from the top whenever a new trace arrives, so the reveal reads as
  // "the model is thinking" rather than a chart that silently swapped.
  useEffect(() => {
    if (!result) return
    setRevealed(0)
    let layer = 0
    const timer = window.setInterval(() => {
      layer += 1
      setRevealed(layer)
      if (layer > result.layers.length) window.clearInterval(timer)
    }, 130)
    return () => window.clearInterval(timer)
  }, [result])

  if (!result) {
    return (
      <Panel
        title="Watch it think"
        subtitle="Decodes the model's predicted next token at every layer, not just the last one."
      >
        <EmptyState>
          Run a prompt to watch the answer form, layer by layer.
        </EmptyState>
      </Panel>
    )
  }

  const answer = clean(result.final_prediction.token)
  const readToken = clean(result.tokens[result.position] ?? '')

  return (
    <Panel
      title="Watch it think"
      subtitle="Every layer's residual stream, pushed through the model's own output head. Confidence is this layer's belief in the final answer; undecided is how spread out the rest of its guess is."
      controls={<Replay onClick={() => setRevealed(0)} />}
    >
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="space-y-8">
          <Verdict answer={answer} prob={result.final_prediction.prob} readToken={readToken} />

          <Column
            layers={result.layers}
            revealed={revealed}
            finalTokenId={result.final_prediction.token_id}
          />

          <div className="border-t border-line pt-6">
            <h3 className="mb-3 font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
              Every candidate, every layer
            </h3>
            <TrajectoryRibbon
              trajectories={result.trajectories}
              layerLabels={result.layers.map((l) => l.label)}
              finalTokenId={result.final_prediction.token_id}
              highlightId={highlightId}
              onHighlight={setHighlightId}
            />
          </div>
        </div>

        <Narration findings={result.narration} />
      </div>
    </Panel>
  )
}

function Verdict({
  answer,
  prob,
  readToken,
}: {
  answer: string
  prob: number
  readToken: string
}) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
      <span className="font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        after &ldquo;{readToken}&rdquo; it says
      </span>
      <span
        className="font-display text-5xl leading-none text-ink"
        style={{ textShadow: `0 0 34px ${PALETTE.base}55` }}
      >
        {answer}
      </span>
      <span className="tabular font-mono text-[13px] text-base-accent">
        {(prob * 100).toFixed(1)}%
      </span>
    </div>
  )
}

function Column({
  layers,
  revealed,
  finalTokenId,
}: {
  layers: LayerLens[]
  revealed: number
  finalTokenId: number
}) {
  // Entropy is unbounded in principle; scale against what this trace actually
  // reached so the bar spans the range this prompt genuinely covers.
  const maxEntropy = useMemo(
    () => Math.max(1, ...layers.map((l) => l.entropy)),
    [layers],
  )

  return (
    <div>
      {/* Outside the positioned wrapper below, so the spine still starts at the
          first layer row rather than being pushed down past the labels. */}
      <ColumnHeader />

      <div className="relative">
        {/* The residual stream itself. Explicit stops rather than a
            transparent->colour->transparent ramp: over a 13-row column that ramp
            spends most of its length near-invisible, and the spine is structure,
            not decoration. */}
        <div className="absolute top-2 bottom-2 left-[76px] w-px bg-[linear-gradient(to_bottom,transparent,var(--color-line-bright)_10%,var(--color-line-bright)_90%,transparent)]" />
        <div className="pointer-events-none absolute top-2 bottom-2 left-[76px] w-px overflow-hidden">
          <div className="animate-descend h-8 w-px bg-gradient-to-b from-transparent via-base-accent to-transparent" />
        </div>

        <ol className="space-y-1">
          {layers.map((layer, index) => (
            <LayerRow
              key={layer.layer}
              layer={layer}
              visible={index < revealed}
              delay={index * 40}
              maxEntropy={maxEntropy}
              isFinal={index === layers.length - 1}
              leaderIsAnswer={layer.top[0]?.token_id === finalTokenId}
            />
          ))}
        </ol>
      </div>
    </div>
  )
}

/**
 * Two bars sit at the end of every row and mean opposite things, so they have
 * to be named. Widths here must track the bar widths in `LayerRow`.
 */
function ColumnHeader() {
  return (
    <div className="mb-2 hidden items-center gap-4 px-2 font-mono text-[9px] tracking-[0.14em] text-ink-faint uppercase sm:flex">
      <span className="w-[46px] shrink-0" />
      <span className="w-8 shrink-0" />
      <span className="min-w-0 flex-1">Best guess · also in play</span>
      <span className="w-20 shrink-0 lg:w-24" title="Probability this layer already assigns to the model's final answer">
        Confidence
      </span>
      <span className="w-20 shrink-0 lg:w-24" title="Shannon entropy: how spread out this layer's distribution still is">
        Undecided
      </span>
    </div>
  )
}

function LayerRow({
  layer,
  visible,
  delay,
  maxEntropy,
  isFinal,
  leaderIsAnswer,
}: {
  layer: LayerLens
  visible: boolean
  delay: number
  maxEntropy: number
  isFinal: boolean
  leaderIsAnswer: boolean
}) {
  const leader = layer.top[0]
  if (!leader) return null

  // Confidence in the eventual answer drives the node's size and glow; entropy
  // is left entirely to its own bar, where it can be read as a number.
  const heat = Math.min(1, layer.target_prob)
  const spread = Math.min(1, layer.entropy / maxEntropy)
  const nodeSize = 9 + heat * 13
  const isEmbed = layer.layer === 0

  return (
    <li
      className={visible ? 'animate-settle' : 'opacity-0'}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-center gap-4 rounded-lg px-2 py-1.5 transition-colors hover:bg-raised/60">
        <span
          className={`w-[46px] shrink-0 text-right font-mono text-[11px] ${
            isFinal ? 'text-base-accent' : 'text-ink-faint'
          }`}
        >
          {layer.label}
        </span>

        {/* Node on the spine */}
        <span className="relative flex h-8 w-8 shrink-0 items-center justify-center">
          <span
            aria-hidden
            className="absolute rounded-full"
            style={{
              width: nodeSize + 14,
              height: nodeSize + 14,
              background: PALETTE.base,
              opacity: 0.1 + heat * 0.4,
              filter: 'blur(6px)',
            }}
          />
          <span
            className={heat > 0.5 ? 'animate-breathe rounded-full' : 'rounded-full'}
            style={{
              width: nodeSize,
              height: nodeSize,
              background: isEmbed ? PALETTE.inkFaint : PALETTE.base,
              opacity: isEmbed ? 0.45 : 0.35 + heat * 0.65,
              boxShadow: heat > 0.3 ? `0 0 ${8 + heat * 22}px ${PALETTE.base}` : undefined,
            }}
          />
        </span>

        {/* The guess */}
        <span className="flex min-w-0 flex-1 items-baseline gap-2.5">
          <span
            className={`truncate font-mono text-[15px] ${
              isEmbed
                ? 'text-ink-faint italic'
                : leaderIsAnswer
                  ? 'text-ink'
                  : 'text-ink-muted'
            }`}
          >
            {clean(leader.token)}
          </span>
          <span className="tabular shrink-0 font-mono text-[11px] text-ink-faint">
            {(leader.prob * 100).toFixed(0)}%
          </span>
          {layer.changed && !isEmbed && (
            <span
              className="shrink-0 font-mono text-[9px] tracking-[0.12em] text-tuned-accent uppercase"
              title="The leading candidate changed at this layer"
            >
              ↻<span className="hidden lg:inline"> switched</span>
            </span>
          )}
          {/* The runners-up are the point of the view as much as the leader is —
              they're what the model considered and dropped. Truncating them is
              fine; hiding them is not, so they stay from `sm` up and the two
              meters give back the width instead. */}
          <span className="hidden min-w-0 truncate font-mono text-[10px] text-ink-faint/70 sm:inline">
            {layer.top.slice(1, 4).map((t) => clean(t.token)).join(' · ')}
          </span>
        </span>

        <Meter
          className="hidden sm:flex"
          fraction={heat}
          color={PALETTE.base}
          value={`${(layer.target_prob * 100).toFixed(0)}%`}
          title={`This layer puts ${(layer.target_prob * 100).toFixed(1)}% on the model's final answer`}
        />
        <Meter
          className="hidden sm:flex"
          fraction={spread}
          color={PALETTE.tuned}
          value={layer.entropy.toFixed(1)}
          title={`${layer.entropy.toFixed(2)} bits of entropy — a full bar is the most undecided this run gets. Each bit removed halves the number of tokens genuinely in play.`}
        />
      </div>
    </li>
  )
}

/** One labelled bar + its number. Both row metrics render through this. */
function Meter({
  fraction,
  color,
  value,
  title,
  className = '',
}: {
  fraction: number
  color: string
  value: string
  title: string
  className?: string
}) {
  return (
    <span className={`w-20 shrink-0 items-center gap-2 lg:w-24 ${className}`} title={title}>
      <span className="h-1 flex-1 overflow-hidden rounded-full bg-line">
        <span
          className="block h-full rounded-full transition-[width] duration-500"
          style={{ width: `${Math.max(0, Math.min(1, fraction)) * 100}%`, background: color }}
        />
      </span>
      <span className="tabular w-9 text-right font-mono text-[10px] text-ink-faint">{value}</span>
    </span>
  )
}

function Narration({ findings }: { findings: string[] }) {
  if (findings.length === 0) return null
  return (
    <aside className="lg:border-l lg:border-line lg:pl-7">
      <h3 className="mb-4 font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        What happened
      </h3>
      <ol className="space-y-4">
        {findings.map((finding, index) => (
          <li key={finding} className="flex gap-3">
            <span className="tabular mt-0.5 shrink-0 font-mono text-[10px] text-base-accent/70">
              {String(index + 1).padStart(2, '0')}
            </span>
            <p className="text-[13px] leading-relaxed text-ink-muted">{finding}</p>
          </li>
        ))}
      </ol>
    </aside>
  )
}

function Replay({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-lg border border-line-bright bg-raised px-3.5 py-2 font-mono text-[11px] tracking-[0.1em] text-ink-muted uppercase transition-colors hover:border-base-accent/50 hover:text-ink"
    >
      ▸ Replay
    </button>
  )
}

/** Strip the visible-whitespace marker for prose; axes still want it. */
function clean(token: string): string {
  return token.replace('␣', '').replace('⏎', '\\n') || token
}

export default LogitLens
