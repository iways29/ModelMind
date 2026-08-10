/**
 * "Watch it think" — the model's answer decoded at every layer.
 *
 * The column is the residual stream, drawn top-to-bottom: embeddings enter at
 * the top, each block writes into the stream, the answer falls out at the
 * bottom. That orientation is a reading-order choice; the structure it shows is
 * the real one.
 *
 * Two encodings carry the meaning, and neither is decorative:
 *   - luminosity = how much probability that layer puts on the final answer
 *   - blur       = entropy. An undecided layer is literally out of focus.
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
      subtitle="Every layer's residual stream, pushed through the model's own output head. Brightness is confidence in the final answer; blur is uncertainty."
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
  // reached so the blur range is meaningful for this prompt.
  const maxEntropy = useMemo(
    () => Math.max(1, ...layers.map((l) => l.entropy)),
    [layers],
  )

  return (
    <div className="relative">
      {/* The residual stream itself */}
      <div className="absolute top-2 bottom-2 left-[76px] w-px bg-gradient-to-b from-transparent via-line-bright to-transparent" />
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

  // Confidence in the eventual answer drives glow; entropy drives blur.
  const heat = Math.min(1, layer.target_prob)
  const focus = 1 - Math.min(1, layer.entropy / maxEntropy)
  const blur = (1 - focus) * 7
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
              filter: `blur(${4 + blur}px)`,
            }}
          />
          <span
            className={heat > 0.5 ? 'animate-breathe rounded-full' : 'rounded-full'}
            style={{
              width: nodeSize,
              height: nodeSize,
              background: isEmbed ? PALETTE.inkFaint : PALETTE.base,
              opacity: isEmbed ? 0.45 : 0.35 + heat * 0.65,
              filter: `blur(${blur * 0.5}px)`,
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
            style={{ filter: `blur(${blur * 0.18}px)` }}
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
              ↻ switched
            </span>
          )}
          <span className="hidden truncate font-mono text-[10px] text-ink-faint/70 md:inline">
            {layer.top.slice(1, 4).map((t) => clean(t.token)).join(' · ')}
          </span>
        </span>

        {/* Confidence in the final answer */}
        <span className="hidden w-24 shrink-0 items-center gap-2 sm:flex">
          <span className="h-1 flex-1 overflow-hidden rounded-full bg-line">
            <span
              className="block h-full rounded-full transition-[width] duration-500"
              style={{ width: `${heat * 100}%`, background: PALETTE.base }}
            />
          </span>
          <span className="tabular w-9 text-right font-mono text-[10px] text-ink-faint">
            {(layer.target_prob * 100).toFixed(0)}%
          </span>
        </span>
      </div>
    </li>
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
