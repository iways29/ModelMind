/**
 * "Find the cause" — the two questions that come after "what happened".
 *
 * The lens shows *when* an answer forms. This shows what built it, and what
 * changed it. Both halves are causal in a way the rest of the app isn't:
 *
 *   Attribution splits the answer into one number per part of the network. The
 *   numbers are exact and they add back up to the answer, so nothing is hidden
 *   in a fudge factor — the panel prints the reconciliation.
 *
 *   Reverting runs your model with another checkpoint's weights in exactly one
 *   place. If the behaviour comes back, that part was carrying it. This is the
 *   only view that answers "what do I tweak" with evidence rather than a hunch.
 *
 * Everything is a labelled bar with the number printed next to it. Signed
 * quantities get a centre line so "argued against" is visible as direction
 * rather than inferred from a minus sign.
 */

import { useEffect, useMemo, useState } from 'react'

import { attribution, patch } from '../api/client'
import type {
  AttributionResponse,
  Contribution,
  LayerPatch,
  ModelInfo,
  PatchResponse,
} from '../api/types'
import { PALETTE } from './chartTheme'
import { Button, EmptyState, ErrorNote, Field, Note, Panel, Select } from './ui'

export function CauseView({
  models,
  modelId,
  prompt,
}: {
  models: ModelInfo[]
  modelId: string
  prompt: string
}) {
  return (
    <div className="space-y-6">
      <BuiltThisAnswer modelId={modelId} prompt={prompt} />
      <WhatChanged models={models} modelId={modelId} prompt={prompt} />
    </div>
  )
}

// --------------------------------------------------------------------------
// Half one — what built this answer
// --------------------------------------------------------------------------

function BuiltThisAnswer({ modelId, prompt }: { modelId: string; prompt: string }) {
  const [result, setResult] = useState<AttributionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [showHeads, setShowHeads] = useState(false)

  // One forward pass, so this can run on its own as soon as there's a prompt
  // rather than making the user press a button to see the headline half.
  useEffect(() => {
    if (!modelId || !prompt.trim()) return
    let cancelled = false
    setLoading(true)
    setError(null)
    attribution({ model_id: modelId, prompt })
      .then((response) => {
        if (!cancelled) setResult(response)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setResult(null)
        setError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [modelId, prompt])

  if (error) {
    return (
      <Panel title="What built this answer" subtitle={ATTRIBUTION_SUBTITLE}>
        <ErrorNote message={error} />
      </Panel>
    )
  }

  if (!result) {
    return (
      <Panel title="What built this answer" subtitle={ATTRIBUTION_SUBTITLE}>
        <EmptyState>
          {loading ? 'Splitting the answer…' : 'Enter a prompt above to see which parts built it.'}
        </EmptyState>
      </Panel>
    )
  }

  const shown = showHeads ? result.heads.slice(0, 20) : result.blocks
  const scale = Math.max(...shown.map((c) => Math.abs(c.logits)), 0.001)
  const partsOnly = result.blocks.reduce((total, c) => total + c.logits, 0)
  const sum = partsOnly + result.unattributed

  return (
    <Panel
      title="What built this answer"
      subtitle={ATTRIBUTION_SUBTITLE}
      controls={
        <div className="flex gap-1 rounded-lg border border-line-bright bg-raised p-1">
          {[
            { id: false, label: `Parts (${result.blocks.length})` },
            { id: true, label: `Heads (${result.heads.length})` },
          ].map(({ id, label }) => (
            <button
              key={String(id)}
              type="button"
              onClick={() => setShowHeads(id)}
              className={`rounded px-3 py-1.5 font-mono text-[11px] tracking-[0.08em] uppercase transition-colors ${
                showHeads === id ? 'bg-base-accent text-ground' : 'text-ink-faint hover:text-ink'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      }
    >
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_19rem]">
        <div className="space-y-6">
          <Decision answer={result.answer.token} contrast={result.contrast.token} margin={result.margin} />

          <ol className="space-y-1">
            {shown.map((c) => (
              <ContributionRow key={c.label} contribution={c} scale={scale} />
            ))}
          </ol>

          {showHeads && result.heads.length > 20 && (
            <p className="font-mono text-[10px] text-ink-faint">
              Showing the 20 strongest of {result.heads.length} heads.
            </p>
          )}

          {/* The arithmetic is the credibility of the whole panel: if the parts
              didn't add up to the answer, the split would be a guess. */}
          <div className="border-t border-line pt-4 font-mono text-[10px] leading-relaxed text-ink-faint">
            The {result.blocks.length} parts sum to {partsOnly.toFixed(3)}
            {Math.abs(result.unattributed) > 0.0005 &&
              `, plus ${result.unattributed.toFixed(3)} of bias that belongs to no part`}
            , giving {sum.toFixed(3)} against a real margin of {result.margin.toFixed(3)}. Nothing is
            estimated.
          </div>
        </div>

        <Findings findings={result.narration} />
      </div>
    </Panel>
  )
}

const ATTRIBUTION_SUBTITLE =
  'Every part of the network adds into one running total, and the read-out is linear — so the answer splits into exact per-part numbers that add back up.'

function Decision({
  answer,
  contrast,
  margin,
}: {
  answer: string
  contrast: string
  margin: number
}) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <span className="font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        it chose
      </span>
      <span className="font-display text-4xl leading-none text-ink">{clean(answer)}</span>
      <span className="font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">over</span>
      <span className="font-display text-2xl leading-none text-ink-muted">{clean(contrast)}</span>
      <span className="tabular font-mono text-[12px] text-base-accent">
        by {margin.toFixed(2)} logits
      </span>
    </div>
  )
}

/**
 * A signed bar growing from a centre line. Direction carries the sign, so
 * "pushed toward" and "argued against" are distinguishable at a glance instead
 * of only in the printed number.
 */
function ContributionRow({ contribution, scale }: { contribution: Contribution; scale: number }) {
  const width = (Math.abs(contribution.logits) / scale) * 50
  const positive = contribution.logits >= 0
  const colour = positive ? PALETTE.base : PALETTE.danger

  return (
    <li className="flex items-center gap-3 rounded px-2 py-1 transition-colors hover:bg-raised/60">
      <span className="w-[68px] shrink-0 text-right font-mono text-[11px] text-ink-muted">
        {contribution.label}
      </span>

      <span className="relative h-3 min-w-0 flex-1">
        <span className="absolute inset-y-0 left-1/2 w-px bg-line-bright" />
        <span
          className="absolute inset-y-0 rounded-sm"
          style={{
            width: `${width}%`,
            background: colour,
            left: positive ? '50%' : `${50 - width}%`,
          }}
        />
      </span>

      <span
        className="tabular w-14 shrink-0 text-right font-mono text-[11px]"
        style={{ color: positive ? PALETTE.ink : PALETTE.danger }}
      >
        {contribution.logits >= 0 ? '+' : ''}
        {contribution.logits.toFixed(2)}
      </span>
    </li>
  )
}

// --------------------------------------------------------------------------
// Half two — what changed, against another checkpoint
// --------------------------------------------------------------------------

function WhatChanged({
  models,
  modelId,
  prompt,
}: {
  models: ModelInfo[]
  modelId: string
  prompt: string
}) {
  const [donorId, setDonorId] = useState('')
  const [result, setResult] = useState<PatchResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [focusLayer, setFocusLayer] = useState<number | null>(null)

  // Default the donor to a base checkpoint that isn't the model under test —
  // reverting a model to itself is the one comparison that can never say
  // anything.
  const donors = useMemo(() => models.filter((m) => m.id !== modelId), [models, modelId])
  useEffect(() => {
    if (donors.length === 0) return
    setDonorId((current) => {
      if (current && donors.some((m) => m.id === current)) return current
      return (donors.find((m) => m.role === 'base') ?? donors[0]).id
    })
  }, [donors])

  async function run(layer: number | null) {
    if (!modelId || !donorId || !prompt.trim() || loading) return
    setLoading(true)
    setError(null)
    setFocusLayer(layer)
    try {
      setResult(
        await patch({
          recipient_model_id: modelId,
          donor_model_id: donorId,
          prompt,
          ...(layer === null ? {} : { layer }),
        }),
      )
    } catch (err: unknown) {
      setResult(null)
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const controls = (
    <>
      <Field label="Revert toward">
        <Select
          value={donorId}
          onChange={(event) => setDonorId(event.target.value)}
          disabled={donors.length === 0}
        >
          {donors.map((model) => (
            <option key={model.id} value={model.id}>
              {model.display_name}
            </option>
          ))}
        </Select>
      </Field>
      <div className="flex flex-col gap-1.5">
        <span aria-hidden className="font-mono text-[10px] uppercase opacity-0 select-none">
          run
        </span>
        <Button onClick={() => void run(null)} disabled={loading || !donorId || !prompt.trim()}>
          {loading ? 'Reverting…' : 'Run sweep'}
        </Button>
      </div>
    </>
  )

  return (
    <Panel title="What changed vs another model" subtitle={PATCH_SUBTITLE} controls={controls}>
      {error && <ErrorNote message={error} />}

      {!error && !result && (
        <EmptyState>
          {loading
            ? 'Reverting one part at a time…'
            : 'Pick a checkpoint to revert toward, then run the sweep. Each part of your model is swapped for that one in turn.'}
        </EmptyState>
      )}

      {!error && result && (
        <div className="space-y-6">
          <Disagreement result={result} />

          {result.agreed ? (
            <Note>
              Both checkpoints predict the same token here, so there's no difference to trace. The
              Behavior tab finds prompts where they disagree.
            </Note>
          ) : (
            <>
              <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_19rem]">
                <div className="space-y-4">
                  <Header />
                  <ol className="space-y-1">
                    {result.layers.map((row) => (
                      <PatchRow
                        key={row.label}
                        row={row}
                        best={row.layer === result.best_layer}
                        active={row.layer === focusLayer}
                        busy={loading}
                        onInspect={() => void run(row.layer)}
                      />
                    ))}
                  </ol>
                </div>
                <Findings findings={result.narration} />
              </div>

              {result.focus && <FocusText focus={result.focus} result={result} />}
            </>
          )}
        </div>
      )}
    </Panel>
  )
}

const PATCH_SUBTITLE =
  "Runs your model with another checkpoint's weights in exactly one place. If the behaviour comes back, that part was carrying it."

function Disagreement({ result }: { result: PatchResponse }) {
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border border-line bg-raised/50 px-4 py-3">
      <Side name={result.recipient_name} token={result.recipient_answer.token} prob={result.recipient_answer.prob} colour={PALETTE.tuned} />
      <span className="font-mono text-[11px] text-ink-faint">vs</span>
      <Side name={result.donor_name} token={result.donor_answer.token} prob={result.donor_answer.prob} colour={PALETTE.base} />
    </div>
  )
}

function Side({
  name,
  token,
  prob,
  colour,
}: {
  name: string
  token: string
  prob: number
  colour: string
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="font-mono text-[10px] tracking-[0.14em] uppercase" style={{ color: colour }}>
        {name}
      </span>
      <span className="flex items-baseline gap-2">
        <span className="font-mono text-[15px] text-ink">{clean(token)}</span>
        <span className="tabular font-mono text-[11px] text-ink-faint">
          {(prob * 100).toFixed(1)}%
        </span>
      </span>
    </div>
  )
}

function Header() {
  return (
    <div className="flex items-center gap-3 px-2 font-mono text-[9px] tracking-[0.14em] text-ink-faint uppercase">
      <span className="w-[68px] shrink-0 text-right">Reverted</span>
      <span
        className="min-w-0 flex-1"
        title="How far reverting this one part moves your model toward the other one's answer"
      >
        How much of the difference it recovers
      </span>
      <span className="w-14 shrink-0 text-right">Says</span>
      <span className="w-[70px] shrink-0" />
    </div>
  )
}

function PatchRow({
  row,
  best,
  active,
  busy,
  onInspect,
}: {
  row: LayerPatch
  best: boolean
  active: boolean
  busy: boolean
  onInspect: () => void
}) {
  const recovery = row.recovery ?? 0
  // Recovery can overshoot past the donor or go negative; the bar is clamped so
  // the axis stays comparable, and the printed number stays honest.
  const width = Math.max(0, Math.min(1, recovery)) * 100

  return (
    <li
      className={`flex items-center gap-3 rounded px-2 py-1.5 transition-colors ${
        active ? 'bg-raised' : 'hover:bg-raised/60'
      }`}
    >
      <span
        className={`w-[68px] shrink-0 text-right font-mono text-[11px] ${
          row.kind === 'block' ? 'text-ink-muted' : 'text-tuned-accent'
        }`}
        title={
          row.kind === 'embed'
            ? 'How the model reads your words, before any thinking'
            : row.kind === 'readout'
              ? 'The step that turns the finished calculation back into a word'
              : undefined
        }
      >
        {row.label}
      </span>

      <span className="relative h-3 min-w-0 flex-1 overflow-hidden rounded-sm bg-line">
        <span
          className="absolute inset-y-0 left-0 rounded-sm"
          style={{ width: `${width}%`, background: best ? PALETTE.base : PALETTE.inkFaint }}
        />
      </span>

      <span
        className={`tabular w-14 shrink-0 text-right font-mono text-[11px] ${
          best ? 'text-base-accent' : 'text-ink-faint'
        }`}
      >
        {(recovery * 100).toFixed(1)}%
      </span>

      <span className="flex w-[70px] shrink-0 items-center gap-1.5">
        <span className="truncate font-mono text-[11px] text-ink-muted">{clean(row.answer.token)}</span>
        {row.flipped && (
          <span className="shrink-0 font-mono text-[9px] text-base-accent" title="Reverting this part flips the answer back">
            ↺
          </span>
        )}
      </span>

      <button
        type="button"
        onClick={onInspect}
        disabled={busy}
        className="shrink-0 rounded border border-line-bright px-2 py-1 font-mono text-[9px] tracking-[0.1em] text-ink-faint uppercase transition-colors hover:border-base-accent/50 hover:text-ink disabled:opacity-30"
      >
        Read
      </button>
    </li>
  )
}

/**
 * The same prompt written three ways. A changed token is evidence; a changed
 * sentence is the thing anyone actually cares about, so this is the payoff of
 * the whole panel.
 */
function FocusText({ focus, result }: { focus: PatchResponse['focus']; result: PatchResponse }) {
  if (!focus) return null
  return (
    <div className="space-y-3 border-t border-line pt-5">
      <h3 className="font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        Written out, with {focus.label} reverted
      </h3>
      <Continuation label={result.recipient_name} text={focus.recipient_text} colour={PALETTE.tuned} />
      <Continuation
        label={`${result.recipient_name} · ${focus.label} reverted`}
        text={focus.patched_text}
        colour={PALETTE.delta}
      />
      <Continuation label={result.donor_name} text={focus.donor_text} colour={PALETTE.base} />
    </div>
  )
}

function Continuation({ label, text, colour }: { label: string; text: string; colour: string }) {
  return (
    <div className="rounded-lg border border-line bg-raised/40 p-3">
      <span className="font-mono text-[9px] tracking-[0.14em] uppercase" style={{ color: colour }}>
        {label}
      </span>
      <p className="mt-1.5 font-mono text-[12px] leading-relaxed text-ink-muted">{text}</p>
    </div>
  )
}

function Findings({ findings }: { findings: string[] }) {
  if (findings.length === 0) return null
  return (
    <aside className="lg:border-l lg:border-line lg:pl-7">
      <h3 className="mb-4 font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        What this means
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

/** Strip the visible-whitespace marker for prose; axes still want it. */
function clean(token: string): string {
  return token.replace('␣', '').replace('⏎', '\\n') || token
}

export default CauseView
