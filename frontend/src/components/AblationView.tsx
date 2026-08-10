/**
 * "Ablate" — switch a component off, re-run, and see what the answer depended on.
 *
 * The lens shows *when* an answer formed. This shows *what it needed*, which is
 * the only question in the app answered by a counterfactual rather than by
 * reading the intact forward pass.
 *
 * Two granularities, one mechanism: a whole block passes the residual stream
 * through untouched, a single head has its slice zeroed before the attention
 * output projection mixes the heads together.
 */

import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

import { ablate, attribution } from '../api/client'
import type {
  AblateResponse,
  AnalyzeResponse,
  AttributionResponse,
  ComponentEffect,
  LayerLens,
  TokenShift,
} from '../api/types'
import { PALETTE } from './chartTheme'
import { Button, EmptyState, ErrorNote, Field, Panel, Select } from './ui'

/** One target for a run: a whole block, or one head inside it. */
interface Target {
  layer: number
  head: number | null
}

/**
 * Mirrors `_block_label` on the backend. Blocks are named after the residual
 * row they write, so block 4 reads as "L5" — the same row the lens shows it
 * landing in. Numbering both 0-based would make every ablation look off by one.
 */
function blockLabel(layer: number): string {
  return `L${layer + 1}`
}

export function AblationView({
  modelId,
  prompt,
  analysis,
}: {
  modelId: string
  prompt: string
  analysis: AnalyzeResponse | null
}) {
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState<number | null>(null)

  const [result, setResult] = useState<AblateResponse | null>(null)
  const [sweep, setSweep] = useState<AttributionResponse | null>(null)
  const [busy, setBusy] = useState<'ablate' | 'sweep' | null>(null)
  const [error, setError] = useState<string | null>(null)

  // A new analysis may come from a different prompt or a deeper model, so any
  // result on screen is stale the moment it lands.
  useEffect(() => {
    setResult(null)
    setSweep(null)
    setError(null)
  }, [analysis])

  const layerCount = analysis?.num_layers ?? 0
  const headCount = analysis?.num_heads ?? 0
  const safeLayer = Math.min(layer, Math.max(0, layerCount - 1))
  const safeHead = head === null ? null : Math.min(head, Math.max(0, headCount - 1))

  async function run(target: Target) {
    setBusy('ablate')
    setError(null)
    setLayer(target.layer)
    setHead(target.head)
    try {
      setResult(
        await ablate({
          model_id: modelId,
          prompt,
          ablations: [{ layer: target.layer, head: target.head }],
          top_k: 5,
        }),
      )
    } catch (err) {
      setResult(null)
      setError(err instanceof Error ? err.message : String(err))
    }
    setBusy(null)
  }

  async function runSweep(scope: 'layers' | 'heads') {
    setBusy('sweep')
    setError(null)
    try {
      setSweep(
        await attribution({
          model_id: modelId,
          prompt,
          scope,
          ...(scope === 'heads' ? { layer: safeLayer } : {}),
        }),
      )
    } catch (err) {
      setSweep(null)
      setError(err instanceof Error ? err.message : String(err))
    }
    setBusy(null)
  }

  if (!analysis) {
    return (
      <Panel
        title="Ablate"
        subtitle="Switch a block or a single attention head off, re-run the prompt, and diff the result against the intact model."
      >
        <EmptyState>
          Run a prompt first. Ablation needs the model&rsquo;s shape — how many blocks it has,
          and how many heads sit in each one.
        </EmptyState>
      </Panel>
    )
  }

  return (
    <Panel
      title="Ablate"
      subtitle="Switch a component off, re-run the prompt, and diff against the intact model. A whole block passes the residual stream through untouched; a single head has its slice zeroed before the heads are mixed."
      controls={
        <>
          <Field label="Block">
            <Select
              value={safeLayer}
              onChange={(event) => setLayer(Number(event.target.value))}
              disabled={busy !== null}
            >
              {Array.from({ length: layerCount }, (_, index) => (
                <option key={index} value={index}>
                  {blockLabel(index)}
                </option>
              ))}
            </Select>
          </Field>

          <Field label="Component">
            <Select
              value={safeHead === null ? 'block' : safeHead}
              onChange={(event) =>
                setHead(event.target.value === 'block' ? null : Number(event.target.value))
              }
              disabled={busy !== null}
            >
              <option value="block">Whole block</option>
              {Array.from({ length: headCount }, (_, index) => (
                <option key={index} value={index}>
                  Head {index}
                </option>
              ))}
            </Select>
          </Field>

          <Button
            onClick={() => run({ layer: safeLayer, head: safeHead })}
            disabled={busy !== null}
          >
            {busy === 'ablate' ? 'Running…' : 'Ablate'}
          </Button>
        </>
      }
    >
      <div className="space-y-8">
        {error && <ErrorNote message={error} />}

        {result ? (
          <AblationResult result={result} />
        ) : (
          <EmptyState>
            Pick a component and hit Ablate. Try {blockLabel(layerCount - 1)} first — the last
            block is usually where an answer gets spent on grammar.
          </EmptyState>
        )}

        <Sweep
          sweep={sweep}
          busy={busy}
          layerLabel={blockLabel(safeLayer)}
          onRun={runSweep}
          onPick={run}
        />
      </div>
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */

function AblationResult({ result }: { result: AblateResponse }) {
  const { effect, ablation_label: label } = result

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="space-y-7">
        <Verdict result={result} />

        <section>
          <SectionLabel>
            Layer by layer, both runs — highlighted where {label} changed the leader
          </SectionLabel>
          <TraceDiff
            baseline={result.baseline.layers}
            ablated={result.ablated.layers}
            label={label}
          />
        </section>

        <section>
          <SectionLabel>Where the probability went</SectionLabel>
          <Shifts shifts={effect.top_shifts} />
        </section>
      </div>

      <Narration findings={result.narration} title="What it means" />
    </div>
  )
}

function Verdict({ result }: { result: AblateResponse }) {
  const { effect, ablation_label: label } = result

  return (
    <div className="rounded-lg border border-line-bright bg-raised/60 p-5">
      <div className="flex flex-wrap items-center gap-x-8 gap-y-5">
        <Answer
          caption="Intact"
          token={clean(effect.baseline_answer.token)}
          prob={effect.baseline_answer.prob}
          color={PALETTE.base}
        />

        <span className="font-mono text-[18px] text-ink-faint" aria-hidden>
          →
        </span>

        <Answer
          caption={`Without ${label}`}
          token={clean(effect.ablated_answer.token)}
          prob={effect.ablated_answer.prob}
          color={effect.answer_changed ? PALETTE.tuned : PALETTE.base}
        />

        <div className="ml-auto flex flex-wrap items-center gap-6">
          <Readout
            label="Divergence"
            value={`${effect.kl_bits.toFixed(2)} bits`}
            title="KL(intact ‖ ablated) over the full vocabulary. Zero means this component made no difference to this prompt."
          />
          <Readout
            label={`"${clean(effect.baseline_answer.token)}" now`}
            value={`${(effect.baseline_answer_prob_after * 100).toFixed(1)}%`}
            delta={effect.prob_delta}
            title="Probability the ablated run still assigns to the answer the intact model gave."
          />
          <span
            className={`rounded-full border px-3 py-1 font-mono text-[10px] tracking-[0.12em] uppercase ${
              effect.answer_changed
                ? 'border-tuned-accent/40 bg-tuned-accent/10 text-tuned-accent'
                : 'border-line-bright bg-line/40 text-ink-faint'
            }`}
          >
            {effect.answer_changed ? 'Answer changed' : 'Answer held'}
          </span>
        </div>
      </div>
    </div>
  )
}

function Answer({
  caption,
  token,
  prob,
  color,
}: {
  caption: string
  token: string
  prob: number
  color: string
}) {
  return (
    <div>
      <div className="font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        {caption}
      </div>
      <div className="mt-1 flex items-baseline gap-2.5">
        <span className="font-display text-3xl leading-none" style={{ color }}>
          {token}
        </span>
        <span className="tabular font-mono text-[12px] text-ink-faint">
          {(prob * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  )
}

function Readout({
  label,
  value,
  delta,
  title,
}: {
  label: string
  value: string
  delta?: number
  title: string
}) {
  return (
    <div title={title}>
      <div className="font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        {label}
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="tabular font-mono text-[14px] text-ink">{value}</span>
        {delta !== undefined && Math.abs(delta) >= 0.0005 && (
          <span
            className="tabular font-mono text-[11px]"
            style={{ color: delta < 0 ? PALETTE.danger : PALETTE.base }}
          >
            {delta > 0 ? '+' : ''}
            {(delta * 100).toFixed(1)}
          </span>
        )}
      </div>
    </div>
  )
}

/**
 * The two runs side by side, one row per layer.
 *
 * Rows above the ablated component are identical by construction — a component
 * cannot affect what came before it — so the first highlighted row is exactly
 * where the ablation took hold.
 */
function TraceDiff({
  baseline,
  ablated,
  label,
}: {
  baseline: LayerLens[]
  ablated: LayerLens[]
  label: string
}) {
  const rows = baseline.slice(0, Math.min(baseline.length, ablated.length))

  return (
    <div className="overflow-hidden rounded-lg border border-line">
      <div className="grid grid-cols-[3.5rem_1fr_1fr] items-center gap-3 border-b border-line bg-raised/50 px-3 py-2 font-mono text-[9px] tracking-[0.14em] text-ink-faint uppercase">
        <span>Layer</span>
        <span>Intact</span>
        <span>Without {label}</span>
      </div>
      <ol>
        {rows.map((base, index) => {
          const other = ablated[index]
          const differs = base.top[0]?.token_id !== other.top[0]?.token_id
          return (
            <li
              key={base.layer}
              className={`grid grid-cols-[3.5rem_1fr_1fr] items-center gap-3 border-b border-line/60 px-3 py-1.5 last:border-b-0 ${
                differs ? 'bg-tuned-accent/[0.06]' : ''
              }`}
            >
              <span
                className={`font-mono text-[11px] ${
                  differs ? 'text-tuned-accent' : 'text-ink-faint'
                }`}
              >
                {base.label}
              </span>
              <Guess prediction={base.top[0]} color={PALETTE.base} />
              <Guess
                prediction={other.top[0]}
                color={differs ? PALETTE.tuned : PALETTE.base}
                muted={!differs}
              />
            </li>
          )
        })}
      </ol>
    </div>
  )
}

function Guess({
  prediction,
  color,
  muted = false,
}: {
  prediction: { token: string; prob: number } | undefined
  color: string
  muted?: boolean
}) {
  if (!prediction) return <span />
  return (
    <span className="flex min-w-0 items-baseline gap-2">
      <span
        className="truncate font-mono text-[13px]"
        style={{ color: muted ? PALETTE.inkMuted : color }}
      >
        {clean(prediction.token)}
      </span>
      <span className="tabular shrink-0 font-mono text-[10px] text-ink-faint">
        {(prediction.prob * 100).toFixed(0)}%
      </span>
    </span>
  )
}

/** Diverging bars: probability the ablation took away, and where it went instead. */
function Shifts({ shifts }: { shifts: TokenShift[] }) {
  const scale = Math.max(...shifts.map((s) => Math.abs(s.delta)), 0.001)

  return (
    <ol className="space-y-1.5">
      {shifts.map((shift) => {
        const width = (Math.abs(shift.delta) / scale) * 50
        const gained = shift.delta > 0
        return (
          <li key={shift.token_id} className="flex items-center gap-3">
            <span className="w-28 shrink-0 truncate font-mono text-[12px] text-ink-muted">
              {clean(shift.token)}
            </span>

            <span className="relative h-1.5 flex-1 rounded-full bg-line/60">
              <span className="absolute inset-y-0 left-1/2 w-px bg-line-bright" />
              <span
                className="absolute inset-y-0 rounded-full"
                style={{
                  width: `${width}%`,
                  left: gained ? '50%' : `${50 - width}%`,
                  background: gained ? PALETTE.base : PALETTE.danger,
                }}
              />
            </span>

            <span className="tabular w-36 shrink-0 text-right font-mono text-[10px] text-ink-faint">
              {(shift.baseline_prob * 100).toFixed(1)}% → {(shift.ablated_prob * 100).toFixed(1)}%
            </span>
            <span
              className="tabular w-14 shrink-0 text-right font-mono text-[11px]"
              style={{ color: gained ? PALETTE.base : PALETTE.danger }}
            >
              {gained ? '+' : ''}
              {(shift.delta * 100).toFixed(1)}
            </span>
          </li>
        )
      })}
    </ol>
  )
}

/* -------------------------------------------------------------------------- */

/**
 * The attribution sweep: ablate every component of one kind, rank by effect.
 *
 * A single ablation answers "did this matter?". With 12 heads per block, only a
 * sweep answers "which one mattered?" — trying them by hand is not a workflow.
 */
function Sweep({
  sweep,
  busy,
  layerLabel,
  onRun,
  onPick,
}: {
  sweep: AttributionResponse | null
  busy: 'ablate' | 'sweep' | null
  layerLabel: string
  onRun: (scope: 'layers' | 'heads') => void
  onPick: (target: Target) => void
}) {
  return (
    <section className="border-t border-line pt-7">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
        <div>
          <SectionLabel>Rank by contribution</SectionLabel>
          <p className="max-w-xl text-[12px] leading-relaxed text-ink-faint">
            Ablates each component in turn and measures how far the output distribution moves.
            Ranked by KL divergence from the intact run, strongest first. Click any row to open
            it in full above.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" onClick={() => onRun('layers')} disabled={busy !== null}>
            {busy === 'sweep' ? 'Sweeping…' : 'All blocks'}
          </Button>
          <Button variant="ghost" onClick={() => onRun('heads')} disabled={busy !== null}>
            Heads in {layerLabel}
          </Button>
        </div>
      </div>

      {sweep ? (
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <Ranking components={sweep.components} onPick={onPick} />
          <Narration findings={sweep.narration} title="What it means" />
        </div>
      ) : (
        <EmptyState>
          Sweep the whole network, or one block&rsquo;s heads, to see which components this
          prediction actually depends on.
        </EmptyState>
      )}
    </section>
  )
}

function Ranking({
  components,
  onPick,
}: {
  components: ComponentEffect[]
  onPick: (target: Target) => void
}) {
  const scale = useMemo(
    () => Math.max(...components.map((c) => c.kl_bits), 0.001),
    [components],
  )

  // Bars are square-rooted, not linear. Ablating the first block destroys the
  // whole computation downstream and scores an order of magnitude above
  // everything else, which on a linear scale flattens the other eleven bars to
  // invisible slivers. The exact figure sits next to every bar, so compressing
  // the outlier costs no precision and makes the ranking readable.
  const width = (kl: number) => Math.sqrt(Math.max(kl, 0) / scale) * 100

  return (
    <ol className="space-y-1">
      {components.map((component) => (
        <li key={component.label}>
          <button
            type="button"
            onClick={() => onPick({ layer: component.layer, head: component.head })}
            className="flex w-full items-center gap-3 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-raised/70"
          >
            <span
              className={`w-[4.5rem] shrink-0 font-mono text-[11px] ${
                component.answer_changed ? 'text-tuned-accent' : 'text-ink-faint'
              }`}
            >
              {component.label}
            </span>

            <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-line/60">
              <span
                className="block h-full rounded-full"
                style={{
                  width: `${width(component.kl_bits)}%`,
                  background: component.answer_changed ? PALETTE.tuned : PALETTE.base,
                }}
              />
            </span>

            <span className="tabular w-16 shrink-0 text-right font-mono text-[11px] text-ink-muted">
              {component.kl_bits.toFixed(3)}
            </span>

            <span className="hidden w-32 shrink-0 truncate font-mono text-[11px] text-ink-faint sm:block">
              {component.answer_changed ? `→ ${clean(component.top_token)}` : 'answer held'}
            </span>
          </button>
        </li>
      ))}
    </ol>
  )
}

/* -------------------------------------------------------------------------- */

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <h3 className="mb-3 font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
      {children}
    </h3>
  )
}

function Narration({ findings, title }: { findings: string[]; title: string }) {
  if (findings.length === 0) return null
  return (
    <aside className="lg:border-l lg:border-line lg:pl-7">
      <SectionLabel>{title}</SectionLabel>
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

export default AblationView
