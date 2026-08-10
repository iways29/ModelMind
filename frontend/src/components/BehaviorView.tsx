/**
 * "Behavior" — what the model actually writes, across many prompts at once.
 *
 * Every other view in this app is a microscope: it needs you to already know
 * which prompt is interesting. This is the view that finds the prompt. You give
 * it a list, it generates real continuations from one or two checkpoints, and it
 * shows you where they part ways — then hands the interesting ones to the lens.
 *
 * Generation is greedy, never sampled. Sampling would mean two runs of the same
 * model disagree with each other, and a diff between two checkpoints would then
 * be measuring the dice rather than the weights.
 */

import { useMemo, useState } from 'react'

import { behavior } from '../api/client'
import type { BehaviorResponse, ModelInfo, PromptBehavior } from '../api/types'
import { PALETTE } from './chartTheme'
import { Button, EmptyState, ErrorNote, Field, Panel, Select } from './ui'

/**
 * Preset suites. The off-domain set is the one that earns its keep: a fine-tune
 * damages a model most visibly on inputs it was *not* tuned for, and that damage
 * is invisible if you only ever test on the training domain.
 */
const SUITES: Record<string, { label: string; note: string; prompts: string[] }> = {
  offDomain: {
    label: 'Off-domain',
    note: 'Nothing to do with movies. This is where a movie-review fine-tune shows its damage.',
    prompts: [
      'My review of the restaurant:',
      'The capital of France is',
      'The weather tomorrow will be',
      'To install the package, run',
      'Dear Sir or Madam, I am writing to',
      'The worst thing about this product is',
      'She opened the door and saw',
      'The recipe calls for two cups of',
    ],
  },
  sentiment: {
    label: 'Sentiment',
    note: 'The domain gpt2-imdb was tuned on. Expect the fine-tune to look better here.',
    prompts: [
      'The movie was absolutely',
      'I watched this film and honestly it was',
      'The acting in this movie is',
      'Would I recommend it? Well,',
      'The plot was completely',
      'Overall my rating is',
    ],
  },
  factual: {
    label: 'Factual',
    note: 'Statements with a right answer, to catch a fine-tune trading knowledge for style.',
    prompts: [
      'The capital of France is',
      'Water boils at a temperature of',
      'The largest planet in our solar system is',
      'The author of Romeo and Juliet is',
      'There are seven continents, and they are',
    ],
  },
}

export function BehaviorView({
  models,
  modelId,
  onInspect,
}: {
  models: ModelInfo[]
  modelId: string
  /** Hands a prompt back to the composer and switches to the lens. */
  onInspect: (prompt: string) => void
}) {
  const [compareId, setCompareId] = useState('')
  const [text, setText] = useState(SUITES.offDomain.prompts.join('\n'))
  const [result, setResult] = useState<BehaviorResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const prompts = useMemo(
    () => text.split('\n').map((line) => line.trim()).filter(Boolean).slice(0, 25),
    [text],
  )

  // Two models on 8 prompts is ~16 CPU generations. Worth warning about before
  // the user clicks rather than leaving them staring at a spinner.
  const estimate = Math.ceil(prompts.length * (compareId ? 0.9 : 0.45))

  async function run() {
    if (prompts.length === 0 || loading) return
    setLoading(true)
    setError(null)
    try {
      setResult(
        await behavior({
          model_id: modelId,
          ...(compareId ? { compare_model_id: compareId } : {}),
          prompts,
          max_new_tokens: 20,
        }),
      )
    } catch (err) {
      setResult(null)
      setError(err instanceof Error ? err.message : String(err))
    }
    setLoading(false)
  }

  return (
    <Panel
      title="Behavior"
      subtitle="Generates real text for a whole list of prompts and diffs two checkpoints against each other. Greedy decoding, so a difference here is the weights, never the dice."
      controls={
        <>
          <Field label="Compare against">
            <Select
              value={compareId}
              onChange={(event) => setCompareId(event.target.value)}
              disabled={loading}
            >
              <option value="">— nothing, just read output —</option>
              {models
                .filter((model) => model.id !== modelId)
                .map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.display_name}
                  </option>
                ))}
            </Select>
          </Field>
          <Button onClick={run} disabled={prompts.length === 0 || loading}>
            {loading ? `Generating… ~${estimate}s` : `Run ${prompts.length} prompts`}
          </Button>
        </>
      }
    >
      <div className="space-y-7">
        <Composer text={text} onChange={setText} count={prompts.length} disabled={loading} />

        {error && <ErrorNote message={error} />}

        {result ? (
          <Results result={result} onInspect={onInspect} />
        ) : (
          <EmptyState>
            {loading
              ? `Generating ${prompts.length} continuation${prompts.length === 1 ? '' : 's'}${
                  compareId ? ' from each of two models' : ''
                } on CPU. Roughly ${estimate} seconds.`
              : 'Pick a suite, choose a checkpoint to compare against, and run. This is the view that tells you which prompts are worth looking at closely.'}
          </EmptyState>
        )}
      </div>
    </Panel>
  )
}

function Composer({
  text,
  onChange,
  count,
  disabled,
}: {
  text: string
  onChange: (value: string) => void
  count: number
  disabled: boolean
}) {
  const [suite, setSuite] = useState('offDomain')

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
        <span className="font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
          Prompts — one per line ({count})
        </span>
        <div className="flex gap-1.5">
          {Object.entries(SUITES).map(([key, value]) => (
            <button
              key={key}
              type="button"
              disabled={disabled}
              onClick={() => {
                setSuite(key)
                onChange(value.prompts.join('\n'))
              }}
              className={`rounded-lg border px-2.5 py-1 font-mono text-[10px] tracking-[0.08em] uppercase transition-colors disabled:opacity-40 ${
                suite === key
                  ? 'border-base-accent/50 bg-base-accent/10 text-ink'
                  : 'border-line-bright bg-raised text-ink-faint hover:text-ink-muted'
              }`}
            >
              {value.label}
            </button>
          ))}
        </div>
      </div>

      <textarea
        value={text}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        rows={6}
        spellCheck={false}
        className="w-full resize-y rounded-lg border border-line-bright bg-raised px-3.5 py-2.5 font-mono text-[12px] leading-relaxed text-ink placeholder:text-ink-faint/60 focus:border-base-accent/70 focus:ring-1 focus:ring-base-accent/25 focus:outline-none disabled:opacity-50"
      />
      <p className="mt-2 text-[11px] leading-relaxed text-ink-faint">{SUITES[suite]?.note}</p>
    </div>
  )
}

/* -------------------------------------------------------------------------- */

function Results({
  result,
  onInspect,
}: {
  result: BehaviorResponse
  onInspect: (prompt: string) => void
}) {
  // Surprise is only meaningful next to the other rows in the same run, so the
  // bar is scaled to this run rather than to any absolute ceiling.
  const worst = Math.max(...result.rows.map((r) => r.surprise_bits ?? 0), 0.001)
  const ordered = useMemo(
    () => [...result.rows].sort((a, b) => (b.surprise_bits ?? 0) - (a.surprise_bits ?? 0)),
    [result.rows],
  )

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <div>
        <h3 className="mb-3 font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
          {result.compare_display_name
            ? 'Most changed first — click any row to open it in the lens'
            : 'What the model wrote'}
        </h3>
        <ol className="space-y-2">
          {ordered.map((row) => (
            <Row key={row.prompt} row={row} worst={worst} onInspect={onInspect} />
          ))}
        </ol>
      </div>

      <Narration findings={result.narration} />
    </div>
  )
}

function Row({
  row,
  worst,
  onInspect,
}: {
  row: PromptBehavior
  worst: number
  onInspect: (prompt: string) => void
}) {
  const inspectable = row.divergence !== null

  return (
    <li className="overflow-hidden rounded-lg border border-line bg-raised/40">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-line px-4 py-2.5">
        <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-ink">{row.prompt}</span>
        {row.flags.map((flag) => (
          <Flag key={flag} kind={flag} />
        ))}
        {row.surprise_bits !== null && (
          <span className="flex items-center gap-2" title="Average bits of surprise — how far this row's continuation moved. Only comparable against the other rows in this run.">
            <span className="h-1 w-16 overflow-hidden rounded-full bg-line">
              <span
                className="block h-full rounded-full"
                style={{
                  width: `${(row.surprise_bits / worst) * 100}%`,
                  background: PALETTE.delta,
                }}
              />
            </span>
            <span className="tabular font-mono text-[10px] text-ink-faint">
              {row.surprise_bits.toFixed(1)}
            </span>
          </span>
        )}
        {inspectable && (
          <button
            type="button"
            onClick={() => onInspect(row.divergence!.shared_prefix)}
            className="shrink-0 rounded border border-line-bright px-2 py-0.5 font-mono text-[9px] tracking-[0.1em] text-ink-faint uppercase transition-colors hover:border-base-accent/50 hover:text-ink"
          >
            Inspect →
          </button>
        )}
      </div>

      <div className={row.compare ? 'grid gap-px bg-line sm:grid-cols-2' : ''}>
        <Said label={row.primary.display_name} prompt={row.prompt} said={row.primary.text} />
        {row.compare && (
          <Said
            label={row.compare.display_name}
            prompt={row.prompt}
            said={row.compare.text}
            accent
          />
        )}
      </div>

      {row.divergence && (
        <div className="border-t border-line px-4 py-2 font-mono text-[10px] text-ink-faint">
          Parted ways at word {row.divergence.index + 1}:{' '}
          <span style={{ color: PALETTE.base }}>{clean(row.divergence.token.token)}</span>{' '}
          ({(row.divergence.token.prob * 100).toFixed(0)}%) vs{' '}
          <span style={{ color: PALETTE.tuned }}>{clean(row.divergence.compare_token.token)}</span>{' '}
          ({(row.divergence.compare_token.prob * 100).toFixed(0)}%)
        </div>
      )}
    </li>
  )
}

/** One model's output, with the prompt greyed so the generated part stands out. */
function Said({
  label,
  prompt,
  said,
  accent = false,
}: {
  label: string
  prompt: string
  said: string
  accent?: boolean
}) {
  return (
    <div className="bg-panel/60 px-4 py-3">
      <div
        className="mb-1.5 font-mono text-[9px] tracking-[0.14em] uppercase"
        style={{ color: accent ? PALETTE.tuned : PALETTE.base }}
      >
        {label}
      </div>
      <p className="font-mono text-[12px] leading-relaxed break-words">
        <span className="text-ink-faint/60">{prompt}</span>
        <span className="text-ink-muted">{said}</span>
      </p>
    </div>
  )
}

const FLAG_COPY: Record<string, { text: string; tone: 'warn' | 'mute' }> = {
  repeats: { text: 'repeats', tone: 'warn' },
  drifted: { text: 'drifted', tone: 'warn' },
  identical: { text: 'identical', tone: 'mute' },
}

function Flag({ kind }: { kind: string }) {
  const copy = FLAG_COPY[kind] ?? { text: kind, tone: 'mute' as const }
  return (
    <span
      className={`shrink-0 rounded-full border px-2 py-0.5 font-mono text-[9px] tracking-[0.1em] uppercase ${
        copy.tone === 'warn'
          ? 'border-tuned-accent/40 bg-tuned-accent/10 text-tuned-accent'
          : 'border-line-bright bg-line/40 text-ink-faint'
      }`}
    >
      {copy.text}
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

function clean(token: string): string {
  return token.replace('␣', '').replace('⏎', '\\n') || token
}

export default BehaviorView
