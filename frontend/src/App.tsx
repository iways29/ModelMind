/**
 * Single-page composition: prompt composer at the top, one analysis run
 * feeding the Attention and Activation panels, and a self-contained Compare
 * panel that shares the prompt.
 */

import { useEffect, useState } from 'react'

import { analyze, API_BASE_URL, fetchModels } from './api/client'
import type { AnalyzeResponse, ModelInfo } from './api/types'
import ActivationChart from './components/ActivationChart'
import AttentionHeatmap from './components/AttentionHeatmap'
import CompareView from './components/CompareView'
import ModelSelector from './components/ModelSelector'
import { Button, ErrorNote, Note } from './components/ui'

const TABS = [
  { id: 'attention', label: 'Attention' },
  { id: 'activations', label: 'Activations' },
  { id: 'compare', label: 'Compare' },
] as const

type TabId = (typeof TABS)[number]['id']

const DEFAULT_PROMPT = 'The movie was absolutely'

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [modelId, setModelId] = useState('')

  const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [tab, setTab] = useState<TabId>('attention')

  useEffect(() => {
    let cancelled = false
    fetchModels()
      .then((catalog) => {
        if (cancelled) return
        setModels(catalog)
        setModelId((current) => current || catalog[0]?.id || '')
      })
      .catch((err: unknown) => {
        if (!cancelled) setModelsError(err instanceof Error ? err.message : String(err))
      })
    return () => {
      cancelled = true
    }
  }, [])

  const canRun = Boolean(prompt.trim()) && Boolean(modelId) && !loading

  async function run() {
    if (!canRun) return
    setLoading(true)
    setError(null)
    try {
      setResult(await analyze({ model_id: modelId, prompt }))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-grid min-h-full">
      <div className="mx-auto max-w-[1180px] px-6 pb-24">
        <Header modelCount={models.length} />

        <main className="space-y-6">
          <section className="rounded-xl border border-line bg-panel/80 p-5">
            <div className="flex flex-wrap items-end gap-4">
              <label className="flex min-w-[22rem] flex-1 flex-col gap-1.5">
                <span className="font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
                  Prompt
                </span>
                <input
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') run()
                  }}
                  placeholder={DEFAULT_PROMPT}
                  className="w-full rounded-lg border border-line-bright bg-raised px-3.5 py-2.5 font-mono text-[13px] text-ink placeholder:text-ink-faint/60 focus:border-base-accent/70 focus:ring-1 focus:ring-base-accent/25 focus:outline-none"
                />
              </label>

              <ModelSelector
                label="Model"
                models={models}
                value={modelId}
                onChange={setModelId}
                disabled={models.length === 0}
              />

              {/* Spacer label keeps the button on the same baseline as the
                  inputs at any width — a fixed bottom padding leaves a dead
                  gap as soon as the row wraps. */}
              <div className="flex flex-col gap-1.5">
                <span aria-hidden className="font-mono text-[10px] tracking-[0.16em] uppercase opacity-0 select-none">
                  Run
                </span>
                <Button onClick={run} disabled={!canRun}>
                  {loading ? 'Running…' : 'Analyze'}
                </Button>
              </div>
            </div>

            {result && (
              <div className="mt-4 border-t border-line pt-4">
                <TokenStrip tokens={result.tokens} />
              </div>
            )}
          </section>

          {modelsError && (
            <ErrorNote message={`Could not load the model catalog. ${modelsError}`} />
          )}
          {error && <ErrorNote message={error} />}
          {result?.truncated && (
            <Note>
              Prompt was truncated to {result.tokens.length} tokens. Attention payloads grow with the
              square of sequence length — raise the server's MAX_PROMPT_TOKENS to see more.
            </Note>
          )}

          <nav className="flex gap-1 border-b border-line">
            {TABS.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                onClick={() => setTab(id)}
                className={`-mb-px border-b-2 px-4 py-2.5 font-mono text-[12px] tracking-[0.1em] uppercase transition-colors ${
                  tab === id
                    ? 'border-base-accent text-ink'
                    : 'border-transparent text-ink-faint hover:text-ink-muted'
                }`}
              >
                {label}
              </button>
            ))}
          </nav>

          {tab === 'attention' && <AttentionHeatmap result={result} />}
          {tab === 'activations' && <ActivationChart result={result} />}
          {tab === 'compare' && (
            <CompareView models={models} prompt={prompt} onPromptChange={setPrompt} />
          )}
        </main>

        <footer className="mt-14 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-6 font-mono text-[10px] text-ink-faint">
          <span>CPU inference · transformers · no persistence, nothing leaves your machine</span>
          <span>api {API_BASE_URL}</span>
        </footer>
      </div>
    </div>
  )
}

function Header({ modelCount }: { modelCount: number }) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-4 py-10">
      <div className="flex items-center gap-3.5">
        <div className="grid h-9 w-9 shrink-0 grid-cols-3 gap-[2px] rounded-lg border border-line-bright bg-raised p-[5px]">
          {[1, 0.45, 0.15, 0.45, 1, 0.3, 0.15, 0.6, 1].map((opacity, index) => (
            <span
              key={index}
              className="rounded-[1px] bg-base-accent"
              style={{ opacity }}
            />
          ))}
        </div>
        <div>
          <h1 className="font-mono text-[15px] tracking-[0.1em] text-ink uppercase">
            Model Internals
          </h1>
          <p className="mt-0.5 text-[12px] text-ink-faint">
            Attention maps and activation profiles for GPT-2 family checkpoints
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 rounded-full border border-line-bright bg-raised px-3.5 py-1.5 font-mono text-[10px] tracking-[0.1em] text-ink-muted uppercase">
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            modelCount > 0 ? 'bg-base-accent' : 'bg-danger'
          }`}
        />
        {modelCount > 0 ? `${modelCount} checkpoints` : 'backend offline'}
      </div>
    </header>
  )
}

/** The exact token sequence the model saw — the ground truth for both charts. */
function TokenStrip({ tokens }: { tokens: string[] }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="mr-1 font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        {tokens.length} tokens
      </span>
      {tokens.map((token, index) => (
        <span
          key={`${index}-${token}`}
          className="rounded border border-line-bright bg-raised px-1.5 py-0.5 font-mono text-[11px] text-ink-muted"
          title={`index ${index}`}
        >
          {token}
        </span>
      ))}
    </div>
  )
}
