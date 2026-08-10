/**
 * Base vs fine-tuned activation profiles on a shared prompt, plus their delta.
 *
 * This panel owns its own request because the two model choices are local to
 * it — but the request itself is `compare()` from `api/client`, never a raw
 * fetch.
 */

import { useEffect, useMemo, useState } from 'react'
import { ResponsiveLine } from '@nivo/line'

import { compare } from '../api/client'
import type { CompareResponse, ModelInfo } from '../api/types'
import { ModelSelector } from './ModelSelector'
import { chartTheme, PALETTE } from './chartTheme'
import { Button, EmptyState, ErrorNote, Note, Panel, Stat } from './ui'

export function CompareView({
  models,
  prompt,
  onPromptChange,
}: {
  models: ModelInfo[]
  /** Shared with the composer at the top of the page — same state, two inputs. */
  prompt: string
  onPromptChange: (prompt: string) => void
}) {
  const [baseId, setBaseId] = useState('')
  const [finetunedId, setFinetunedId] = useState('')
  const [result, setResult] = useState<CompareResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Seed the selectors once the catalog arrives: first 'base' vs first 'finetuned'.
  useEffect(() => {
    if (models.length === 0) return
    setBaseId((current) => current || (models.find((m) => m.role === 'base') ?? models[0]).id)
    setFinetunedId(
      (current) => current || (models.find((m) => m.role === 'finetuned') ?? models[0]).id,
    )
  }, [models])

  async function run() {
    if (!prompt.trim() || !baseId || !finetunedId) return
    setLoading(true)
    setError(null)
    try {
      setResult(await compare({ base_model_id: baseId, finetuned_model_id: finetunedId, prompt }))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const series = useMemo(() => {
    if (!result) return []
    // Serie ids double as legend labels, so they stay short — the display
    // names already carry the "(base)" / "(fine-tuned)" qualifier.
    return [
      {
        id: result.base.display_name,
        data: result.base.hidden_state_magnitudes.map((y, x) => ({ x, y })),
      },
      {
        id: result.finetuned.display_name,
        data: result.finetuned.hidden_state_magnitudes.map((y, x) => ({ x, y })),
      },
      { id: 'delta', data: result.delta.map((y, x) => ({ x, y })) },
    ]
  }, [result])

  const largestShift = useMemo(() => {
    if (!result || result.delta.length === 0) return null
    let index = 0
    for (let i = 1; i < result.delta.length; i += 1) {
      if (Math.abs(result.delta[i]) > Math.abs(result.delta[index])) index = i
    }
    return { index, value: result.delta[index] }
  }, [result])

  const sameModel = baseId === finetunedId
  const canRun = Boolean(prompt.trim()) && Boolean(baseId) && Boolean(finetunedId) && !loading

  return (
    <Panel
      title="Compare"
      subtitle="Both checkpoints see the identical prompt. The delta line is where fine-tuning actually moved the residual stream."
      controls={
        <>
          <ModelSelector label="Base" models={models} value={baseId} onChange={setBaseId} />
          <ModelSelector
            label="Fine-tuned"
            models={models}
            value={finetunedId}
            onChange={setFinetunedId}
          />
          <Button onClick={run} disabled={!canRun}>
            {loading ? 'Running…' : 'Compare'}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <label className="flex flex-col gap-1.5">
          <span className="font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
            Shared prompt
          </span>
          <input
            value={prompt}
            onChange={(event) => onPromptChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && canRun) run()
            }}
            placeholder="The movie was absolutely"
            className="w-full rounded-lg border border-line-bright bg-raised px-3.5 py-2.5 font-mono text-[13px] text-ink placeholder:text-ink-faint/60 focus:border-base-accent/70 focus:ring-1 focus:ring-base-accent/25 focus:outline-none"
          />
        </label>

        {sameModel && (
          <Note>Base and fine-tuned are the same checkpoint — the delta will be flat zero.</Note>
        )}
        {error && <ErrorNote message={error} />}
        {result?.note && <Note>{result.note}</Note>}

        {!result ? (
          <EmptyState>
            {loading
              ? 'Loading both checkpoints. The first run downloads weights from Hugging Face.'
              : 'Pick two checkpoints and hit Compare.'}
          </EmptyState>
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap items-end gap-x-8 gap-y-3 border-b border-line pb-4">
              <Stat label="Layers compared" value={String(result.layers_compared)} />
              <Stat label="Tokens" value={String(result.tokens.length)} />
              {largestShift && (
                <>
                  <Stat label="Largest shift at" value={`layer ${largestShift.index}`} />
                  <Stat
                    label="Shift size"
                    value={`${largestShift.value > 0 ? '+' : ''}${largestShift.value.toFixed(4)}`}
                    accent={PALETTE.delta}
                  />
                </>
              )}
            </div>

            <div className="h-[420px]">
              <ResponsiveLine
                data={series}
                theme={chartTheme}
                colors={[PALETTE.base, PALETTE.tuned, PALETTE.delta]}
                margin={{ top: 20, right: 28, bottom: 88, left: 68 }}
                xScale={{ type: 'linear', min: 0, max: 'auto' }}
                yScale={{ type: 'linear', min: 'auto', max: 'auto' }}
                curve="monotoneX"
                lineWidth={2}
                pointSize={5}
                pointColor={{ from: 'serieColor' }}
                pointBorderWidth={0}
                enableGridX={false}
                // Reference line so the sign of the delta reads at a glance.
                markers={[
                  {
                    axis: 'y',
                    value: 0,
                    lineStyle: { stroke: PALETTE.lineBright, strokeWidth: 1, strokeDasharray: '4 4' },
                  },
                ]}
                axisBottom={{
                  tickSize: 0,
                  tickPadding: 10,
                  legend: 'hidden state index (0 = embeddings)',
                  legendOffset: 42,
                  legendPosition: 'middle',
                }}
                axisLeft={{
                  tickSize: 0,
                  tickPadding: 10,
                  legend: 'mean |activation|',
                  legendOffset: -54,
                  legendPosition: 'middle',
                }}
                useMesh
                enableSlices={false}
                tooltip={({ point }) => (
                  <div className="px-2.5 py-2 leading-relaxed">
                    <div className="flex items-center gap-1.5">
                      <span
                        className="inline-block h-1.5 w-1.5 rounded-full"
                        style={{ background: point.seriesColor }}
                      />
                      <span className="text-ink">{point.seriesId}</span>
                    </div>
                    <div className="mt-1 text-ink-faint">
                      layer <span className="text-ink">{String(point.data.x)}</span>
                    </div>
                    <div className="tabular text-ink">{Number(point.data.y).toFixed(4)}</div>
                  </div>
                )}
                legends={[
                  {
                    anchor: 'bottom',
                    direction: 'row',
                    translateY: 76,
                    itemWidth: 190,
                    itemHeight: 18,
                    itemsSpacing: 12,
                    symbolSize: 8,
                    symbolShape: 'circle',
                    itemTextColor: PALETTE.inkMuted,
                  },
                ]}
              />
            </div>

            <p className="font-mono text-[10px] leading-relaxed text-ink-faint">
              Base and fine-tuned lines share an axis, so the delta sits near zero by construction —
              its shape, not its height, is the signal.
            </p>
          </div>
        )}
      </div>
    </Panel>
  )
}

export default CompareView
