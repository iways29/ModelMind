/**
 * Attention weights for one (layer, head) pair, as a token x token heatmap.
 *
 * Rows are query tokens ("the token doing the attending"), columns are key
 * tokens. GPT-2 is causal, so the upper triangle is structurally zero.
 */

import { useMemo, useState } from 'react'
import { ResponsiveHeatMap } from '@nivo/heatmap'

import type { AnalyzeResponse } from '../api/types'
import { attentionInterpolator, chartTheme, PALETTE } from './chartTheme'
import { EmptyState, Field, Panel, Select, Stat } from './ui'

/**
 * One cell. `x` carries the *column index as a string* rather than the token
 * text, because nivo needs unique keys within a row and a prompt can easily
 * repeat a token. The real label is looked up by index at render time.
 */
interface AttentionCell {
  x: string
  y: number
  /** Column index, for tooltips and axis lookups. */
  ki: number
}

type ScaleMode = 'auto' | 'fixed'

export function AttentionHeatmap({ result }: { result: AnalyzeResponse | null }) {
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [scaleMode, setScaleMode] = useState<ScaleMode>('auto')

  // A new result may come from a model with a different depth, so clamp rather
  // than trusting the selector state from the previous run.
  const layerIndex = result ? Math.min(layer, result.attentions.length - 1) : 0
  const headIndex = result ? Math.min(head, result.num_heads - 1) : 0
  const matrix = result?.attentions[layerIndex]?.[headIndex]

  const tokens = result?.tokens ?? []

  const data = useMemo(() => {
    if (!matrix) return []
    return matrix.map((row, qi) => ({
      id: String(qi),
      data: row.map((value, ki) => ({ x: String(ki), y: value, ki })) as AttentionCell[],
    }))
  }, [matrix])

  const maxWeight = useMemo(() => {
    if (!matrix) return 1
    let max = 0
    for (const row of matrix) for (const v of row) if (v > max) max = v
    return max || 1
  }, [matrix])

  const controls = (
    <>
      <Field label="Layer">
        <Select
          value={layerIndex}
          disabled={!result}
          onChange={(e) => setLayer(Number(e.target.value))}
        >
          {Array.from({ length: result?.attentions.length ?? 0 }, (_, i) => (
            <option key={i} value={i}>
              Layer {i}
            </option>
          ))}
        </Select>
      </Field>
      <Field label="Head">
        <Select
          value={headIndex}
          disabled={!result}
          onChange={(e) => setHead(Number(e.target.value))}
        >
          {Array.from({ length: result?.num_heads ?? 0 }, (_, i) => (
            <option key={i} value={i}>
              Head {i}
            </option>
          ))}
        </Select>
      </Field>
      <Field label="Colour scale">
        <Select value={scaleMode} onChange={(e) => setScaleMode(e.target.value as ScaleMode)}>
          <option value="auto">Auto — max in this head</option>
          <option value="fixed">Fixed — 0 to 1</option>
        </Select>
      </Field>
    </>
  )

  return (
    <Panel
      title="Attention"
      subtitle="Row = query token, column = key token. Cell brightness is the softmax weight the query put on that key."
      controls={controls}
    >
      {!result || !matrix ? (
        <EmptyState>Run a prompt to load attention weights.</EmptyState>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-end gap-x-8 gap-y-3 border-b border-line pb-4">
            <Stat label="Tokens" value={String(tokens.length)} />
            <Stat label="Layer / head" value={`${layerIndex} / ${headIndex}`} />
            <Stat
              label="Peak weight"
              value={maxWeight.toFixed(3)}
              accent={PALETTE.base}
            />
            <Stat
              label="Scale ceiling"
              value={scaleMode === 'auto' ? maxWeight.toFixed(3) : '1.000'}
            />
          </div>

          {/* Height scales with token count so cells stay roughly square. */}
          <div style={{ height: Math.max(360, tokens.length * 22 + 130) }}>
            {/* Record<never, never> rather than nivo's Record<string, never> default:
                the latter's index signature collides with the serie's own `id`. */}
            <ResponsiveHeatMap<AttentionCell, Record<never, never>>
              data={data}
              margin={{ top: 96, right: 24, bottom: 24, left: 104 }}
              theme={chartTheme}
              colors={{
                type: 'sequential',
                interpolator: attentionInterpolator,
                minValue: 0,
                maxValue: scaleMode === 'auto' ? maxWeight : 1,
              }}
              emptyColor={PALETTE.panel}
              // Square cells: an attention matrix read as wide rectangles
              // makes the causal triangle much harder to see.
              forceSquare
              borderWidth={1}
              borderColor={PALETTE.ground}
              borderRadius={2}
              enableLabels={false}
              xInnerPadding={0.04}
              yInnerPadding={0.04}
              animate={false}
              hoverTarget="rowColumn"
              inactiveOpacity={0.28}
              axisTop={{
                tickSize: 0,
                tickPadding: 8,
                tickRotation: -55,
                legend: 'key token',
                legendOffset: -78,
                format: (value: string) => tokens[Number(value)] ?? value,
              }}
              axisLeft={{
                tickSize: 0,
                tickPadding: 8,
                legend: 'query token',
                legendOffset: -92,
                format: (value: string) => tokens[Number(value)] ?? value,
              }}
              axisRight={null}
              axisBottom={null}
              tooltip={({ cell }) => (
                <div className="px-2.5 py-2 leading-relaxed">
                  <div className="text-ink-faint">
                    query <span className="text-ink">{tokens[Number(cell.serieId)]}</span>
                  </div>
                  <div className="text-ink-faint">
                    key <span className="text-ink">{tokens[cell.data.ki]}</span>
                  </div>
                  <div className="tabular mt-1 text-base-accent">
                    {(cell.value ?? 0).toFixed(4)}
                  </div>
                </div>
              )}
            />
          </div>

          <p className="font-mono text-[10px] leading-relaxed text-ink-faint">
            {'␣'} marks a leading space and {'⏎'} a newline in the GPT-2 BPE vocabulary. Cells above
            the diagonal are empty because a causal model cannot attend forward.
          </p>
        </div>
      )}
    </Panel>
  )
}

export default AttentionHeatmap
