/**
 * Mean absolute activation per layer for a single model.
 *
 * x = hidden-state index (0 is the embedding output, i>0 is the output of
 * transformer block i), y = mean |activation| across tokens and dimensions.
 */

import { useMemo } from 'react'
import { ResponsiveLine } from '@nivo/line'

import type { AnalyzeResponse } from '../api/types'
import { chartTheme, PALETTE } from './chartTheme'
import { EmptyState, Panel, Stat } from './ui'

export function ActivationChart({ result }: { result: AnalyzeResponse | null }) {
  const series = useMemo(() => {
    if (!result) return []
    return [
      {
        id: result.model_id,
        data: result.hidden_state_magnitudes.map((y, x) => ({ x, y })),
      },
    ]
  }, [result])

  const magnitudes = result?.hidden_state_magnitudes ?? []
  const peakLayer = magnitudes.indexOf(Math.max(...(magnitudes.length ? magnitudes : [0])))

  return (
    <Panel
      title="Activation profile"
      subtitle="Residual-stream magnitude as the prompt moves through the network. GPT-2 characteristically ramps monotonically, then drops at the very end."
    >
      {!result ? (
        <EmptyState>Run a prompt to plot per-layer activation magnitude.</EmptyState>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-end gap-x-8 gap-y-3 border-b border-line pb-4">
            <Stat label="Model" value={result.model_id} accent={PALETTE.base} />
            <Stat label="Hidden states" value={String(magnitudes.length)} />
            <Stat label="Peak at" value={`layer ${peakLayer}`} />
            <Stat label="Peak value" value={(magnitudes[peakLayer] ?? 0).toFixed(3)} />
          </div>

          <div className="h-[380px]">
            <ResponsiveLine
              data={series}
              theme={chartTheme}
              colors={[PALETTE.base]}
              margin={{ top: 20, right: 28, bottom: 56, left: 68 }}
              xScale={{ type: 'linear', min: 0, max: 'auto' }}
              yScale={{ type: 'linear', min: 0, max: 'auto' }}
              curve="monotoneX"
              lineWidth={2}
              // Solid dots in the serie colour. Hollow points read as breaks
              // in the line at this density.
              pointSize={5}
              pointColor={{ from: 'serieColor' }}
              pointBorderWidth={0}
              enableArea
              areaOpacity={0.08}
              enableGridX={false}
              axisBottom={{
                tickSize: 0,
                tickPadding: 10,
                legend: 'hidden state index (0 = embeddings)',
                legendOffset: 42,
                legendPosition: 'middle',
                tickValues: magnitudes.map((_, i) => i).filter((i) => i % 2 === 0),
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
                  <div className="text-ink-faint">
                    layer <span className="text-ink">{String(point.data.x)}</span>
                  </div>
                  <div className="tabular text-base-accent">
                    {Number(point.data.y).toFixed(4)}
                  </div>
                </div>
              )}
            />
          </div>

          <p className="font-mono text-[10px] leading-relaxed text-ink-faint">
            Index 0 is the embedding output. The final point is measured after GPT-2&rsquo;s last
            layer norm (<span className="text-ink-muted">ln_f</span>), which rescales the stream —
            that drop is the architecture, not an artefact.
          </p>
        </div>
      )}
    </Panel>
  )
}

export default ActivationChart
