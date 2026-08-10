/**
 * Racing probability lines — every candidate token's odds at every layer.
 *
 * Hand-rolled SVG rather than nivo: this chart needs per-line glow, labels
 * anchored at each line's peak rather than its end, and a winner that reads
 * differently from the field. That's fighting a charting library, not using one.
 *
 * The story this exists to tell: a token can lead mid-network and lose. GPT-2
 * puts 18% on "Paris" at layer 10 and 3% by layer 12.
 */

import { useMemo } from 'react'

import type { TokenTrajectory } from '../api/types'
import { PALETTE } from './chartTheme'

const WIDTH = 720
const HEIGHT = 300
const PAD = { top: 18, right: 116, bottom: 34, left: 46 }

export function TrajectoryRibbon({
  trajectories,
  layerLabels,
  finalTokenId,
  highlightId,
  onHighlight,
  maxLines = 7,
}: {
  trajectories: TokenTrajectory[]
  layerLabels: string[]
  finalTokenId: number
  highlightId: number | null
  onHighlight: (tokenId: number | null) => void
  maxLines?: number
}) {
  const lines = useMemo(() => {
    // The embedding row is an artifact (it just echoes the input token), so a
    // candidate that only peaks there is noise and shouldn't take a slot.
    const meaningful = trajectories.filter((t) => t.peak_layer > 0 || t.token_id === finalTokenId)
    const pool = meaningful.length > 0 ? meaningful : trajectories
    return pool.slice(0, maxLines)
  }, [trajectories, finalTokenId, maxLines])

  const innerW = WIDTH - PAD.left - PAD.right
  const innerH = HEIGHT - PAD.top - PAD.bottom

  // Scale to the tallest line actually drawn, not to 1.0 — most of these
  // distributions never exceed 0.3 and a 0-1 axis would flatten everything.
  const yMax = useMemo(
    () => Math.max(0.05, ...lines.map((l) => l.peak_prob)),
    [lines],
  )

  const nLayers = layerLabels.length
  const x = (layer: number) => PAD.left + (nLayers <= 1 ? 0 : (layer / (nLayers - 1)) * innerW)
  const y = (prob: number) => PAD.top + innerH - (prob / yMax) * innerH

  // Most candidates finish near zero, so their end-of-line labels would all
  // land on the same pixel row. Spread them just enough to stay legible.
  const labelY = useMemo(() => {
    const wanted = lines.map((line, i) => ({
      i,
      y: PAD.top + innerH - (line.probs[line.probs.length - 1] / yMax) * innerH,
    }))
    wanted.sort((a, b) => a.y - b.y)

    const MIN_GAP = 12
    for (let k = 1; k < wanted.length; k += 1) {
      if (wanted[k].y - wanted[k - 1].y < MIN_GAP) wanted[k].y = wanted[k - 1].y + MIN_GAP
    }
    // If pushing down overflowed the plot, slide the whole stack back up.
    const overflow = wanted.length ? wanted[wanted.length - 1].y - (PAD.top + innerH) : 0
    if (overflow > 0) wanted.forEach((w) => (w.y -= overflow))

    const out: number[] = new Array(lines.length).fill(0)
    wanted.forEach((w) => (out[w.i] = w.y))
    return out
  }, [lines, innerH, yMax])

  if (lines.length === 0) return null

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full"
        role="img"
        aria-label="Probability of each candidate token at every layer"
      >
        <defs>
          <filter id="ribbon-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3.2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Horizontal reference grid */}
        {[0, 0.5, 1].map((frac) => {
          const gy = PAD.top + innerH - frac * innerH
          return (
            <g key={frac}>
              <line
                x1={PAD.left}
                x2={PAD.left + innerW}
                y1={gy}
                y2={gy}
                stroke={PALETTE.line}
                strokeDasharray="2 5"
              />
              <text
                x={PAD.left - 9}
                y={gy + 3.5}
                textAnchor="end"
                fontSize="9"
                fill={PALETTE.inkFaint}
                fontFamily="var(--font-mono)"
              >
                {Math.round(frac * yMax * 100)}%
              </text>
            </g>
          )
        })}

        {/* Layer ticks — thinned so labels never collide */}
        {layerLabels.map((label, i) => {
          const step = nLayers > 14 ? 3 : 2
          if (i % step !== 0 && i !== nLayers - 1) return null
          return (
            <text
              key={label}
              x={x(i)}
              y={HEIGHT - 12}
              textAnchor="middle"
              fontSize="9"
              fill={PALETTE.inkFaint}
              fontFamily="var(--font-mono)"
            >
              {label}
            </text>
          )
        })}

        {lines.map((line, index) => {
          const isWinner = line.token_id === finalTokenId
          const dimmed = highlightId !== null && highlightId !== line.token_id
          const color = isWinner ? PALETTE.base : RIVAL_COLORS[index % RIVAL_COLORS.length]
          const d = smoothPath(line.probs.map((p, i) => [x(i), y(p)]))

          return (
            <g
              key={line.token_id}
              onMouseEnter={() => onHighlight(line.token_id)}
              onMouseLeave={() => onHighlight(null)}
              style={{ cursor: 'pointer' }}
              opacity={dimmed ? 0.18 : 1}
            >
              <path
                d={d}
                fill="none"
                stroke={color}
                strokeWidth={isWinner ? 2.4 : 1.4}
                strokeLinecap="round"
                strokeLinejoin="round"
                filter={isWinner || highlightId === line.token_id ? 'url(#ribbon-glow)' : undefined}
                className="animate-trace"
                style={
                  {
                    '--trace-length': 2200,
                    animationDelay: `${index * 90}ms`,
                  } as React.CSSProperties
                }
              />
              {/* Marker + label at the PEAK, which is where the story is. */}
              <circle cx={x(line.peak_layer)} cy={y(line.peak_prob)} r={isWinner ? 3.4 : 2.4} fill={color} />
              {/* Leader line, since a de-collided label no longer sits on its trace. */}
              <line
                x1={x(nLayers - 1) + 2}
                y1={y(line.probs[line.probs.length - 1])}
                x2={x(nLayers - 1) + 8}
                y2={labelY[index]}
                stroke={color}
                strokeWidth="0.75"
                opacity="0.5"
              />
              <text
                x={x(nLayers - 1) + 11}
                y={labelY[index] + 3.5}
                fontSize="10"
                fill={dimmed ? PALETTE.inkFaint : color}
                fontFamily="var(--font-mono)"
              >
                {line.token.replace('␣', '')}
              </text>
            </g>
          )
        })}
      </svg>

      <figcaption className="mt-2 font-mono text-[10px] leading-relaxed text-ink-faint">
        Dots mark each token&rsquo;s peak layer. A line that rises and falls is a candidate the
        model considered and then talked itself out of.
      </figcaption>
    </figure>
  )
}

/** Rivals get warm/cool alternation so adjacent lines stay tellable apart. */
const RIVAL_COLORS = [
  PALETTE.tuned,
  PALETTE.delta,
  '#5eead4',
  '#f472b6',
  '#a3e635',
  '#fb923c',
]

/**
 * Catmull-Rom through the points, converted to cubic béziers.
 *
 * Straight polylines make these traces look like sawtooth noise; a smooth
 * curve reads as a trajectory. Tension is kept low so the curve never
 * overshoots into implying a probability the model never assigned.
 */
function smoothPath(points: Array<[number, number]>): string {
  if (points.length === 0) return ''
  if (points.length < 3) return `M ${points.map(([px, py]) => `${px},${py}`).join(' L ')}`

  const parts = [`M ${points[0][0]},${points[0][1]}`]
  for (let i = 0; i < points.length - 1; i += 1) {
    const p0 = points[i - 1] ?? points[i]
    const p1 = points[i]
    const p2 = points[i + 1]
    const p3 = points[i + 2] ?? p2
    const t = 6 // higher = flatter control arms = less overshoot
    parts.push(
      `C ${p1[0] + (p2[0] - p0[0]) / t},${p1[1] + (p2[1] - p0[1]) / t}` +
        ` ${p2[0] - (p3[0] - p1[0]) / t},${p2[1] - (p3[1] - p1[1]) / t}` +
        ` ${p2[0]},${p2[1]}`,
    )
  }
  return parts.join(' ')
}

export default TrajectoryRibbon
