/**
 * Shared nivo theming.
 *
 * nivo defaults to dark-on-light, which is unreadable on this ground colour,
 * so every chart in the app pulls its axis/grid/tooltip styling from here.
 */

import type { PartialTheme } from '@nivo/theming'

export const PALETTE = {
  ground: '#07090d',
  panel: '#0c1015',
  raised: '#11161d',
  line: '#1a212b',
  lineBright: '#27313e',
  ink: '#e8eef5',
  inkMuted: '#8d9bab',
  inkFaint: '#5c6a7a',
  base: '#38bdf8',
  tuned: '#fbbf24',
  delta: '#a78bfa',
} as const

const MONO = "'SF Mono', 'JetBrains Mono', ui-monospace, Menlo, monospace"

export const chartTheme: PartialTheme = {
  text: { fontSize: 11, fontFamily: MONO, fill: PALETTE.inkMuted },
  axis: {
    domain: { line: { stroke: PALETTE.line, strokeWidth: 1 } },
    ticks: {
      line: { stroke: PALETTE.line, strokeWidth: 1 },
      text: { fontSize: 10, fontFamily: MONO, fill: PALETTE.inkFaint },
    },
    legend: {
      text: { fontSize: 11, fontFamily: MONO, fill: PALETTE.inkMuted, letterSpacing: 0.6 },
    },
  },
  grid: { line: { stroke: PALETTE.line, strokeWidth: 1, strokeDasharray: '2 4' } },
  legends: { text: { fontSize: 11, fontFamily: MONO, fill: PALETTE.inkMuted } },
  tooltip: {
    container: {
      background: PALETTE.raised,
      color: PALETTE.ink,
      fontSize: 11,
      fontFamily: MONO,
      border: `1px solid ${PALETTE.lineBright}`,
      borderRadius: 6,
      boxShadow: '0 10px 30px rgb(0 0 0 / 0.55)',
    },
  },
  crosshair: { line: { stroke: PALETTE.lineBright, strokeWidth: 1, strokeDasharray: '3 3' } },
}

/** Stops for the attention ramp: panel black -> deep blue -> cyan -> hot white. */
const ATTENTION_STOPS: Array<[number, number, number]> = [
  [12, 17, 23],
  [15, 46, 74],
  [17, 110, 158],
  [56, 189, 248],
  [214, 245, 255],
]

/**
 * Sequential interpolator for the heatmap.
 *
 * Built by hand rather than pulled from a nivo scheme so the heatmap sits in
 * the same palette as the rest of the UI, and so `t = 0` lands exactly on the
 * panel colour — cells with no attention then read as empty, not as data.
 */
export function attentionInterpolator(t: number): string {
  const clamped = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0))
  const scaled = clamped * (ATTENTION_STOPS.length - 1)
  const lower = Math.min(Math.floor(scaled), ATTENTION_STOPS.length - 2)
  const frac = scaled - lower

  const [r1, g1, b1] = ATTENTION_STOPS[lower]
  const [r2, g2, b2] = ATTENTION_STOPS[lower + 1]
  const mix = (a: number, b: number) => Math.round(a + (b - a) * frac)

  return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`
}
