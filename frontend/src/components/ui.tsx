/**
 * Small presentational primitives shared by the panels.
 *
 * Purely visual — no state, no data fetching. Keeping them here stops the
 * feature components from re-declaring the same border/label styling.
 */

import type { ReactNode, SelectHTMLAttributes } from 'react'

export function Panel({
  title,
  subtitle,
  controls,
  children,
}: {
  title: string
  subtitle?: string
  controls?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="rounded-xl border border-line bg-panel/80 shadow-[0_1px_0_0_rgba(255,255,255,0.03)_inset]">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-line px-5 py-4">
        <div>
          <h2 className="font-mono text-[13px] tracking-[0.12em] text-ink uppercase">{title}</h2>
          {subtitle && <p className="mt-1 max-w-2xl text-[12px] text-ink-faint">{subtitle}</p>}
        </div>
        {controls && <div className="flex flex-wrap items-end gap-3">{controls}</div>}
      </header>
      <div className="p-5">{children}</div>
    </section>
  )
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        {label}
      </span>
      {children}
    </label>
  )
}

type SelectProps = SelectHTMLAttributes<HTMLSelectElement> & { children: ReactNode }

export function Select({ className = '', children, ...props }: SelectProps) {
  return (
    <select
      {...props}
      className={`min-w-[9rem] rounded-lg border border-line-bright bg-raised px-3 py-2 font-mono text-[12px] text-ink outline-none transition-colors hover:border-base-accent/50 focus:border-base-accent/80 focus:ring-1 focus:ring-base-accent/30 disabled:cursor-not-allowed disabled:opacity-40 ${className}`}
    >
      {children}
    </select>
  )
}

export function Button({
  children,
  onClick,
  disabled,
  variant = 'primary',
  type = 'button',
}: {
  children: ReactNode
  onClick?: () => void
  disabled?: boolean
  variant?: 'primary' | 'ghost'
  type?: 'button' | 'submit'
}) {
  const styles =
    variant === 'primary'
      ? 'bg-base-accent text-ground hover:bg-sky-300 disabled:bg-line-bright disabled:text-ink-faint'
      : 'border border-line-bright bg-raised text-ink-muted hover:border-base-accent/50 hover:text-ink'

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-4 py-2 font-mono text-[12px] tracking-[0.08em] uppercase transition-colors disabled:cursor-not-allowed ${styles}`}
    >
      {children}
    </button>
  )
}

/** Inline error surface. Shows the backend's own `detail` text verbatim. */
export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-danger/30 bg-danger/5 px-4 py-3 font-mono text-[12px] leading-relaxed text-danger">
      {message}
    </div>
  )
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-tuned-accent/25 bg-tuned-accent/5 px-4 py-2.5 font-mono text-[11px] leading-relaxed text-tuned-accent/90">
      {children}
    </div>
  )
}

/** Neutral filler shown before the first run and while a request is in flight. */
export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-[340px] items-center justify-center rounded-lg border border-dashed border-line text-center font-mono text-[12px] text-ink-faint">
      <p className="max-w-sm px-6 leading-relaxed">{children}</p>
    </div>
  )
}

export function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="font-mono text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        {label}
      </span>
      <span className="tabular font-mono text-[13px]" style={accent ? { color: accent } : undefined}>
        {value}
      </span>
    </div>
  )
}
