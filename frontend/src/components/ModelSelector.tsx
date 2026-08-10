/**
 * Dropdown over the catalog returned by `GET /models`.
 *
 * The fetch itself lives in `api/client.ts` and happens in `App`; this
 * component only renders the options it is handed.
 */

import { Field, Select } from './ui'
import type { ModelInfo } from '../api/types'

export function ModelSelector({
  label,
  models,
  value,
  onChange,
  disabled = false,
  /** Restrict the options to 'base' or 'finetuned' entries. */
  role,
}: {
  label: string
  models: ModelInfo[]
  value: string
  onChange: (modelId: string) => void
  disabled?: boolean
  role?: string
}) {
  const options = role ? models.filter((m) => m.role === role) : models
  const selected = models.find((m) => m.id === value)

  return (
    <div className="flex flex-col gap-1.5">
      <Field label={label}>
        <Select
          value={value}
          disabled={disabled || options.length === 0}
          onChange={(event) => onChange(event.target.value)}
          className="min-w-[15rem]"
        >
          {options.length === 0 && <option value="">No models available</option>}
          {options.map((model) => (
            <option key={model.id} value={model.id}>
              {model.display_name}
            </option>
          ))}
        </Select>
      </Field>
      {selected && (
        <p className="max-w-[15rem] font-mono text-[10px] leading-relaxed text-ink-faint">
          {selected.hf_model_id}
        </p>
      )}
    </div>
  )
}

export default ModelSelector
