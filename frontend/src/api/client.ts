/**
 * The only module in the app that knows the backend exists.
 *
 * Components import the three functions at the bottom and never call `fetch`
 * themselves.
 */

import type {
  AblateRequest,
  AblateResponse,
  AnalyzeRequest,
  AnalyzeResponse,
  AttributionRequest,
  AttributionResponse,
  BehaviorRequest,
  BehaviorResponse,
  CompareRequest,
  CompareResponse,
  LensRequest,
  LensResponse,
  ModelInfo,
} from './types'

/**
 * 127.0.0.1 rather than localhost: on macOS `localhost` can resolve to ::1
 * first, and uvicorn's default bind is IPv4-only, which shows up as a
 * connection refused that looks like a CORS error.
 */
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

/** An error carrying the backend's own `detail` message, so the UI can show it. */
export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    // fetch only rejects on network-level failure — server down, DNS, or a
    // CORS preflight the browser refused. All three look identical from here.
    throw new ApiError(
      `Could not reach the backend at ${API_BASE_URL}. Is uvicorn running, and does its CORS allowlist include this origin (${window.location.origin})?`,
      0,
    )
  }

  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }

  return (await response.json()) as T
}

/** FastAPI returns `{detail: string}` for HTTPException and `{detail: [...]}` for validation errors. */
async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json()
    const detail = body?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail.map((d: { msg?: string }) => d?.msg ?? JSON.stringify(d)).join('; ')
    }
  } catch {
    // Non-JSON error body (a proxy page, a stack trace) — fall through.
  }
  return `${response.status} ${response.statusText}`
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: 'POST', body: JSON.stringify(body) })
}

/** GET /models — the hardcoded catalog that populates the selectors. */
export function fetchModels(): Promise<ModelInfo[]> {
  return request<ModelInfo[]>('/models')
}

/** POST /analyze — tokens, attention, and per-layer activations for one model. */
export function analyze(body: AnalyzeRequest): Promise<AnalyzeResponse> {
  return post<AnalyzeResponse>('/analyze', body)
}

/** POST /lens — the model's predicted next token decoded at every layer. */
export function lens(body: LensRequest): Promise<LensResponse> {
  return post<LensResponse>('/lens', body)
}

/** POST /ablate — the same prompt with components switched off, diffed against intact. */
export function ablate(body: AblateRequest): Promise<AblateResponse> {
  return post<AblateResponse>('/ablate', body)
}

/** POST /attribution — ablate every block, or every head in one block, ranked by effect. */
export function attribution(body: AttributionRequest): Promise<AttributionResponse> {
  return post<AttributionResponse>('/attribution', body)
}

/** POST /behavior — real generated text for many prompts, diffed between two models. */
export function behavior(body: BehaviorRequest): Promise<BehaviorResponse> {
  return post<BehaviorResponse>('/behavior', body)
}

/** POST /compare — two models on one prompt, plus their per-layer delta. */
export function compare(body: CompareRequest): Promise<CompareResponse> {
  return post<CompareResponse>('/compare', body)
}
