/**
 * Wire types for the FastAPI backend.
 *
 * These mirror `backend/app/schemas.py` field for field. There is no codegen at
 * this stage — if you change a Pydantic model, change the matching interface
 * here in the same commit.
 */

/** Mirrors `ModelInfo`. */
export interface ModelInfo {
  id: string
  display_name: string
  hf_model_id: string
  /** 'base' | 'finetuned' — kept as a string so a new role doesn't break parsing. */
  role: string
  description: string
}

/** Mirrors `AnalyzeRequest`. */
export interface AnalyzeRequest {
  model_id: string
  prompt: string
  max_tokens?: number
}

/**
 * Mirrors `AnalyzeResponse`.
 *
 * `attentions` is indexed layer -> head -> query token -> key token. GPT-2 is
 * causal, so every cell above the diagonal is 0.
 *
 * `hidden_state_magnitudes` has `num_layers + 1` entries: index 0 is the
 * embedding output, index i>0 is the output of transformer block i. The last
 * entry is measured after GPT-2's final layer norm, so it dips.
 */
export interface AnalyzeResponse {
  model_id: string
  tokens: string[]
  attentions: number[][][][]
  hidden_state_magnitudes: number[]
  num_layers: number
  num_heads: number
  truncated: boolean
  /** Set when the server adjusted the prompt, e.g. trimming whitespace. */
  prompt_notice: string | null
}

/** Mirrors `TokenPrediction`. */
export interface TokenPrediction {
  token: string
  token_id: number
  prob: number
}

/** Mirrors `LayerLens` — the model's guess read out of one layer. */
export interface LayerLens {
  layer: number
  label: string
  top: TokenPrediction[]
  /** Shannon entropy in bits. High = undecided across many tokens. */
  entropy: number
  /** Probability this layer assigns to the model's FINAL answer. */
  target_prob: number
  changed: boolean
}

/** Mirrors `TokenTrajectory` — one candidate's gap-free probability line. */
export interface TokenTrajectory {
  token: string
  token_id: number
  probs: number[]
  peak_layer: number
  peak_prob: number
  final_prob: number
}

/** Mirrors `LensRequest`. */
export interface LensRequest {
  model_id: string
  prompt: string
  top_k?: number
  position?: number
  max_tokens?: number
}

/** Mirrors `LensResponse`. */
export interface LensResponse {
  model_id: string
  display_name: string
  tokens: string[]
  position: number
  layers: LayerLens[]
  trajectories: TokenTrajectory[]
  final_prediction: TokenPrediction
  narration: string[]
  truncated: boolean
  prompt_notice: string | null
}

/**
 * Mirrors `Ablation`.
 *
 * `layer` is the 0-based block index the backend actually hooks. Display labels
 * shift by one so a block is named after the residual row it writes — block 4
 * shows as "L5", matching the lens. Never render `layer` raw; use the `label`
 * the backend sends back.
 */
export interface Ablation {
  layer: number
  /** Omit or null to ablate the whole block. */
  head?: number | null
}

/** Mirrors `LensTrace` — the layer readout of one forward pass. */
export interface LensTrace {
  layers: LayerLens[]
  trajectories: TokenTrajectory[]
  final_prediction: TokenPrediction
}

/** Mirrors `TokenShift`. */
export interface TokenShift {
  token: string
  token_id: number
  baseline_prob: number
  ablated_prob: number
  /** ablated_prob - baseline_prob. Negative means the ablation suppressed it. */
  delta: number
}

/** Mirrors `AblationEffect`. */
export interface AblationEffect {
  answer_changed: boolean
  baseline_answer: TokenPrediction
  ablated_answer: TokenPrediction
  baseline_answer_prob_after: number
  prob_delta: number
  /** KL(baseline || ablated) in bits — 0 means the component changed nothing. */
  kl_bits: number
  top_shifts: TokenShift[]
}

/** Mirrors `AblateRequest`. */
export interface AblateRequest {
  model_id: string
  prompt: string
  ablations: Ablation[]
  top_k?: number
  position?: number
  max_tokens?: number
}

/** Mirrors `AblateResponse`. */
export interface AblateResponse {
  model_id: string
  display_name: string
  tokens: string[]
  position: number
  ablations: Ablation[]
  ablation_label: string
  baseline: LensTrace
  ablated: LensTrace
  effect: AblationEffect
  narration: string[]
  truncated: boolean
  prompt_notice: string | null
}

/** Mirrors `ComponentEffect` — one component's measured contribution. */
export interface ComponentEffect {
  layer: number
  head: number | null
  label: string
  baseline_answer_prob_after: number
  prob_delta: number
  kl_bits: number
  top_token: string
  top_token_id: number
  answer_changed: boolean
}

/** Mirrors `AttributionRequest`. `layer` is required when scope is 'heads'. */
export interface AttributionRequest {
  model_id: string
  prompt: string
  scope: 'layers' | 'heads'
  layer?: number
  position?: number
  max_tokens?: number
}

/** Mirrors `AttributionResponse`. `components` arrives ranked by kl_bits. */
export interface AttributionResponse {
  model_id: string
  display_name: string
  tokens: string[]
  position: number
  scope: 'layers' | 'heads'
  layer: number | null
  baseline_answer: TokenPrediction
  components: ComponentEffect[]
  runs: number
  narration: string[]
  truncated: boolean
  prompt_notice: string | null
}

/** Mirrors `CompareRequest`. */
export interface CompareRequest {
  base_model_id: string
  finetuned_model_id: string
  prompt: string
  max_tokens?: number
}

/** Mirrors `ModelMagnitudes`. */
export interface ModelMagnitudes {
  model_id: string
  display_name: string
  hidden_state_magnitudes: number[]
}

/** Mirrors `CompareResponse`. */
export interface CompareResponse {
  prompt: string
  tokens: string[]
  base: ModelMagnitudes
  finetuned: ModelMagnitudes
  /** finetuned[i] - base[i], over the layers the two models share. */
  delta: number[]
  layers_compared: number
  note: string | null
  prompt_notice: string | null
}
