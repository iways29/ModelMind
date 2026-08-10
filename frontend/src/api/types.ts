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
}
