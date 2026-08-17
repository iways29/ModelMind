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
 */
export interface AnalyzeResponse {
  model_id: string
  tokens: string[]
  attentions: number[][][][]
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
/**
 * The residual stream is a running sum — every part of the network adds to it
 * and nothing is overwritten — and the final read-out is linear. So the answer
 * splits into exactly one number per part, in a single forward pass.
 */
export interface Contribution {
  kind: 'embed' | 'attn' | 'mlp' | 'head'
  layer: number | null
  head: number | null
  label: string
  /** Signed push toward the answer, in logits. Negative means it argued against. */
  logits: number
  /** `logits` as a fraction of the margin. Exceeds 1 when other parts push back. */
  share: number
}

export interface AttributionRequest {
  model_id: string
  prompt: string
  contrast_token_id?: number
  position?: number
  max_tokens?: number
}

export interface AttributionResponse {
  model_id: string
  display_name: string
  tokens: string[]
  position: number
  answer: TokenPrediction
  contrast: TokenPrediction
  /** answer logit - contrast logit. Every contribution sums into this. */
  margin: number
  blocks: Contribution[]
  heads: Contribution[]
  unattributed: number
  narration: string[]
  truncated: boolean
  prompt_notice?: string | null
}

/** One part of the recipient reverted to the donor's weights. */
export interface LayerPatch {
  /** 0-based block index; -1 is the embeddings, block count is the read-out. */
  layer: number
  kind: 'embed' | 'block' | 'readout'
  label: string
  answer: TokenPrediction
  donor_answer_prob: number
  /** 0 = no movement toward the donor, 1 = fully the donor's answer. */
  recovery: number | null
  flipped: boolean
}

export interface PatchFocus {
  layer: number
  label: string
  recipient_text: string
  donor_text: string
  patched_text: string
}

export interface PatchRequest {
  recipient_model_id: string
  donor_model_id: string
  prompt: string
  layer?: number
  position?: number
  max_tokens?: number
  max_new_tokens?: number
}

export interface PatchResponse {
  recipient_model_id: string
  recipient_name: string
  donor_model_id: string
  donor_name: string
  tokens: string[]
  position: number
  recipient_answer: TokenPrediction
  donor_answer: TokenPrediction
  agreed: boolean
  layers: LayerPatch[]
  best_layer: number | null
  focus?: PatchFocus | null
  narration: string[]
  truncated: boolean
  prompt_notice?: string | null
}

export interface BehaviorRequest {
  model_id: string
  compare_model_id?: string
  prompts: string[]
  max_new_tokens?: number
}

/** Mirrors `Continuation` — what one model actually wrote after the prompt. */
export interface Continuation {
  model_id: string
  display_name: string
  /** Generated text only, prompt excluded. */
  text: string
  /** True when the continuation falls into a repetition loop. */
  repeats: boolean
}

/** Mirrors `Divergence` — the first token where the two models part ways. */
export interface Divergence {
  index: number
  shared_prefix: string
  /** Token count of `shared_prefix`; the position to open the lens at. */
  prefix_token_count: number
  token: TokenPrediction
  compare_token: TokenPrediction
}

/** Mirrors `PromptBehavior`. */
export interface PromptBehavior {
  prompt: string
  primary: Continuation
  compare: Continuation | null
  identical: boolean
  divergence: Divergence | null
  /** Avg bits the first model assigns the second's text. Higher = drifted further. */
  surprise_bits: number | null
  flags: string[]
}

/** Mirrors `BehaviorResponse`. */
export interface BehaviorResponse {
  model_id: string
  display_name: string
  compare_model_id: string | null
  compare_display_name: string | null
  rows: PromptBehavior[]
  max_new_tokens: number
  narration: string[]
}

/** Mirrors `CompareRequest`. */
