# Backend — FastAPI

CPU-only inference over GPT-2 family checkpoints. Exposes the tokens, attention
weights, and per-layer activation magnitudes for a single forward pass.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Interactive docs: <http://127.0.0.1:8000/docs>

## Layout

| File | Responsibility |
| --- | --- |
| `app/main.py` | FastAPI app, CORS, four routes. Validates and delegates — no model code. |
| `app/inference.py` | Everything that touches `torch`/`transformers`: loading, caching, forward pass, extraction. |
| `app/models.py` | The hardcoded catalog and id lookup. |
| `app/schemas.py` | Pydantic request/response models. The HTTP contract lives here. |
| `app/insights.py` | Turns a finished layer trace into plain-English findings. Pure functions, no torch. |

The one architectural rule: routes never contain inference logic, and
`inference.py` never imports FastAPI.

## Endpoints

### `GET /models`

Returns the catalog as `ModelInfo[]`:

```json
[{ "id": "gpt2", "display_name": "GPT-2 (base)", "hf_model_id": "gpt2", "role": "base", "description": "…" }]
```

### `POST /analyze`

```json
{ "model_id": "gpt2", "prompt": "The movie was absolutely" }
```

Returns `tokens`, `attentions` (layer → head → seq × seq), and
`hidden_state_magnitudes` (mean absolute activation per hidden state), plus
`num_layers`, `num_heads`, and `truncated`.

### `POST /lens`

```json
{ "model_id": "gpt2", "prompt": "The capital of France is", "top_k": 5 }
```

The logit lens: projects every layer's residual stream through the model's own
final norm + unembedding to get a next-token distribution per layer. Returns
`layers` (top-k, entropy in bits, and the probability that layer assigns the
final answer), `trajectories` (gap-free probability lines built from the union
of all layers' top-k, so a token that leads at L10 and drops out by L12 still
has a complete series), `final_prediction`, and `narration`.

Two correctness details worth preserving if you edit `logit_lens`:

- The **last** hidden state is already post-`ln_f`. Norming it again silently
  corrupts the final distribution — it must pass through untouched.
- The **embedding** row is an artifact. GPT-2 ties input and output embeddings,
  so the lens there recovers the input token at ~100%. It is not a prediction,
  and `insights.py` excludes it from every baseline it computes.

### `POST /compare`

```json
{ "base_model_id": "gpt2", "finetuned_model_id": "gpt2-imdb", "prompt": "The movie was absolutely" }
```

Runs both models on the same prompt and returns each one's
`hidden_state_magnitudes` plus a per-layer `delta` (`finetuned − base`).

## Model catalog

Every `hf_model_id` was checked against the Hugging Face Hub API and reports
`model_type: gpt2` / `GPT2LMHeadModel`, so per-layer comparisons are between
like architectures.

| id | Hub repo | Blocks | Note |
| --- | --- | --- | --- |
| `gpt2` | `gpt2` → `openai-community/gpt2` | 12 | Reference anchor. |
| `gpt2-imdb` | `lvwerra/gpt2-imdb` | 12 | GPT-2 small fine-tuned on IMDB reviews. |
| `dialogpt-small` | `microsoft/DialoGPT-small` | 12 | GPT-2 small fine-tuned on Reddit dialogue. |
| `gpt2-medium` | `gpt2-medium` → `openai-community/gpt2-medium` | 24 | Deeper; comparisons cover the shared prefix. |

Weights download on first use into the standard Hugging Face cache
(`~/.cache/huggingface`) and are held in an in-process dict afterwards, so the
second request for a model is fast.

## Notable details

- **Device is explicitly CPU.** MPS is available on Apple Silicon but has
  returned subtly wrong values for some attention ops; this tool exists to show
  real numbers.
- **Prompts are truncated to 32 tokens** by default. Attention payloads scale
  as `layers × heads × seq²` — GPT-2 small at 32 tokens is already ~147k floats.
  Raise it with `MAX_PROMPT_TOKENS=64` if you want longer prompts and can accept
  a heavier response.
- **`hidden_state_magnitudes` has `num_layers + 1` entries.** Index 0 is the
  embedding output; index *i* is the output of transformer block *i*.
- **GPT-2 has no pad token**, so `tokenizer.pad_token` is set to `eos_token` at
  load time. Nothing here batches, but the tokenizer warns without it.
- **Routes are sync `def`**, so Starlette runs them in a worker thread rather
  than blocking the event loop during a CPU forward pass.

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `MAX_PROMPT_TOKENS` | `32` | Ceiling on tokens per forward pass. |
| `CORS_ORIGINS` | `localhost`/`127.0.0.1` on ports 5173 and 3000 | Comma-separated allowlist. |
