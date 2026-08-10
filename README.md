# Model Internals Viz

A local tool for looking inside GPT-2 family checkpoints. Type a prompt, pick a
model, and see the actual tokens the model consumed, the attention weights for
any layer/head pair, and how activation magnitude builds through the residual
stream — then diff a fine-tune against its base to see where fine-tuning
actually moved things.

Everything runs on your machine. No auth, no database, no persistence: one
FastAPI process holding models in memory, one Vite dev server, CORS between
them.

```
├── backend/          FastAPI + transformers, CPU inference
│   └── app/          main.py · models.py · inference.py · schemas.py
├── frontend/         React + TypeScript + Tailwind + nivo
│   └── src/          api/ · components/ · App.tsx · main.tsx
└── README.md
```

## Requirements

- Python 3.10+ (3.12 recommended — `torch` wheels for 3.9 lag behind)
- Node 20+
- ~2 GB of disk for model weights, and a network connection for the first run

## Install from a fresh clone

Two terminals. Backend first — the frontend shows "backend offline" until
`GET /models` answers.

**Terminal 1 — backend**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — frontend**

```bash
cd frontend
npm install
npm run dev
```

Then open <http://localhost:5173>. The API docs are at
<http://127.0.0.1:8000/docs>.

The first `/analyze` call downloads weights from Hugging Face (~550 MB for
GPT-2 small) and takes 10–15 seconds. Every call after that is served from an
in-memory cache and takes about a second.

## What each view shows

**Attention** — a token × token heatmap for one (layer, head) pair, with real
token labels on both axes. Rows are query tokens, columns are keys. The upper
triangle is empty because GPT-2 is causal and cannot attend forward. `␣` marks
a leading space in the BPE vocabulary and `⏎` a newline.

**Activations** — mean absolute activation at each hidden state, x = layer.
Index 0 is the embedding output. The last point is measured after GPT-2's final
layer norm, so it drops rather than continuing the ramp — architecture, not a
bug.

**Compare** — two checkpoints on one prompt, drawn as three lines: base,
fine-tuned, and the per-layer delta. Comparing `gpt2` against `lvwerra/gpt2-imdb`
on a movie-review-shaped prompt is the intended demo.

## Models

All four are GPT-2 architecture, verified against the Hugging Face Hub API
(`model_type: gpt2`, `GPT2LMHeadModel`) before being hardcoded:

| id | Hub repo | Blocks |
| --- | --- | --- |
| `gpt2` | `openai-community/gpt2` | 12 |
| `gpt2-imdb` | `lvwerra/gpt2-imdb` | 12 |
| `dialogpt-small` | `microsoft/DialoGPT-small` | 12 |
| `gpt2-medium` | `openai-community/gpt2-medium` | 24 |

Comparing models of different depth (`gpt2` vs `gpt2-medium`) works — the delta
covers the shared prefix and the response carries a note saying so.

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/models` | — | `ModelInfo[]` |
| `POST` | `/analyze` | `{model_id, prompt}` | `{tokens, attentions, hidden_state_magnitudes, …}` |
| `POST` | `/compare` | `{base_model_id, finetuned_model_id, prompt}` | both profiles + per-layer `delta` |

`attentions` is nested layer → head → seq × seq.

## How the code is organised

Two rules, enforced by hand:

1. **No inference logic in route handlers.** `app/main.py` validates, calls into
   `app/inference.py`, and maps domain errors to status codes. `inference.py`
   never imports FastAPI.
2. **No API logic in React components.** `src/api/client.ts` is the only module
   that calls `fetch`. Components receive data as props or call
   `analyze`/`compare`/`fetchModels`.

Response shapes are defined once in `backend/app/schemas.py` and mirrored by
hand in `frontend/src/api/types.ts`. No codegen at this stage — change both in
the same commit.

## Configuration

| Env var | Side | Default | Purpose |
| --- | --- | --- | --- |
| `MAX_PROMPT_TOKENS` | backend | `32` | Tokens per forward pass. Attention payloads scale as `layers × heads × seq²`, so 32 tokens is already ~1.8 MB of JSON. |
| `CORS_ORIGINS` | backend | ports 5173 and 3000 on both `localhost` and `127.0.0.1` | Comma-separated allowlist. |
| `VITE_API_BASE_URL` | frontend | `http://127.0.0.1:8000` | Backend origin. |

## Troubleshooting

**"Could not reach the backend"** — the browser cannot tell a dead server from a
blocked CORS preflight, so this one message covers both. Check uvicorn is up
(`curl http://127.0.0.1:8000/models`), then check the port: `vite.config.ts` sets
`strictPort: true` precisely so a busy 5173 fails loudly instead of quietly
moving to 5174, which the backend's allowlist would then reject.

**Port 8000 already taken** (macOS, or another local service) — run the backend
on another port and tell the frontend:

```bash
uvicorn app.main:app --reload --port 8010
```

then create `frontend/.env.local` with `VITE_API_BASE_URL=http://127.0.0.1:8010`
and restart Vite.

**First request hangs** — it is downloading weights. Watch the uvicorn log.
Weights land in `~/.cache/huggingface`; delete that directory to force a
re-download if a fetch was interrupted and left a partial file.

## Not in this version

No auth, no database, no persistence. No sparse autoencoders or feature
extraction. No deployment or CI config — this is local dev only, two processes
on two ports.
