# Model Internals Viz

A local tool for watching GPT-2 family checkpoints think. Type a prompt and see
the model's answer *form* — layer by layer, with the candidates it considered
and discarded on the way — then switch any block or attention head off and watch
what the answer depended on. Plus attention weights for any layer/head pair,
activation magnitude through the residual stream, and a diff of a fine-tune
against its base.

The thing that makes it worth using over a notebook: it decodes what the model
believes at **every** layer, not just the last one, and says what happened in
plain English.

Everything runs on your machine. No auth, no database, no persistence: one
FastAPI process holding models in memory, one Vite dev server, CORS between
them.

```
├── backend/          FastAPI + transformers, CPU inference
│   └── app/          main.py · models.py · inference.py · insights.py · schemas.py
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

**Behavior** — the entry point, and the only view that doesn't require you to
already know which prompt is interesting. Give it a list of prompts, pick a
second checkpoint, and it generates real continuations from both and shows you
where they part ways.

Decoding is greedy, never sampled. That is the whole reason a diff here means
anything: with sampling, two runs of the *same* model disagree, and a comparison
between two checkpoints would be measuring the dice rather than the weights.

Rows are ranked by how far the second model drifted — measured as the average
bits of surprise the first model assigns to the second's continuation, which is
high exactly when the fine-tune wrote something the base would rarely produce.
Rows are also flagged for repetition loops, which greedy decoding produces
readily and which are the most common way a small model's output goes bad.

The **Off-domain** suite is the one that earns its keep. Run it against
`lvwerra/gpt2-imdb` and the fine-tune answers *"My review of the restaurant:"*
with `"The best thing about this movie is…"` — it drags unrelated prompts back
toward movie reviews. That damage is invisible if you only test on the domain a
model was tuned for.

**Inspect** on any row loads that prompt into the lens, cut at exactly the token
where the two models disagreed, so the readout lands on the decision that
differed. That is the hand-off from *what changed* to *which part of the network
changed it*.

**Watch it think** — the headline view. At every layer, the model's residual
stream is pushed through its own final layer norm and unembedding matrix, which
turns each layer into a readable next-token prediction. You watch the answer
form: GPT-2 on *"The movie was absolutely"* cycles through `clear` → `quite` →
locks onto `amazing` at L5 → peaks at **80% by L9** → then decays to 10% as it
hedges across `fantastic`/`brilliant`/`phenomenal`.

Each row ends in two labelled bars: **confidence** is how much probability that
layer puts on the final answer, **undecided** is entropy in bits. Plain-English
findings are generated alongside, so you don't have to read the numbers to get
the story.

The technique is the *logit lens* (nostalgebraist, 2020). Two honest caveats it
surfaces for you: the `embed` row is an artifact — GPT-2 ties its input and
output embeddings, so the lens there just echoes the input token — and early
layers are unreliable because the residual stream isn't yet aligned with the
output embedding.

Try `The capital of France is`: GPT-2 puts 62% on `France` and 18% on `Paris` at
layer 10, then throws it away by layer 12 to predict `the`. The knowledge is in
there; the last block spends it on grammar.

**Find the cause** — two questions the rest of the app can't answer, both causal.

*What built this answer* splits the prediction into one number per part of the
network, in a single forward pass. The residual stream is a running sum — every
part adds into it and nothing is overwritten — and the read-out is linear, so the
answer's margin over the runner-up decomposes exactly. The panel prints the
reconciliation, because a split that didn't add back up to the real margin would
be a guess dressed as a measurement.

It's measured as a *difference* between two tokens rather than one token's raw
logit: adding a constant to every logit leaves the model's answer unchanged, so a
single logit means nothing on its own. Shares can exceed 100% — that isn't an
error, it's parts cancelling each other out, and the narration says so when it
happens.

Heads are a drill-down on the attention rows. Each head owns a disjoint slice of
the attention output and therefore a disjoint band of the output projection, so
its contribution can be recovered exactly — but only before `c_proj` sums them,
which is the last moment the heads are separable at all.

*What changed vs another model* reverts one part of your model to another
checkpoint's weights, leaves everything else alone, and re-runs. If the behaviour
comes back, that part was carrying it. This is a weight swap rather than an
activation transplant on purpose: transplanting activations replaces the whole
residual stream at that depth, which carries everything the earlier blocks did
too, so it measures the accumulated difference up to that point rather than the
part itself.

The sweep covers the embeddings and the read-out alongside the blocks. Leaving
them out would be a quiet lie — a fine-tune that moved its output embedding shows
nothing anywhere in a block sweep, and "the change isn't in the blocks" reads
identically to "there is no change to find". Swapping every target at once
reproduces the donor exactly, which is the property that makes the sweep
trustworthy.

Against `lvwerra/gpt2-imdb` on *"My review of the restaurant:"*, every one of the
12 blocks recovers ~0% and the **read-out recovers 28%** — the fine-tune changed
how the model words things, not what it knows. **Read** on that row generates the
proof: the fine-tune writes *"The best thing about this movie is…"*, and with the
read-out reverted the same model writes *"The restaurant is a little too cozy,
but it's not too bad. The food is good…"*. The movie obsession is gone.

Two honest limits: an effect is measured *on this prompt*, so the ranking moves
when the prompt does; and reverting parts one at a time cannot see a change
spread thinly across many of them, which is what a broad fine-tune often is.

**Attention** — a token × token heatmap for one (layer, head) pair, with real
token labels on both axes. Rows are query tokens, columns are keys. The upper
triangle is empty because GPT-2 is causal and cannot attend forward. `␣` marks
a leading space in the BPE vocabulary and `⏎` a newline.

## Models

All four are GPT-2 architecture, verified against the Hugging Face Hub API
(`model_type: gpt2`, `GPT2LMHeadModel`) before being hardcoded:

| id | Hub repo | Blocks |
| --- | --- | --- |
| `gpt2` | `openai-community/gpt2` | 12 |
| `gpt2-imdb` | `lvwerra/gpt2-imdb` | 12 |
| `dialogpt-small` | `microsoft/DialoGPT-small` | 12 |
| `gpt2-medium` | `openai-community/gpt2-medium` | 24 |

Reverting weights requires matching depth and width, so `gpt2-medium` can't be
paired with the 12-block checkpoints — the API says so rather than returning a
meaningless number.

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/models` | — | `ModelInfo[]` |
| `POST` | `/analyze` | `{model_id, prompt}` | `{tokens, attentions, num_layers, num_heads}` |
| `POST` | `/lens` | `{model_id, prompt, top_k?}` | per-layer top-k predictions, gap-free `trajectories`, and `narration` |
| `POST` | `/behavior` | `{model_id, prompts[], compare_model_id?}` | per-prompt continuations, divergence point, drift score |
| `POST` | `/attribution` | `{model_id, prompt, contrast_token_id?}` | `blocks` and `heads`, each a signed logit contribution |
| `POST` | `/patch` | `{recipient_model_id, donor_model_id, prompt, layer?}` | one `recovery` per part, plus generated text when `layer` is named |

`attentions` is nested layer → head → seq × seq.

`layer` is the honest 0-based block index, while every display label is shifted
by one so a block is named after the residual row it writes — block 11 reads as
`L12`, the same row the lens shows it landing in. Numbering both 0-based made
every intervention look off by one.

`/patch` addresses any part of the model with a single integer: `-1` is the
embeddings, `0…n-1` are the blocks, and `n` is the read-out. Omit `layer` to
sweep everything; name one to also get generated text for that swap.

## How the code is organised

Two rules, enforced by hand:

1. **No inference logic in route handlers.** `app/main.py` validates, calls into
   `app/inference.py`, and maps domain errors to status codes. `inference.py`
   never imports FastAPI.
2. **No API logic in React components.** `src/api/client.ts` is the only module
   that calls `fetch`. Components receive data as props or call
   `analyze`/`lens`/`attribution`/`patch`/`behavior`/`fetchModels`.

Analysis that produces *prose* rather than numbers lives in
`backend/app/insights.py`, which is deliberately torch-free — it takes a
finished layer trace and returns sentences, so it can be reasoned about and
changed without touching inference.

Every intervention — the attribution hooks and the weight swaps behind `/patch`
— is installed by a context manager and torn down in a `finally`. The models are
process-wide cached singletons, so a hook or a swap that outlived its request
would silently corrupt every later run, and a half-reverted checkpoint is a
chimera of two models that reports itself as one.

`/behavior` is the only endpoint that generates text. It batches with **left**
padding built by hand: decoder-only models continue from the last position, so
right-padding would have them continue from pad tokens, and building the batch
in-place avoids mutating `padding_side` on a shared cached tokenizer. Batching
is worth it — 0.16s per prompt against 0.44s one at a time.

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

**Fonts don't load** — the UI pulls Instrument Serif and IBM Plex Mono from
Google Fonts, the only external request the app makes. Offline, it falls back to
system serif/mono and everything still works.

**First request hangs** — it is downloading weights. Watch the uvicorn log.
Weights land in `~/.cache/huggingface`; delete that directory to force a
re-download if a fetch was interrupted and left a partial file.

## Not in this version

No auth, no database, no persistence. No sparse autoencoders or feature
extraction. No deployment or CI config — this is local dev only, two processes
on two ports.
