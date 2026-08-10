# Frontend — React + TypeScript + Tailwind

Vite dev server on port 5173. Talks to the FastAPI backend over CORS; there is
no proxy, no SSR, no build-time coupling between the two services.

## Run

```bash
npm install
npm run dev
```

The backend must already be running on port 8000 — the header pill reads
"backend offline" until `GET /models` succeeds.

## Layout

```
src/
├── api/
│   ├── client.ts     # the only module that calls fetch
│   └── types.ts      # hand-mirrored copies of the Pydantic schemas
├── components/
│   ├── ActivationChart.tsx   # per-layer activation magnitude (nivo line)
│   ├── AttentionHeatmap.tsx  # token x token attention (nivo heatmap)
│   ├── CompareView.tsx       # base vs fine-tuned vs delta
│   ├── ModelSelector.tsx     # dropdown over GET /models
│   ├── chartTheme.ts         # shared nivo dark theme + heatmap colour ramp
│   └── ui.tsx                # Panel / Field / Select / Button primitives
├── App.tsx
└── main.tsx
```

The one architectural rule: **no component calls `fetch`.** `App` and
`CompareView` call `fetchModels`, `analyze`, and `compare` from `api/client.ts`;
every other component is handed its data as props.

## Charting

`nivo` rather than `visx`, chosen for the heatmap: `@nivo/heatmap` renders
labelled axes on both edges out of the box, where `visx` would need hand-built
axis components for the same result. `@nivo/line` then covers both line charts
so the whole app shares one theme object.

Two details worth knowing if you edit `AttentionHeatmap`:

- Cell keys are **column indices as strings**, not token text. A prompt can
  repeat a token, and nivo requires unique keys within a row. The real labels
  are looked up by index in the axis `format` callback.
- The colour ramp is a custom interpolator in `chartTheme.ts` rather than a
  built-in nivo scheme, so `t = 0` lands exactly on the panel background and
  zero-attention cells read as empty rather than as data.

## Keeping types in sync

`api/types.ts` mirrors `backend/app/schemas.py` by hand — no codegen at this
stage. If you change a Pydantic model, change the matching interface in the
same commit.

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Backend origin. |

`127.0.0.1` rather than `localhost` on purpose: on macOS `localhost` can resolve
to `::1` first while uvicorn binds IPv4 only, and the resulting connection
failure looks exactly like a CORS error in the console.
