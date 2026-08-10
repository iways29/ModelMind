"""FastAPI entrypoint.

Routes are deliberately thin: validate, delegate to `inference`, translate
domain errors into HTTP status codes. No model logic lives here.
"""

import logging
import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import inference, models
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CompareRequest,
    CompareResponse,
    LensRequest,
    LensResponse,
    ModelInfo,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Model Internals Viz",
    description="Attention and activation internals for GPT-2 family checkpoints.",
    version="1.0.0",
)

# Vite defaults to 5173; CRA/Next default to 3000. Both spellings of localhost
# are listed because the browser treats them as distinct origins.
DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_env_origins = os.getenv("CORS_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _env_origins.split(",") if o.strip()] or DEFAULT_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/models", response_model=List[ModelInfo])
def get_models() -> List[ModelInfo]:
    """The hardcoded model catalog that populates the frontend selector."""
    return models.list_models()


# Routes are sync `def`, so Starlette runs them in a worker thread. A CPU
# forward pass would otherwise block the event loop for every other request.
@app.post("/analyze", response_model=AnalyzeResponse)
def post_analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run one model on one prompt; return tokens, attention, activations."""
    try:
        return inference.analyze(request.model_id, request.prompt, request.max_tokens)
    except models.UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except inference.EmptyPromptError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except inference.ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/lens", response_model=LensResponse)
def post_lens(request: LensRequest) -> LensResponse:
    """Decode the model's predicted next token at every layer, not just the last."""
    try:
        return inference.logit_lens(
            request.model_id,
            request.prompt,
            top_k=request.top_k,
            position=request.position,
            max_tokens=request.max_tokens,
        )
    except models.UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except inference.EmptyPromptError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except inference.LensUnsupportedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except inference.ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/compare", response_model=CompareResponse)
def post_compare(request: CompareRequest) -> CompareResponse:
    """Run two models on the same prompt; return both activation profiles and their delta."""
    try:
        return inference.compare(
            request.base_model_id,
            request.finetuned_model_id,
            request.prompt,
            request.max_tokens,
        )
    except models.UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except inference.EmptyPromptError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except inference.ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
