"""
main.py — FastAPI Entry Point for Arabic NLP API.

Owner:   Student A
Phase:   5 — Backend & API (Week 5–6)

Endpoints:
    POST /api/v1/ner        — Named Entity Recognition
    POST /api/v1/pos        — Part-of-Speech Tagging
    POST /api/v1/coref      — Coreference Resolution
    POST /api/v1/analyze    — Unified multi-task endpoint
    GET  /api/v1/health     — Health check
    GET  /api/v1/models     — Available models info

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Docs auto-generated at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)

Requirements:
    pip install fastapi uvicorn[standard] pydantic transformers
"""

from __future__ import annotations

import time
from pathlib import Path
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("arabic_nlp_api")

# ---------------------------------------------------------------------------
# Schemas (inline to keep single-file runnable)
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field, field_validator

class ArabicTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("text must not be blank")
        return v.strip()

class NERSpan(BaseModel):
    text: str; type: str; start: int; end: int

class NERToken(BaseModel):
    token: str; tag: str; start: int; end: int

class NERRequest(ArabicTextRequest):
    include_tokens: bool = False

class NERResponse(BaseModel):
    text: str; spans: list[NERSpan]; tokens: list[NERToken] | None = None
    latency_ms: float = 0.0

class POSToken(BaseModel):
    token: str; pos_tag: str; morph: dict | None = None

class POSRequest(ArabicTextRequest):
    include_morphology: bool = False

class POSResponse(BaseModel):
    text: str; tokens: list[POSToken]; latency_ms: float = 0.0

class CorefMention(BaseModel):
    text: str; sent_idx: int = 0; start: int = 0; end: int = 0

class CorefCluster(BaseModel):
    cluster_id: int; mentions: list[CorefMention]

class CorefRequest(ArabicTextRequest):
    pass

class CorefResponse(BaseModel):
    text: str; clusters: list[CorefCluster]; latency_ms: float = 0.0

class AnalyzeRequest(ArabicTextRequest):
    tasks: list[str] = ["ner", "pos", "coref"]

    @field_validator("tasks")
    @classmethod
    def valid_tasks(cls, v):
        bad = set(v) - {"ner", "pos", "coref"}
        if bad:
            raise ValueError(f"Unknown tasks: {bad}")
        return v

class AnalyzeResponse(BaseModel):
    text: str
    ner:   NERResponse   | None = None
    pos:   POSResponse   | None = None
    coref: CorefResponse | None = None
    latency_ms: float = 0.0

class HealthResponse(BaseModel):
    status: str; model_loaded: bool; version: str

class ModelsResponse(BaseModel):
    models: list[dict]

class ErrorResponse(BaseModel):
    error: str; message: str; request_id: str = ""

# ---------------------------------------------------------------------------
# Stub inference engine
# ---------------------------------------------------------------------------
class RealEngine:
    """Real MTL inference engine wrapping InferenceEngine."""

    def __init__(self, checkpoint_path: str):
        from api.inference import InferenceEngine
        self.engine = InferenceEngine.load(checkpoint_path)

    def predict(self, text: str, tasks=None, **kwargs):
        tasks = tasks or ["ner", "pos", "coref"]
        results = self.engine.predict_all([text])
        r = results[0]

        # NER spans
        ner_spans = [
            {"text": e.text, "type": e.label,
             "start": e.start_tok, "end": e.end_tok, "score": e.score}
            for e in r.ner
        ]
        # NER tokens (IOB2 style for token-level view)
        ner_tokens = [
            {"token": t.token, "tag": t.tag, "start": i, "end": i, "score": t.score}
            for i, t in enumerate(r.pos)  # reuse word list
        ]
        # POS tokens
        pos_tokens = [
            {"token": t.token, "pos_tag": t.tag, "morph": None, "score": t.score}
            for t in r.pos
        ]
        # Coref clusters
        coref_clusters = [
            {"mentions": c.mentions}
            for c in r.coref
        ]

        return type("R", (), {
            "ner_spans":      ner_spans,
            "ner_tokens":     ner_tokens,
            "pos_tokens":     pos_tokens,
            "coref_clusters": coref_clusters,
            "latency_ms":     r.latency_ms,
        })()

class StubEngine:
    """Fallback stub if model file is missing."""
    def predict(self, text, tasks=None, **kwargs):
        words = text.split()
        return type("R", (), {
            "ner_spans":      [],
            "ner_tokens":     [{"token": w, "tag": "O", "start": i, "end": i} for i, w in enumerate(words)],
            "pos_tokens":     [{"token": w, "pos_tag": "NN", "morph": None} for w in words],
            "coref_clusters": [],
            "latency_ms":     0.5,
        })()

# ---------------------------------------------------------------------------
# Lifespan: load engine once
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Arabic NLP API...")
    import os
    checkpoint = os.environ.get(
        "MODEL_CHECKPOINT",
        str(Path(__file__).parent.parent / "checkpoints" / "FINAL_MODEL.pt")
    )
    try:
        if Path(checkpoint).exists():
            app.state.engine = RealEngine(checkpoint)
            logger.info(f"✓ Real model loaded from {checkpoint}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint} — using stub")
            app.state.engine = StubEngine()
    except Exception as e:
        logger.warning(f"Engine load failed: {e} — using stub")
        app.state.engine = StubEngine()
    yield
    logger.info("API shutdown")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Arabic NLP API",
    description=(
        "Multi-task Arabic NLP: Named Entity Recognition, "
        "Part-of-Speech Tagging, Coreference Resolution. "
        "Built on AraBERT v2 with a shared MTL backbone."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Request ID + latency middleware ────────────────────────────────────────
@app.middleware("http")
async def request_metadata(request: Request, call_next):
    rid   = str(uuid.uuid4())[:8]
    t0    = time.perf_counter()
    request.state.request_id = rid
    resp  = await call_next(request)
    ms    = (time.perf_counter() - t0) * 1000
    resp.headers["X-Request-ID"] = rid
    resp.headers["X-Latency-Ms"] = f"{ms:.1f}"
    logger.info(f"[{rid}] {request.method} {request.url.path} → {resp.status_code} ({ms:.0f}ms)")
    return resp

# ── Global error handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={
        "error": "internal_server_error",
        "message": "An unexpected error occurred.",
        "request_id": getattr(request.state, "request_id", ""),
    })

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Arabic NLP API — visit /docs"}


@app.get("/api/v1/health", response_model=HealthResponse)
async def health(request: Request):
    ok = getattr(request.app.state, "engine", None) is not None
    return HealthResponse(status="ok" if ok else "degraded", model_loaded=ok, version=app.version)


@app.get("/api/v1/models", response_model=ModelsResponse)
async def models_info():
    return ModelsResponse(models=[{
        "id": "arabert-v2-mtl",
        "backbone": "aubmindlab/bert-base-arabertv02",
        "tasks": ["ner", "pos", "coref"],
        "ner_f1": 0.8487,
        "pos_accuracy": 0.9706,
        "coref_avg_f1": 0.6715,
    }])


@app.post("/api/v1/ner", response_model=NERResponse, summary="Named Entity Recognition")
async def ner(body: NERRequest, request: Request):
    """Detect named entities (PER, ORG, LOC, GPE, DATE...) in Arabic text."""
    engine = getattr(request.app.state, "engine", None)
    r = engine.predict(body.text, tasks=["ner"], include_tokens=body.include_tokens)
    return NERResponse(
        text=body.text,
        spans=[NERSpan(**s) for s in r.ner_spans],
        tokens=[NERToken(**t) for t in r.ner_tokens] if body.include_tokens else None,
        latency_ms=r.latency_ms,
    )


@app.post("/api/v1/pos", response_model=POSResponse, summary="Part-of-Speech Tagging")
async def pos(body: POSRequest, request: Request):
    """Tag each Arabic token with its PATB POS label."""
    engine = getattr(request.app.state, "engine", None)
    r = engine.predict(body.text, tasks=["pos"], include_morphology=body.include_morphology)
    return POSResponse(
        text=body.text,
        tokens=[POSToken(**t) for t in r.pos_tokens],
        latency_ms=r.latency_ms,
    )


@app.post("/api/v1/coref", response_model=CorefResponse, summary="Coreference Resolution")
async def coref(body: CorefRequest, request: Request):
    """Identify coreference chains in Arabic text."""
    engine = getattr(request.app.state, "engine", None)
    r = engine.predict(body.text, tasks=["coref"])
    return CorefResponse(
        text=body.text,
        clusters=[
            CorefCluster(cluster_id=i, mentions=[CorefMention(text=m.get("text",""), sent_idx=0, start=m.get("start",0), end=m.get("end",0)) for m in c.get("mentions",[])])
            for i, c in enumerate(r.coref_clusters)
        ],
        latency_ms=r.latency_ms,
    )


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, summary="Unified multi-task analysis")
async def analyze(body: AnalyzeRequest, request: Request):
    """
    Run NER + POS + Coreference in one request.
    Single backbone forward pass — fastest option for the demo.
    """
    engine = getattr(request.app.state, "engine", None)
    r = engine.predict(body.text, tasks=body.tasks)
    resp = AnalyzeResponse(text=body.text, latency_ms=r.latency_ms)

    if "ner" in body.tasks:
        resp.ner = NERResponse(text=body.text, spans=[NERSpan(**s) for s in r.ner_spans], latency_ms=r.latency_ms)
    if "pos" in body.tasks:
        resp.pos = POSResponse(text=body.text, tokens=[POSToken(**t) for t in r.pos_tokens], latency_ms=r.latency_ms)
    if "coref" in body.tasks:
        resp.coref = CorefResponse(
            text=body.text,
            clusters=[
                CorefCluster(cluster_id=i, mentions=[CorefMention(text=m.get("text",""), sent_idx=0, start=m.get("start",0), end=m.get("end",0)) for m in c.get("mentions",[])])
                for i, c in enumerate(r.coref_clusters)
            ],
            latency_ms=r.latency_ms,
        )
    return resp


# ---------------------------------------------------------------------------
# Sanity check  (python main.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from fastapi.testclient import TestClient

    print("=== FastAPI Sanity Check ===\n")

    with TestClient(app) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        print(f"health: {r.json()['status']}")

        r = client.get("/api/v1/models")
        assert r.status_code == 200
        print(f"models: {len(r.json()['models'])} model(s)")

        r = client.post("/api/v1/ner", json={"text": "visited Cairo today"})
        assert r.status_code == 200
        print(f"NER: {r.json()['spans']} spans")

        r = client.post("/api/v1/pos", json={"text": "visited Cairo today"})
        assert r.status_code == 200
        print(f"POS: {len(r.json()['tokens'])} tokens")

        r = client.post("/api/v1/coref", json={"text": "Mohamed said he is happy"})
        assert r.status_code == 200
        print(f"Coref: {r.json()['clusters']} clusters")

        r = client.post("/api/v1/analyze", json={"text": "visited Cairo", "tasks": ["ner", "pos"]})
        assert r.status_code == 200
        data = r.json()
        assert data["ner"] is not None and data["pos"] is not None
        print(f"Analyze (ner+pos): ok, {data['latency_ms']:.2f}ms")

        r = client.post("/api/v1/analyze", json={"text": "test", "tasks": ["bad_task"]})
        assert r.status_code == 422
        print("Validation bad task: 422 ok")

        r = client.post("/api/v1/ner", json={"text": "   "})
        assert r.status_code == 422
        print("Validation blank text: 422 ok")

    print("\n==> All checks passed — place in api/main.py")