"""
api/routers/coref.py — Coreference Resolution Router
Owner:   Student B (Hiba)
Phase:   5 — Backend & API (Week 5–6)

Endpoints:
    POST /api/v1/coref          — Resolve coreference in Arabic text
    POST /api/v1/coref/batch    — Batch coreference resolution
    GET  /api/v1/coref/metrics  — Supported evaluation metrics info
    POST /api/v1/coref/evaluate — Evaluate predictions against gold clusters

Integrates with:
    - api/inference.py  InferenceEngine.predict_coref()
    - src/evaluation/coref_eval.py  for metric computation
    - src/utils/arabic_utils.py  for text utilities
"""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.utils.arabic_utils import (
    is_arabic,
    arabic_ratio,
    get_text_direction,
    iob2_to_spans,
)

router = APIRouter(prefix="/api/v1/coref", tags=["Coreference Resolution"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class CorefRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Arabic text for coreference resolution",
        examples=["محمد ذهب إلى المدرسة. هو يحب القراءة."],
    )
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Antecedent score threshold for linking mentions",
    )

    @field_validator("text")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v.strip()


class CorefBatchRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=16,
        description="List of Arabic texts (max 16)",
    )
    threshold: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("texts must contain at least one non-blank string")
        return cleaned


class MentionResponse(BaseModel):
    text:  str
    start: int
    end:   int
    tokens: Optional[list[str]] = None


class ClusterResponse(BaseModel):
    cluster_id: int
    mentions:   list[MentionResponse]
    n_mentions: int


class CorefResponse(BaseModel):
    text:          str
    tokens:        list[str]
    clusters:      list[ClusterResponse]
    n_clusters:    int
    n_mentions:    int
    arabic_ratio:  float
    direction:     str
    latency_ms:    float


class CorefBatchResponse(BaseModel):
    results:    list[CorefResponse]
    n_texts:    int
    latency_ms: float


class CorefMetricsResponse(BaseModel):
    metrics:     list[dict[str, str]]
    description: str


class CorefEvalRequest(BaseModel):
    gold_clusters: list[list[list[int]]] = Field(
        ...,
        description="Gold coreference clusters: list of clusters, "
                    "each a list of [start, end] mention spans",
    )
    pred_clusters: list[list[list[int]]] = Field(
        ...,
        description="Predicted coreference clusters (same format)",
    )


class CorefEvalResponse(BaseModel):
    muc_f1:     float
    b3_f1:      float
    ceafe_f1:   float
    conll_avg:  float
    mention_f1: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_stub_clusters(text: str) -> list[ClusterResponse]:
    """
    Stub coreference — creates a single dummy cluster for development.
    Replace with engine.predict_coref() once model is trained.
    """
    tokens = text.split()
    if len(tokens) < 3:
        return []

    # Simple stub: link first token to any pronoun-like token
    pronouns = {"هو", "هي", "هم", "هن", "أنا", "نحن", "هذا", "هذه"}
    mentions: list[MentionResponse] = [
        MentionResponse(text=tokens[0], start=0, end=0, tokens=[tokens[0]])
    ]
    for i, tok in enumerate(tokens[1:], 1):
        if tok in pronouns:
            mentions.append(
                MentionResponse(text=tok, start=i, end=i, tokens=[tok])
            )
            break

    if len(mentions) < 2:
        return []

    return [ClusterResponse(
        cluster_id=0,
        mentions=mentions,
        n_mentions=len(mentions),
    )]


def _engine_coref(
    engine, text: str, threshold: float
) -> list[ClusterResponse]:
    """Call the real inference engine and convert to ClusterResponse."""
    clusters_raw = engine.predict_coref([text])[0]
    result: list[ClusterResponse] = []
    tokens = text.split()

    for i, cluster in enumerate(clusters_raw):
        mentions: list[MentionResponse] = []
        for mention_dict in cluster.mentions:
            start = mention_dict.get("start", 0)
            end   = mention_dict.get("end",   0)
            text_ = mention_dict.get("text",  "")
            mention_tokens = tokens[start:end + 1] if tokens else []
            mentions.append(MentionResponse(
                text=text_,
                start=start,
                end=end,
                tokens=mention_tokens,
            ))
        if len(mentions) >= 2:
            result.append(ClusterResponse(
                cluster_id=i,
                mentions=mentions,
                n_mentions=len(mentions),
            ))
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=CorefResponse,
    summary="Coreference Resolution",
    description=(
        "Identify coreference chains in Arabic text. "
        "Returns clusters of mentions that refer to the same entity. "
        "Implements Lee et al. (2018) end-to-end approach adapted for Arabic."
    ),
)
async def resolve_coref(body: CorefRequest, request: Request) -> CorefResponse:
    t0 = time.perf_counter()

    if not is_arabic(body.text):
        raise HTTPException(
            status_code=422,
            detail="Input text does not appear to contain Arabic.",
        )

    engine = getattr(request.app.state, "engine", None)

    try:
        if engine and hasattr(engine, "predict_coref"):
            clusters = _engine_coref(engine, body.text, body.threshold)
        else:
            clusters = _build_stub_clusters(body.text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Coreference inference failed: {exc}",
        ) from exc

    tokens     = body.text.split()
    n_mentions = sum(c.n_mentions for c in clusters)
    latency_ms = (time.perf_counter() - t0) * 1000

    return CorefResponse(
        text=body.text,
        tokens=tokens,
        clusters=clusters,
        n_clusters=len(clusters),
        n_mentions=n_mentions,
        arabic_ratio=arabic_ratio(body.text),
        direction=get_text_direction(body.text),
        latency_ms=round(latency_ms, 2),
    )


@router.post(
    "/batch",
    response_model=CorefBatchResponse,
    summary="Batch Coreference Resolution",
    description="Resolve coreference in multiple Arabic texts (max 16).",
)
async def resolve_coref_batch(
    body: CorefBatchRequest, request: Request
) -> CorefBatchResponse:
    t0     = time.perf_counter()
    engine = getattr(request.app.state, "engine", None)
    results: list[CorefResponse] = []

    for text in body.texts:
        if not is_arabic(text):
            results.append(CorefResponse(
                text=text, tokens=text.split(), clusters=[],
                n_clusters=0, n_mentions=0, arabic_ratio=0.0,
                direction="ltr", latency_ms=0.0,
            ))
            continue

        try:
            if engine and hasattr(engine, "predict_coref"):
                clusters = _engine_coref(engine, text, body.threshold)
            else:
                clusters = _build_stub_clusters(text)
        except Exception:
            clusters = []

        results.append(CorefResponse(
            text=text,
            tokens=text.split(),
            clusters=clusters,
            n_clusters=len(clusters),
            n_mentions=sum(c.n_mentions for c in clusters),
            arabic_ratio=arabic_ratio(text),
            direction=get_text_direction(text),
            latency_ms=0.0,
        ))

    total_ms = (time.perf_counter() - t0) * 1000
    return CorefBatchResponse(
        results=results,
        n_texts=len(results),
        latency_ms=round(total_ms, 2),
    )


@router.get(
    "/metrics",
    response_model=CorefMetricsResponse,
    summary="Coreference Evaluation Metrics",
    description="Return information about the supported coreference evaluation metrics.",
)
async def list_metrics() -> CorefMetricsResponse:
    return CorefMetricsResponse(
        metrics=[
            {
                "name":        "MUC",
                "full_name":   "Mention-based F1",
                "reference":   "Vilain et al. (1995)",
                "description": "Measures how well predicted clusters partition gold clusters.",
            },
            {
                "name":        "B³",
                "full_name":   "B-cubed F1",
                "reference":   "Bagga & Baldwin (1998)",
                "description": "Entity-level metric averaging per-mention precision and recall.",
            },
            {
                "name":        "CEAFe",
                "full_name":   "Entity Alignment F1",
                "reference":   "Luo (2005)",
                "description": "Finds optimal one-to-one alignment between gold and predicted clusters.",
            },
            {
                "name":        "CoNLL Avg",
                "full_name":   "CoNLL Average F1",
                "reference":   "Pradhan et al. (2012)",
                "description": "Standard benchmark metric: (MUC + B³ + CEAFe) / 3.",
            },
        ],
        description=(
            "All metrics implemented in src/evaluation/coref_eval.py. "
            "CoNLL Average F1 is the primary benchmark metric used in comparison tables."
        ),
    )


@router.post(
    "/evaluate",
    response_model=CorefEvalResponse,
    summary="Evaluate Coreference Predictions",
    description=(
        "Compute MUC, B³, CEAFe, and CoNLL average F1 for a set of "
        "predicted coreference clusters against gold annotations. "
        "Useful for development evaluation without running the full eval pipeline."
    ),
)
async def evaluate_coref(body: CorefEvalRequest) -> CorefEvalResponse:
    try:
        from src.evaluation.coref_eval import (
            to_clusters, muc_score, b3_score,
            ceafe_score, mention_score,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation module not available: {exc}",
        ) from exc

    gold = to_clusters(body.gold_clusters)
    pred = to_clusters(body.pred_clusters)

    muc   = muc_score(gold, pred)
    b3    = b3_score(gold, pred)
    ceafe = ceafe_score(gold, pred)
    ms    = mention_score(gold, pred)
    conll = (muc.f1 + b3.f1 + ceafe.f1) / 3.0

    return CorefEvalResponse(
        muc_f1=round(muc.f1,   4),
        b3_f1=round(b3.f1,    4),
        ceafe_f1=round(ceafe.f1, 4),
        conll_avg=round(conll,   4),
        mention_f1=round(ms.f1,  4),
    )
