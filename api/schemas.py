"""
api/schemas.py — Pydantic Request/Response Schemas
Owner:   Student B (Hiba)
Phase:   5 — Backend & API (Week 5–6)

Centralized schema definitions used across all FastAPI routers.
Importing from here keeps main.py and routers clean and consistent.

All schemas are Pydantic v2 compatible.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# Base
# ══════════════════════════════════════════════════════════════════════════════

class ArabicTextRequest(BaseModel):
    """Base request model for all Arabic NLP endpoints."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Arabic input text (MSA or dialectal)",
        examples=["زار الرئيس محمد بن سلمان مدينة الرياض."],
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace only")
        return v.strip()


class LatencyMixin(BaseModel):
    """Mixin that adds latency_ms to any response."""
    latency_ms: float = Field(
        0.0,
        description="End-to-end inference latency in milliseconds",
    )


# ══════════════════════════════════════════════════════════════════════════════
# NER schemas
# ══════════════════════════════════════════════════════════════════════════════

class NERRequest(ArabicTextRequest):
    """Request body for the /ner endpoint."""
    include_tokens: bool = Field(
        False,
        description="If True, return full token list alongside entity spans",
    )
    include_html: bool = Field(
        False,
        description="If True, return HTML-annotated text with entity markup",
    )


class NERSpan(BaseModel):
    """A single named entity span."""
    text:      str   = Field(..., description="Surface form of the entity")
    type:      str   = Field(..., description="Entity type: PER, ORG, LOC, GPE, DATE, MISC")
    start:     int   = Field(..., description="Start token index (0-based)")
    end:       int   = Field(..., description="End token index (inclusive)")
    score:     float = Field(1.0, description="Confidence score [0, 1]")


class NERToken(BaseModel):
    """A single token with its IOB2 label."""
    token: str = Field(..., description="Surface word form")
    tag:   str = Field(..., description="IOB2 tag: B-TYPE, I-TYPE, or O")
    start: int = Field(..., description="Token index")
    end:   int = Field(..., description="Token index (same as start for single tokens)")


class NERResponse(LatencyMixin):
    """Response from the /ner endpoint."""
    text:        str                  = Field(..., description="Original input text")
    spans:       list[NERSpan]        = Field(..., description="Detected entity spans")
    tokens:      Optional[list[NERToken]] = Field(None, description="Full token list (if requested)")
    html:        Optional[str]        = Field(None, description="HTML-annotated text (if requested)")
    n_entities:  int                  = Field(0,    description="Total entity count")
    entity_types: dict[str, int]      = Field(default_factory=dict,
                                              description="Entity count per type")


# ══════════════════════════════════════════════════════════════════════════════
# POS schemas
# ══════════════════════════════════════════════════════════════════════════════

class POSRequest(ArabicTextRequest):
    """Request body for the /pos endpoint."""
    include_morphology: bool = Field(
        False,
        description="Include morphological category per token (verb/noun/clitic/etc.)",
    )
    include_direction: bool = Field(
        False,
        description="Include RTL/LTR direction annotation per token",
    )
    include_segments: bool = Field(
        False,
        description="Include morphological segmentation if available (e.g. ب+يحب+ك)",
    )


class POSTokenSchema(BaseModel):
    """A single token with its POS tag and optional features."""
    token:          str            = Field(..., description="Surface word form")
    pos_tag:        str            = Field(..., description="POS tag (QCRI tagset)")
    morph_category: Optional[str] = Field(None, description="Morphological category")
    direction:      Optional[str] = Field(None, description="Text direction: rtl/ltr/neutral")
    segment:        Optional[str] = Field(None, description="Morphological segmentation")
    is_stopword:    bool           = Field(False, description="True if token is a stopword")
    token_type:     str            = Field("arabic_word", description="Token category")


class POSResponse(LatencyMixin):
    """Response from the /pos endpoint."""
    text:          str                = Field(..., description="Original input text")
    tokens:        list[POSTokenSchema] = Field(..., description="Tagged tokens")
    n_tokens:      int                = Field(0,    description="Token count")
    arabic_ratio:  float              = Field(0.0,  description="Fraction of Arabic characters")
    direction:     str                = Field("rtl", description="Primary text direction")
    tag_counts:    dict[str, int]     = Field(default_factory=dict,
                                              description="Token count per POS tag")


# ══════════════════════════════════════════════════════════════════════════════
# Coreference schemas
# ══════════════════════════════════════════════════════════════════════════════

class CorefRequest(ArabicTextRequest):
    """Request body for the /coref endpoint."""
    threshold: float = Field(
        0.5,
        ge=0.0, le=1.0,
        description="Antecedent score threshold for linking mentions [0, 1]",
    )
    return_mention_scores: bool = Field(
        False,
        description="If True, include raw mention scores in response",
    )


class CorefMentionSchema(BaseModel):
    """A single mention in a coreference cluster."""
    text:      str        = Field(..., description="Surface form of the mention")
    start:     int        = Field(..., description="Start token index")
    end:       int        = Field(..., description="End token index (inclusive)")
    tokens:    list[str]  = Field(default_factory=list,
                                  description="Individual tokens in this mention")
    score:     Optional[float] = Field(None, description="Mention score (if requested)")


class CorefClusterSchema(BaseModel):
    """A single coreference cluster — all mentions refer to the same entity."""
    cluster_id:  int                    = Field(..., description="Cluster index (0-based)")
    mentions:    list[CorefMentionSchema] = Field(..., description="All mentions in this cluster")
    n_mentions:  int                    = Field(..., description="Number of mentions")
    head_mention: Optional[str]         = Field(None, description="Most prominent mention text")


class CorefResponse(LatencyMixin):
    """Response from the /coref endpoint."""
    text:          str                      = Field(..., description="Original input text")
    tokens:        list[str]                = Field(..., description="Tokenized words")
    clusters:      list[CorefClusterSchema] = Field(..., description="Coreference clusters")
    n_clusters:    int                      = Field(0,   description="Number of clusters")
    n_mentions:    int                      = Field(0,   description="Total mention count")
    arabic_ratio:  float                    = Field(0.0, description="Arabic character ratio")
    direction:     str                      = Field("rtl", description="Text direction")


# ══════════════════════════════════════════════════════════════════════════════
# Unified analyze schemas
# ══════════════════════════════════════════════════════════════════════════════

VALID_TASKS = {"ner", "pos", "coref"}

class AnalyzeRequest(ArabicTextRequest):
    """Request body for the unified /analyze endpoint."""
    tasks: list[str] = Field(
        ["ner", "pos", "coref"],
        description="Tasks to run. Any subset of: ner, pos, coref",
    )
    include_morphology: bool = Field(False)
    include_tokens:     bool = Field(False)
    coref_threshold:    float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, v: list[str]) -> list[str]:
        invalid = set(v) - VALID_TASKS
        if invalid:
            raise ValueError(
                f"Invalid tasks: {invalid}. "
                f"Choose from: {VALID_TASKS}"
            )
        if not v:
            raise ValueError("tasks must contain at least one task")
        return list(set(v))  # deduplicate


class AnalyzeResponse(LatencyMixin):
    """Response from the unified /analyze endpoint."""
    text:   str                      = Field(..., description="Original input text")
    tokens: list[str]                = Field(default_factory=list)
    ner:    Optional[NERResponse]    = Field(None, description="NER results (if requested)")
    pos:    Optional[POSResponse]    = Field(None, description="POS results (if requested)")
    coref:  Optional[CorefResponse]  = Field(None, description="Coref results (if requested)")


# ══════════════════════════════════════════════════════════════════════════════
# Health & model info
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status:       str  = Field(..., description="ok | degraded | error")
    model_loaded: bool = Field(..., description="True if inference engine is ready")
    version:      str  = Field(..., description="API version string")
    uptime_s:     Optional[float] = Field(None, description="Seconds since startup")


class ModelInfo(BaseModel):
    """Metadata for one loaded model."""
    id:           str       = Field(..., description="Model identifier")
    backbone:     str       = Field(..., description="Transformer backbone name")
    tasks:        list[str] = Field(..., description="Supported tasks")
    ner_f1:       Optional[float] = Field(None, description="NER entity F1 on test set")
    pos_accuracy: Optional[float] = Field(None, description="POS token accuracy on test set")
    coref_avg_f1: Optional[float] = Field(None, description="CoNLL avg F1 on test set")
    latency_ms:   Optional[float] = Field(None, description="Mean inference latency")


class ModelsResponse(BaseModel):
    models: list[ModelInfo] = Field(..., description="Available models")


# ══════════════════════════════════════════════════════════════════════════════
# Batch schemas
# ══════════════════════════════════════════════════════════════════════════════

class BatchNERRequest(BaseModel):
    """Batch NER request — up to 32 texts."""
    texts: list[str] = Field(..., min_length=1, max_length=32)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("texts must contain at least one non-blank string")
        return cleaned


class BatchNERResponse(BaseModel):
    results:    list[NERResponse] = Field(..., description="One NERResponse per input text")
    n_texts:    int               = Field(..., description="Number of texts processed")
    latency_ms: float             = Field(0.0, description="Total batch latency")


class BatchPOSRequest(BaseModel):
    """Batch POS request — up to 32 texts."""
    texts: list[str] = Field(..., min_length=1, max_length=32)
    include_morphology: bool = False

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        return [t.strip() for t in v if t.strip()]


class BatchPOSResponse(BaseModel):
    results:    list[POSResponse] = Field(...)
    n_texts:    int               = Field(...)
    latency_ms: float             = Field(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Error schemas
# ══════════════════════════════════════════════════════════════════════════════

class ErrorDetail(BaseModel):
    """Standard error response body."""
    error:      str            = Field(..., description="Error code")
    message:    str            = Field(..., description="Human-readable message")
    details:    Optional[Any]  = Field(None, description="Additional error context")
    request_id: Optional[str]  = Field(None, description="Request ID for tracing")


# ══════════════════════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Schemas Sanity Check ===\n")

    # NER request validation
    req = NERRequest(text="  زار الرئيس مدينة الرياض  ")
    assert req.text == "زار الرئيس مدينة الرياض"
    print("✓ NERRequest: text stripped correctly")

    # Blank text rejected
    try:
        NERRequest(text="   ")
        assert False, "Should have raised"
    except Exception:
        print("✓ NERRequest: blank text rejected")

    # POS request
    req = POSRequest(text="الطالب يدرس", include_morphology=True)
    assert req.include_morphology is True
    print("✓ POSRequest: morphology flag works")

    # Analyze request task validation
    req = AnalyzeRequest(text="نص عربي", tasks=["ner", "pos"])
    assert set(req.tasks) == {"ner", "pos"}
    print("✓ AnalyzeRequest: task list validated")

    # Invalid task rejected
    try:
        AnalyzeRequest(text="نص", tasks=["ner", "unknown_task"])
        assert False, "Should have raised"
    except Exception:
        print("✓ AnalyzeRequest: invalid task rejected")

    # Coref request
    req = CorefRequest(text="محمد يعمل وهو سعيد", threshold=0.7)
    assert req.threshold == 0.7
    print("✓ CorefRequest: threshold validated")

    # Response construction
    ner_resp = NERResponse(
        text="زار محمد القاهرة",
        spans=[NERSpan(text="محمد", type="PER", start=1, end=1, score=0.95),
               NERSpan(text="القاهرة", type="LOC", start=2, end=2, score=0.91)],
        n_entities=2,
        entity_types={"PER": 1, "LOC": 1},
        latency_ms=42.1,
    )
    assert ner_resp.n_entities == 2
    print(f"✓ NERResponse: {ner_resp.n_entities} entities, {ner_resp.latency_ms}ms")

    # Health response
    health = HealthResponse(status="ok", model_loaded=True, version="1.0.0")
    assert health.status == "ok"
    print("✓ HealthResponse: valid")

    # Batch request
    batch = BatchNERRequest(texts=["نص أول", "  ", "نص ثاني"])
    assert len(batch.texts) == 2  # blank filtered out
    print(f"✓ BatchNERRequest: {len(batch.texts)} valid texts (blank filtered)")

    print("\n✅ All schema checks passed — api/schemas.py ready")
