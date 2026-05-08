"""
api/routers/pos.py — POS Tagging Router
Owner:   Student B (Hiba)
Phase:   5 — Backend & API (Week 5–6)

Endpoints:
    POST /api/v1/pos          — Tag a single Arabic text
    POST /api/v1/pos/batch    — Tag multiple texts in one request
    GET  /api/v1/pos/tags     — List all supported POS tags
    POST /api/v1/pos/validate — Validate a POS tag sequence

Integrates with:
    - api/inference.py  InferenceEngine.predict_pos()
    - src/utils/arabic_utils.py  for RTL annotation
    - src/models/pos_head.py  LABELS / LABEL2ID
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
    add_rtl_markers_to_tokens,
    is_stopword,
    token_type,
)
from src.models.pos_head import LABELS as POS_LABELS, LABEL2ID

router = APIRouter(prefix="/api/v1/pos", tags=["POS Tagging"])

# ── Morphological category mapping ────────────────────────────────────────────
MORPH_CATEGORIES = {
    "verb":     {"V", "FUT_PART+V", "PROG_PART+V",
                 "FUT_PART+V+PRON", "PROG_PART+V+PRON", "CONJ+V"},
    "noun":     {"NOUN", "DET+NOUN", "CONJ+NOUN",
                 "NOUN+PRON", "NOUN+PREP+PRON"},
    "clitic":   {"PREP+PRON", "CONJ+PART", "CONJ+PRON",
                 "PART+PRON", "PREP+DET"},
    "adjective":{"ADJ", "CONJ+ADJ", "ADJ+PRON"},
    "pronoun":  {"PRON", "V+PRON"},
    "particle": {"PART", "DET", "PREP", "CONJ", "ADV"},
    "other":    {"NUM", "PUNC", "ABBREV", "FOREIGN", "OTHER"},
}


def get_morph_category(tag: str) -> str:
    for cat, tags in MORPH_CATEGORIES.items():
        if tag in tags:
            return cat
    return "other"


# ── Schemas ───────────────────────────────────────────────────────────────────

class POSRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Arabic text to tag",
        examples=["الطالب يدرس في الجامعة"],
    )
    include_morphology: bool = Field(
        False,
        description="Include morphological category per token",
    )
    include_direction: bool = Field(
        False,
        description="Include RTL/LTR direction annotation per token",
    )

    @field_validator("text")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v.strip()


class POSBatchRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of Arabic texts to tag (max 32)",
    )
    include_morphology: bool = False

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("texts must contain at least one non-blank string")
        return cleaned


class POSTokenResponse(BaseModel):
    token:     str
    pos_tag:   str
    morph_category: Optional[str] = None
    direction: Optional[str] = None
    is_stopword: bool = False
    token_type:  str  = "arabic_word"


class POSResponse(BaseModel):
    text:          str
    tokens:        list[POSTokenResponse]
    n_tokens:      int
    arabic_ratio:  float
    latency_ms:    float
    direction:     str


class POSBatchResponse(BaseModel):
    results:    list[POSResponse]
    n_texts:    int
    latency_ms: float


class POSTagsResponse(BaseModel):
    tags:       list[str]
    n_tags:     int
    categories: dict[str, list[str]]


class POSValidateRequest(BaseModel):
    tokens:   list[str]
    pos_tags: list[str]


class POSValidateResponse(BaseModel):
    valid:    bool
    n_tokens: int
    n_tags:   int
    issues:   list[str]
    coverage: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_stub_pos(text: str, include_morphology: bool,
                    include_direction: bool) -> list[POSTokenResponse]:
    """
    Stub POS tagger — cycles through common tags.
    Replace with engine.predict_pos() once model is trained.
    """
    words     = text.split()
    stub_tags = ["NOUN", "V", "PREP", "NOUN", "ADJ",
                 "PRON", "CONJ", "PART", "ADV", "NUM"]
    result    = []
    for i, word in enumerate(words):
        tag = stub_tags[i % len(stub_tags)]
        tok = POSTokenResponse(
            token=word,
            pos_tag=tag,
            is_stopword=is_stopword(word),
            token_type=token_type(word),
        )
        if include_morphology:
            tok.morph_category = get_morph_category(tag)
        if include_direction:
            tok.direction = get_text_direction(word)
        result.append(tok)
    return result


def _engine_pos(
    engine,
    text: str,
    include_morphology: bool,
    include_direction: bool,
) -> list[POSTokenResponse]:
    """Call the real inference engine and convert to POSTokenResponse."""
    pos_tokens = engine.predict_pos([text])[0]
    result = []
    for pt in pos_tokens:
        tok = POSTokenResponse(
            token=pt.token,
            pos_tag=pt.tag,
            is_stopword=is_stopword(pt.token),
            token_type=token_type(pt.token),
        )
        if include_morphology:
            tok.morph_category = get_morph_category(pt.tag)
        if include_direction:
            tok.direction = get_text_direction(pt.token)
        result.append(tok)
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=POSResponse,
    summary="Part-of-Speech Tagging",
    description=(
        "Tag each token in an Arabic sentence with its POS label. "
        "Supports MSA and dialectal Arabic (Egyptian, Gulf, Levantine, Maghrebi). "
        "Uses the QCRI tagset with compound clitic tags (NOUN+PRON, PREP+PRON, etc.)."
    ),
)
async def pos_tag(body: POSRequest, request: Request) -> POSResponse:
    t0 = time.perf_counter()

    if not is_arabic(body.text):
        raise HTTPException(
            status_code=422,
            detail="Input text does not appear to contain Arabic. "
                   "This endpoint processes Arabic text only.",
        )

    engine = getattr(request.app.state, "engine", None)

    try:
        if engine is not None and hasattr(engine, "predict_pos"):
            tokens = _engine_pos(engine, body.text,
                                  body.include_morphology,
                                  body.include_direction)
        else:
            # Stub mode — development without trained model
            tokens = _build_stub_pos(body.text,
                                     body.include_morphology,
                                     body.include_direction)
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"POS inference failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000

    return POSResponse(
        text=body.text,
        tokens=tokens,
        n_tokens=len(tokens),
        arabic_ratio=arabic_ratio(body.text),
        direction=get_text_direction(body.text),
        latency_ms=round(latency_ms, 2),
    )


@router.post(
    "/batch",
    response_model=POSBatchResponse,
    summary="Batch POS Tagging",
    description="Tag multiple Arabic texts in a single request (max 32).",
)
async def pos_tag_batch(
    body: POSBatchRequest, request: Request
) -> POSBatchResponse:
    t0     = time.perf_counter()
    engine = getattr(request.app.state, "engine", None)
    results: list[POSResponse] = []

    for text in body.texts:
        if not is_arabic(text):
            results.append(POSResponse(
                text=text, tokens=[], n_tokens=0,
                arabic_ratio=0.0, direction="ltr", latency_ms=0.0,
            ))
            continue

        try:
            if engine and hasattr(engine, "predict_pos"):
                tokens = _engine_pos(engine, text,
                                     body.include_morphology, False)
            else:
                tokens = _build_stub_pos(text, body.include_morphology, False)
        except Exception:
            tokens = []

        results.append(POSResponse(
            text=text,
            tokens=tokens,
            n_tokens=len(tokens),
            arabic_ratio=arabic_ratio(text),
            direction=get_text_direction(text),
            latency_ms=0.0,
        ))

    total_ms = (time.perf_counter() - t0) * 1000
    return POSBatchResponse(
        results=results,
        n_texts=len(results),
        latency_ms=round(total_ms, 2),
    )


@router.get(
    "/tags",
    response_model=POSTagsResponse,
    summary="List POS Tags",
    description="Return all supported POS tags with their morphological categories.",
)
async def list_pos_tags() -> POSTagsResponse:
    categories: dict[str, list[str]] = {}
    for tag in POS_LABELS:
        cat = get_morph_category(tag)
        categories.setdefault(cat, []).append(tag)

    return POSTagsResponse(
        tags=POS_LABELS,
        n_tags=len(POS_LABELS),
        categories=categories,
    )


@router.post(
    "/validate",
    response_model=POSValidateResponse,
    summary="Validate POS Sequence",
    description=(
        "Validate that a POS tag sequence is consistent with the supported tagset. "
        "Useful for checking annotation quality before training."
    ),
)
async def validate_pos_sequence(body: POSValidateRequest) -> POSValidateResponse:
    issues: list[str] = []

    if len(body.tokens) != len(body.pos_tags):
        issues.append(
            f"Length mismatch: {len(body.tokens)} tokens "
            f"vs {len(body.pos_tags)} tags"
        )

    unknown_tags = [t for t in body.pos_tags if t not in LABEL2ID]
    if unknown_tags:
        issues.append(f"Unknown tags: {list(set(unknown_tags))}")

    known = sum(1 for t in body.pos_tags if t in LABEL2ID)
    coverage = known / len(body.pos_tags) if body.pos_tags else 1.0

    return POSValidateResponse(
        valid=len(issues) == 0,
        n_tokens=len(body.tokens),
        n_tags=len(body.pos_tags),
        issues=issues,
        coverage=round(coverage, 4),
    )
