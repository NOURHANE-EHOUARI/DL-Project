"""
arabic_utils.py — Arabic Script Utilities & RTL Helpers
Owner:   Student B (Hiba)
Phase:   2 — Preprocessing & Infrastructure (Week 2–3)

Utilities for:
  - Arabic text detection and validation
  - RTL/LTR text direction handling
  - Arabic numeral conversion
  - Token-level Arabic analysis
  - IOB2 sequence utilities
  - Text statistics for EDA
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Optional

# ── Unicode ranges ─────────────────────────────────────────────────────────────
ARABIC_BLOCK        = range(0x0600, 0x0700)
ARABIC_SUPPLEMENT   = range(0x0750, 0x0780)
ARABIC_EXTENDED_A   = range(0x08A0, 0x08FF)
ARABIC_PRESENTATION_A = range(0xFB50, 0xFE00)
ARABIC_PRESENTATION_B = range(0xFE70, 0xFF00)

# ── Regex patterns ─────────────────────────────────────────────────────────────
RE_ARABIC         = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')
RE_ARABIC_WORD    = re.compile(r'^[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+$')
RE_DIACRITICS     = re.compile(r'[\u064B-\u065F\u0670]')
RE_TATWEEL        = re.compile(r'\u0640+')
RE_ARABIC_PUNCT   = re.compile(r'[\u060C\u061B\u061F\u06D4]')
RE_ARABIC_DIGITS  = re.compile(r'[\u0660-\u0669]')
RE_LATIN          = re.compile(r'[a-zA-Z]')
RE_DIGITS         = re.compile(r'\d')

# ── Arabic-Indic digit map ─────────────────────────────────────────────────────
ARABIC_INDIC_TO_WESTERN = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
WESTERN_TO_ARABIC_INDIC = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')

# ── Common Arabic stopwords ────────────────────────────────────────────────────
ARABIC_STOPWORDS = frozenset({
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
    'التي', 'الذي', 'الذين', 'اللواتي', 'وقد', 'كان', 'كانت', 'يكون',
    'أن', 'إن', 'لأن', 'حتى', 'إذا', 'لكن', 'أو', 'و', 'ف', 'ب', 'ل',
    'ك', 'م', 'لم', 'لن', 'قد', 'ما', 'لا', 'هو', 'هي', 'هم', 'هن',
    'أنا', 'نحن', 'أنت', 'أنتم', 'أنتن', 'هؤلاء', 'أولئك',
    'كل', 'بعض', 'غير', 'حين', 'بعد', 'قبل', 'فوق', 'تحت', 'بين',
})


# ══════════════════════════════════════════════════════════════════════════════
# Text detection & validation
# ══════════════════════════════════════════════════════════════════════════════

def is_arabic(text: str) -> bool:
    """Return True if text contains any Arabic characters."""
    return bool(RE_ARABIC.search(text))


def is_arabic_word(token: str) -> bool:
    """Return True if token consists entirely of Arabic characters."""
    return bool(RE_ARABIC_WORD.match(token)) if token else False


def arabic_ratio(text: str) -> float:
    """
    Fraction of characters in text that are Arabic script.
    Useful for dialect detection and filtering mixed-script text.

    Returns:
        Float in [0.0, 1.0]. 1.0 = fully Arabic, 0.0 = no Arabic.
    """
    if not text:
        return 0.0
    arabic_chars = len(RE_ARABIC.findall(text))
    total_chars  = len([c for c in text if not c.isspace()])
    return arabic_chars / total_chars if total_chars > 0 else 0.0


def is_mixed_script(text: str, threshold: float = 0.1) -> bool:
    """
    Return True if text contains significant amounts of both
    Arabic and Latin characters.

    Used to flag code-switched text for special handling.

    Args:
        text:      Input string.
        threshold: Minimum fraction for a script to be 'significant'.
    """
    if not text:
        return False
    total = len([c for c in text if not c.isspace()])
    if total == 0:
        return False
    ar_ratio  = len(RE_ARABIC.findall(text)) / total
    lat_ratio = len(RE_LATIN.findall(text))  / total
    return ar_ratio >= threshold and lat_ratio >= threshold


def has_diacritics(text: str) -> bool:
    """Return True if text contains Arabic diacritical marks (tashkeel)."""
    return bool(RE_DIACRITICS.search(text))


def is_stopword(token: str) -> bool:
    """Return True if token is a common Arabic stopword."""
    return token in ARABIC_STOPWORDS


# ══════════════════════════════════════════════════════════════════════════════
# Numeral conversion
# ══════════════════════════════════════════════════════════════════════════════

def arabic_indic_to_western(text: str) -> str:
    """Convert Arabic-Indic digits (٠١٢٣) to Western digits (0123)."""
    return text.translate(ARABIC_INDIC_TO_WESTERN)


def western_to_arabic_indic(text: str) -> str:
    """Convert Western digits (0123) to Arabic-Indic digits (٠١٢٣)."""
    return text.translate(WESTERN_TO_ARABIC_INDIC)


# ══════════════════════════════════════════════════════════════════════════════
# RTL / LTR helpers
# ══════════════════════════════════════════════════════════════════════════════

# Unicode bidirectional control characters
RTL_MARK  = '\u200F'   # Right-to-Left Mark
LTR_MARK  = '\u200E'   # Left-to-Right Mark
RTL_EMBED = '\u202B'   # Right-to-Left Embedding
LTR_EMBED = '\u202A'   # Left-to-Right Embedding
PDF       = '\u202C'   # Pop Directional Formatting


def wrap_rtl(text: str) -> str:
    """
    Wrap text in RTL embedding markers.
    Use this when displaying Arabic text in a mixed-direction context
    (e.g., inside a JSON response that will be rendered in a React UI).
    """
    return f"{RTL_EMBED}{text}{PDF}"


def get_text_direction(text: str) -> str:
    """
    Detect the primary text direction.

    Returns:
        'rtl' if text is primarily Arabic/Hebrew,
        'ltr' if primarily Latin,
        'neutral' if mixed or no strong direction.
    """
    ar_count  = len(RE_ARABIC.findall(text))
    lat_count = len(RE_LATIN.findall(text))

    if ar_count == 0 and lat_count == 0:
        return 'neutral'
    if ar_count > lat_count:
        return 'rtl'
    if lat_count > ar_count:
        return 'ltr'
    return 'neutral'


def add_rtl_markers_to_tokens(
    tokens: list[str],
) -> list[dict[str, str]]:
    """
    Annotate a list of tokens with their text direction.
    Used by the React frontend to apply correct CSS direction per token.

    Returns:
        List of {"token": str, "direction": "rtl"|"ltr"|"neutral"}
    """
    return [
        {"token": tok, "direction": get_text_direction(tok)}
        for tok in tokens
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Token-level analysis
# ══════════════════════════════════════════════════════════════════════════════

def token_type(token: str) -> str:
    """
    Classify a token into a broad category.
    Useful for error analysis and morphological feature extraction.

    Returns one of:
        'arabic_word', 'digit', 'arabic_digit', 'punctuation',
        'latin', 'mixed', 'whitespace', 'other'
    """
    if not token or token.isspace():
        return 'whitespace'
    if RE_ARABIC_DIGITS.fullmatch(token):
        return 'arabic_digit'
    if token.isdigit():
        return 'digit'
    if RE_ARABIC_WORD.fullmatch(token):
        return 'arabic_word'
    if token.isalpha() and RE_LATIN.fullmatch(token):
        return 'latin'
    if RE_ARABIC_PUNCT.fullmatch(token) or token in '.,!?;:()[]{}"\'-':
        return 'punctuation'
    if is_mixed_script(token):
        return 'mixed'
    return 'other'


def count_arabic_chars(text: str) -> dict[str, int]:
    """
    Count character types in Arabic text.
    Useful for EDA and data quality reports.

    Returns:
        Dict with keys: arabic, latin, digit, punctuation,
                        diacritic, tatweel, whitespace, other
    """
    counts: dict[str, int] = {
        'arabic':     0,
        'latin':      0,
        'digit':      0,
        'punctuation':0,
        'diacritic':  0,
        'tatweel':    0,
        'whitespace': 0,
        'other':      0,
    }
    for ch in text:
        cp = ord(ch)
        if ch == '\u0640':
            counts['tatweel']  += 1
        elif '\u064B' <= ch <= '\u065F' or ch == '\u0670':
            counts['diacritic'] += 1
        elif cp in ARABIC_BLOCK or cp in ARABIC_SUPPLEMENT:
            counts['arabic']   += 1
        elif ch.isalpha() and ch.isascii():
            counts['latin']    += 1
        elif ch.isdigit():
            counts['digit']    += 1
        elif ch in '.,!?;:،؟؛،()[]{}"\'-':
            counts['punctuation'] += 1
        elif ch.isspace():
            counts['whitespace'] += 1
        else:
            counts['other']    += 1
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# Sentence & corpus statistics
# ══════════════════════════════════════════════════════════════════════════════

def sentence_stats(sentences: list[list[str]]) -> dict:
    """
    Compute corpus-level statistics from a list of tokenized sentences.
    Used in the EDA notebook and the dataset_manifest.

    Args:
        sentences: List of sentences, each a list of token strings.

    Returns:
        Dict with: total_sentences, total_tokens, vocab_size,
                   avg_length, max_length, min_length,
                   arabic_token_ratio, oov_rate (if vocab provided).
    """
    if not sentences:
        return {}

    lengths      = [len(s) for s in sentences]
    all_tokens   = [t for s in sentences for t in s]
    vocab        = set(all_tokens)
    arabic_count = sum(1 for t in all_tokens if is_arabic_word(t))

    return {
        'total_sentences':  len(sentences),
        'total_tokens':     len(all_tokens),
        'vocab_size':       len(vocab),
        'avg_length':       round(sum(lengths) / len(lengths), 2),
        'max_length':       max(lengths),
        'min_length':       min(lengths),
        'arabic_token_ratio': round(arabic_count / len(all_tokens), 4)
                              if all_tokens else 0.0,
        'top_10_tokens':    [t for t, _ in Counter(all_tokens).most_common(10)],
        'stopword_ratio':   round(
            sum(1 for t in all_tokens if is_stopword(t)) / len(all_tokens), 4
        ) if all_tokens else 0.0,
    }


def ner_corpus_stats(sentences: list[list[str]], labels: list[list[str]]) -> dict:
    """
    Compute NER-specific corpus statistics.
    Used for EDA and the jury presentation's dataset slide.

    Args:
        sentences: Tokenized sentences.
        labels:    Corresponding IOB2 label sequences.

    Returns:
        Dict with entity counts, type distribution, and class imbalance ratio.
    """
    entity_counts: Counter = Counter()
    total_tokens  = 0
    entity_tokens = 0

    for sent, labs in zip(sentences, labels):
        total_tokens += len(sent)
        for tag in labs:
            if tag.startswith('B-'):
                entity_counts[tag[2:]] += 1
            if tag != 'O':
                entity_tokens += 1

    total_entities = sum(entity_counts.values())
    imbalance      = (total_tokens - entity_tokens) / entity_tokens \
                     if entity_tokens > 0 else float('inf')

    return {
        'total_tokens':    total_tokens,
        'total_entities':  total_entities,
        'entity_density':  round(entity_tokens / total_tokens, 4)
                           if total_tokens > 0 else 0.0,
        'class_imbalance_ratio': round(imbalance, 2),
        'entity_type_counts': dict(entity_counts.most_common()),
        'most_common_entity': entity_counts.most_common(1)[0][0]
                              if entity_counts else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# IOB2 utilities
# ══════════════════════════════════════════════════════════════════════════════

def validate_iob2(tags: list[str]) -> tuple[bool, list[str]]:
    """
    Validate and optionally fix an IOB2 tag sequence.

    Fixes:
      - I-TYPE without preceding B-TYPE → converted to B-TYPE

    Args:
        tags: List of IOB2 tag strings.

    Returns:
        (is_valid, fixed_tags) where is_valid is True if no fixes needed.
    """
    fixed  = []
    errors = False
    prev   = None

    for tag in tags:
        if tag.startswith('I-'):
            etype = tag[2:]
            if prev != f'B-{etype}' and prev != f'I-{etype}':
                fixed.append(f'B-{etype}')
                errors = True
            else:
                fixed.append(tag)
        else:
            fixed.append(tag)
        prev = tag

    return not errors, fixed


def iob2_to_spans(
    tokens: list[str],
    tags:   list[str],
) -> list[dict]:
    """
    Convert IOB2 tag sequence to entity span dicts.

    Args:
        tokens: Word strings.
        tags:   IOB2 tags aligned to tokens.

    Returns:
        List of {"type", "start", "end", "text"} dicts.

    Example:
        tokens = ["محمد", "صلاح", "يلعب", "في", "ليفربول"]
        tags   = ["B-PER", "I-PER", "O", "O", "B-ORG"]
        →  [
            {"type": "PER", "start": 0, "end": 1, "text": "محمد صلاح"},
            {"type": "ORG", "start": 4, "end": 4, "text": "ليفربول"},
           ]
    """
    spans: list[dict] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith('B-'):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(tags) and tags[i] == f'I-{etype}':
                i += 1
            end  = i - 1
            text = ' '.join(tokens[start:end + 1])
            spans.append({
                'type':  etype,
                'start': start,
                'end':   end,
                'text':  text,
            })
        else:
            i += 1
    return spans


def spans_to_iob2(
    n_tokens: int,
    spans:    list[dict],
) -> list[str]:
    """
    Convert entity spans back to IOB2 tag sequence.
    Inverse of iob2_to_spans().

    Args:
        n_tokens: Total number of tokens in the sentence.
        spans:    List of {"type", "start", "end"} dicts.

    Returns:
        IOB2 tag list of length n_tokens.
    """
    tags = ['O'] * n_tokens
    for span in sorted(spans, key=lambda s: s['start']):
        start, end, etype = span['start'], span['end'], span['type']
        if start < n_tokens:
            tags[start] = f'B-{etype}'
        for i in range(start + 1, min(end + 1, n_tokens)):
            tags[i] = f'I-{etype}'
    return tags


# ══════════════════════════════════════════════════════════════════════════════
# Frontend helpers (used by React API responses)
# ══════════════════════════════════════════════════════════════════════════════

# Color map for NER entity types — matches the React frontend
NER_COLORS = {
    'PER':     '#4A90D9',   # blue
    'ORG':     '#27AE60',   # green
    'LOC':     '#E07B54',   # coral
    'GPE':     '#E07B54',   # coral (same as LOC)
    'DATE':    '#9B59B6',   # purple
    'TIME':    '#9B59B6',   # purple
    'MONEY':   '#F39C12',   # amber
    'PERCENT': '#F39C12',   # amber
    'MISC':    '#7F8C8D',   # gray
    'DEFAULT': '#95A5A6',   # light gray
}


def get_entity_color(entity_type: str) -> str:
    """
    Return the hex color for an entity type.
    Colors match the React annotation visualizer in the frontend.
    """
    return NER_COLORS.get(entity_type.upper(), NER_COLORS['DEFAULT'])


def annotate_text_html(
    text:   str,
    spans:  list[dict],
) -> str:
    """
    Wrap entity spans in HTML <mark> tags with data attributes.
    Used by the API's /analyze endpoint for HTML rendering mode.

    Args:
        text:  Original Arabic text (space-separated words).
        spans: List of {"type", "start", "end", "text"} dicts.

    Returns:
        HTML string with entity markup.

    Example output:
        '<mark data-type="PER" style="background:#4A90D9">محمد صلاح</mark>
         يلعب في
         <mark data-type="ORG" style="background:#27AE60">ليفربول</mark>'
    """
    tokens = text.split()
    result = []
    i      = 0

    # Build span lookup: token_idx → span
    span_starts = {s['start']: s for s in spans}

    while i < len(tokens):
        if i in span_starts:
            span  = span_starts[i]
            color = get_entity_color(span['type'])
            text_ = ' '.join(tokens[span['start']:span['end'] + 1])
            result.append(
                f'<mark data-type="{span["type"]}" '
                f'style="background-color:{color}22;'
                f'border-bottom:2px solid {color};'
                f'padding:0 2px;border-radius:3px;" '
                f'title="{span["type"]}">{text_}</mark>'
            )
            i = span['end'] + 1
        else:
            result.append(tokens[i])
            i += 1

    return ' '.join(result)


# ══════════════════════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Arabic Utils Sanity Check ===\n")

    # ── Text detection ────────────────────────────────────────────────────
    assert is_arabic("مرحبا")          == True
    assert is_arabic("Hello")          == False
    assert is_arabic_word("مرحبا")     == True
    assert is_arabic_word("Hello")     == False
    assert is_arabic_word("مرحباHello") == False
    assert has_diacritics("مَرْحَبَا")  == True
    assert has_diacritics("مرحبا")     == False
    assert is_stopword("في")           == True
    assert is_stopword("مرحبا")        == False
    print("✓ Text detection")

    # ── Arabic ratio ──────────────────────────────────────────────────────
    ratio = arabic_ratio("مرحبا Hello")
    assert 0.4 < ratio < 0.7, f"Expected ~0.5, got {ratio}"
    assert arabic_ratio("مرحبا") == 1.0
    assert arabic_ratio("Hello") == 0.0
    print(f"✓ Arabic ratio: 'مرحبا Hello' = {ratio:.2f}")

    # ── Numeral conversion ────────────────────────────────────────────────
    assert arabic_indic_to_western("٢٠٢٤") == "2024"
    assert western_to_arabic_indic("2024") == "٢٠٢٤"
    print("✓ Numeral conversion")

    # ── RTL helpers ───────────────────────────────────────────────────────
    assert get_text_direction("مرحبا") == "rtl"
    assert get_text_direction("Hello") == "ltr"
    annotated = add_rtl_markers_to_tokens(["مرحبا", "Hello"])
    assert annotated[0]["direction"] == "rtl"
    assert annotated[1]["direction"] == "ltr"
    print("✓ RTL helpers")

    # ── IOB2 utilities ────────────────────────────────────────────────────
    tokens = ["محمد", "صلاح", "يلعب", "في", "ليفربول"]
    tags   = ["B-PER", "I-PER", "O", "O", "B-ORG"]
    spans  = iob2_to_spans(tokens, tags)
    assert len(spans) == 2
    assert spans[0] == {"type": "PER", "start": 0, "end": 1, "text": "محمد صلاح"}
    assert spans[1] == {"type": "ORG", "start": 4, "end": 4, "text": "ليفربول"}
    print(f"✓ iob2_to_spans: {spans}")

    # Round-trip: spans → IOB2 → spans
    recovered_tags = spans_to_iob2(len(tokens), spans)
    assert recovered_tags == tags
    print(f"✓ spans_to_iob2 round-trip: {recovered_tags}")

    # Fix broken IOB2
    broken = ["O", "I-PER", "I-PER", "O"]
    valid, fixed = validate_iob2(broken)
    assert not valid
    assert fixed == ["O", "B-PER", "I-PER", "O"]
    print(f"✓ validate_iob2: {broken} → {fixed}")

    # ── Corpus stats ──────────────────────────────────────────────────────
    sents = [["محمد", "يعمل", "في", "القاهرة"],
             ["شركة", "أرامكو", "أعلنت", "النتائج"]]
    stats = sentence_stats(sents)
    assert stats["total_sentences"] == 2
    assert stats["total_tokens"]    == 8
    assert stats["vocab_size"]      == 8
    print(f"✓ sentence_stats: {stats['total_tokens']} tokens, "
          f"vocab={stats['vocab_size']}")

    # ── NER stats ─────────────────────────────────────────────────────────
    lab = [["B-PER","O","O","B-LOC"], ["B-ORG","I-ORG","O","O"]]
    ner = ner_corpus_stats(sents, lab)
    assert ner["total_entities"] == 3
    print(f"✓ ner_corpus_stats: {ner['total_entities']} entities, "
          f"density={ner['entity_density']}")

    # ── HTML annotation ───────────────────────────────────────────────────
    html = annotate_text_html(
        "محمد صلاح يلعب في ليفربول",
        [{"type": "PER", "start": 0, "end": 1, "text": "محمد صلاح"},
         {"type": "ORG", "start": 4, "end": 4, "text": "ليفربول"}],
    )
    assert '<mark data-type="PER"' in html
    assert '<mark data-type="ORG"' in html
    print(f"✓ annotate_text_html: {html[:80]}...")

    # ── Entity color ──────────────────────────────────────────────────────
    assert get_entity_color("PER")  == "#4A90D9"
    assert get_entity_color("ORG")  == "#27AE60"
    assert get_entity_color("UNKNOWN") == "#95A5A6"
    print("✓ Entity colors")

    print("\n✅ All checks passed — src/utils/arabic_utils.py ready")
