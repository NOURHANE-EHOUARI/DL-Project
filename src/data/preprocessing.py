"""
preprocessing.py — Arabic Text Normalization & Data Preparation Pipeline
Owner: Student B (Hiba)
Phase: 2 — Preprocessing & Infrastructure (Week 2–3)

Responsibilities:
  - Unicode normalization for Arabic text
  - Hamza normalization (أ إ آ ا → ا)
  - Diacritic (tashkeel) removal
  - Tatweel removal
  - Punctuation normalization
  - Converting CoNLL files → JSONL format for datasets.py
  - Morphological feature extraction using pyarabic

Output format (JSONL) matches ArabicNERDataset / ArabicPOSDataset:
  NER : {"tokens": [...], "ner_tags": [...]}
  POS : {"tokens": [...], "pos_tags": [...], "segments": [...]}
  Coref: {"tokens": [...], "clusters": [...]}
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional
import pyarabic.araby as araby

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# ── Arabic Unicode ranges ──────────────────────────────────────────────────────
ARABIC_LETTERS   = "\u0600-\u06FF"
ARABIC_EXTENDED  = "\u0750-\u077F"
ARABIC_SUPPL     = "\uFB50-\uFDFF"
ARABIC_COMPAT    = "\uFE70-\uFEFF"

# ── Normalization maps ─────────────────────────────────────────────────────────
# Hamza variants → bare alef
HAMZA_MAP = {
    "\u0623": "\u0627",  # أ → ا
    "\u0625": "\u0627",  # إ → ا
    "\u0622": "\u0627",  # آ → ا
    "\u0671": "\u0627",  # ٱ → ا (wasla)
    "\u0624": "\u0648",  # ؤ → و
    "\u0626": "\u064A",  # ئ → ي
}

# Ta marbuta → ha
TA_MARBUTA_MAP = {
    "\u0629": "\u0647",  # ة → ه
}

# Alef maqsura → ya
ALEF_MAQSURA_MAP = {
    "\u0649": "\u064A",  # ى → ي
}

# Arabic punctuation → standard
PUNCT_MAP = {
    "\u060C": ",",   # ، → ,
    "\u061B": ";",   # ؛ → ;
    "\u061F": "?",   # ؟ → ?
    "\u0660": "0", "\u0661": "1", "\u0662": "2",  # Arabic-Indic digits
    "\u0663": "3", "\u0664": "4", "\u0665": "5",
    "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
}

# Diacritics (tashkeel) to remove
DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

# Tatweel (elongation)
TATWEEL = re.compile(r"\u0640+")

# Non-Arabic, non-space characters (for aggressive cleaning)
NON_ARABIC = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\w\d\-\.\،\؟\!\،]"
)


# ══════════════════════════════════════════════════════════════════════════════
# Core normalization functions
# ══════════════════════════════════════════════════════════════════════════════

def remove_diacritics(text: str) -> str:
    """Remove all Arabic diacritical marks (tashkeel/harakat)."""
    return DIACRITICS.sub("", text)

def remove_tatweel(text: str) -> str:
    """Remove Arabic tatweel (elongation character ـ)."""
    return TATWEEL.sub("", text)

def normalize_hamza(text: str) -> str:
    """Normalize all Hamza variants to bare alef."""
    for src, tgt in HAMZA_MAP.items():
        text = text.replace(src, tgt)
    return text

def normalize_ta_marbuta(text: str) -> str:
    """Normalize ta marbuta (ة) to ha (ه)."""
    for src, tgt in TA_MARBUTA_MAP.items():
        text = text.replace(src, tgt)
    return text

def normalize_alef_maqsura(text: str) -> str:
    """Normalize alef maqsura (ى) to ya (ي)."""
    for src, tgt in ALEF_MAQSURA_MAP.items():
        text = text.replace(src, tgt)
    return text

def normalize_unicode(text: str) -> str:
    """Apply Unicode NFC normalization."""
    return unicodedata.normalize("NFC", text)

def normalize_punctuation(text: str) -> str:
    """Normalize Arabic punctuation and digits to standard forms."""
    for src, tgt in PUNCT_MAP.items():
        text = text.replace(src, tgt)
    return text

def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces, strip leading/trailing whitespace."""
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Main normalization pipeline
# ══════════════════════════════════════════════════════════════════════════════

def normalize_arabic(
    text: str,
    remove_diacritics_flag: bool = True,
    remove_tatweel_flag: bool = True,
    normalize_hamza_flag: bool = True,
    normalize_ta_marbuta_flag: bool = False,
    normalize_alef_maqsura_flag: bool = False,
    normalize_punctuation_flag: bool = True,
) -> str:
    """
    Full Arabic normalization pipeline.

    Args:
        text: Raw Arabic text
        remove_diacritics_flag: Remove tashkeel (default True)
        remove_tatweel_flag: Remove elongation (default True)
        normalize_hamza_flag: Normalize hamza variants (default True)
        normalize_ta_marbuta_flag: Normalize ة→ه (default False — changes meaning)
        normalize_alef_maqsura_flag: Normalize ى→ي (default False — changes meaning)
        normalize_punctuation_flag: Normalize Arabic punctuation (default True)

    Returns:
        Normalized Arabic text

    Example:
        >>> normalize_arabic('الرَّئِيسُ الأَمْرِيكِيُّ')
        'الرييس الامريكي'
    """
    # Step 1: Unicode normalization
    text = normalize_unicode(text)

    # Step 2: Remove diacritics (tashkeel)
    if remove_diacritics_flag:
        text = remove_diacritics(text)

    # Step 3: Remove tatweel
    if remove_tatweel_flag:
        text = remove_tatweel(text)

    # Step 4: Normalize Hamza variants
    if normalize_hamza_flag:
        text = normalize_hamza(text)

    # Step 5: Normalize ta marbuta (optional)
    if normalize_ta_marbuta_flag:
        text = normalize_ta_marbuta(text)

    # Step 6: Normalize alef maqsura (optional)
    if normalize_alef_maqsura_flag:
        text = normalize_alef_maqsura(text)

    # Step 7: Normalize punctuation
    if normalize_punctuation_flag:
        text = normalize_punctuation(text)

    # Step 8: Normalize whitespace
    text = normalize_whitespace(text)

    return text


def normalize_token(token: str) -> str:
    """
    Normalize a single Arabic token.
    Same pipeline as normalize_arabic but designed for token-level use.
    Preserves the token even if it becomes empty after normalization.
    """
    if not token or not token.strip():
        return token
    normalized = normalize_arabic(token)
    # If normalization empties the token, return original
    return normalized if normalized else token


# ══════════════════════════════════════════════════════════════════════════════
# Morphological features (using pyarabic)
# ══════════════════════════════════════════════════════════════════════════════

def extract_morphological_features(token: str) -> dict:
    """
    Extract morphological features from an Arabic token using pyarabic.

    Returns a dict with:
        - is_arabicword: bool
        - has_prefix: bool
        - has_suffix: bool
        - token_length: int
        - is_stopword: bool
        - normalized: str

    These features are used by the POS and NER heads for
    morphology-aware feature injection (innovative feature #2).
    """
    features = {
        "is_arabicword":  araby.is_arabicword(token),
        "has_prefix":     araby.has_arabic_letters(token[:2]) if len(token) > 2 else False,
        "token_length":   len(token),
        "is_stopword":    token in araby.STOPWORDS if hasattr(araby, "STOPWORDS") else False,
        "normalized":     normalize_token(token),
        "is_digit":       token.isdigit(),
        "is_punct":       not araby.is_arabicword(token) and not token.isalnum(),
    }
    return features


# ══════════════════════════════════════════════════════════════════════════════
# CoNLL → JSONL converters
# ══════════════════════════════════════════════════════════════════════════════

def conll_ner_to_jsonl(
    input_path: Path,
    output_path: Path,
    token_col: int = 0,
    tag_col: int = 1,
    normalize: bool = True,
    include_morph: bool = False,
) -> int:
    """
    Convert a 2-column CoNLL NER file to JSONL format for ArabicNERDataset.

    Input format (CoNLL):
        محمد    B-PER
        يعمل    O
        <blank line = sentence boundary>

    Output format (JSONL):
        {"tokens": ["محمد", "يعمل"], "ner_tags": ["B-PER", "O"]}

    Args:
        input_path: Path to input CoNLL file
        output_path: Path to output JSONL file
        token_col: Column index for tokens (default 0)
        tag_col: Column index for NER tags (default 1)
        normalize: Apply Arabic normalization (default True)
        include_morph: Include morphological features (default False)

    Returns:
        Number of sentences written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sentences_written = 0
    current_tokens, current_tags = [], []

    with open(input_path, encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        def flush_sentence():
            nonlocal sentences_written
            if current_tokens:
                record = {
                    "tokens":   current_tokens[:],
                    "ner_tags": current_tags[:],
                }
                if include_morph:
                    record["morph_features"] = [
                        extract_morphological_features(t) for t in current_tokens
                    ]
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                sentences_written += 1
                current_tokens.clear()
                current_tags.clear()

        for line in fin:
            line = line.rstrip("\n")
            if line.strip() == "" or line.startswith("#"):
                flush_sentence()
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue
            token = parts[token_col]
            tag   = parts[tag_col]
            if normalize:
                token = normalize_token(token)
            current_tokens.append(token)
            current_tags.append(tag)

        flush_sentence()  # final sentence

    logger.info("NER CoNLL→JSONL: %d sentences → %s", sentences_written, output_path)
    return sentences_written


def conll_pos_to_jsonl(
    input_path: Path,
    output_path: Path,
    token_col: int = 0,
    pos_col: int = 1,
    seg_col: Optional[int] = 2,
    normalize: bool = True,
) -> int:
    """
    Convert a CoNLL POS file to JSONL format for ArabicPOSDataset.

    Input format (CoNLL, 2 or 3 columns):
        ليه    PART    ليه
        تحب    V       تحب

    Output format (JSONL):
        {"tokens": ["ليه", "تحب"], "pos_tags": ["PART", "V"],
         "segments": ["ليه", "تحب"]}

    Returns:
        Number of sentences written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sentences_written = 0
    current_tokens, current_tags, current_segs = [], [], []

    with open(input_path, encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        def flush_sentence():
            nonlocal sentences_written
            if current_tokens:
                record = {
                    "tokens":   current_tokens[:],
                    "pos_tags": current_tags[:],
                }
                if current_segs:
                    record["segments"] = current_segs[:]
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                sentences_written += 1
                current_tokens.clear()
                current_tags.clear()
                current_segs.clear()

        for line in fin:
            line = line.rstrip("\n")
            if line.strip() == "" or line.startswith("#"):
                flush_sentence()
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue
            token = parts[token_col]
            tag   = parts[pos_col]
            if normalize:
                token = normalize_token(token)
            current_tokens.append(token)
            current_tags.append(tag)
            if seg_col is not None and len(parts) > seg_col:
                current_segs.append(parts[seg_col])

        flush_sentence()

    logger.info("POS CoNLL→JSONL: %d sentences → %s", sentences_written, output_path)
    return sentences_written


def conll_coref_to_jsonl(
    input_path: Path,
    output_path: Path,
    normalize: bool = True,
) -> int:
    """
    Convert a CoNLL 2012 coreference file to JSONL format for
    ArabicCoreferenceDataset.

    Input: Standard CoNLL 2012 format with coreference column.
    Output format (JSONL):
        {
            "tokens": ["محمد", "يعمل", "هو"],
            "clusters": [[[0, 0], [2, 2]]]
        }

    Returns:
        Number of documents written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    docs_written = 0

    current_tokens  = []
    # cluster_id → list of [start, end] mention spans
    open_mentions: dict[str, int] = {}   # cluster_id → start token idx
    clusters: dict[str, list] = {}       # cluster_id → list of [start, end]

    def flush_document():
        nonlocal docs_written
        if current_tokens:
            cluster_list = [spans for spans in clusters.values() if len(spans) >= 2]
            record = {
                "tokens":   current_tokens[:],
                "clusters": cluster_list,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            docs_written += 1

    with open(input_path, encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.rstrip("\n").strip()

            if line.startswith("#begin document"):
                current_tokens = []
                open_mentions  = {}
                clusters       = {}
                continue

            if line.startswith("#end document"):
                flush_document()
                current_tokens = []
                open_mentions  = {}
                clusters       = {}
                continue

            if line == "" or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            # Token is column 3 (0-indexed) in CoNLL 2012
            token_idx = len(current_tokens)
            if len(parts) >= 4:
                token = parts[3]
            else:
                token = parts[0]

            if normalize:
                token = normalize_token(token)
            current_tokens.append(token)

            # Parse coreference column (last column)
            coref_col = parts[-1]
            if coref_col != "-":
                # Handle patterns like (1, (1|(2, 1), (1)
                segments = coref_col.split("|")
                for seg in segments:
                    # Opening: (1 or (1)
                    opens = re.findall(r"\((\d+)", seg)
                    for cid in opens:
                        open_mentions[cid] = token_idx
                        if cid not in clusters:
                            clusters[cid] = []
                    # Closing: 1) or (1)
                    closes = re.findall(r"(\d+)\)", seg)
                    for cid in closes:
                        if cid in open_mentions:
                            start = open_mentions.pop(cid)
                            clusters.setdefault(cid, []).append([start, token_idx])

        # Handle file without #end document marker
        if current_tokens:
            flush_document()

    logger.info("Coref CoNLL→JSONL: %d documents → %s", docs_written, output_path)
    return docs_written


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline — convert all datasets
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    """
    Convert all raw CoNLL datasets to JSONL format ready for datasets.py.
    Run this script directly to prepare all data splits.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)

    sep = "─" * 62
    print(f"\n{'═'*62}")
    print("  Arabic NLP — Preprocessing Pipeline")
    print("  Student B: Hiba El Ouazi")
    print(f"{'═'*62}")

    # ── 1. ANERcorp NER ───────────────────────────────────────────
    print(f"\n{sep}\n  1. ANERcorp → NER JSONL\n{sep}")
    anercorp_dir = RAW / "anercorp"
    for split in ["train", "dev", "test"]:
        src = anercorp_dir / f"anercorp_{split}.conll"
        dst = PROCESSED / f"ner_anercorp_{split}.jsonl"
        if src.exists() and src.stat().st_size > 10:
            n = conll_ner_to_jsonl(src, dst, normalize=True)
            print(f"  ✓ {split:5s}: {n:,} sentences → {dst.name}")
        else:
            print(f"  ⚠ {split:5s}: {src.name} not found or empty — skipping")

    # ── 2. AQMAR NER ──────────────────────────────────────────────
    print(f"\n{sep}\n  2. AQMAR NER → JSONL\n{sep}")
    patb_dir = RAW / "patb"
    for split in ["train", "dev", "test"]:
        src = patb_dir / f"aqmar_ner_{split}.conll"
        dst = PROCESSED / f"ner_aqmar_{split}.jsonl"
        if src.exists() and src.stat().st_size > 10:
            n = conll_ner_to_jsonl(src, dst, normalize=True)
            print(f"  ✓ {split:5s}: {n:,} sentences → {dst.name}")
        else:
            print(f"  ⚠ {split:5s}: {src.name} not found — skipping")

    # ── 3. QCRI Dialect POS ───────────────────────────────────────
    print(f"\n{sep}\n  3. QCRI Arabic Dialect POS → JSONL\n{sep}")
    for split in ["train", "dev", "test"]:
        src = patb_dir / f"arabic_pos_dialect_{split}.conll"
        dst = PROCESSED / f"pos_dialect_{split}.jsonl"
        if src.exists() and src.stat().st_size > 10:
            n = conll_pos_to_jsonl(src, dst,
                                   token_col=0, pos_col=1, seg_col=2,
                                   normalize=True)
            print(f"  ✓ {split:5s}: {n:,} sentences → {dst.name}")
        else:
            print(f"  ⚠ {split:5s}: {src.name} not found — skipping")

    # ── 4. WikiCoref → Coreference JSONL ─────────────────────────
    print(f"\n{sep}\n  4. WikiCoref → Coreference JSONL\n{sep}")
    coref_dir = RAW / "wikicoref_ar"
    coref_files = list(coref_dir.glob("*.conll")) + list(coref_dir.glob("*_conll"))
    if coref_files:
        # Merge all conll files then split
        all_docs = []
        for cf in sorted(coref_files):
            dst_tmp = PROCESSED / f"coref_{cf.stem}.jsonl"
            n = conll_coref_to_jsonl(cf, dst_tmp, normalize=True)
            if dst_tmp.exists():
                with open(dst_tmp, encoding="utf-8") as f:
                    all_docs.extend(f.readlines())
                dst_tmp.unlink()  # remove temp file

        # Split into train/dev/test
        import random
        random.seed(42)
        random.shuffle(all_docs)
        n      = len(all_docs)
        tr, dv = int(n * 0.8), int(n * 0.1)
        splits = {"train": all_docs[:tr],
                  "dev":   all_docs[tr:tr+dv],
                  "test":  all_docs[tr+dv:]}
        for split_name, docs in splits.items():
            dst = PROCESSED / f"coref_{split_name}.jsonl"
            with open(dst, "w", encoding="utf-8") as f:
                f.writelines(docs)
            print(f"  ✓ {split_name:5s}: {len(docs):,} docs → {dst.name}")
    else:
        print("  ⚠ No CoNLL files found in wikicoref_ar/ — using placeholder")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{sep}\n  PROCESSED FILES\n{sep}")
    jsonl_files = sorted(PROCESSED.glob("*.jsonl"))
    total_size  = 0
    for jf in jsonl_files:
        size_kb = jf.stat().st_size / 1024
        total_size += size_kb
        with open(jf, encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        print(f"  ✓ {jf.name:<40} {lines:>6,} examples  {size_kb:>8.1f} KB")
    print(f"\n  Total: {len(jsonl_files)} files, {total_size:.0f} KB")
    print(f"\n{'═'*62}")
    print("  Done. data/processed/ is ready for datasets.py")
    print(f"{'═'*62}\n")


# ── Quick normalization demo ───────────────────────────────────────────────────
def demo():
    """Show normalization examples — useful for the jury presentation."""
    print("\n── Normalization Demo ──────────────────────────────────")
    examples = [
        "الرَّئِيسُ الأَمْرِيكِيُّ",   # with full tashkeel
        "مُحَمَّدٌ يَعْمَلُ فِي الْقَاهِرَةِ",
        "الاتِّحَادُ الأُورُوبِّيُّ",
        "إِنَّ اللَّهَ عَلَى كُلِّ شَيْءٍ قَدِيرٌ",
        "ـــالكتابـــ",                # with tatweel
        "أُسْتَاذٌ في الجَامِعَةِ",
    ]
    for text in examples:
        normalized = normalize_arabic(text)
        print(f"  Input : {text}")
        print(f"  Output: {normalized}")
        print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        run_pipeline()
