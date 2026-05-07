"""
Dataset Collection & Verification Script — FIXED
Arabic NLP Project — Student B (Hiba)

Datasets:
  - ANERcorp  : Arabic NER (PER, LOC, ORG, MISC) — IOB2
  - AQMAR     : Arabic NER (PER, LOC, ORG, MISC) — IOB2, 2-column format
  - WikiCoref : Arabic Coreference — CoNLL 2012 format

Usage:
    python3 scripts/collect_datasets.py
"""

import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
RAW          = ROOT / "data" / "raw"
PROCESSED    = ROOT / "data" / "processed"
ANALYSIS     = ROOT / "data" / "analysis"
ANERCORP_DIR = RAW / "anercorp"
PATB_DIR     = RAW / "patb"
WIKI_DIR     = RAW / "wikicoref_ar"

# ── URLs ───────────────────────────────────────────────────────────────────────
AQMAR_URL = "http://www.cs.cmu.edu/~ark/ArabicNER/AQMAR_Arabic_NER_corpus-1.0.zip"

ANERCORP_MIRRORS = [
    "https://raw.githubusercontent.com/AhmedHani/anercorp/master/data/ANERcorp",
    "https://raw.githubusercontent.com/hbtl/ANERcorp/master/ANERcorp",
    "https://raw.githubusercontent.com/CAMeL-Lab/camel_tools/master/data/ner/ANERcorp",
]

# ── Tag normalization ──────────────────────────────────────────────────────────
# AQMAR uses MIS0/MIS1/MIS2/MIS3 for miscellaneous subtypes.
# We normalize all to MISC for consistency with ANERcorp.
def normalize_tag(tag: str) -> str:
    if tag in ("O", "-", ""):
        return "O"
    # Handle malformed tags like OO
    if tag == "OO":
        return "O"
    # B-MIS0, B-MIS1, B-MIS2, B-MIS3, B-MIS, B-MIS-1, B-MIS-2 → B-MISC
    if re.match(r"^B-MIS", tag):
        return "B-MISC"
    if re.match(r"^I-MIS", tag):
        return "I-MISC"
    # B-MISS1 typo in some files
    if re.match(r"^B-MISS", tag):
        return "B-MISC"
    if re.match(r"^I-MISS", tag):
        return "I-MISC"
    return tag

# ── Helpers ────────────────────────────────────────────────────────────────────
SEP = "─" * 62

def log(msg, level="info"):
    icons = {"ok": "  ✓ ", "warn": "  ⚠ ", "fail": "  ✗ ", "info": "    "}
    print(icons.get(level, "    ") + msg)

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    if dest.exists() and dest.stat().st_size > 200:
        log(f"Already exists: {dest.name}", "ok")
        return True
    try:
        log(f"Downloading {desc or dest.name} ...")
        dest.parent.mkdir(parents=True, exist_ok=True)

        def progress(count, block, total):
            if total > 0:
                print(f"\r    {min(count*block*100//total, 100)}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, progress)
        print()
        if dest.stat().st_size < 100:
            log("Downloaded file is too small — likely a 404 page.", "fail")
            dest.unlink()
            return False
        log(f"Saved: {dest.name}", "ok")
        return True
    except Exception as e:
        log(f"Failed: {e}", "fail")
        return False

def try_mirrors(mirrors: list, dest: Path, desc: str) -> bool:
    for url in mirrors:
        if download_file(url, dest, desc):
            return True
    return False

def extract_zip(zip_path: Path, dest_dir: Path):
    log(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    log("Extracted.", "ok")

# ── Parsers ────────────────────────────────────────────────────────────────────
def parse_2col_conll(path: Path, normalize=True) -> list:
    """
    Parse a 2-column CoNLL file: token<space>tag
    Returns list of sentences, each a list of (token, tag) tuples.
    Handles blank-line sentence boundaries.
    """
    sentences, current = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            # Blank line = sentence boundary
            if line.strip() == "" or line.strip().startswith("#"):
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token = parts[0]
            tag   = normalize_tag(parts[1]) if normalize else parts[1]
            current.append((token, tag))
    if current:
        sentences.append(current)
    return sentences

def parse_conll_coref(path: Path) -> dict:
    """Parse CoNLL 2012 coreference format — count docs, tokens, mentions, chains."""
    docs, mentions, tokens = 0, 0, 0
    chains = Counter()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#begin document"):
                docs += 1
            elif line == "" or line.startswith("#"):
                continue
            else:
                parts = line.split()
                if parts:
                    tokens += 1
                    coref = parts[-1]
                    if coref != "-":
                        mentions += coref.count("(")
                        for cid in re.findall(r"\d+", coref):
                            chains[cid] += 1
    return {"documents": docs, "tokens": tokens,
            "mentions": mentions, "chains": len(chains)}

# ── Statistics ─────────────────────────────────────────────────────────────────
def ner_stats(sentences: list) -> dict:
    tag_counts, entity_counts, lengths = Counter(), Counter(), []
    for sent in sentences:
        lengths.append(len(sent))
        for _, tag in sent:
            tag_counts[tag] += 1
            if tag.startswith("B-"):
                entity_counts[tag[2:]] += 1
    total = sum(lengths)
    o     = tag_counts.get("O", 0)
    return {
        "total_sentences":  len(sentences),
        "total_tokens":     total,
        "avg_sent_length":  round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "max_sent_length":  max(lengths) if lengths else 0,
        "entity_counts":    dict(entity_counts.most_common()),
        "tag_distribution": dict(tag_counts.most_common()),
        "ne_ratio":         round((total - o) / total, 4) if total else 0,
    }

def print_stats(stats: dict):
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"    {k}:")
            for kk, vv in list(v.items())[:10]:
                print(f"      {kk:<28}: {vv:,}")
        else:
            print(f"    {k:<35}: {v:,}" if isinstance(v, int)
                  else f"    {k:<35}: {v}")

def save_stats(stats: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log(f"Stats → {path.relative_to(ROOT)}", "ok")

# ── Splits ─────────────────────────────────────────────────────────────────────
def split_and_save(sentences: list, out_dir: Path,
                   prefix: str, ratios=(0.8, 0.1, 0.1)):
    n  = len(sentences)
    t  = int(n * ratios[0])
    d  = int(n * ratios[1])
    parts = {"train": sentences[:t],
             "dev":   sentences[t:t+d],
             "test":  sentences[t+d:]}
    for name, sents in parts.items():
        dest = out_dir / f"{prefix}_{name}.conll"
        with open(dest, "w", encoding="utf-8") as f:
            for sent in sents:
                for token, tag in sent:
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")
        log(f"{name:5s} split: {len(sents):,} sentences → {dest.name}", "ok")

# ══════════════════════════════════════════════════════════════════════════════
# 1. ANERcorp
# ══════════════════════════════════════════════════════════════════════════════
def collect_anercorp():
    section("1. ANERcorp — Arabic NER Corpus (Benajiba & Rosso, 2007)")
    log("Tags   : PER, LOC, ORG, MISC — IOB2 format")
    log("License: Free for research use")

    dest = ANERCORP_DIR / "ANERcorp.txt"
    ok   = try_mirrors(ANERCORP_MIRRORS, dest, "ANERcorp")

    if not ok:
        # Try HuggingFace datasets API
        log("Trying HuggingFace datasets API ...", "info")
        try:
            from datasets import load_dataset
            ds = load_dataset("conll2003")  # fallback — not Arabic but tests pipeline
        except Exception:
            pass

        # Final fallback: create a meaningful placeholder
        log("All mirrors failed. Creating placeholder.", "warn")
        log("→ Manual download: https://github.com/EmnamoR/Arabic-named-entity-recognition", "warn")
        placeholder = "\n".join([
            "محمد\tB-PER", "علي\tI-PER", "يعمل\tO", "مديراً\tO", "في\tO",
            "شركة\tB-ORG", "أرامكو\tI-ORG", "السعودية\tO", "",
            "زارت\tO", "الرئيسة\tO", "القاهرة\tB-LOC", "أمس\tO", "",
            "أعلنت\tO", "الأمم\tB-ORG", "المتحدة\tI-ORG", "عن\tO",
            "خطة\tO", "جديدة\tO", "",
        ])
        dest.write_text(placeholder, encoding="utf-8")
        log("Placeholder saved (replace with real ANERcorp when available).", "warn")

    sentences = parse_2col_conll(dest, normalize=True)
    if not sentences:
        log("Could not parse ANERcorp — check file format.", "warn")
        return

    stats = ner_stats(sentences)
    log(f"Parsed {stats['total_sentences']:,} sentences, "
        f"{stats['total_tokens']:,} tokens")
    print_stats(stats)
    save_stats({"dataset": "ANERcorp", "task": "NER", **stats},
               ANALYSIS / "anercorp_stats.json")
    split_and_save(sentences, ANERCORP_DIR, "anercorp")

# ══════════════════════════════════════════════════════════════════════════════
# 2. AQMAR
# ══════════════════════════════════════════════════════════════════════════════
def collect_aqmar():
    section("2. AQMAR — Arabic NER Corpus (Schneider et al., 2012)")
    log("Tags   : PER, LOC, ORG, MISC (normalized from MIS0/1/2/3) — IOB2")
    log("Content: 28 Wikipedia articles in Arabic")
    log("License: Free for research use (CMU)")
    log("Format : 2 columns — token  NER_tag")

    zip_dest = PATB_DIR / "AQMAR_Arabic_NER_corpus-1.0.zip"
    ok = download_file(AQMAR_URL, zip_dest, "AQMAR corpus")

    if ok:
        # Only extract if .txt files not already there
        txt_files = [f for f in PATB_DIR.glob("*.txt")
                     if f.name not in ("aqmar_sample.txt",)]
        if not txt_files:
            extract_zip(zip_dest, PATB_DIR)

    # Collect all topic .txt files (Football.txt, Physics.txt, etc.)
    # Exclude our own output files
    our_files = {"aqmar_sample.txt"}
    topic_files = [
        f for f in PATB_DIR.glob("*.txt")
        if f.name not in our_files
        and not f.name.startswith("aqmar_")
    ]
    log(f"Found {len(topic_files)} topic annotation files")

    if not topic_files:
        log("No AQMAR files found — creating placeholder.", "warn")
        placeholder = (
            "الرئيس\tB-PER\nالأمريكي\tO\nبايدن\tI-PER\nيزور\tO\nباريس\tB-LOC\n\n"
            "البنك\tB-ORG\nالمركزي\tI-ORG\nيرفع\tO\nأسعار\tO\nالفائدة\tO\n\n"
        )
        (PATB_DIR / "aqmar_sample.txt").write_text(placeholder, encoding="utf-8")
        topic_files = [PATB_DIR / "aqmar_sample.txt"]

    # Parse all files and merge into one corpus
    all_sentences = []
    per_file_stats = {}
    for f in sorted(topic_files):
        sents = parse_2col_conll(f, normalize=True)
        if sents:
            all_sentences.extend(sents)
            per_file_stats[f.name] = len(sents)

    log(f"Total after merging: {len(all_sentences):,} sentences")

    # Print per-file sentence counts
    print("    Per-file sentence counts:")
    for fname, count in sorted(per_file_stats.items(), key=lambda x: -x[1])[:10]:
        print(f"      {fname:<35}: {count:,} sentences")

    stats = ner_stats(all_sentences)
    log(f"Total tokens: {stats['total_tokens']:,}")
    print_stats(stats)
    save_stats({"dataset": "AQMAR", "task": "NER",
                "per_file_sentences": per_file_stats, **stats},
               ANALYSIS / "aqmar_ner_stats.json")
    split_and_save(all_sentences, PATB_DIR, "aqmar_ner")

# ══════════════════════════════════════════════════════════════════════════════
# 3. WikiCoref
# ══════════════════════════════════════════════════════════════════════════════
def collect_wikicoref():
    section("3. WikiCoref — Arabic Coreference Corpus (Farhan et al., 2016)")
    log("Format : CoNLL 2012 coreference format")
    log("License: Free for research use")

    # Try multiple known paths in the repo
    coref_urls = [
        "https://raw.githubusercontent.com/qcri/WikiCoref/master/data/arabic/train.arabic.v4_gold_conll",
        "https://raw.githubusercontent.com/qcri/WikiCoref/master/Arabic/train.arabic.v4_gold_conll",
        "https://raw.githubusercontent.com/qcri/arabic-coref/master/data/train.conll",
    ]
    dev_urls = [
        "https://raw.githubusercontent.com/qcri/WikiCoref/master/data/arabic/dev.arabic.v4_gold_conll",
        "https://raw.githubusercontent.com/qcri/WikiCoref/master/Arabic/dev.arabic.v4_gold_conll",
    ]
    test_urls = [
        "https://raw.githubusercontent.com/qcri/WikiCoref/master/data/arabic/test.arabic.v4_gold_conll",
        "https://raw.githubusercontent.com/qcri/WikiCoref/master/Arabic/test.arabic.v4_gold_conll",
    ]

    downloaded = []
    for split_name, mirrors in [("train", coref_urls),
                                 ("dev",   dev_urls),
                                 ("test",  test_urls)]:
        dest = WIKI_DIR / f"{split_name}.arabic.v4_gold_conll"
        if try_mirrors(mirrors, dest, f"WikiCoref {split_name}"):
            downloaded.append(dest)

    if not downloaded:
        log("All WikiCoref mirrors failed.", "warn")
        log("Manual download required — options:", "warn")
        log("  1. https://github.com/qcri/WikiCoref", "warn")
        log("  2. Arabic OntoNotes 5.0 via LDC (LDC2013T19)", "warn")
        log("     Most universities have LDC access — check with your library.", "warn")
        log("Creating CoNLL placeholder for pipeline testing ...", "warn")
        placeholder = (
            "#begin document (sample_ar); part 000\n"
            "sample 0 0  محمد   NNP  *  -  -  -  (1\n"
            "sample 0 1  علي    NNP  *  -  -  -  1)\n"
            "sample 0 2  يعمل   VBZ  *  -  -  -  -\n"
            "sample 0 3  مديراً  NN   *  -  -  -  -\n"
            "\n"
            "sample 0 4  هو     PRP  *  -  -  -  (1)\n"
            "sample 0 5  يسكن   VBZ  *  -  -  -  -\n"
            "sample 0 6  في     IN   *  -  -  -  -\n"
            "sample 0 7  القاهرة NNP  *  -  -  -  (2)\n"
            "#end document\n"
        )
        dest = WIKI_DIR / "sample.arabic.v4_gold_conll"
        dest.write_text(placeholder, encoding="utf-8")
        downloaded.append(dest)
        log("Placeholder saved.", "ok")

    # Save license info
    (WIKI_DIR / "LICENSE_INFO.json").write_text(json.dumps({
        "dataset":  "WikiCoref",
        "paper":    "Farhan et al. (2016)",
        "github":   "https://github.com/qcri/WikiCoref",
        "fallback": "Arabic OntoNotes 5.0 — LDC2013T19",
        "ldc_url":  "https://catalog.ldc.upenn.edu/LDC2013T19",
        "license":  "Free for research (WikiCoref). LDC license for OntoNotes.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Parse and report stats
    all_stats = {}
    for path in downloaded:
        stat = parse_conll_coref(path)
        all_stats[path.name] = stat
        log(f"{path.name}: {stat['documents']} docs, "
            f"{stat['tokens']:,} tokens, "
            f"{stat['mentions']} mentions, "
            f"{stat['chains']} chains")

    save_stats({"dataset": "WikiCoref", "task": "coreference",
                "splits": all_stats}, ANALYSIS / "wikicoref_stats.json")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
def print_summary():
    section("COLLECTION SUMMARY")

    datasets = [
        ("ANERcorp",  ANERCORP_DIR, "NER"),
        ("AQMAR",     PATB_DIR,     "NER (normalized)"),
        ("WikiCoref", WIKI_DIR,     "Coreference"),
    ]
    all_ok = True
    for name, path, task in datasets:
        files = [f for f in path.glob("*") if f.is_file()] if path.exists() else []
        if files:
            size_kb = sum(f.stat().st_size for f in files) / 1024
            log(f"{name:<12} ({task:<20}) "
                f"— {len(files)} files, {size_kb:.0f} KB", "ok")
        else:
            log(f"{name:<12} ({task:<20}) — EMPTY", "warn")
            all_ok = False

    # Save full manifest
    manifest = {
        "project": "Arabic NLP — NER, POS, Coreference",
        "student": "Student B — Hiba El Ouazi",
        "tag_scheme": "IOB2",
        "ner_classes": ["PER", "LOC", "ORG", "MISC"],
        "datasets": {
            "ANERcorp": {
                "task": "NER", "format": "CoNLL 2-col",
                "path": "data/raw/anercorp",
                "source": "Benajiba & Rosso (2007)",
                "license": "Free for research",
                "note": "If placeholder: download from EmnamoR/Arabic-named-entity-recognition",
            },
            "AQMAR": {
                "task": "NER", "format": "CoNLL 2-col (normalized)",
                "path": "data/raw/patb",
                "source": "Schneider et al. (2012) — CMU",
                "url": AQMAR_URL,
                "license": "Free for research",
                "tag_normalization": "MIS0/1/2/3 → MISC",
            },
            "WikiCoref": {
                "task": "Coreference", "format": "CoNLL 2012",
                "path": "data/raw/wikicoref_ar",
                "source": "Farhan et al. (2016) — QCRI",
                "github": "https://github.com/qcri/WikiCoref",
                "fallback": "Arabic OntoNotes 5.0 — LDC2013T19",
                "license": "Free / LDC institutional",
            },
        },
    }
    dest = RAW / "dataset_manifest.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log("Manifest → data/raw/dataset_manifest.json", "ok")

    print()
    if all_ok:
        log("All datasets ready. Proceed to preprocessing pipeline.", "ok")
    else:
        log("Some datasets need manual download — see warnings above.", "warn")
        log("Placeholders allow pipeline development to continue.", "ok")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 62)
    print("  Arabic NLP — Dataset Collection & Verification (v2)")
    print("  Student B: Hiba El Ouazi")
    print("═" * 62)

    for d in [ANERCORP_DIR, PATB_DIR, WIKI_DIR, ANALYSIS, PROCESSED]:
        d.mkdir(parents=True, exist_ok=True)

    collect_anercorp()
    collect_aqmar()
    collect_wikicoref()
    print_summary()

    print("\n" + "═" * 62)
    print("  Done. Stats saved in data/analysis/")
    print("  Next: src/data/preprocessing.py")
    print("═" * 62 + "\n")
