"""
datasets.py — PyTorch Dataset classes for NER, POS, and Coreference tasks.

Owner: Student A
Phase: 2 — Preprocessing & Infrastructure (Week 2–3)

Each dataset class handles:
  - Loading pre-tokenized, aligned data from disk
  - Subword-to-word label alignment (first-subword strategy)
  - Returning tensors ready for the MTL collator
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NER Dataset
# ---------------------------------------------------------------------------


class ArabicNERDataset(Dataset):
    """Dataset for Arabic Named Entity Recognition.

    Expects data in JSONL format, one example per line:
    {
        "tokens": ["محمد", "يعمل", "في", "القاهرة"],
        "ner_tags": ["B-PER", "O", "O", "B-LOC"]
    }

    Label scheme: IOB2 — B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O
    """

    LABEL2ID = {
        "O": 0,
        "B-PER": 1, "I-PER": 2,
        "B-ORG": 3, "I-ORG": 4,
        "B-LOC": 5, "I-LOC": 6,
        "B-MISC": 7, "I-MISC": 8,
    }
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}
    IGNORE_INDEX = -100  # ignored by CrossEntropyLoss

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load(Path(data_path))
        logger.info("NER dataset loaded: %d examples from %s", len(self.examples), data_path)

    # ------------------------------------------------------------------
    def _load(self, path: Path) -> list[dict[str, Any]]:
        examples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.examples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        tokens: list[str] = example["tokens"]
        ner_tags: list[str] = example["ner_tags"]

        # Tokenize with word-level alignment tracking
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # collator handles padding
            return_tensors="pt",
        )

        # Align labels: first-subword gets the true label, others get IGNORE_INDEX
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # [CLS] or [SEP] tokens
                aligned_labels.append(self.IGNORE_INDEX)
            elif word_idx != previous_word_idx:
                # First subword of a word → assign the true label
                tag = ner_tags[word_idx]
                aligned_labels.append(self.LABEL2ID.get(tag, 0))
            else:
                # Continuation subword → ignore
                aligned_labels.append(self.IGNORE_INDEX)
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(1)).squeeze(0),
            "ner_labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# POS Dataset
# ---------------------------------------------------------------------------


class ArabicPOSDataset(Dataset):
    """Dataset for Arabic Part-of-Speech tagging using PATB tagset.

    Expects JSONL format:
    {
        "tokens": ["الكتاب", "الجديد"],
        "pos_tags": ["DT+NN", "DT+JJ"]
    }
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        label2id: dict[str, int],
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.max_length = max_length
        self.examples = self._load(Path(data_path))
        self.ignore_index = -100
        logger.info("POS dataset loaded: %d examples from %s", len(self.examples), data_path)

    def _load(self, path: Path) -> list[dict[str, Any]]:
        examples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        tokens: list[str] = example["tokens"]
        pos_tags: list[str] = example["pos_tags"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(self.ignore_index)
            elif word_idx != previous_word_idx:
                tag = pos_tags[word_idx]
                aligned_labels.append(self.label2id.get(tag, 0))
            else:
                aligned_labels.append(self.ignore_index)
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(1)).squeeze(0),
            "pos_labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Coreference Dataset
# ---------------------------------------------------------------------------


class ArabicCoreferenceDataset(Dataset):
    """Dataset for Arabic Coreference Resolution.

    Expects JSONL format based on CoNLL-2012 annotation:
    {
        "tokens": ["قال", "محمد", "إنه", "مريض"],
        "clusters": [[[1, 1], [2, 2]]]   // list of mention clusters
                                          // each mention is [start, end] (inclusive, word-level)
    }
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        max_span_width: int = 30,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_span_width = max_span_width
        self.examples = self._load(Path(data_path))
        logger.info("Coref dataset loaded: %d examples from %s", len(self.examples), data_path)

    def _load(self, path: Path) -> list[dict[str, Any]]:
        examples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        tokens: list[str] = example["tokens"]
        clusters: list[list[list[int]]] = example.get("clusters", [])

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        # Map word-level spans to subword-level spans
        word_ids = encoding.word_ids(batch_index=0)
        word_to_subword_start: dict[int, int] = {}
        word_to_subword_end: dict[int, int] = {}

        for subword_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx not in word_to_subword_start:
                    word_to_subword_start[word_idx] = subword_idx
                word_to_subword_end[word_idx] = subword_idx

        # Convert clusters to subword-level
        subword_clusters = []
        for cluster in clusters:
            subword_cluster = []
            for start_word, end_word in cluster:
                if start_word in word_to_subword_start and end_word in word_to_subword_end:
                    subword_cluster.append(
                        [word_to_subword_start[start_word], word_to_subword_end[end_word]]
                    )
            if subword_cluster:
                subword_clusters.append(subword_cluster)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(1)).squeeze(0),
            "clusters": subword_clusters,  # kept as list of lists (variable length)
            "sentence_map": word_ids,       # for the span scorer
        }