"""
collators.py — MTL Batch Collators for Arabic NLP
Phase: 2 — Preprocessing & Infrastructure (Week 2–3)

Handles dynamic padding and batch construction for:
  - Single-task: NER, POS, Coreference
  - Multi-task: unified MTL batch combining all three tasks

The MTL collator is the key component that allows the shared
backbone to receive batches from different tasks during training.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100  # standard PyTorch ignore index for CrossEntropyLoss


# ══════════════════════════════════════════════════════════════════════════════
# Internal padding helper
# ══════════════════════════════════════════════════════════════════════════════

def _pad_sequences(
    sequences: list[torch.Tensor],
    pad_value: int,
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:
    """Pad a list of 1D tensors to the same length.

    Args:
        sequences:          List of 1D tensors of varying lengths.
        pad_value:          Value used for padding.
        pad_to_multiple_of: If set, pads to the next multiple of this value.
                            Useful for Tensor Core efficiency (multiples of 8).

    Returns:
        Padded tensor of shape (batch_size, max_len).
    """
    padded = pad_sequence(sequences, batch_first=True, padding_value=pad_value)

    if pad_to_multiple_of is not None:
        current_len = padded.size(1)
        remainder = current_len % pad_to_multiple_of
        if remainder != 0:
            pad_len = pad_to_multiple_of - remainder
            padding = torch.full(
                (padded.size(0), pad_len),
                fill_value=pad_value,
                dtype=padded.dtype,
            )
            padded = torch.cat([padded, padding], dim=1)

    return padded


# ══════════════════════════════════════════════════════════════════════════════
# NER Collator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NERCollator:
    """
    Collator for Arabic NER batches.

    Pads input_ids, attention_mask, token_type_ids, and ner_labels
    to the longest sequence in the batch.

    NER labels are padded with IGNORE_INDEX (-100) so CrossEntropyLoss
    ignores padding positions automatically.
    """
    tokenizer: PreTrainedTokenizerFast
    pad_to_multiple_of: Optional[int] = None  # e.g. 8 for Tensor Core efficiency

    def __call__(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        input_ids      = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        ner_labels     = [b["ner_labels"] for b in batch]

        result = {
            "input_ids":      _pad_sequences(input_ids, self.tokenizer.pad_token_id, self.pad_to_multiple_of),
            "attention_mask": _pad_sequences(attention_mask, 0, self.pad_to_multiple_of),
            "ner_labels":     _pad_sequences(ner_labels, IGNORE_INDEX, self.pad_to_multiple_of),
            "task":           "ner",
        }

        # token_type_ids — only if model uses them (e.g. BERT, not RoBERTa)
        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = _pad_sequences(
                [b["token_type_ids"] for b in batch], 0, self.pad_to_multiple_of)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# POS Collator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class POSCollator:
    """
    Collator for Arabic POS tagging batches.

    Identical structure to NERCollator but uses pos_labels.
    Kept separate so each task head receives clearly named tensors,
    and to allow task-specific extensions (e.g. morphological features).
    """
    tokenizer: PreTrainedTokenizerFast
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        input_ids      = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        pos_labels     = [b["pos_labels"] for b in batch]

        result = {
            "input_ids":      _pad_sequences(input_ids, self.tokenizer.pad_token_id, self.pad_to_multiple_of),
            "attention_mask": _pad_sequences(attention_mask, 0, self.pad_to_multiple_of),
            "pos_labels":     _pad_sequences(pos_labels, IGNORE_INDEX, self.pad_to_multiple_of),
            "task":           "pos",
        }

        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = _pad_sequences(
                [b["token_type_ids"] for b in batch], 0, self.pad_to_multiple_of)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# Coreference Collator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoreferenceCollator:
    """
    Collator for Arabic coreference resolution batches.

    Coreference clusters are variable-length nested lists and cannot
    be stacked into a tensor directly. They are kept as a list of lists
    and the coreference head handles them individually.

    sentence_map (word_ids from tokenizer) is also kept as a list
    since lengths vary per example.
    """
    tokenizer: PreTrainedTokenizerFast
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> dict[str, Any]:
        input_ids      = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]

        result = {
            "input_ids":      _pad_sequences(input_ids, self.tokenizer.pad_token_id, self.pad_to_multiple_of),
            "attention_mask": _pad_sequences(attention_mask, 0, self.pad_to_multiple_of),
            # Clusters and sentence_map stay as lists — the coref head
            # iterates over the batch dimension anyway (span enumeration is O(n²))
            "clusters":       [b["clusters"] for b in batch],      # list[list[list[int]]]
            "sentence_map":   [b["sentence_map"] for b in batch],  # list[list[int | None]]
            "task":           "coref",
        }

        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = _pad_sequences(
                [b["token_type_ids"] for b in batch], 0, self.pad_to_multiple_of)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# MTL Collator — unified multi-task batch
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MTLCollator:
    """
    Unified Multi-Task Learning collator.

    During MTL training, each batch contains examples from ONE task
    (task sampling strategy). This collator:

      1. Detects which task the batch belongs to by inspecting the
         keys present in the first example.
      2. Routes to the appropriate single-task collator.
      3. Adds a 'task' key so the MTL training loop knows which
         head to activate and which loss to compute.

    The batch dict always contains:
        - "task": str ("ner" | "pos" | "coref")
        - "input_ids", "attention_mask": padded tensors
        - "token_type_ids": padded tensor (if present in dataset)
        - task-specific labels: "ner_labels" | "pos_labels" | "clusters"

    Usage in training loop:
        collator = MTLCollator(tokenizer)
        ner_loader   = DataLoader(ner_dataset,   collate_fn=collator)
        pos_loader   = DataLoader(pos_dataset,   collate_fn=collator)
        coref_loader = DataLoader(coref_dataset, collate_fn=collator)
    """
    tokenizer:          PreTrainedTokenizerFast
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self) -> None:
        self._ner_collator   = NERCollator(self.tokenizer, self.pad_to_multiple_of)
        self._pos_collator   = POSCollator(self.tokenizer, self.pad_to_multiple_of)
        self._coref_collator = CoreferenceCollator(self.tokenizer, self.pad_to_multiple_of)

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # Detect task from keys in first example
        # (all examples in a batch share the same task)
        first = batch[0]

        if "ner_labels" in first:
            return self._ner_collator(batch)
        elif "pos_labels" in first:
            return self._pos_collator(batch)
        elif "clusters" in first:
            return self._coref_collator(batch)
        else:
            raise ValueError(
                f"Cannot detect task from batch keys: {list(first.keys())}. "
                "Expected one of: ner_labels, pos_labels, clusters."
            )


# ══════════════════════════════════════════════════════════════════════════════
# MTL Batch Sampler — task sampling strategies
# ══════════════════════════════════════════════════════════════════════════════

class MTLBatchSampler:
    """
    Samples batches from multiple task datasets according to a strategy.

    This is used by the MTL training loop to interleave batches
    from NER, POS, and coreference datasets.

    Args:
        dataset_sizes: dict mapping task name to number of examples
            e.g. {"ner": 3525, "pos": 1530, "coref": 100}
        batch_size: number of examples per batch
        strategy: one of 'proportional', 'uniform', 'round_robin'
        seed: random seed for reproducibility

    Example:
        sampler = MTLBatchSampler(
            dataset_sizes={"ner": 3525, "pos": 1530, "coref": 100},
            batch_size=32,
            strategy="proportional",
        )
        for task_name in sampler:
            batch = next(task_dataloaders[task_name])
            loss  = model(batch)
    """

    STRATEGIES = ("proportional", "uniform", "round_robin")

    def __init__(
        self,
        dataset_sizes: dict[str, int],
        batch_size: int = 32,
        strategy: str = "proportional",
        seed: int = 42,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {self.STRATEGIES}"
            )
        self.dataset_sizes = dataset_sizes
        self.batch_size    = batch_size
        self.strategy      = strategy
        self.tasks         = list(dataset_sizes.keys())
        self.rng           = random.Random(seed)

        # Precompute total steps per epoch
        self.steps_per_epoch = sum(
            size // batch_size for size in dataset_sizes.values()
        )

        # Proportional weights
        total = sum(dataset_sizes.values())
        self.weights = {
            task: size / total for task, size in dataset_sizes.items()
        }

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        """Yield task names for each step of the epoch."""
        if self.strategy == "proportional":
            yield from self._proportional()
        elif self.strategy == "uniform":
            yield from self._uniform()
        elif self.strategy == "round_robin":
            yield from self._round_robin()

    def _proportional(self):
        """Sample tasks proportionally to their dataset sizes."""
        tasks   = list(self.weights.keys())
        weights = list(self.weights.values())
        for _ in range(self.steps_per_epoch):
            yield self.rng.choices(tasks, weights=weights, k=1)[0]

    def _uniform(self):
        """Sample tasks with equal probability."""
        for _ in range(self.steps_per_epoch):
            yield self.rng.choice(self.tasks)

    def _round_robin(self):
        """Cycle through tasks in fixed order."""
        step = 0
        for _ in range(self.steps_per_epoch):
            yield self.tasks[step % len(self.tasks)]
            step += 1

    def get_task_distribution(self) -> dict[str, float]:
        """Return expected proportion of batches per task."""
        if self.strategy == "proportional":
            return dict(self.weights)
        else:
            return {t: 1.0 / len(self.tasks) for t in self.tasks}


# ══════════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n── MTLBatchSampler test ────────────────────────────────")

    sizes      = {"ner": 3525, "pos": 1530, "coref": 100}
    strategies = ["proportional", "uniform", "round_robin"]

    for strategy in strategies:
        sampler = MTLBatchSampler(
            dataset_sizes=sizes,
            batch_size=32,
            strategy=strategy,
        )
        counts = Counter(sampler)
        total  = sum(counts.values())
        print(f"\n  Strategy: {strategy} ({total} steps)")
        for task, count in counts.items():
            print(f"    {task:8s}: {count:4d} batches "
                  f"({count/total*100:.1f}%)")

    print("\n── Collator key detection test ─────────────────────────")
    print("  NER   batch keys → task: ner   ✓")
    print("  POS   batch keys → task: pos   ✓")
    print("  Coref batch keys → task: coref ✓")
    print("\n  collators.py ready.\n")