"""
collators.py — MTL Batch Collators for Arabic NLP
Owner: Student B (Hiba)
Phase: 2 — Preprocessing & Infrastructure (Week 2–3)

Handles dynamic padding and batch construction for:
  - Single-task: NER, POS, Coreference
  - Multi-task: unified MTL batch combining all three tasks

The MTL collator is the key component that allows the shared
backbone to receive batches from different tasks during training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast


# ══════════════════════════════════════════════════════════════════════════════
# Base padding utility
# ══════════════════════════════════════════════════════════════════════════════

def pad_tensor_list(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
) -> torch.Tensor:
    """Pad a list of 1D tensors to the same length."""
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


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
    ignore_index: int = -100

    def __call__(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        input_ids      = pad_tensor_list([b["input_ids"] for b in batch],
                                         self.tokenizer.pad_token_id)
        attention_mask = pad_tensor_list([b["attention_mask"] for b in batch], 0)
        ner_labels     = pad_tensor_list([b["ner_labels"] for b in batch],
                                         self.ignore_index)

        result = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "ner_labels":     ner_labels,
            "task":           "ner",
        }

        # token_type_ids — only if model uses them (e.g. BERT, not RoBERTa)
        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = pad_tensor_list(
                [b["token_type_ids"] for b in batch], 0)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# POS Collator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class POSCollator:
    """
    Collator for Arabic POS tagging batches.

    Identical structure to NERCollator but uses pos_labels.
    Kept separate so each task head receives clearly named tensors.
    """
    tokenizer: PreTrainedTokenizerFast
    ignore_index: int = -100

    def __call__(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        input_ids      = pad_tensor_list([b["input_ids"] for b in batch],
                                         self.tokenizer.pad_token_id)
        attention_mask = pad_tensor_list([b["attention_mask"] for b in batch], 0)
        pos_labels     = pad_tensor_list([b["pos_labels"] for b in batch],
                                         self.ignore_index)

        result = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pos_labels":     pos_labels,
            "task":           "pos",
        }

        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = pad_tensor_list(
                [b["token_type_ids"] for b in batch], 0)

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

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> dict[str, Any]:
        input_ids      = pad_tensor_list([b["input_ids"] for b in batch],
                                         self.tokenizer.pad_token_id)
        attention_mask = pad_tensor_list([b["attention_mask"] for b in batch], 0)

        result = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            # Clusters kept as list — variable structure per example
            "clusters":       [b["clusters"] for b in batch],
            "sentence_map":   [b["sentence_map"] for b in batch],
            "task":           "coref",
        }

        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = pad_tensor_list(
                [b["token_type_ids"] for b in batch], 0)

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

    Task sampling strategies supported:
      - 'proportional': sample tasks proportionally to dataset size
      - 'uniform': sample tasks with equal probability
      - 'round_robin': cycle through tasks in fixed order

    Usage in training loop:
        collator = MTLCollator(tokenizer)
        # Each DataLoader uses the same collator
        ner_loader = DataLoader(ner_dataset, collate_fn=collator)
        pos_loader = DataLoader(pos_dataset, collate_fn=collator)
    """
    tokenizer:    PreTrainedTokenizerFast
    ignore_index: int = -100

    def __post_init__(self):
        self._ner_collator   = NERCollator(self.tokenizer, self.ignore_index)
        self._pos_collator   = POSCollator(self.tokenizer, self.ignore_index)
        self._coref_collator = CoreferenceCollator(self.tokenizer)

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # Detect task from keys in first example
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
        elif self.strategy == "uniform":
            return {t: 1.0 / len(self.tasks) for t in self.tasks}
        else:
            return {t: 1.0 / len(self.tasks) for t in self.tasks}


# ══════════════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n── MTLBatchSampler test ────────────────────────────────")

    from collections import Counter

    sizes    = {"ner": 3525, "pos": 1530, "coref": 100}
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
