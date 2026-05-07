"""
backbone.py — Arabic Transformer Encoder Wrapper.

Owner: Student A
Phase: 3 — Modeling & Training (Week 3–5)

Wraps AraBERT v2 / CAMeLBERT / AraELECTRA / XLM-R with:
  - Subword-to-word output pooling (first subword strategy)
  - Optional feature extraction mode (freeze backbone weights)
  - Clean interface for the MTL model to call
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

logger = logging.getLogger(__name__)

# Recommended Arabic backbone checkpoints
SUPPORTED_BACKBONES = {
    "arabert-v2": "aubmindlab/bert-large-arabertv2",
    "camelbert-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "camelbert-msa": "CAMeL-Lab/bert-base-arabic-camelbert-msa",
    "araelectra": "aubmindlab/araelectra-base-discriminator",
    "xlm-r": "xlm-roberta-large",
}


class ArabicBackbone(nn.Module):
    """Shared Arabic transformer encoder for the MTL model.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
            Use keys from SUPPORTED_BACKBONES for convenience.
        hidden_dropout_prob: Dropout applied to the encoder output.
        freeze: If True, freezes all backbone parameters (feature extraction mode).
        gradient_checkpointing: Enables gradient checkpointing to save memory
            at the cost of recomputation during backward pass.
    """

    def __init__(
        self,
        model_name_or_path: str = "arabert-v2",
        hidden_dropout_prob: float = 0.1,
        freeze: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        # Resolve shorthand names
        checkpoint = SUPPORTED_BACKBONES.get(model_name_or_path, model_name_or_path)
        logger.info("Loading backbone: %s", checkpoint)

        config = AutoConfig.from_pretrained(checkpoint)
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = hidden_dropout_prob

        self.encoder: PreTrainedModel = AutoModel.from_pretrained(
            checkpoint, config=config
        )
        self.hidden_size: int = config.hidden_size

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        if freeze:
            self._freeze_all()
            logger.info("Backbone parameters frozen")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer encoder.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len) — optional for RoBERTa-style models

        Returns:
            sequence_output: (batch, seq_len, hidden_size)
                Token-level representations including [CLS] and [SEP].
        """
        kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # XLM-R and ELECTRA do not use token_type_ids
        if token_type_ids is not None and self._has_token_type_ids():
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs, return_dict=True)
        return outputs.last_hidden_state  # (batch, seq_len, hidden_size)

    def get_word_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids: list[list[Optional[int]]],
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subword-to-word alignment using the first-subword strategy.

        Arabic BPE tokenizers split words into multiple subword pieces.
        For token-level tasks (NER, POS), we need ONE representation per
        original word. Strategy: keep only the hidden state of the FIRST
        subword of each word, discard the rest.

        This matches the standard approach used in the NER literature
        (Aloraini et al. 2021, Inoue et al. 2022) and is what HuggingFace
        offset_mapping / word_ids() implements natively.

        Args:
            input_ids:      (batch, subword_seq_len)
            attention_mask: (batch, subword_seq_len)
            word_ids:       list of length batch_size.
                            Each element is the list returned by
                            tokenizer(text).word_ids() — a list of length
                            subword_seq_len where each entry is:
                              - None  → special token ([CLS], [SEP], [PAD])
                              - int   → index of the original word this
                                        subword belongs to
            token_type_ids: (batch, subword_seq_len) — optional

        Returns:
            word_output:  (batch, max_words, hidden_size)
                          Padded tensor of word-level representations.
            word_mask:    (batch, max_words)  — 1 for real words, 0 for padding.

        Example (what word_ids looks like for one sentence):
            Original words : ["زار"  , "الرئيس", "باريس"]
            After BPE      : [CLS, "زار", "ال", "##رئيس", "با", "##ريس", SEP]
            word_ids       : [None,  0,    1,    1,        2,    2,       None]
            → We keep positions 1, 2, 4  (first subword of each word)
            → word_output shape: (1, 3, hidden_size)
        """
        # (batch, subword_seq_len, hidden_size)
        sequence_output = self.forward(input_ids, attention_mask, token_type_ids)

        batch_size = sequence_output.size(0)
        hidden_size = sequence_output.size(2)

        # For each sentence: collect the indices of first-subword positions
        first_subword_indices: list[list[int]] = []
        for b in range(batch_size):
            seen: set[int] = set()
            indices: list[int] = []
            for pos, wid in enumerate(word_ids[b]):
                if wid is None:
                    continue          # skip [CLS], [SEP], [PAD]
                if wid not in seen:
                    seen.add(wid)
                    indices.append(pos)
            first_subword_indices.append(indices)

        # Pad to the longest word sequence in the batch
        max_words = max(len(idx) for idx in first_subword_indices)

        word_output = torch.zeros(
            batch_size, max_words, hidden_size,
            dtype=sequence_output.dtype,
            device=sequence_output.device,
        )
        word_mask = torch.zeros(
            batch_size, max_words,
            dtype=torch.long,
            device=sequence_output.device,
        )

        for b, indices in enumerate(first_subword_indices):
            for word_pos, subword_pos in enumerate(indices):
                word_output[b, word_pos] = sequence_output[b, subword_pos]
                word_mask[b, word_pos] = 1

        return word_output, word_mask  # (batch, max_words, H), (batch, max_words)

    def get_cls_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns [CLS] token representation. Useful for sentence-level tasks."""
        sequence_output = self.forward(input_ids, attention_mask, token_type_ids)
        return sequence_output[:, 0, :]  # (batch, hidden_size)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _freeze_all(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters (for gradual unfreezing schedules)."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def unfreeze_top_n_layers(self, n: int) -> None:
        """Unfreeze the top n transformer layers and the embedding layer."""
        self._freeze_all()

        for param in self.encoder.embeddings.parameters():
            param.requires_grad = True

        if hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
            for layer in layers[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        logger.info("Unfroze top %d layers + embeddings", n)

    def _has_token_type_ids(self) -> bool:
        """Check if this model architecture uses token_type_ids."""
        return hasattr(self.encoder, "embeddings") and hasattr(
            self.encoder.embeddings, "token_type_embeddings"
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())