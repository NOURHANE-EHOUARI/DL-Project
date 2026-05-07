"""
ner_head.py — BiLSTM-CRF Named Entity Recognition Head.

Owner: Student A
Phase: 3 — Modeling & Training (Week 3–5)

Architecture:
  backbone output → dropout → BiLSTM → dropout → Linear → CRF decoder

The CRF layer enforces valid IOB2 tag transitions at inference time,
significantly improving precision on entity boundary detection.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchcrf import CRF

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class NERHead(nn.Module):
    """BiLSTM-CRF head for Arabic NER.

    Args:
        hidden_size:   Backbone output dimension (e.g., 768 or 1024).
        num_labels:    Number of IOB2 tag classes (e.g., 9 for PER/ORG/LOC/MISC).
        lstm_hidden:   Hidden size of the BiLSTM (each direction).
        lstm_layers:   Number of stacked BiLSTM layers.
        dropout:       Dropout probability applied before and after BiLSTM.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)

        # BiLSTM: input is backbone hidden states
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Projection to tag space
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

        # CRF layer — handles transition constraints between tags
        self.crf = CRF(num_tags=num_labels, batch_first=True)

        logger.info(
            "NERHead initialized: hidden=%d, lstm_hidden=%d, num_labels=%d",
            hidden_size, lstm_hidden, num_labels,
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        sequence_output: torch.Tensor,          # (batch, seq_len, hidden_size)
        attention_mask: torch.Tensor,            # (batch, seq_len)
        labels: Optional[torch.Tensor] = None,  # (batch, seq_len), IGNORE_INDEX = -100
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            During training (labels provided):
                {"loss": scalar, "emissions": (batch, seq_len, num_labels)}
            During inference:
                {"emissions": ..., "predictions": list[list[int]]}
        """
        x = self.dropout(sequence_output)

        # Pack for efficiency (handles variable-length sequences)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        emissions = self.classifier(lstm_out)  # (batch, seq_len, num_labels)

        # Build boolean mask for CRF (True = real token, False = padding)
        crf_mask = attention_mask.bool()

        if labels is not None:
            # Replace IGNORE_INDEX labels with 0 for CRF (they are masked out)
            crf_labels = labels.clone()
            crf_labels[crf_labels == IGNORE_INDEX] = 0

            # CRF loss: negative log-likelihood (averaged over batch)
            loss = -self.crf(
                emissions=emissions,
                tags=crf_labels,
                mask=crf_mask,
                reduction="mean",
            )
            return {"loss": loss, "emissions": emissions}

        # Viterbi decoding at inference time
        predictions = self.crf.decode(emissions, mask=crf_mask)
        return {"emissions": emissions, "predictions": predictions}

    # ------------------------------------------------------------------

    def decode(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[list[int]]:
        """Convenience method: returns decoded tag sequences only."""
        output = self.forward(sequence_output, attention_mask, labels=None)
        return output["predictions"]