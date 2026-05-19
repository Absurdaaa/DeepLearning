"""Shared encoder module."""

from __future__ import annotations

import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, dropout: float, pad_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, source_tokens: torch.Tensor, source_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(source_tokens))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths=source_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden
