"""Conditional character-level RNN for name generation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalNameGenerator(nn.Module):
    def __init__(
        self,
        num_categories: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        combined_size = num_categories + input_size + hidden_size
        # 每一步都把“类别条件 + 当前字符 + 上一时刻 hidden”拼到一起
        self.i2h = nn.Linear(combined_size, hidden_size)
        self.i2o = nn.Linear(combined_size, output_size)
        # 再把新 hidden 和中间输出合并一次，让生成结果更稳定一些
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, category: torch.Tensor, input_step: torch.Tensor, hidden: torch.Tensor):
        combined = torch.cat((category, input_step, hidden), dim=1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.i2o(combined)
        output_combined = torch.cat((hidden, output), dim=1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.log_softmax(output)
        return output, hidden

    def init_hidden(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, self.hidden_size, device=device)
