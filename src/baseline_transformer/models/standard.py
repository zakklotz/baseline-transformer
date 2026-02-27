from __future__ import annotations

import torch
import torch.nn as nn

from nncore.models import Transformer
from nncore.models import TransformerConfig


class StandardTransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.model = Transformer(cfg)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # nn-core already returns a dict-like output pattern; keep stable here
        return out
