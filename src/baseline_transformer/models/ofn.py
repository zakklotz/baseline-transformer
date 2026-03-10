from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nncore.models import OFNConfig
from nncore.models import OFNLM as NNCoreOFNLM


class OFNTransformerLM(nn.Module):
    """
    Adapter around nn-core's OFNLM to support HF-style LM batches:
      input_ids, attention_mask, labels
    """

    def __init__(self, cfg: OFNConfig):
        super().__init__()
        self.model = NNCoreOFNLM(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        recurrence_steps: int | None = None,
    ):
        _ = recurrence_steps
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.to(torch.bool)

        logits = self.model(
            input_ids=input_ids,
            key_padding_mask=key_padding_mask,
        )
        if not torch.is_tensor(logits):
            raise TypeError(f"Unexpected OFN output type: {type(logits)}")

        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return result
