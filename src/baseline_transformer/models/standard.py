from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nncore.models import Transformer
from nncore.models import TransformerConfig


class StandardTransformerLM(nn.Module):
    """
    Adapter around nn-core's Transformer to support HF-style LM batches:
      input_ids, attention_mask, labels

    nn-core forward signature:
      forward(self, src_ids, tgt_ids=None, *, src_key_padding_mask=None, ...)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.model = Transformer(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        recurrence_steps: int | None = None,
    ):
        # nn-core uses key_padding_mask=True for PAD positions
        src_kpm = None
        if attention_mask is not None:
            src_kpm = ~attention_mask.to(torch.bool)

        out = self.model(
            src_ids=input_ids,
            tgt_ids=None,
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=None,
            recurrence_steps=recurrence_steps,
            return_aux=False,
        )

        # Normalize output to logits tensor
        if isinstance(out, dict):
            logits = out.get("logits", None)
            if logits is None:
                # fall back: first tensor-like value
                for v in out.values():
                    if torch.is_tensor(v):
                        logits = v
                        break
        else:
            logits = out

        if logits is None or not torch.is_tensor(logits):
            raise TypeError(f"Unexpected Transformer output type: {type(out)}")

        result = {"logits": logits}

        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result
