from __future__ import annotations

import torch

from baseline_transformer.models import OFNTransformerLM
from baseline_transformer.nncore_bridge import build_ofn_config


def test_build_ofn_config_round_trips_expected_fields():
    cfg = build_ofn_config(
        {
            "ofn": {
                "vocab_size": 32,
                "d_model": 16,
                "n_heads": 4,
                "max_seq_len": 8,
                "num_layers": 2,
                "dropout": 0.1,
                "ffn_mult": 2.0,
                "field": {
                    "enabled": True,
                    "slots": 4,
                    "d_field": 64,
                    "builder": "ema",
                    "ema_timescales": [8, 32, 128, 512],
                    "conditioning": "film",
                    "feedback": True,
                    "feedback_scale": 1.0,
                },
                "operators": {
                    "local": {"enabled": True, "d_hidden": 96, "kernel_size": 5},
                    "attention": {"enabled": True, "mode": "window", "window_size": 8},
                },
                "mediator": {"mode": "gated_sum", "d_imaginal": 32, "gate_hidden": 24},
            }
        }
    )

    assert cfg.vocab_size == 32
    assert cfg.d_model == 16
    assert cfg.n_heads == 4
    assert cfg.num_layers == 2
    assert cfg.field.builder == "ema"
    assert cfg.field.ema_timescales == [8, 32, 128, 512]
    assert cfg.operators.attention.mode == "window"
    assert cfg.mediator.mode == "gated_sum"


def test_ofn_transformer_lm_forward_cpu():
    cfg = build_ofn_config(
        {
            "ofn": {
                "vocab_size": 64,
                "d_model": 16,
                "n_heads": 4,
                "max_seq_len": 8,
                "num_layers": 2,
                "positional": "rope",
                "dropout": 0.0,
                "ffn_mult": 2.0,
                "field": {
                    "enabled": True,
                    "slots": 4,
                    "d_field": 64,
                    "builder": "ema",
                    "ema_timescales": [8, 32, 128, 512],
                    "conditioning": "film",
                    "feedback": True,
                    "feedback_scale": 1.0,
                },
                "operators": {
                    "local": {"enabled": True, "d_hidden": 64, "kernel_size": 5},
                    "attention": {"enabled": True, "mode": "window", "window_size": 8},
                },
                "mediator": {"mode": "barzakh", "d_imaginal": 32, "gate_hidden": 32},
            }
        }
    )
    model = OFNTransformerLM(cfg)
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 64, (2, 8), dtype=torch.long)
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out["logits"].shape == (2, 8, 64)
    assert torch.isfinite(out["loss"]).item()
