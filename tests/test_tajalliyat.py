from __future__ import annotations

import pytest
import torch

from baseline_transformer.models import StandardTransformerLM, TajalliyatTransformerLM
from baseline_transformer.nncore_bridge import build_tajalliyat_config
from nncore.models import TransformerConfig
from nncore.models.config import BlockConfig


def test_standard_transformer_respects_keep_mask_semantics():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=16,
        num_heads=4,
        max_seq_len=8,
        num_encoder_layers=0,
        num_decoder_layers=2,
        block=BlockConfig(mlp_dims=[16, 32, 16]),
    )
    model = StandardTransformerLM(cfg)
    input_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 32, (2, 8), dtype=torch.long)
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out["logits"].shape == (2, 8, 32)
    assert torch.isfinite(out["loss"]).item()


def test_build_tajalliyat_config_round_trips_expected_fields():
    cfg = build_tajalliyat_config(
        {
            "tajalliyat": {
                "vocab_size": 32,
                "d_model": 16,
                "n_heads": 4,
                "max_seq_len": 8,
                "num_layers": 2,
                "dropout": 0.1,
                "ffn_mult": 2.0,
                "use_attention": True,
                "use_cnn": True,
                "use_mamba": False,
                "fusion_type": "gated_sum",
                "cnn_kernel_size": 5,
                "branch_dropout": 0.2,
                "branch_scheduler": "cuda_streams",
            }
        }
    )
    assert cfg.vocab_size == 32
    assert cfg.d_model == 16
    assert cfg.n_heads == 4
    assert cfg.num_layers == 2
    assert cfg.use_attention is True
    assert cfg.use_cnn is True
    assert cfg.use_mamba is False
    assert cfg.fusion_type == "gated_sum"
    assert cfg.cnn_kernel_size == 5
    assert cfg.branch_dropout == 0.2
    assert cfg.branch_scheduler == "cuda_streams"


def test_tajalliyat_transformer_lm_forward_cpu():
    cfg = build_tajalliyat_config(
        {
            "tajalliyat": {
                "vocab_size": 64,
                "d_model": 16,
                "n_heads": 4,
                "max_seq_len": 8,
                "num_layers": 2,
                "positional": "rope",
                "dropout": 0.0,
                "ffn_mult": 2.0,
                "use_attention": True,
                "use_cnn": True,
                "use_mamba": False,
                "fusion_type": "sum",
            }
        }
    )
    model = TajalliyatTransformerLM(cfg)
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 64, (2, 8), dtype=torch.long)
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out["logits"].shape == (2, 8, 64)
    assert torch.isfinite(out["loss"]).item()

    status = model.branch_scheduler_status(device="cpu", compiled=False)
    assert status["configured"] == "auto"
    assert status["resolved"] == "sequential"
    assert status["active_branches"] == ["attention", "cnn"]


def test_tajalliyat_transformer_lm_mamba_forward_cpu():
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_available():
        pytest.skip("mamba-ssm forward path requires CUDA in this environment")

    cfg = build_tajalliyat_config(
        {
            "tajalliyat": {
                "vocab_size": 64,
                "d_model": 64,
                "n_heads": 4,
                "max_seq_len": 8,
                "num_layers": 1,
                "positional": "rope",
                "dropout": 0.0,
                "ffn_mult": 2.0,
                "use_attention": True,
                "use_cnn": True,
                "use_mamba": True,
                "fusion_type": "sum",
            }
        }
    )
    model = TajalliyatTransformerLM(cfg).to("cuda")
    input_ids = torch.randint(0, 64, (1, 8), dtype=torch.long, device="cuda")
    labels = torch.randint(0, 64, (1, 8), dtype=torch.long, device="cuda")
    attention_mask = torch.ones(1, 8, dtype=torch.long, device="cuda")

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out["logits"].shape == (1, 8, 64)
    assert torch.isfinite(out["loss"]).item()
