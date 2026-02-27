from __future__ import annotations

from typing import Any, Dict

from nncore.models import TransformerConfig
from nncore.models.config import BlockConfig, AttentionConfig


def build_transformer_config(model_cfg: Dict[str, Any]) -> TransformerConfig:
    """
    Convert our project-friendly transformer config into nn-core's TransformerConfig.

    Our YAML under model.transformer:
      vocab_size: int
      max_seq_len: int
      d_model: int
      n_heads: int
      n_layers: int            # interpreted as decoder-only layers for causal LM
      d_ff: int                # maps to BlockConfig.mlp_dims = [d_model, d_ff, d_model]
      dropout: float           # maps to AttentionConfig.dropout_p and resid_dropout_p

    nn-core TransformerConfig signature:
      (vocab_size, d_model, num_heads, max_seq_len,
       num_encoder_layers, num_decoder_layers,
       positional, ..., attn: AttentionConfig, block: BlockConfig)
    """
    if "transformer" not in model_cfg:
        raise KeyError("model.transformer missing from config")

    t = dict(model_cfg["transformer"])

    vocab_size = int(t.get("vocab_size", 32000))
    d_model = int(t.get("d_model", 512))
    num_heads = int(t.get("n_heads", t.get("num_heads", 8)))
    max_seq_len = int(t.get("max_seq_len", 2048))

    # decoder-only default for LM
    n_layers = int(t.get("n_layers", 0))
    if n_layers > 0:
        num_encoder_layers = 0
        num_decoder_layers = n_layers
    else:
        num_encoder_layers = int(t.get("num_encoder_layers", 0))
        num_decoder_layers = int(t.get("num_decoder_layers", 0))

    # MLP dims in nn-core are an explicit chain: [in_dim, hidden_dim, out_dim]
    d_ff = t.get("d_ff", None)
    if d_ff is not None:
        d_ff = int(d_ff)
        mlp_dims = [d_model, d_ff, d_model]
    else:
        mlp_dims = None

    dropout = float(t.get("dropout", 0.0))

    block = BlockConfig(
        mlp_dims=mlp_dims,
        # keep norm defaults unless you add them to YAML later
    )

    attn = AttentionConfig(
        dropout_p=dropout,
        resid_dropout_p=dropout,
    )

    positional = t.get("positional", "absolute")
    tie_weights = bool(t.get("tie_weights", True))
    return_hidden = bool(t.get("return_hidden", False))

    recursive = bool(t.get("recursive", False))
    recurrence_steps = int(t.get("recurrence_steps", 1))

    return TransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        positional=positional,
        recursive=recursive,
        recurrence_steps=recurrence_steps,
        tie_weights=tie_weights,
        return_hidden=return_hidden,
        attn=attn,
        block=block,
    )
