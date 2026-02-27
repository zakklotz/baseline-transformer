# baseline-transformer

Reusable transformer baseline harness built on `nn-core`.

## Install

Recommended (sibling repos):
```bash
# in a venv
pip install -e ../nn-core
pip install -e .
Train
python scripts/train.py --config configs/parity/wt103_512d.yaml
Eval
python scripts/eval.py --config configs/parity/wt103_512d.yaml
Config overview

model.type: standard_transformer or recursive_transformer

model.transformer: passed into nncore.models.TransformerConfig

data: dataset/tokenizer/max_seq_len

train: optimizer and loop settings
