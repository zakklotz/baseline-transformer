# baseline-transformer

Reusable transformer baseline harness built on `nn-core`.

## Install

Create a virtual environment, then install this project in editable mode with dev extras:

```bash
pip install -e .[dev]
```

This project expects `nn-core` to be available as a dependency in your environment (for example via editable install from a sibling checkout).

## Train

Use either the installed console script:

```bash
bt-train --config configs/parity/wt103_512d.yaml
```

or the repository script wrapper:

```bash
python scripts/train.py --config configs/parity/wt103_512d.yaml
```

## Evaluate

Use either the installed console script:

```bash
bt-eval --config configs/parity/wt103_512d.yaml
```

or the repository script wrapper:

```bash
python scripts/eval.py --config configs/parity/wt103_512d.yaml
```

Optionally pass a checkpoint for evaluation:

```bash
bt-eval --config configs/parity/wt103_512d.yaml --ckpt /path/to/model.pt
```

## Parity config

The parity baseline config is at:

- `configs/parity/wt103_512d.yaml`

It defines model shape, data/tokenizer settings, and training loop hyperparameters used for fair comparisons.

## Recursive mode

Set `model.type: recursive_transformer` to enable recursion using `nn-core` recurrence flags.

When enabled, baseline-transformer maps config values onto nn-core transformer settings by:

- setting `recursive=True`
- setting `recurrence_steps=model.depth`
- forcing `num_decoder_layers=1` so recursion is unambiguously a single shared block applied for `N` steps
