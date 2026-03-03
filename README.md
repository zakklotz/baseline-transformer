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
bt-train --config configs/parity/wt103_512d_standard.yaml
```

or the repository script wrapper:

```bash
python scripts/train.py --config configs/parity/wt103_512d_standard.yaml
```

Training logs include token counts and throughput (`tokens_step`, `tokens_total`, `tok_s`) for fair comparisons, and are also written to `<out_dir>/train_log.csv`.

## Evaluate

Use either the installed console script:

```bash
bt-eval --config configs/parity/wt103_512d_standard.yaml
```

or the repository script wrapper:

```bash
python scripts/eval.py --config configs/parity/wt103_512d_standard.yaml
```

Optionally pass a checkpoint for evaluation:

```bash
bt-eval --config configs/parity/wt103_512d_standard.yaml --ckpt /path/to/model.pt
```

## Parity config

The explicit parity baseline configs are:

- `configs/parity/wt103_512d_standard.yaml`
- `configs/parity/wt103_512d_recursive.yaml`
- `configs/parity/wt103_768d_recursive_tajalli_match.yaml`

The original config remains available for backward compatibility:

- `configs/parity/wt103_512d.yaml`

These configs define model shape, data/tokenizer settings, and training loop hyperparameters for fair comparisons.
The `wt103_768d_recursive_tajalli_match.yaml` config is the closest paper-control match for Tajalli Phase 1:
same GPT-2 tokenizer, `wikitext-103-v1`, sequence length 512, effective batch size 32, 300k steps, 768/12/3072 width, cosine decay to `1e-5`, and RoPE attention.

## Packed LM mode

For standard language-model perplexity comparisons (especially on WikiText-103), packed token-stream mode is available.

When enabled, the dataset pipeline tokenizes each non-empty row, appends `eos_token_id` separators when available, concatenates all tokens into one stream, and chunks contiguous fixed-size blocks.

Enable via config:

```yaml
data:
  packing: true
  block_size: 512  # defaults to data.max_seq_len if omitted
  # optional Hugging Face dataset config override
  # dataset_config: wikitext-103-v1
  # optional eval overlap (defaults to non-overlapping)
  # stride: 256
  # optional training overlap override (default: non-overlapping)
  # train_stride: 256
```

`data.stride` is interpreted as eval stride by default; training stays non-overlapping unless `data.train_stride` is explicitly set.
`data.num_workers` controls DataLoader workers in normal runs, while pytest runs force `num_workers=0` to avoid fork warnings.

## Reproducibility and run metadata

Training supports an optional deterministic mode:

```yaml
train:
  deterministic: true  # default false
```

When enabled, training uses deterministic PyTorch algorithms and deterministic cuDNN settings.

Every training run also writes run artifacts into `out_dir`:

- `config.resolved.yaml` (resolved config used for the run)
- `command.txt` (CLI command used)
- `git_commit.txt` (best-effort current commit hash when available)
- `model_stats.txt` (model type/recurrence metadata and parameter counts)

## Recursive mode

Set `model.type: recursive_transformer` to enable recursion using `nn-core` recurrence flags.

When enabled, baseline-transformer maps config values onto nn-core transformer settings by:

- setting `recursive=True`
- setting `recurrence_steps=model.depth`
- forcing `num_decoder_layers=1` so recursion is unambiguously a single shared block applied for `N` steps
