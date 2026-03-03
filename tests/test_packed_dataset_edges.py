import torch

from baseline_transformer.data.packed_lm import PackedLMDataset


class _TinyTokenizer:
    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_packed_dataset_skips_empty_lines():
    tok = _TinyTokenizer(eos_token_id=None)
    ds = PackedLMDataset(
        name="unused",
        split="train",
        tokenizer=tok,
        block_size=3,
        texts=["", "   ", "ab", None, "cde"],
    )

    assert len(ds) > 0
    item = ds[0]
    assert item["input_ids"].shape == torch.Size([3])
    assert item["attention_mask"].sum().item() == 3


def test_packed_dataset_inserts_eos_between_rows_when_available():
    tok = _TinyTokenizer(eos_token_id=99)
    ds = PackedLMDataset(
        name="unused",
        split="train",
        tokenizer=tok,
        block_size=4,
        texts=["ab", "cd"],
    )

    assert ds.stream.tolist() == [97, 98, 99, 99, 100, 99]


def test_packed_dataset_stride_increases_number_of_blocks():
    tok = _TinyTokenizer(eos_token_id=None)
    texts = ["abcdefgh", "ijklmnop"]

    non_overlap = PackedLMDataset(
        name="unused",
        split="train",
        tokenizer=tok,
        block_size=8,
        stride=None,
        texts=texts,
    )
    overlap = PackedLMDataset(
        name="unused",
        split="train",
        tokenizer=tok,
        block_size=8,
        stride=4,
        texts=texts,
    )

    assert len(overlap) > len(non_overlap)


def test_packed_dataset_forwards_dataset_config(monkeypatch):
    tok = _TinyTokenizer(eos_token_id=None)
    seen = {}

    def fake_load_lm_dataset(name, split, config_name=None):
        seen["name"] = name
        seen["split"] = split
        seen["config_name"] = config_name
        return [{"text": "abcd"}]

    monkeypatch.setattr("baseline_transformer.data.packed_lm.load_lm_dataset", fake_load_lm_dataset)

    ds = PackedLMDataset(
        name="wikitext103",
        split="train",
        tokenizer=tok,
        block_size=2,
        dataset_config="wikitext-103-v1",
    )

    assert len(ds) > 0
    assert seen == {
        "name": "wikitext103",
        "split": "train",
        "config_name": "wikitext-103-v1",
    }
