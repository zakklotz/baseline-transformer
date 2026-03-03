from baseline_transformer.config import ExperimentConfig


def test_config_load_parity():
    cfg = ExperimentConfig.load("configs/parity/wt103_512d.yaml")
    assert cfg.model["transformer"]["d_model"] == 512
    assert cfg.model["transformer"]["n_heads"] == 8
    assert cfg.data["max_seq_len"] == 512


def test_config_load_tajalli_matched_recursive_parity():
    cfg = ExperimentConfig.load("configs/parity/wt103_768d_recursive_tajalli_match.yaml")
    assert cfg.model["type"] == "recursive_transformer"
    assert cfg.model["depth"] == 8
    assert cfg.model["transformer"]["d_model"] == 768
    assert cfg.model["transformer"]["n_heads"] == 12
    assert cfg.data["dataset_config"] == "wikitext-103-v1"
    assert cfg.train["batch_size"] == 1
    assert cfg.train["grad_accum_steps"] == 32
