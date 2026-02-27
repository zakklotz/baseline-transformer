from baseline_transformer.config import ExperimentConfig


def test_config_load_parity():
    cfg = ExperimentConfig.load("configs/parity/wt103_512d.yaml")
    assert cfg.model["transformer"]["d_model"] == 512
    assert cfg.model["transformer"]["n_heads"] == 8
    assert cfg.data["max_seq_len"] == 512
