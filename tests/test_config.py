import pytest

from src.config import load_config


def test_load_config_valid():
    config = load_config("configs/example.yaml")

    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.generation_model == "google/flan-t5-small"
    assert config.top_k > 0
    assert "billing" in config.categories
    assert config.data_path == "data/sample_tickets.csv"


def test_load_config_missing_key_raises(tmp_path):
    incomplete_yaml = tmp_path / "incomplete.yaml"
    incomplete_yaml.write_text(
        "embedding_model: some-model\n"
        "generation_model: some-model\n"
        "top_k: 5\n"
        "categories: [a, b]\n"
    )

    with pytest.raises(ValueError, match="data_path"):
        load_config(incomplete_yaml)
