from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED_KEYS = (
    "embedding_model",
    "generation_model",
    "top_k",
    "categories",
    "data_path",
)


@dataclass
class Config:
    embedding_model: str
    generation_model: str
    top_k: int
    categories: list[str]
    data_path: str


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"Config at {path} is missing required key(s): {', '.join(missing)}")

    return Config(
        embedding_model=raw["embedding_model"],
        generation_model=raw["generation_model"],
        top_k=raw["top_k"],
        categories=raw["categories"],
        data_path=raw["data_path"],
    )
