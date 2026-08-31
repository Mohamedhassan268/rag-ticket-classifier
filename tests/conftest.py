import pandas as pd
import pytest

from src.config import load_config
from src.embeddings import Embedder
from src.pipeline import TicketClassifierPipeline
from src.retrieval import TicketRetriever


@pytest.fixture(scope="session")
def config():
    return load_config("configs/example.yaml")


@pytest.fixture(scope="session")
def embedder(config):
    return Embedder(config.embedding_model)


@pytest.fixture(scope="session")
def retriever(config, embedder):
    df = pd.read_csv(config.data_path)
    embeddings = embedder.embed(df["text"].tolist())
    return TicketRetriever(
        texts=df["text"].tolist(),
        categories=df["category"].tolist(),
        embeddings=embeddings,
    )


@pytest.fixture(scope="session")
def pipeline(config):
    return TicketClassifierPipeline(config)
