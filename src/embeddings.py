import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
