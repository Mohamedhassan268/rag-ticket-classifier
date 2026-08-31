from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class RetrievedTicket:
    text: str
    category: str
    score: float


class TicketRetriever:
    def __init__(self, texts: list[str], categories: list[str], embeddings: np.ndarray):
        if not (len(texts) == len(categories) == embeddings.shape[0]):
            raise ValueError("texts, categories, and embeddings must have matching lengths")

        self._texts = texts
        self._categories = categories
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def retrieve(self, query_embedding: np.ndarray, k: int) -> list[RetrievedTicket]:
        k = min(k, len(self._texts))
        scores, indices = self._index.search(query_embedding.reshape(1, -1), k)

        return [
            RetrievedTicket(
                text=self._texts[i],
                category=self._categories[i],
                score=float(scores[0][rank]),
            )
            for rank, i in enumerate(indices[0])
        ]
