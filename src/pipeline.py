import argparse
from collections import Counter

import pandas as pd

from src.config import Config, load_config
from src.embeddings import Embedder
from src.generation import Generator, build_prompt
from src.retrieval import RetrievedTicket, TicketRetriever


def _vote_category(retrieved: list[RetrievedTicket]) -> str:
    votes = Counter()
    for r in retrieved:
        votes[r.category] += r.score
    return votes.most_common(1)[0][0]


class TicketClassifierPipeline:
    def __init__(self, config: Config):
        self._config = config
        self._embedder = Embedder(config.embedding_model)
        self._generator = Generator(config.generation_model)

        df = pd.read_csv(config.data_path)
        unknown = set(df["category"]) - set(config.categories)
        if unknown:
            raise ValueError(f"Data contains categories not in config: {unknown}")

        embeddings = self._embedder.embed(df["text"].tolist())
        self._retriever = TicketRetriever(
            texts=df["text"].tolist(),
            categories=df["category"].tolist(),
            embeddings=embeddings,
        )

    def classify(self, ticket_text: str) -> dict:
        query_embedding = self._embedder.embed([ticket_text])[0]
        retrieved = self._retriever.retrieve(query_embedding, self._config.top_k)

        # Retrieval, not generation, decides the category: flan-t5-small was
        # empirically unreliable at picking a category directly (see build
        # notes), but consistently correct via nearest-neighbor majority vote.
        # Generation is repurposed to produce a human-readable rationale.
        category = _vote_category(retrieved)
        prompt = build_prompt(ticket_text, retrieved, category)
        rationale = self._generator.generate(prompt)

        return {
            "category": category,
            "rationale": rationale,
            "retrieved": retrieved,
        }


def main():
    parser = argparse.ArgumentParser(description="Classify a support ticket using RAG.")
    parser.add_argument("--ticket", required=True, help="The ticket text to classify.")
    parser.add_argument("--config", default="configs/example.yaml", help="Path to the config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = TicketClassifierPipeline(config)
    result = pipeline.classify(args.ticket)

    print(f"Category: {result['category']}")
    print(f"Rationale: {result['rationale']}\n")
    print("Retrieved context:")
    for r in result["retrieved"]:
        print(f"  [{r.score:.3f}] ({r.category}) {r.text}")


if __name__ == "__main__":
    main()
