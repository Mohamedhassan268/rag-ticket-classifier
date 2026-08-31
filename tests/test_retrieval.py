def test_retrieve_returns_k_results(retriever, embedder):
    query_embedding = embedder.embed(["My card was declined at checkout"])[0]
    results = retriever.retrieve(query_embedding, k=3)

    assert len(results) == 3


def test_retrieve_sorted_by_descending_similarity(retriever, embedder):
    query_embedding = embedder.embed(["My card was declined at checkout"])[0]
    results = retriever.retrieve(query_embedding, k=5)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_finds_relevant_category(retriever, embedder):
    query_embedding = embedder.embed(["I was charged twice for my subscription"])[0]
    results = retriever.retrieve(query_embedding, k=5)

    categories = [r.category for r in results]
    assert "billing" in categories
