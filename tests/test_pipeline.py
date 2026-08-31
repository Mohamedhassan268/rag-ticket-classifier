SAMPLE_TICKETS = [
    "I was charged twice for my subscription this month",
    "The dashboard throws an error whenever I try to download a CSV export",
    "I can't get into my account, the login page just spins forever",
    "Could you add a dark theme to the app?",
    "What plans do you offer and how do they differ?",
]


def test_classify_returns_valid_category(pipeline, config):
    # Guaranteed by construction: _vote_category() can only return a category
    # that exists in the loaded data, never free text from the generation
    # model. This test documents that guarantee across varied inputs rather
    # than catching a failure mode that the design already rules out.
    for ticket in SAMPLE_TICKETS:
        result = pipeline.classify(ticket)
        assert result["category"] in config.categories


def test_classify_smoke(pipeline):
    result = pipeline.classify("My invoice looks wrong this month")

    assert result["category"]
    assert result["retrieved"]
    assert isinstance(result["rationale"], str)
