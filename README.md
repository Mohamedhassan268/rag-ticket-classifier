# RAG Ticket Classifier

Classifies support tickets into categories by retrieving similar past tickets (FAISS + sentence-transformer embeddings) and using their labels to make the decision, with a small language model (Flan-T5) generating a plain-language rationale for the result.

## Why retrieval instead of a plain classifier

A standard classifier (e.g. a fine-tuned BERT head) needs retraining every time a category is added or the data shifts, and it gives you a label with no explanation. This pipeline instead embeds every ticket, retrieves the `k` most similar past tickets for any new one, and lets those neighbors vote (weighted by similarity) on the category. New categories just need a few labeled examples added to the dataset — no retraining — and every decision comes with the evidence that produced it: "classified as X because it's similar to these tickets."

The generation model (Flan-T5-small) does not make the classification decision. Early testing showed an 80M-parameter model prompted to classify directly was unreliable — it tended to output the same category regardless of input. Retrieval-based voting was consistently correct in testing, so it was made the decision-maker; Flan-T5 was repurposed for a lower-stakes job it's actually suited to: explaining, in one sentence, why the retrieved evidence supports the chosen category.

## Architecture

```
ticket text
    │
    ▼
sentence-transformer embedding (all-MiniLM-L6-v2)
    │
    ▼
FAISS similarity search over the ticket dataset  →  top-k similar tickets
    │
    ▼
weighted majority vote over their categories  →  category
    │
    ▼
Flan-T5-small generates a one-sentence rationale from the ticket + top matches
    │
    ▼
{ category, rationale, retrieved tickets }
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd rag-ticket-classifier
uv venv --python 3.11
uv pip install -r requirements.txt

.venv/Scripts/python -m src.pipeline --ticket "I was charged twice for my subscription this month"
```

(On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python`.)

First run downloads the two models from HuggingFace (~200MB combined) and caches them locally; subsequent runs are fast.

## Configuration

`configs/example.yaml`:

| Key | Meaning |
|---|---|
| `embedding_model` | HuggingFace sentence-transformer used to embed tickets |
| `generation_model` | HuggingFace seq2seq model used to generate the rationale |
| `top_k` | Number of similar tickets retrieved per query, and the number of votes cast |
| `categories` | The closed set of valid categories; data containing anything else is rejected at load time |
| `data_path` | Path to the CSV of labeled tickets the classifier retrieves against |

Pass a different config with `--config path/to/your.yaml`.

## Worked example

```
$ python -m src.pipeline --ticket "I was charged twice for my subscription this month"

Category: billing
Rationale: "My invoice shows a charge higher than my plan price"

Retrieved context:
  [0.853] (billing) I was charged twice for my subscription this month and need a refund
  [0.455] (billing) My invoice shows a charge higher than my plan price
  [0.264] (billing) I want to update my credit card on file before the next billing cycle
```

The scores are cosine similarity (0 to 1). Three of the top matches are `billing`, and they carry the highest similarity, so `billing` wins the weighted vote — and you can see exactly which past tickets drove that decision.

## Limitations

- **Synthetic data.** `data/sample_tickets.csv` is 45 fabricated support tickets, not real user data — this is a demonstration dataset, not a trained/validated production classifier.
- **Small models, by design.** `all-MiniLM-L6-v2` and `flan-t5-small` were chosen to keep the pipeline fast and CPU-friendly (matters for CI); a production system would likely use larger embedding and generation models.
- **The rationale is decorative, not authoritative.** Flan-T5-small often paraphrases or quotes the nearest retrieved ticket rather than composing a genuinely new explanation. It doesn't affect the classification decision, which comes entirely from retrieval.
- **No persistence.** The FAISS index is rebuilt in memory on every run. Fine at 45 rows; would need to change for a larger, growing dataset.
- **No API or tests yet.** This is a CLI-only pipeline. A test suite and CI workflow are a planned follow-up.
