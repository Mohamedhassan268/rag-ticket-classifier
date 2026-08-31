import torch
from transformers import pipeline as hf_pipeline

from src.retrieval import RetrievedTicket


def build_prompt(ticket_text: str, retrieved: list[RetrievedTicket], category: str) -> str:
    examples = "\n".join(f'- "{r.text}"' for r in retrieved[:3])

    return (
        f'A support ticket was classified as "{category}" based on these similar past tickets:\n'
        f"{examples}\n"
        f'New ticket: "{ticket_text}"\n'
        f'In one short sentence, explain why this ticket fits the "{category}" category.'
    )


class Generator:
    def __init__(self, model_name: str):
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = hf_pipeline("text2text-generation", model=model_name, device=device)

    def generate(self, prompt: str) -> str:
        output = self._pipe(prompt, max_new_tokens=16)
        return output[0]["generated_text"]
