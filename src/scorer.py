import torch
from rich import print
from torch.nn.functional import cosine_similarity, normalize
from transformers import AutoModel, AutoTokenizer


class BERTScore:
    def __init__(self, model_id: str):
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True
        )
        self.model: AutoModel = AutoModel.from_pretrained(model_id)

    def _mean_pooling(
        self, output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def score(self, ref: str, candidate: str) -> torch.Tensor:
        encoded_input = self.tokenizer(
            [ref, candidate], padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = normalize(embeddings, p=2, dim=1)

        return cosine_similarity(embeddings[0:1], embeddings[1:2], dim=1)


if __name__ == "__main__":
    scorer = BERTScore(model_id="sentence-transformers/all-MiniLM-L6-v2")
    tiny_aya = "A mole is a small, usually circular or oval-shaped, non-cancerous growth on the skin, often caused by a concentration of pigment cells."

    defs = [
        "A small congenital pigmented spot on the skin",
        "A benign growth on the skin (usually tan, brown, or flesh-colored) that contains a cluster of melanocytes and may form a slight relief",
        "A pigmented spot on the skin, a naevus, slightly raised, and sometimes hairy",
        "Dark spot on the skin",
        "A spy who works against enemy espionage",
        "Spy under deep cover",
        "Small velvety-furred burrowing mammal having small eyes and fossorial forefeet",
        "The molecular weight of a substance expressed in grams; the basic unit of amount of substance adopted under the Systeme International d'Unites",
        "The SI unit for the amount of substance",
    ]

    for d in defs:
        score = scorer.score(d, tiny_aya)
        print(f"SENT: {d}\nSCORE: {score.mean()}\n")
