from bert_score import BERTScorer
from bleurt import score as Bleurt
from sentence_transformers import SentenceTransformer


class BLEURTScore:
    def __init__(self, model_id: str):
        self.model = Bleurt.BleurtScorer(model_id)

    def score(self, ref: str, pred: str) -> float:
        return self.model.score(references=[ref], candidates=[pred])

    def score_batch(self, refs: list[str], preds: list[str]) -> list[float]:
        return self.model.score(references=refs, candidates=preds)


class SBERTScore:
    def __init__(self, model_id: str, device: str = "mps"):
        self.model = SentenceTransformer(model_id, device=device)
        self.device = device
        self.encode_opts = {"convert_to_tensor": True, "device": self.device}

    def score(self, ref: str, pred: str) -> float:
        encoded_ref = self.model.encode(ref, convert_to_tensor=True, device=self.device)
        encoded_pred = self.model.encode(
            pred, convert_to_tensor=True, device=self.device
        )
        return self.model.similarity(encoded_ref, encoded_pred).item()

    def score_batch(self, refs: list[str], preds: list[str]) -> list[float]:
        encoded_ref = self.model.encode(
            refs, convert_to_tensor=True, device=self.device
        )
        encoded_preds = self.model.encode(
            preds, convert_to_tensor=True, device=self.device
        )
        return [
            score.item()
            for score in self.model.similarity_pairwise(encoded_ref, encoded_preds)
        ]


class BERTScore:
    def __init__(self, model_id: str, device: str = "mps"):
        self.model = BERTScorer(
            lang="en",
            model_type=model_id,
            device=device,
            rescale_with_baseline=True,
        )

    def score(self, ref: str, pred: str) -> float:
        P, R, F1 = self.model.score(pred, ref)
        return F1.item()

    def score_batch(self, refs: list[str], preds: list[str]) -> list[float]:
        P, R, F1 = self.model.score(preds, refs)
        return F1


if __name__ == "__main__":
    scorer = SBERTScore(model_id="paraphrase-multilingual-mpnet-base-v2")
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
