import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

import requests_cache
from requests_cache import DO_NOT_CACHE, NEVER_EXPIRE

logging.getLogger("babelnet").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

urls_expire_after = {
    "*babelnet.io*": NEVER_EXPIRE,
    "*": DO_NOT_CACHE,
}
requests_cache.install_cache(
    "babelnet_cache",
    allowable_methods=["GET", "POST"],
    urls_expire_after=urls_expire_after,
    allowable_codes=[200],
)


@dataclass
class Word:
    text: str
    lemma: str
    pos: str
    is_instance: bool
    instance_id: str | None = None
    bn_ids: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.lemma}#{self.pos}"


@dataclass
class Sentence:
    id: str
    text_id: str
    text: str
    words: list[Word] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


@dataclass
class Corpus:
    lang: str
    source: str
    sentences: list[Sentence] = field(default_factory=list)


def parse_inventory(fpath: str, polysemy: bool = True) -> dict[str, list[str]]:
    inventory: dict[str, list[str]] = {}
    with open(fpath) as f:
        for line in f:
            fields = line.split("\t")
            ids = [s.strip() for s in fields[1:]]
            if polysemy and len(ids) <= 1:
                continue
            inventory[fields[0]] = ids
    return inventory


def parse_doc(
    dataset: Literal["test", "dev"], lang: str, lemma: str | None = None
) -> Corpus:
    from lxml import etree

    n = f"{dataset}-{lang}"
    gold_fpath = f"./xl-wsd/evaluation_datasets/{n}/{n}.gold.key.txt"
    xml_fpath = f"./xl-wsd/evaluation_datasets/{n}/{n}.data.xml"

    ids: dict[str, list[str]] = {}

    with open(gold_fpath, newline="") as f:
        for line in f:
            fields = line.split(" ")
            ids[fields[0]] = [s.strip() for s in fields[1:]]
    tree = etree.parse(xml_fpath)
    root = tree.getroot()

    corpus = Corpus(
        lang=root.attrib.get("lang", ""),
        source=root.attrib.get("source", ""),
    )

    if lemma is None:
        sent_nodes = root.xpath(".//sentence")
    else:
        sent_nodes = root.xpath(f".//sentence[instance/@lemma='{lemma}']")

    for sent_node in sent_nodes:
        parent = sent_node.getparent()
        text_id = (
            parent.attrib.get("id", "unknown") if parent is not None else "unknown"
        )

        raw_text = "".join(sent_node.itertext()).strip()
        raw_text = " ".join(raw_text.split())
        sentence = Sentence(
            id=sent_node.attrib.get("id", "unknown"),
            text_id=text_id,
            text=raw_text,
        )

        for child in sent_node:
            if child.tag in ("wf", "instance"):
                instance_id = (
                    child.attrib.get("id") if child.tag == "instance" else None
                )

                word = Word(
                    bn_ids=ids[instance_id] if instance_id else [],
                    text=child.text or "",
                    lemma=child.attrib.get("lemma", ""),
                    pos=child.attrib.get("pos", ""),
                    is_instance=(child.tag == "instance"),
                    instance_id=instance_id,
                )
                sentence.words.append(word)
        corpus.sentences.append(sentence)

    return corpus


@dataclass
class Data:
    lemma: str
    pos: str
    gloss: str
    sense: str


@lru_cache(maxsize=None)
def get_babelnet_data(id: str) -> Data | None:
    import babelnet as bn
    from babelnet.language import Language
    from babelnet.pos import POS
    from babelnet.resources import BabelSynsetID

    wsd2bn_lang = {
        "en": Language.EN,
    }

    wsd2bn_pos = {
        "NOUN": POS.NOUN,
        "VERB": POS.VERB,
        "ADJ": POS.ADJ,
        "ADV": POS.ADV,
    }

    synset = bn.get_synset(BabelSynsetID(id))
    if synset is None:
        print(f"synset for '{id}' not found")
        return None

    gloss = synset.main_gloss()
    if gloss is None:
        print(f"gloss for '{id}' not found")
        return None

    sense = synset.main_sense()
    if sense is None:
        print(f"sense for '{id}' not found")
        return None

    return Data(
        lemma=str(sense.full_lemma),
        pos=str(synset.pos),
        gloss=str(gloss.gloss),
        sense="",
    )


@dataclass
class Eval(Mapping):
    lang: str
    target_word: str
    context: str
    pos: str
    ref: str
    pred: str | None = None
    sbert_score: float | None = None
    bert_score: float | None = None
    bleurt_score: float | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)


def clean(text: str) -> str:
    if text.startswith("Definition:"):
        text = text.replace("Definition:", "", 1)
        text = text.replace("*", "", -1)
        text = text.strip()
    return text


def gather(set: Literal["test", "dev"], lang: str) -> list[Eval]:
    print(f"ORGANISING DATA {set}-{lang}...")
    corpus = parse_doc(set, lang)

    eval_items: list[Eval] = []
    loop = True
    for i, sent in enumerate(corpus.sentences):
        for word in sent.words:
            if not word.is_instance:
                continue

            # Use gold synset
            bn_ids = word.bn_ids
            if len(bn_ids) == 0:
                continue

            try:
                bn_data = get_babelnet_data(bn_ids[0])
                if bn_data is None:
                    continue
            except Exception:
                print("BabelNet limit reached")
                loop = False
                break

            eval_items.append(
                Eval(
                    lang=lang,
                    context=sent.text,
                    target_word=word.text,
                    pos=word.pos,
                    ref=bn_data.gloss,
                )
            )
        if not loop:
            break
    return eval_items


def predict(model: str, eval_items: list[Eval]) -> list[Eval]:
    """
    Returns:
        predictions
        preds
        refs
    """
    from aya import format_msg

    print(f"PREDICTING SENSES [{model}]...")

    if "aya" in model:
        from aya import AyaClient

        client = AyaClient()
    else:
        from lmstudio import LMStudioClient

        client = LMStudioClient()

    predictions: list[Eval] = []

    for item in eval_items:
        pred = client(model, format_msg(item.target_word, item.context))
        pred = clean(pred)
        item.pred = pred
        predictions.append(item)

    return predictions


def score(
    refs: list[str], preds: list[str]
) -> tuple[list[float], list[float], list[float]]:
    """
    Returns:
        sbert_scores
        bleurt_scores
        bert_scores
    """
    assert len(refs) == len(preds)
    from scorer import BERTScore, BLEURTScore, SBERTScore

    print("COMPUTING BERT SCORES...")
    bert = BERTScore("roberta-large")
    bert_scores = bert.score_batch(refs, preds)
    del bert

    print("COMPUTING SBERT SCORES...")
    sbert = SBERTScore("paraphrase-multilingual-mpnet-base-v2")
    sbert_scores = sbert.score_batch(refs, preds)
    del sbert

    print("COMPUTING BLEURT SCORES...")
    bleurt = BLEURTScore("BLEURT-20-D6")
    bleurt_scores = bleurt.score_batch(refs, preds)
    del bleurt

    return sbert_scores, bleurt_scores, bert_scores


def save_checkpoint(fpath: str, data: list[Eval]):
    import os

    dirname, fname = os.path.split(fpath)
    os.makedirs(dirname, exist_ok=True)
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(Eval.__dataclass_fields__)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved at {fpath}")


def load_checkpoint(fpath: str) -> list[Eval]:
    items: list[Eval] = []
    with open(fpath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = Eval(
                lang=row["lang"],
                target_word=row["target_word"],
                context=row["context"],
                pos=row["pos"],
                ref=row["ref"],
                pred=row.get("pred") or None,
                sbert_score=float(row["sbert_score"])
                if row.get("sbert_score")
                else None,
                bert_score=float(row["bert_score"]) if row.get("bert_score") else None,
                bleurt_score=float(row["bleurt_score"])
                if row.get("bleurt_score")
                else None,
            )
            items.append(item)
    print(f"Loaded {fpath}")
    return items


if __name__ == "__main__":
    import csv
    import os

    from dotenv import load_dotenv

    load_dotenv()

    portion = "test-ur"
    dataset, lang = portion.split("-")

    sense_path = f"./evals/{portion}/senses.csv"
    if not os.path.isfile(sense_path):
        eval_items = gather(dataset, lang)
        save_checkpoint(sense_path, eval_items)

    else:
        eval_items = load_checkpoint(sense_path)
    print(f"Got {len(eval_items)} items")

    models = [
        "tiny-aya-fire",
        "tiny-aya-global",
        "tiny-aya-water",
        "gemma-2-2b-it",
        "qwen2.5-3b-instruct",
        "Qwen3.5-4B-GGUF",
        "Qwen3.5-2B-GGUF",
    ]

    for model in models:
        pred_path = f"./evals/{portion}/{model}-predictions.csv"
        if not os.path.isfile(pred_path):
            predictions = predict(model, eval_items)
            save_checkpoint(pred_path, predictions)

        else:
            predictions = load_checkpoint(pred_path)
        print(f"Got {len(eval_items)} items")

        refs = [item.ref for item in predictions]
        preds = [item.pred for item in predictions]

        score_path = f"./evals/{portion}/{model}-scores.csv"
        if not os.path.isfile(score_path):
            sbert_scores, bleurt_scores, bert_scores = score(refs, preds)
            for i, item in enumerate(eval_items):
                item.sbert_score = sbert_scores[i]
                item.bert_score = bert_scores[i]
                item.bleurt_score = bleurt_scores[i]
            save_checkpoint(score_path, eval_items)

        else:
            print("Done")
