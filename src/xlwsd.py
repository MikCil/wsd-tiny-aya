from dataclasses import dataclass, field
from typing import Literal

import logging

import requests_cache
from dotenv import load_dotenv
from lxml import etree
from requests_cache import DO_NOT_CACHE, NEVER_EXPIRE
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from aya import AyaClient, format_msg
from scorer import SBERTScore

logging.getLogger("babelnet").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

urls_expire_after = {
    "*babelnet.io*": NEVER_EXPIRE,
    "*": DO_NOT_CACHE,
}
requests_cache.install_cache(
    "babelnet_cache",
    allowable_methods=["GET", "POST"],
    urls_expire_after=urls_expire_after,
)


def display_wsd_result(
    console: Console,
    sentence: str,
    target_word: str,
    lemma: str,
    pos: str,
    prediction: str,
) -> None:
    """Display a WSD result in a simple table format."""
    # Highlight target word in sentence
    highlighted_text = Text()
    remaining = sentence
    target_lower = target_word.lower()

    while remaining:
        idx = remaining.lower().find(target_lower)
        if idx == -1:
            highlighted_text.append(remaining)
            break
        if idx > 0:
            highlighted_text.append(remaining[:idx])
        end_idx = idx + len(target_word)
        target_match = remaining[idx:end_idx]
        highlighted_text.append(target_match, style="bold bright_yellow")
        remaining = remaining[end_idx:]

    # Create table with word info, sentence, and prediction
    table = Table(
        show_header=False,
        box=box.SIMPLE_HEAD,
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column(style="dim", justify="right", no_wrap=True)
    table.add_column()

    table.add_row("Word", f"[bold]{lemma}[/bold]#[dim]{pos}[/dim]")
    table.add_row("Context", highlighted_text)
    table.add_row("Prediction", f"[green]{prediction}[/green]")

    console.print(table)
    console.print()


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
class Eval:
    model: str
    lang: str
    target_word: str
    context: str
    pos: str
    ref: str
    pred: str
    score: float


if __name__ == "__main__":
    load_dotenv()

    models = [
        "tiny-aya-global",
        "tiny-aya-fire",
    ]
    langs = ["en"]
    instances: list[Eval] = []

    for lang in langs:
        corpus = parse_doc("dev", lang)
        sense_inventory = parse_inventory(f"./xl-wsd/inventories/inventory.{lang}.txt")
        n = len(corpus.sentences)

        # Collect
        for model in models:
            print(f"Processing dev set for '{lang}' on '{model}'")

            for i, sent in enumerate(corpus.sentences):
                for word in sent.words:
                    if not word.is_instance:
                        continue

                    # Use gold synset
                    bn_ids = word.bn_ids
                    if len(bn_ids) == 0:
                        continue

                    bn_data = get_babelnet_data(bn_ids[0])
                    if bn_data is None:
                        continue

                    instances.append(
                        Eval(
                            model=model,
                            lang=lang,
                            context=sent.text,
                            target_word=word.text,
                            pos=word.pos,
                            ref=bn_data.gloss,
                            pred="",
                            score=0.0,
                        )
                    )

    # Process
    aya_client = AyaClient()
    scorer = SBERTScore("paraphrase-multilingual-mpnet-base-v2")
    preds = [
        aya_client(inst.model, format_msg(inst.target_word, inst.context))
        for inst in instances
    ]
    refs = [inst.ref for inst in instances]
    scores = scorer.score_batch(refs, preds)
    for inst, pred, score in zip(instances, preds, scores):
        inst.pred = pred
        inst.score = float(score)
