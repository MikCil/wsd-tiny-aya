from dataclasses import dataclass, field
from typing import Literal

import requests_cache
from requests_cache import DO_NOT_CACHE, NEVER_EXPIRE
from dotenv import load_dotenv
from lxml import etree

from aya import AyaClient, format_msg

urls_expire_after = {
    "*.babelnet.io": NEVER_EXPIRE,
    "*": DO_NOT_CACHE,
}
requests_cache.install_cache(
    "babelnet_cache",
    allowable_methods=["GET", "POST"],
    urls_expire_after=urls_expire_after,
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


def get_data(id: str) -> Data | None:
    import babelnet as bn
    from babelnet.pos import POS
    from babelnet.language import Language
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


if __name__ == "__main__":
    load_dotenv()

    corpus = parse_doc("test", "en")
    sense_inventory = parse_inventory(
        "./xl-wsd/inventories/inventory.en.txt", polysemy=True
    )

    aya_client = AyaClient()

    for sent in corpus.sentences:
        for word in sent.words:
            if not word.is_instance:
                continue

            senses = sense_inventory.get(word.__str__())
            if senses is None or len(senses) <= 1:
                continue

            msg = format_msg(word.text, sent.text)
            response = aya_client("tiny-aya-global", msg)
    # for sent in corpus.sentences:
    #     print(sent.text)
    #     for word in sent.words:
    #         if word.bn_ids:
    #             print(word.text)
    #             pos = wsd2bn_pos[word.pos]
    #
    #             for bn_id in word.bn_ids:
    #                 gloss = get_glosses(bn_id)
    #                 print(gloss)
    #             break
