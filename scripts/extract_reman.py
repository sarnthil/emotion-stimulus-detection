from collections import defaultdict, Counter
from itertools import zip_longest
from xml.etree.ElementTree import parse

from cause_io import Instance


def join_spans(spans):
    last = None
    for s2, e2 in sorted(set(spans)):
        if not last:
            last = s2, e2
            continue
        s1, e1 = last
        if e1 < s2:
            yield last
            last = s2, e2
            continue
        last = s1, e2
    if last:
        yield last


def span_borders(span, text):
    return span.text == text[int(span.attrib["cbegin"]) : int(span.attrib["cend"])]


def clean_text(string):
    return string.replace("\t", " ")


def offsets_from_relations(relations, id_to_span, side="target"):
    offsets = []
    for relation in relations:
        # source_span = id_to_span[relation.attrib["source_annotation_id"]]  # cue
        # target_span = id_to_span[
        #     relation.attrib["target_annotation_id"]
        # ]  # target/experiencer/...
        # ಠ_ಠ
        # assert span_borders(source_span, text)
        # assert span_borders(target_span, text)

        try:
            offsets.append(
                (
                    int(
                        id_to_span[relation.attrib[f"{side}_annotation_id"]].attrib[
                            "cbegin"
                        ]
                    ),
                    int(
                        id_to_span[relation.attrib[f"{side}_annotation_id"]].attrib[
                            "cend"
                        ]
                    ),
                )
            )
        except KeyError:
            pass  # ಠ_ಠ
    return list(join_spans(offsets))


def extract():
    tree = parse("sources/reman/reman-version1.0.xml")

    for document in tree.iterfind("document"):
        text = document.find("text").text.replace("\t", " ")
        types = {
            span.attrib["type"]
            for span in document.find("adjudicated").find("spans").iterfind("span")
        }
        emotions = types & {
            "joy",
            "sadness",
            "disgust",
            "anger",
            "surprise",
            "fear",
            "anticipation",
            "trust",
            "other",  # ಠ_ಠ
        }
        if not emotions:
            emotions = ["noemo"]
        id_to_span = {
            span.attrib["annotation_id"]: span
            for span in document.find("adjudicated").find("spans").iterfind("span")
        }
        cause_relations = []
        for relation in (
            document.find("adjudicated").find("relations").iterfind("relation")
        ):
            relation_type = relation.attrib["type"]
            if relation_type != "cause":
                continue
            cause_relations.append(relation)

        offsets = offsets_from_relations(cause_relations, id_to_span, side="target")
        cursor = 0
        tokens = []
        for start, end in sorted(offsets):
            for word in text[cursor:start].split():
                tokens.append((word, "O"))
            for word, tag in zip_longest(text[start:end].split(), "B", fillvalue="I"):
                tokens.append((word, tag))
            cursor = end
        for word in text[cursor:].split():
            tokens.append((word, "O"))
        yield Instance(
            text=text, tokens=tokens, emotions=list(emotions),
        )


meta = {"domain": "literature", "annotation": "expert"}
dataset = "reman"
