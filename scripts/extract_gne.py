import re
import json
import os

from cause_io import Instance, tokens_from_text_and_annotations

meta = {"domain": "headlines", "annotation": "crowdsourcing"}

dataset = "gne"


def extract():
    with open("sources/gne/gne.jsonl") as f:
        for line in f:
            data = json.loads(line)
            text = data["headline"]
            headline = text
            emotion = data["annotations"]["dominant_emotion"]["gold"]
            if len(data["annotations"]["cause"]["gold"]) == 0:
                annotations = []
            else:
                annotations = data["annotations"]["cause"]["gold"][0]
            if annotations == ["none"]:
                annotations = []
            tokens = tokens_from_text_and_annotations(headline, annotations)
            yield Instance(
                text=text,
                emotions=[emotion],
                tokens=tokens,
                tags=["cause-" + ("yes" if annotations else "no")],
                extra={"annotations": annotations},
            )
