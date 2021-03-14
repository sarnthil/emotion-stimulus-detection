import json
from pathlib import Path
import itertools


def is_divider(line):
    return line.strip() == ""


datasets = ["emotion-stimulus", "electoral_tweets", "eca", "reman", "gne"]
models = ["sl"]  # later "jcc", "icc

for dataset in datasets:
    for model in models:
        file_in = Path(f"workdata/{model}/{dataset}.test")
        file_out = Path(f"workdata/{model}/{dataset}.json.test")
        if not file_in.exists():
            continue
        with file_out.open("w") as out:
            with file_in.open() as f:
                for divider, lines in itertools.groupby(f, is_divider):
                    if divider:
                        continue
                    fields = [l.strip().split() for l in lines]
                    fields = [l for l in zip(*fields)]
                    tokens, tags = fields
                    sentence = " ".join(fields[0])
                    tags = " ".join(fields[1])
                    json.dump({"sentence": sentence, "tags": tags}, out)
                    out.write("\n")

# for divider, lines in itertools.groupby(conll_file, is_divider):
#                 # skip over any dividing lines
#                 if divider: continue
#                 # get the CoNLL fields, each token is a list of fields
#                 fields = [l.strip().split() for l in lines]
#                 # switch it so that each field is a list of tokens/labels
#                 fields = [l for l in zip(*fields)]
#                 # only keep the tokens and NER labels
#                 tokens, _, _, ner_tags = fields

#                 yield self.text_to_instance(tokens, ner_tags)
