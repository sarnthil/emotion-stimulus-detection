import json
from itertools import zip_longest

datasets = ["eca", "emotion-stimulus", "electoral_tweets", "reman", "gne"]


def gold_clauses(dataset):
    with open("workdata/clausified.json", "r") as f:
        for line in f:
            data = json.loads(line)
            if data["dataset"] != dataset or data["split"] != "test":
                continue
            for clause in data["clauses"]:
                yield clause


def pred_clauses(dataset):
    with open(f"workdata/predictions/jcc/{dataset}_predictions", "r") as f:
        for line in f:
            data = json.loads(line)
            id_ = data["metadata"]["id"]
            for clause in data["metadata"]["clauses_text"]:
                for label in data["labels"][0]:
                    result = []
                    for i, token in enumerate(clause):
                        result.append([token, "BI"[bool(i)]])
                yield result


for dataset in datasets:
    with open(f"workdata/predictions/jcc/{dataset}_aligned", "w") as f:
        for gold_clause, pred_clause in zip(
            gold_clauses(dataset), pred_clauses(dataset)
        ):
            for gold, pred in zip_longest(gold_clause, pred_clause):
                try:
                    gtok, gtag = gold
                except TypeError:
                    import ipdb

                    ipdb.set_trace()

                ptok, ptag = pred
                assert ptok == gtok
                print(ptok, gtag, ptag, sep="\t", file=f)
            print(file=f)
