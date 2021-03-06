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
    with open(f"workdata/predictions/icc/{dataset}_predictions", "r") as f:
        for line in f:
            yield json.loads(line)  # label (gold), tokens (list), predicted


def standardize(data):
    if data["predicted"] == 0:
        for token in data["tokens"]:
            yield (token, "O")
    else:
        for i, token in enumerate(data["tokens"]):
            yield (token, "BI"[bool(i)])


for dataset in datasets:
    with open(f"workdata/predictions/icc/{dataset}_aligned", "w") as f:
        for gold_clause, pred_clause in zip(
            gold_clauses(dataset), pred_clauses(dataset)
        ):
            for gold, pred in zip_longest(gold_clause, standardize(pred_clause)):
                gtok, gtag = gold
                ptok, ptag = pred
                assert ptok == gtok
                print(ptok, gtag, ptag, sep="\t", file=f)
            print(file=f)
