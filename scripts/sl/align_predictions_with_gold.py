import json
from itertools import zip_longest

datasets = ["eca", "emotion-stimulus", "electoral_tweets", "reman", "gne"]


def predictions(dataset):
    with open(f"workdata/predictions/sl/{dataset}_predictions", "r") as f:
        for line in f:
            data = json.loads(line)
            tokens = data["tokens"]
            predicted_tokens = data["predicted"]
            gold_tokens = data["labels"]
            yield tokens, gold_tokens, predicted_tokens


for dataset in datasets:
    with open(f"workdata/predictions/sl/{dataset}_aligned", "w") as f:
        for tokens, golds, preds in predictions(dataset):
            for tok, gtok, ptok in zip(tokens, golds, preds):
                print(tok, gtok, ptok, sep="\t", file=f)
            print(file=f)
