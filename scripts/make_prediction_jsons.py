import json
from pathlib import Path

datasets = ["emotion-stimulus", "electoral_tweets", "eca", "reman", "gne"]
models = ["icc"]  # later "jcc", "sl"

for dataset in datasets:
    for model in models:
        file_in = Path(f"workdata/{model}/{dataset}.test")
        file_out = Path(f"workdata/{model}/{dataset}.json.test")
        if not file_in.exists():
            continue
        with file_out.open("w") as out:
            with file_in.open() as f:
                for line in f:
                    sentence, _ = line.strip().split("\t")
                    json.dump({"sentence": sentence}, out)
                    out.write("\n")
