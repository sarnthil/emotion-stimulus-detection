import json
import os
from contextlib import ExitStack
from collections import defaultdict

os.makedirs("workdata/icc", exist_ok=True)
files = defaultdict(dict)

with open("workdata/clausified.json", "r") as f:
    with ExitStack() as stack:
        for ds in ["eca", "emotion-stimulus", "reman", "gne", "electoral_tweets"]:
            for split in ["train", "dev", "test"]:
                files[ds][split] = stack.enter_context(
                    open(f"workdata/icc/{ds}.{split}", "w")
                )
        for line in f:
            data = json.loads(line)
            ds = data["dataset"]
            split = data["split"]
            for clause in data["clauses"]:
                text = " ".join(token for token, tag in clause)
                label = int(bool(set("BI") & {tag for token, tag in clause}))
                files[ds][split].write(f"{text}\t{label}\n")
