import json
import os
from contextlib import ExitStack
from collections import defaultdict

os.makedirs("workdata/sl", exist_ok=True)
files = defaultdict(dict)

with open("workdata/clausified.json", "r") as f:
    with ExitStack() as stack:
        for ds in ["eca", "emotion-stimulus", "reman", "gne", "electoral_tweets"]:
            for split in ["train", "dev", "test"]:
                files[ds][split] = stack.enter_context(
                    open(f"workdata/sl/{ds}.{split}", "w")
                )
        for line in f:
            data = json.loads(line)
            ds = data["dataset"]
            split = data["split"]
            for token in data["tokens"]:
                tok = token[0]
                label = token[1]
                files[ds][split].write(f"{tok}\t{label}\n")
            files[ds][split].write("\n")
