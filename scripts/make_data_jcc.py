import json
import os
from contextlib import ExitStack
from collections import defaultdict

os.makedirs("workdata/jcc", exist_ok=True)
files = defaultdict(dict)
with open("workdata/clausified.json", "r") as f:
    with ExitStack() as stack:
        for ds in ["eca", "emotion-stimulus", "reman", "gne", "electoral_tweets"]:
            for split in ["train", "dev", "test"]:
                files[ds][split] = stack.enter_context(
                    open(f"workdata/jcc/{ds}.{split}", "w")
                )
        for line in f:
            data = json.loads(line)
            ds = data["dataset"]
            split = data["split"]
            id_ = data["id"]
            clauses = []
            labels = []
            for clause in data["clauses"]:
                clause_tokens = [token for token, tag in clause]
                clauses.append(clause_tokens)
                label = int(bool(set("BI") & {tag for token, tag in clause}))
                labels.append("yes" if label else "no")
            json.dump(
                {"id": id_, "clauses": clauses, "labels": labels}, files[ds][split]
            )
            files[ds][split].write("\n")
