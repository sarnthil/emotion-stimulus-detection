HEADER = r"""\begin{tabular}{lrrrrrrrrrrr} \toprule Data set &  Size & \# Stim. & $\mu$ & $\sigma$ & tokS./I. & tokS./cls. & \# Cls. & \# Clss. & \# Cls./I. & \# Clss.(all)/I. & \# Cls. aligned/I.\\
\cmidrule(r){1-1}\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}\cmidrule(l){7-7}\cmidrule(l){8-8}\cmidrule(l){9-9}\cmidrule(l){10-10}\cmidrule(l){11-11}"""
LINE = r"""{ds} & {n_instances:,} & {n_instances_with_stimulus:,} & {mean_length_stimulus:.2f} & {stdev_length_stimulus:.2f} & {mean_token_stimulus_ratio_per_instance:.2f} & {mean_token_stimulus_ratio_per_clause:.2f} & {n_clauses:,} & {n_clauses_with_stimulus:,} & {mean_clauses_per_instance:.2f} & {mean_clauses_all_stimulus_per_instance:.2f} & {mean_clauses_pure_per_instance:.2f}\\"""
FOOTER = r"""\bottomrule"""

import re
import json
import fileinput
from statistics import mean, stdev
from collections import defaultdict, Counter

import click


@click.command()
@click.argument("file", type=click.File("r"))
@click.option("--use-predicted", is_flag=True)
def cli(file, use_predicted):
    transds = {
        "reman": r"\dsREMAN",
        "emotion-stimulus": r"\dsES",
        "eca": r"\dsECA",
        "gne": r"\dsGNE",
        "electoral_tweets": r"\dsET",
    }

    stats = defaultdict(Counter)
    lengths = defaultdict(lambda: defaultdict(list))

    for line in file:
        data = json.loads(line)
        ds = data["dataset"]
        if use_predicted:
            data["clauses"] = (
                data["clauses-predicted"]
                if "clauses-predicted" in data
                else data["clauses"]
            )
        stats[ds]["n_instances"] += 1
        stats[ds]["n_clauses"] += len(data["clauses"])
        n_clauses_with_stimulus = sum(
            1 for clause in data["clauses"] if any(tag in "BI" for token, tag in clause)
        )
        n_clauses_all_stimulus = sum(
            1 for clause in data["clauses"] if all(tag in "BI" for token, tag in clause)
        )
        n_clauses_pure = sum(
            # either all stimulus or none
            1
            for clause in data["clauses"]
            if len({tag.replace("I", "B") for token, tag in clause}) == 1
        )
        stats[ds]["n_clauses_with_stimulus"] += n_clauses_with_stimulus
        stats[ds]["n_clauses_all_stimulus"] += n_clauses_all_stimulus
        stats[ds]["n_clauses_pure"] += n_clauses_pure
        stats[ds]["n_instances_with_stimulus"] += int(bool(n_clauses_with_stimulus))
        tags = "".join(tag for token, tag in data["tokens"])
        lengths[ds]["stimuli"].extend(
            len(stimulus) for stimulus in re.findall("BI*", tags)
        )
        lengths[ds]["token_stimulus_ratio_per_instance"].append(
            len(tags.replace("O", "")) / len(data["tokens"])
        )
        lengths[ds]["token_stimulus_ratio_per_clause"].extend(
            sum(1 if tag in "BI" else 0 for _, tag in clause) / len(clause)
            for clause in data["clauses"]
        )

    print(HEADER)
    for ds in stats:
        stats[ds]["mean_length_stimulus"] = mean(lengths[ds]["stimuli"])
        stats[ds]["stdev_length_stimulus"] = stdev(lengths[ds]["stimuli"])
        stats[ds]["ds"] = transds[ds]
        stats[ds]["mean_clauses_per_instance"] = (
            stats[ds]["n_clauses"] / stats[ds]["n_instances"]
        )
        stats[ds]["mean_clauses_all_stimulus_per_instance"] = (
            stats[ds]["n_clauses_all_stimulus"] / stats[ds]["n_instances"]
        )
        stats[ds]["mean_clauses_pure_per_instance"] = (
            stats[ds]["n_clauses_pure"] / stats[ds]["n_instances"]
        )
        stats[ds]["mean_token_stimulus_ratio_per_clause"] = mean(
            lengths[ds]["token_stimulus_ratio_per_clause"]
        )
        stats[ds]["mean_token_stimulus_ratio_per_instance"] = mean(
            lengths[ds]["token_stimulus_ratio_per_instance"]
        )
        print(LINE.format(**stats[ds]))
    print(FOOTER)


if __name__ == "__main__":
    cli()
