import json
import fileinput

from collections import Counter, defaultdict

import click
from sklearn.metrics import precision_recall_fscore_support


# dataset -> begin/end/both -> correct/incorrect -> int
counts = defaultdict(lambda: defaultdict(Counter))

ytrue, ypred = defaultdict(list), defaultdict(list)


def is_span_begin(clauses, i, j):
    return clauses[i][j][1] == "B"


def is_span_end(clauses, i, j):
    cur = clauses[i][j][1]
    if i == len(clauses) - 1 and j == len(clauses[i]) - 1:
        return cur in "BI"
    if j == len(clauses[i]) - 1:
        nex = clauses[i + 1][0][1]
    else:
        nex = clauses[i][j + 1][1]
    return cur in "BI" and nex in "BO"


def is_span_both(clauses, i, j):
    cur = clauses[i][j][1]
    if i == len(clauses) - 1 and j == len(clauses[i]) - 1:
        return cur in "BI"
    if j == len(clauses[i]) - 1:
        nex = clauses[i + 1][0][1]
    else:
        nex = clauses[i][j + 1][1]
    return (
        (cur in "BI" and nex in "BO")  # next thing ends current span
        or (
            cur == "O" and nex == "B"
        )  # next thing starts span after our outsideness
        or (cur == "B" and i == j == 0)  # special case: start starts span
    )


def add_counts_of_clauses(counts, clauses):
    for i, clause in enumerate(clauses):
        for j, pair in enumerate(clause):
            _, tag = pair
            s_begin = is_span_begin(clauses, i, j)
            s_end = is_span_end(clauses, i, j)
            s_both = is_span_both(clauses, i, j)
            c_begin = j == 0
            c_end = j == len(clause) - 1
            c_both = c_end or i == j == 0
            if s_begin:
                if c_begin:
                    counts["begin"]["correct"] += 1
                else:
                    counts["begin"]["incorrect"] += 1
            if s_end:
                if c_end:
                    counts["end"]["correct"] += 1
                else:
                    counts["end"]["incorrect"] += 1
            if s_both:
                if c_both:
                    counts["both"]["correct"] += 1
                else:
                    counts["both"]["incorrect"] += 1


def Bs_and_Is(clauses):
    for clause in clauses:
        for i, _ in enumerate(clause):
            if i == 0:
                yield "B"
            else:
                yield "I"


def change_IOB_from_clauses(clauses, reference):
    ref_IBs = Bs_and_Is(reference)
    for clause in clauses:
        for pair in clause:
            pair[1] = next(ref_IBs)

    return clauses


def add_ftpn(ds, manual, automatic):
    for man, auto in zip(Bs_and_Is(manual), Bs_and_Is(automatic)):
        ytrue[ds].append(man)
        ypred[ds].append(auto)


@click.command()
@click.argument("file", type=click.File("r"))
@click.option("--manual", is_flag=True)
@click.option("--compare-with-extracted-clauses", is_flag=True)
@click.option("--prf-extracted-annotated", is_flag=True)
def cli(file, manual, compare_with_extracted_clauses, prf_extracted_annotated):
    key = "clauses-manual" if manual else "clauses"
    # key = "clauses-predicted" if manual else "clauses"

    if compare_with_extracted_clauses and not manual:
        raise click.BadOptionUsage(
            "Can't use --compare-with-extracted-clauses without --manual"
        )

    for line in file:
        data = json.loads(line)
        if manual and key not in data:
            continue
        if prf_extracted_annotated:
            if "clauses-manual" in data:
                add_ftpn(
                    data["dataset"],
                    data["clauses-manual"],
                    data["clauses"]
                    if "clauses-predicted" not in data
                    else data["clauses-predicted"],
                )
            continue

        if compare_with_extracted_clauses:
            clauses = change_IOB_from_clauses(
                data["clauses-manual"],
                data["clauses"]
                if "clauses-predicted" not in data
                else data["clauses-predicted"],
            )
        else:
            clauses = data[key]
        add_counts_of_clauses(counts[data["dataset"]], clauses)

    if prf_extracted_annotated:
        for dataset in ytrue:
            p, r, f, _ = precision_recall_fscore_support(ytrue[dataset], ypred[dataset], average="macro")
            print(dataset, f"P: {p:.2f}, R: {r:.2f}, F: {f:.2f}")
        return

    for dataset in counts:
        for type in counts[dataset]:
            print(
                dataset,
                type,
                "{:.2f}".format(
                    counts[dataset][type]["correct"]
                    / (
                        counts[dataset][type]["correct"]
                        + counts[dataset][type]["incorrect"]
                    )
                ),
            )


if __name__ == "__main__":
    cli()
