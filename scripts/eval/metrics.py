import re
from functools import wraps

import click


def strip_modifiers(list_of_lists):
    return [
        [re.sub("-.*", "", word) for word in sentence] for sentence in list_of_lists
    ]


def stripped(metric):
    """Decorator to modify a metric to strip all -modifiers"""

    @wraps(metric)
    def wrapper(*args, **kwargs):
        return metric(*(strip_modifiers(arg) for arg in args), **kwargs)

    return wrapper


def segments(g, p, fix=False):
    if fix:
        g = "".join(g).replace("OI", "OB")
        if g[0] == "I":
            g = "B" + g[1:]
    g = list(g)[::-1]
    p = list(p)[::-1]
    gseg = []
    pseg = []
    assert len(g) == len(p)
    while g:
        G, P = g.pop(), p.pop()
        gseg.append(G)
        pseg.append(P)
        if g and (g[-1] == "B" or g[-1] == "O" and G != "O"):
            yield gseg, pseg
            gseg = []
            pseg = []
    yield gseg, pseg


@stripped
def metric1(gold, predicted):
    """ Partial match """
    res = {"fp": 0, "fn": 0, "tp": 0, "tn": 0}
    for gsent, psent in zip(gold, predicted):
        for gseg, pseg in segments(gsent, psent):
            if gseg[0] == "B":
                if "B" in pseg or "I" in pseg:
                    res["tp"] += 1
                else:
                    res["fn"] += 1
            else:
                if pseg == gseg:
                    res["tn"] += 1
                else:
                    res["fp"] += 1
    return res


@stripped
def metric2(gold, predicted):
    """ Exact match """
    res = {"fp": 0, "fn": 0, "tp": 0, "tn": 0}
    # Data comes as [["B", "I", "O"], ["O","O"], [..]...] * 2
    for gsent, psent in zip(gold, predicted):
        for gseg, pseg in segments(gsent, psent):
            if gseg[0] == "B":
                if gseg == pseg:
                    res["tp"] += 1
                else:
                    res["fn"] += 1
            else:
                if gseg == pseg:
                    res["tn"] += 1
                else:
                    res["fp"] += 1
    return res


@stripped
def metric3(gold, predicted):
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for gsent, psent in zip(gold, predicted):
        for gtok, ptok in zip(gsent, psent):
            pn = "p" if ptok == "B" else "n"
            tf = "f" if [ptok, gtok].count("B") % 2 else "t"
            counts[tf + pn] += 1
    return counts


@stripped
def metric4(gold, predicted):
    def inv(s):
        return list(re.sub("B(I*)", r"\1B", "".join(s))[::-1])

    return metric3(map(inv, gold), map(inv, predicted))


def f1_prec_recall(preds):
    try:
        precision = preds["tp"] / (preds["tp"] + preds["fp"])
        recall = preds["tp"] / (preds["tp"] + preds["fn"])
        f1 = 2 * ((precision * recall) / (precision + recall))
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1: {f1:.2f}")
    except ZeroDivisionError:
        print("The score is Zero")
    return preds


@stripped
def metric7(gold, predicted):
    """Evaluate labeling as being clause-classification on clause level"""
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for gclause, pclause in zip(gold, predicted):
        gold_binary = "B" in gclause or "I" in gclause
        predicted_binary = "B" in pclause or "I" in pclause

        if predicted_binary:
            if gold_binary:
                counts["tp"] += 1
            else:
                counts["fp"] += 1
        else:
            if gold_binary:
                counts["fn"] += 1
            else:
                counts["tn"] += 1
    return counts


@stripped
def metric8(gold, predicted):
    """ Beginning match based on gold segments """
    res = {"fp": 0, "fn": 0, "tp": 0, "tn": 0}
    # Data comes as [["B", "I", "O"], ["O","O"], [..]...] * 2
    for gsent, psent in zip(gold, predicted):
        for gseg, pseg in segments(gsent, psent):
            if gseg[0] == "B":
                if pseg[0] == "B":
                    res["tp"] += 1
                else:
                    res["fn"] += 1
            else:
                if pseg[0] == "B":
                    res["fp"] += 1
                else:
                    res["tn"] += 1
    return res


@stripped
def metric9(gold, predicted):
    def inv(s):
        return list(re.sub("B(I*)", r"\1B", "".join(s))[::-1])

    return metric9(map(inv, gold), map(inv, predicted))


@stripped
def metric10(gold, predicted):
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for gsent, psent in zip(gold, predicted):
        for gtok, ptok in zip(gsent, psent):
            counts["tp" if gtok == ptok else "fp"] += 1
    print("Warning: Precision is Accuracy, ignore all else")
    return counts


from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


@stripped
def metric11(gold, predicted):
    print(classification_report(gold, predicted))


from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    classification_report,
)


@stripped
def metric12(gold, predicted):
    from itertools import chain

    gold = list(chain.from_iterable(gold))
    predicted = list(chain.from_iterable(predicted))
    print(precision_recall_fscore_support(gold, predicted))


@click.command()
@click.option("--metric", type=int)
@click.argument("file", type=click.File("r"))
def cli(file, metric):
    gold, pred = [], []

    gclause, pclause = [], []
    for line in file:
        if not line.strip():
            if gclause or pclause:
                gold.append(gclause)
                pred.append(pclause)
            gclause, pclause = [], []
            continue
        _, giob, piob = line.strip().split("\t")
        gclause.append(giob)
        pclause.append(piob)

    print(f1_prec_recall(globals()[f"metric{metric}"](gold, pred)))


if __name__ == "__main__":
    cli()
