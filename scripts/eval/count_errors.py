import re
import pickle
from collections import Counter
import click
from metrics import strip_modifiers, stripped, segments


@stripped
def error_classes(gold, predicted):
    cases = Counter()
    for gsent, psent in zip(gold, predicted):
        gsegs, psegs = zip(*segments(gsent, psent, fix=True))
        seg_len = len(gsegs)
        for i, (gseg, pseg) in enumerate(zip(gsegs, psegs)):
            pstr = "".join(pseg)
            if gseg[0] == "O":
                if set(pseg) == {"O"}:
                    cases["IX"] += 1
                    continue
                if re.match(r"BI*O+", pstr) or pseg[0] == pseg[-1] == "O":
                    cases["X"] += 1  # False positive
                continue
            assert gseg[0] == "B", gseg
            if re.match(r"BI*$", pstr):
                # supposedly correct, but let's check the next segment
                if i < seg_len - 1 and psegs[i + 1][0] == "I":
                    cases["II"] += 1  # late stop
                else:
                    cases["N"] += 1
                continue
            if re.match(r"BI*O+$", pstr):
                cases["I"] += 1  # early stop
                continue
            if re.match(r"I+", pstr):
                if set(pseg) == {"I"}:
                    if i < seg_len - 1 and psegs[i + 1][0] == "I":
                        cases["V"] += 1  # surround
                    else:
                        cases["IV"] += 1  # early start
                    continue
                if re.match(r"I+O+$", pstr):
                    cases["III"] += 1  # early start and stop
                continue
            if re.match(r"OBI*O$", pstr):
                cases["VI"] += 1  # contained
                continue
            if re.match(r"OBI*$", pstr):
                if i < seg_len - 1 and psegs[i + 1][0] == "I":
                    cases["VIII"] += 1  # late start and stop
                else:
                    cases["VII"] += 1  # late start
                continue
            if set(pstr) == {"O"}:
                cases["XII"] += 1  # false negative
            cases["XI"] += 1  # weird, probably multiple
    return cases


@stripped
def premature_stop(gold, predicted):
    """ Beginning match correct, end not (too early) (based on gold segments) """
    # I
    errors = 0
    # Data comes as [["B", "I", "O"], ["O","O"], [..]...] * 2
    for gsent, psent in zip(gold, predicted):
        for gseg, pseg in segments(gsent, psent):
            if gseg[0] == "B" and re.match(r"BI*O+$", "".join(pseg)):
                errors += 1
    return errors


@stripped
def late_start(gold, predicted):
    """ Right match correct, left match not (based on gold segments)"""
    # VII
    errors = 0
    # Data comes as [["B", "I", "O"], ["O","O"], [..]...] * 2
    for gsent, psent in zip(gold, predicted):
        for gseg, pseg in segments(gsent, psent):
            if gseg[0] == "B" and re.match(r"O+BI*$", "".join(pseg)):
                errors += 1
    return errors


@stripped
def late_stop(gold, predicted):
    """ Left match correct, right match not (too long) (based on gold segments)"""
    errors = 0
    # Data comes as [["B", "I", "O"], ["O","O"], [..]...] * 2
    for gsent, psent in zip(gold, predicted):
        gsegs, psegs = zip(*segments(gsent, psent))
        if len(gsegs) < 2:
            continue
        for i, (gseg, pseg) in enumerate(zip(gsegs[:-1], psegs[:-1])):
            if gseg == pseg and gseg[0] == "B" and psegs[i + 1] == "I":
                errors += 1
    return errors


@click.command()
@click.argument("path", type=click.Path())
def cli(path):
    gold, pred = [], []
    if path.endswith("pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            gold = data["gold_labels"]
            pred = data["predicted_labels"]
    else:
        with open(path) as file:
            data = file.read().split("\n\n\n")
            for sent in data:
                gclause, pclause = [], []
                for line in sent.split("\n"):
                    if not line.strip():
                        continue
                    _, giob, piob = line.strip().split("\t")
                    gclause.append(giob)
                    pclause.append(piob)
                if gclause or pclause:
                    gold.append(gclause)
                    pred.append(pclause)

    print(error_classes(gold, pred))


if __name__ == "__main__":
    cli()
