import pickle
import click
import metrics
from count_errors import error_classes

g_s, g_e = ("\x1b[38;5;220m", "\x1b[39m")
p_s, p_e = ("\x1b[4m", "\x1b[24m")


def print_sentence(text, gold, predicted, annotation=None):
    gold_state = "O"
    predicted_state = "O"
    if annotation:
        try:
            padding = text.index("ENDPAD")
        except ValueError:
            padding = len(text)
        result = annotation([gold[:padding]], [predicted[:padding]])
        print("-".join(sorted(k for k in result.keys() if result[k])), end=": ")
        # print("{}{}{}".format(
        #     "FP" if result["fp"] else "  ",
        #     "FN" if result["fn"] else "  ",
        #     "TP" if result["tp"] else "  ",
        # ), end=" ")
    for i, (word, gtok, ptok) in enumerate(zip(text, gold, predicted)):
        if word == "ENDPAD":
            break
        endings = "{}{}".format(
            g_e if gtok == "O" and gold_state != "O" else "",
            p_e if ptok == "O" and predicted_state != "O" else "",
        )
        whitespace = " " if i else ""
        beginnings = "{}{}".format(
            g_s if gtok.startswith("B") else "", p_s if ptok.startswith("B") else "",
        )
        print(endings, whitespace, beginnings, word, sep="", end="")
        gold_state, predicted_state = gtok, ptok
    print(g_e, p_e)
    print()


def load_textual(fh):
    # warning: fh is binary
    newlines = 0
    t, g, p = [], [], []
    for line in fh:
        line = line.decode("utf-8").strip()
        if not line:
            newlines += 1
            if newlines > 1:
                yield t, g, p
                t, g, p = [], [], []
                newlines = 0
            continue
        newlines = 0
        word, gold, pred = line.split("\t")
        t.append(word)
        g.append(gold)
        p.append(pred)
    if t or g or p:
        yield t, g, p


@click.command()
@click.argument("pkl", type=click.File("rb"))
@click.option(
    "--annotate", "-a", type=click.Choice(["partial", "exact", "p", "e", "cases"]),
)
@click.option("--mode", type=click.Choice(["pkl", "txt"]))
def cli(pkl, annotate, mode="pkl"):
    annotate = (
        annotate
        and {"p": metrics.metric1, "e": metrics.metric2, "c": error_classes,}[
            annotate[0]
        ]
    )
    if mode == "pkl":
        data = pickle.load(pkl)
        # word, gold predicted
        for T, G, P in zip(
            data["test_sentences"], data["gold_labels"], data["predicted_labels"],
        ):
            print_sentence([data["idx2word"][i] for i in T], G, P, annotation=annotate)
    else:
        for T, G, P in load_textual(pkl):
            print_sentence(T, G, P, annotation=annotate)


if __name__ == "__main__":
    cli()
