import json
import click
from sklearn.metrics import classification_report


@click.command()
@click.argument("dataset")
def cli(dataset):
    with open(f"workdata/predictions/icc/{dataset}_predictions") as f:
        y_true, y_pred = [], []
        for line in f:
            data = json.loads(line)
            y_pred.append(data["predicted"].strip())

    with open(f"workdata/icc/{dataset}.test") as f:
        for line in f:
            text, label = line.split("\t")
            y_true.append(label.strip())

    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    cli()
